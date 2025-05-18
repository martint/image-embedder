use std::fs;
use std::path::{Path, PathBuf};
use std::thread;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::clip::{ClipConfig, ClipModel};
use candle_transformers::models::clip::text_model::{Activation, ClipTextConfig};
use candle_transformers::models::clip::vision_model::ClipVisionConfig;
use clap::{arg, Parser, Subcommand};
use crossbeam_channel::bounded;
use image::io::Reader as ImageReader;
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use walkdir::WalkDir;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
    #[arg(long)]
    model_path: PathBuf,
}

#[derive(Subcommand)]
enum Command {
    Embed {
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        output_json: PathBuf,
    },
    Search {
        #[arg(long)]
        query: String,
        #[arg(long)]
        embeddings_json: PathBuf,
        #[arg(long)]
        top_k: usize,
    },
}

#[derive(Serialize, Deserialize)]
struct ImageEmbedding {
    path: String,
    embedding: Vec<f32>,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-8)
}

fn load_image(path: &Path) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let image = ImageReader::open(path)?.decode()?.to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    Ok(resized)
}

fn compute_image_embedding(device: &Device, model: &ClipModel, image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Tensor> {
    let data: Vec<f32> = image
        .pixels()
        .flat_map(|p| p.0.iter().map(|&c| c as f32 / 255.0))
        .collect();

    let tensor = Tensor::from_vec(data, (1, 224, 224, 3), &device)?.permute((0, 3, 1, 2))?;
    let embed = model.get_image_features(&tensor)?;
    Ok(embed)
}

fn collect_images(dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| {
                    matches!(
                        s.to_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "nef" | "heic"
                    )
                })
                .unwrap_or(false)
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = {
        #[cfg(target_os = "macos")]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }

        #[cfg(target_os = "linux")]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Device::Cpu
        }
    };
    let device = &device;

    eprintln!("Using device: {:?}", device.location());

    let mut model_path = PathBuf::from(&args.model_path);
    model_path.push("model.safetensors");

    let config = ClipConfig {
        text_config: ClipTextConfig {
            vocab_size: 49408,
            embed_dim: 1024,
            activation: Activation::QuickGelu,
            intermediate_size: 4096,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            projection_dim: 1024,
        },
        vision_config: ClipVisionConfig {
            embed_dim: 1280,
            activation: Activation::QuickGelu,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            projection_dim: 1024,
            num_channels: 3,
            image_size: 224,
            patch_size: 14,
        },
        logit_scale_init_value: 2.6592,
        image_size: 224,
    };

    let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(model_path)? };

    let var_map = {
        let map = VarMap::new();
        {
            let mut vars = map.data().lock().unwrap();
            for (name, _) in safetensors.tensors() {
                let tensor = safetensors.load(&name, &device)?;
                vars.insert(name.clone(), Var::from_tensor(&tensor)?);
            }
        }
        map
    };

    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model = ClipModel::new(vb, &config)?;

    match args.command {
        Command::Embed {
            input_dir,
            output_json,
        } => {
            let image_paths = collect_images(&input_dir);
            eprintln!("Found {} images to process", image_paths.len());

            // Create a bounded channel with appropriate buffer size
            let buffer_size = 10; 
            let (sender, receiver) = bounded::<(PathBuf, ImageBuffer<Rgb<u8>, Vec<u8>>)>(buffer_size);

            // Create a device for the embedding thread
            let thread_device = device.clone();
            let thread_model = model.clone();

            // Spawn a thread for computing embeddings (single-threaded)
            let embedding_thread = thread::spawn(move || {
                let mut results = Vec::new();
                while let Ok((path, image)) = receiver.recv() {
                    match compute_image_embedding(&thread_device, &thread_model, image) {
                        Ok(embed) => {
                            match embed.squeeze(0).and_then(|t| t.to_vec1()) {
                                Ok(embedding) => {
                                    results.push(ImageEmbedding {
                                        path: path.display().to_string(),
                                        embedding,
                                    });
                                    if results.len() % 10 == 0 {
                                        eprintln!("Processed {} images", results.len());
                                    }
                                }
                                Err(e) => eprintln!("Error processing embedding: {}", e),
                            }
                        }
                        Err(e) => eprintln!("Error computing embedding for {}: {}", path.display(), e),
                    }
                }
                results
            });

            let chunk_size = 50;
            image_paths.par_chunks(chunk_size).for_each_with(sender.clone(), |s, chunk| {
                for path in chunk {
                    match load_image(path) {
                        Ok(image) => {
                            if s.send((path.clone(), image)).is_err() {
                                eprintln!("Error sending image to embedding thread");
                                break;
                            }
                        }
                        Err(e) => eprintln!("Error loading image {}: {}", path.display(), e),
                    }
                }
            });

            // Close the sender to signal the embedding thread to finish
            drop(sender);

            let results = embedding_thread.join().map_err(|_| anyhow!("Embedding thread panicked"))?;

            eprintln!("Writing {} embeddings to {}", results.len(), output_json.display());
            fs::write(output_json, serde_json::to_string_pretty(&results)?)?;
        }
        Command::Search {
            query,
            embeddings_json,
            top_k,
        } => {
            let data = fs::read_to_string(embeddings_json)?;
            let embeddings: Vec<ImageEmbedding> = serde_json::from_str(&data)?;

            let mut tokenizer_path = PathBuf::from(&args.model_path);
            tokenizer_path.push("tokenizer.json");

            let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
                .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

            let encoding = tokenizer
                .encode(query, true)
                .map_err(|e| anyhow!("Failed to encode query: {}", e))?;

            let ids = encoding.get_ids();
            let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &device)?;
            let text_embed = model.get_text_features(&input)?;
            let query_vec = text_embed.squeeze(0)?.to_vec1::<f32>()?;

            let mut scored: Vec<_> = embeddings
                .iter()
                .map(|e| (cosine_similarity(&query_vec, &e.embedding), &e.path))
                .collect();

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for (score, path) in scored.into_iter().take(top_k) {
                println!("{:.4} {}", score, path);
            }
        }
    }

    Ok(())
}
