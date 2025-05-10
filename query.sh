target/release/image-embedder --model-path models/clip-vit-h14 search  --embeddings-json embeddings.json --top-k 1 --query "$1" | awk '{$1=""; sub(/^ /, ""); printf "%s\0", $0 }' | xargs -0 open
