#!/bin/bash

set -e  # Exit on any error

echo "Starting LLaMA setup..."

# Remove existing llama.cpp directory if it exists
rm -rf llama.cpp

# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make clean
make

# Set proper permissions
chmod +x main
chmod 755 main

# Create models directory
cd ..
mkdir -p models

# Download LLaMA-2 7B GGUF model if it doesn't exist
if [ ! -f "models/llama-2-7b-chat.gguf" ]; then
    echo "Downloading LLaMA-2 7B GGUF model..."
    curl -L "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf" \
         -o models/llama-2-7b-chat.gguf
fi

# Set proper permissions for models directory
chmod -R 755 models

echo "Setup complete! Checking permissions:"
ls -l llama.cpp/main
ls -l models/llama-2-7b-chat.gguf
