#!/bin/bash

# URLs to the pretrained weights
WEIGHTS_URL1="https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"
WEIGHTS_URL2="https://huggingface.co/imageomics/BGNN-trait-segmentation/resolve/main/se_resnext50_32x4d-a260b3a4.pth"

# Local paths to save the weights
WEIGHTS_PATH1="pretrained.pth"
WEIGHTS_PATH2="se_resnext50_32x4d-a260b3a4.pth"

# Function to download weights
download_weights() {
    local url=$1
    local path=$2
    echo "Downloading weights from $url..."
    
    # Use curl or wget to download the file
    if command -v curl &> /dev/null; then
        curl -L -o "$path" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$path" "$url"
    else
        echo "Error: Neither curl nor wget is installed."
        exit 1
    fi
    
    echo "Weights saved to $path"
}

# Create cache directory structure
mkdir -p /root/.cache/torch/hub/checkpoints/

# Execute the download functions
download_weights "$WEIGHTS_URL1" "$WEIGHTS_PATH1"
download_weights "$WEIGHTS_URL2" "$WEIGHTS_PATH2"

# Move the second weights file to the torch hub directory
mv "$WEIGHTS_PATH2" "/root/.cache/torch/hub/checkpoints/"
