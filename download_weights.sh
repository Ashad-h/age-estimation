#!/bin/bash

# URL to the pretrained weights
WEIGHTS_URL="https://huggingface.co/public-data/yu4u-age-estimation-pytorch/resolve/main/pretrained.pth"

# Local path to save the weights
WEIGHTS_PATH="pretrained.pth"

# Function to download the weights
download_weights() {
    echo "Downloading weights from $WEIGHTS_URL..."
    
    # Use curl or wget to download the file
    if command -v curl &> /dev/null; then
        curl -L -o "$WEIGHTS_PATH" "$WEIGHTS_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$WEIGHTS_PATH" "$WEIGHTS_URL"
    else
        echo "Error: Neither curl nor wget is installed."
        exit 1
    fi
    
    echo "Weights saved to $WEIGHTS_PATH"
}

# Execute the download function
download_weights