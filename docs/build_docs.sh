#!/bin/bash

# Define paths (relative to the script location)
SOURCE_DIR="source"          # Sphinx source directory
BUILD_DIR="build/html"       # Directory where Sphinx generates HTML output
DEPLOY_DIR="."               # Root of 'docs/' for deployment

# Ensure script runs from the correct directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Step 1: Clean previous builds
echo "Cleaning previous builds..."
if [ -f "Makefile" ]; then
    make clean
else
    echo "Error: Makefile not found in current directory. Exiting."
    exit 1
fi

# Step 2: Build the documentation
echo "Building documentation..."
make html
if [ $? -ne 0 ]; then
    echo "Error: Documentation build failed. Exiting."
    exit 1
fi

# Step 3: Copy the built files to the deployment directory
echo "Copying built files to the deployment directory..."
if [ -d "$BUILD_DIR" ]; then
    cp -r "$BUILD_DIR"/* "$DEPLOY_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy built files. Exiting."
        exit 1
    fi
else
    echo "Error: Build directory '$BUILD_DIR' not found. Exiting."
    exit 1
fi

# Step 4: Confirmation message
echo "Documentation successfully built and deployed!"
