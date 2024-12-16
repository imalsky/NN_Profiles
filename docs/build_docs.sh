#!/bin/bash

# Define key directories
BUILD_DIR="build/html"    # Directory where Sphinx generates HTML output
SOURCE_DIR="source"       # Directory containing Sphinx source files
DEPLOY_DIR="."            # Root directory for GitHub Pages deployment (current directory)

# Step 1: Clean previous builds
if [ -f "Makefile" ]; then
    echo "Cleaning previous builds..."
    make clean
else
    echo "Error: Makefile not found in the current directory. Exiting."
    exit 1
fi

# Step 2: Build the documentation
echo "Building documentation..."
make html
if [ $? -ne 0 ]; then
    echo "Error: Build process failed. Exiting."
    exit 1
fi

# Step 3: Deploy build output to the root of the docs/ directory
if [ -d "$BUILD_DIR" ]; then
    echo "Copying built files to the deployment directory..."
    cp -r "$BUILD_DIR"/* "$DEPLOY_DIR"
else
    echo "Error: Build directory '$BUILD_DIR' not found. Exiting."
    exit 1
fi

echo "Documentation successfully built and deployed!"
