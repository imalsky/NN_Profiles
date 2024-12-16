#!/bin/bash

# Define paths
SOURCE_DIR="source"          # Sphinx source directory
BUILD_DIR="build/html"       # Directory where Sphinx generates HTML output
DEPLOY_DIR="."               # Deployment directory (root of docs/)

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
    cp -r $BUILD_DIR/* $DEPLOY_DIR
else
    echo "Error: Build directory '$BUILD_DIR' not found. Exiting."
    exit 1
fi

# Step 4: Remove any residual 'build/' directory to prevent it from being pushed
echo "Cleaning up build directory..."
rm -rf build/

echo "Documentation successfully built and deployed!"
