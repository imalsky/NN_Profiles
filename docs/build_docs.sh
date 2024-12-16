#!/bin/bash

# Define paths
SOURCE_DIR="source"          # Sphinx source directory
BUILD_DIR="build/html"       # Directory where Sphinx generates HTML output
DEPLOY_DIR="."               # Deployment directory (root of docs/)

# Ensure script runs from the correct directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Step 1: Clean previous builds
echo "Cleaning previous builds..."
if [ -f "Makefile" ]; then
    make clean
else
    echo "Error: Makefile not found. Exiting."
    exit 1
fi

# Step 2: Build the documentation
echo "Building documentation..."
make html
if [ $? -ne 0 ]; then
    echo "Error: Build failed. Exiting."
    exit 1
fi

# Step 3: Remove old files from the root directory
echo "Removing old files..."
find "$DEPLOY_DIR" -maxdepth 1 ! -name 'source' ! -name 'Makefile' \
    ! -name 'make.bat' ! -name 'build_docs.sh' ! -name '.' -exec rm -rf {} +

# Step 4: Copy new built files to the deployment directory
echo "Copying new files..."
cp -r "$BUILD_DIR"/* "$DEPLOY_DIR"

echo "Documentation successfully built and deployed!"
