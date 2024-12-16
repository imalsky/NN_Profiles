#!/bin/bash

# Ensure script runs from the correct directory
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# Clean previous builds
echo "Cleaning previous builds..."
make clean || { echo "Error: Makefile not found or clean failed."; exit 1; }

# Build the documentation
echo "Building documentation..."
make html || { echo "Error: Documentation build failed."; exit 1; }

# Copy new built files to the deployment directory
echo "Deploying documentation..."
rm -rf index.html _static _sources genindex.html search.html searchindex.js  # Clean root first
cp -r build/html/* .  # Copy built files

# Remove the build/ directory to keep the root clean
echo "Cleaning up build directory..."
rm -rf build/

echo "Documentation successfully built and deployed!"
