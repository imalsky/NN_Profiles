#!/bin/bash

# Ensure script runs from the correct directory
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# Clean previous builds
echo "Cleaning previous builds..."
make clean || { echo "Error: Makefile not found or clean failed."; exit 1; }

# Build the documentation
echo "Building documentation..."
make html || { echo "Error: Documentation build failed."; exit 1; }

# Deploy the built files
echo "Deploying documentation..."
rm -rf index.html _static _sources genindex.html search.html searchindex.js
cp -r build/html/* .

echo "Documentation successfully built and deployed!"
