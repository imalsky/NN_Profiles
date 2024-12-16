#!/bin/bash

# Navigate to docs directory
cd docs

# Clean old builds
echo "Cleaning previous build..."
make clean

# Build the documentation
echo "Building new documentation..."
make html

# Copy the built files to the top level of docs/
echo "Copying built files to root of docs directory..."
cp -r _build/html/* .

echo "Documentation build complete."
