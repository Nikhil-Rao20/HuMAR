#!/bin/bash
# Process both train and validation JSON files to add scribble annotations
# Usage: ./run_scribble_generation.sh

echo "========================================"
echo "Generating Scribbles for CrowdPose Dataset"
echo "========================================"

# Set paths
INPUT_DIR="CrowdPose_HumarProcessed"
OUTPUT_DIR="CrowdPose_HumarProcessed"

# Process training set
echo ""
echo "Processing RefHuman_train.json..."
python generate_scribbles.py \
    --input "${INPUT_DIR}/RefHuman_train.json" \
    --output "${OUTPUT_DIR}/RefHuman_train.json" \
    --coverage 0.03 \
    --dilation 2

# Process validation set  
echo ""
echo "Processing RefHuman_val.json..."
python generate_scribbles.py \
    --input "${INPUT_DIR}/RefHuman_val.json" \
    --output "${OUTPUT_DIR}/RefHuman_val.json" \
    --coverage 0.03 \
    --dilation 2

echo ""
echo "========================================"
echo "Scribble generation complete!"
echo "========================================"
