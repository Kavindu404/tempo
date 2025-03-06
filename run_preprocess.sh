#!/bin/bash

# Set paths
INPUT_DIR="./dataset"
OUTPUT_DIR="./processed_dataset"
ANNOTATION_FILE="./dataset/annotations/640.json"

# Process training split
echo "Processing training split..."
python preprocess_custom_dataset.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --annotation_file $ANNOTATION_FILE \
    --split train

# Process validation split
echo "Processing validation split..."
python preprocess_custom_dataset.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --annotation_file $ANNOTATION_FILE \
    --split val

# Visualize some processed examples to verify
echo "Visualizing processed examples..."
python visualize_annotations.py \
    --image_dir "$OUTPUT_DIR/images/train" \
    --annotation_file "$OUTPUT_DIR/annotations/640.json" \
    --output_dir "$OUTPUT_DIR/visualizations" \
    --num_samples 5

echo "Preprocessing complete. Check $OUTPUT_DIR/visualizations for visual verification."
