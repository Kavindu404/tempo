import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import argparse
from tqdm import tqdm

def visualize_annotation(img, annotations, save_path=None):
    """Visualize the image with annotations"""
    plt.figure(figsize=(10, 10))
    
    # Display the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Add annotations
    for ann in annotations:
        # Draw polygons
        for segm in ann['segmentation']:
            pts = np.array(segm).reshape(-1, 2)
            plt.plot(pts[:, 0], pts[:, 1], '-', color='red', linewidth=2)
            
            # Draw first point as a marker to show direction
            plt.plot(pts[0, 0], pts[0, 1], 'o', color='green', markersize=6)
        
        # Draw bounding box
        x, y, w, h = ann['bbox']
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize COCO annotations')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--annotation_file', type=str, required=True, help='COCO annotation file path')
    parser.add_argument('--output_dir', type=str, help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Load COCO annotations
    coco = COCO(args.annotation_file)
    img_ids = coco.getImgIds()
    
    # Limit samples if specified
    if args.num_samples and args.num_samples < len(img_ids):
        img_ids = np.random.choice(img_ids, args.num_samples, replace=False)
    
    print(f"Visualizing {len(img_ids)} images...")
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load image
        img_path = os.path.join(args.image_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error loading image {img_path}")
            continue
        
        # Visualize
        if args.output_dir:
            save_path = os.path.join(args.output_dir, f"{os.path.splitext(img_info['file_name'])[0]}_viz.png")
            visualize_annotation(img, anns, save_path)
        else:
            visualize_annotation(img, anns)
            plt.pause(1)  # Pause to allow visualization

if __name__ == "__main__":
    main()
