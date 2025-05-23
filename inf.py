"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T
import pycocotools.coco as coco
import time

import numpy as np 
from PIL import Image, ImageDraw

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig
import random
from itertools import cycle
from src.data.transforms._transforms import PolyAffine

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Additional imports for batch inference
import glob
from pathlib import Path
from tqdm import tqdm
import json


def draw(im, labels, boxes, coords, scores, img_name, thrh=0.5):
    colors = ["#1F77B4", "#FF7F0E", "#2EA02C", "#D62827", "#9467BD", 
              "#8C564B", "#E377C2", "#7E7E7E", "#BCBD20", "#1ABECF"]
    np.random.shuffle(colors)
    colors = cycle(colors)

    # Convert the image to RGB mode if it's not already in that mode
    if im.mode != 'RGB':
        im = im.convert('RGB')

    scr = scores[0]
    mask = scr > thrh
    lab = labels[0][mask]
    coord = coords[0][mask]
    box = boxes[0][mask]
    scrs = scores[0][mask]

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for j, (b, c) in enumerate(zip(box, coord)):
        color = next(colors)
        draw_poly = c.reshape(-1).tolist()
        polygon = Polygon(np.array(draw_poly).reshape((-1, 2)), linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(polygon)

    plt.axis('off')  # Hide the axes
    plt.savefig(img_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)  # Close the figure after saving to free up memory


def batch_inference(
    config_path,
    checkpoint_path,
    input_folder,
    output_folder,
    num_images=100,
    device='cuda',
    threshold=0.5,
    seed=42
):
    """
    Perform batch inference on randomly selected images from a folder.
    
    Args:
        config_path (str): Path to the model configuration file
        checkpoint_path (str): Path to the model checkpoint (.pth file)
        input_folder (str): Path to folder containing input images
        output_folder (str): Path where annotated images will be saved
        num_images (int): Number of images to randomly select for inference
        device (str): Device to run inference on ('cuda' or 'cpu')
        threshold (float): Confidence threshold for predictions
        seed (int): Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of all image files in the input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(input_folder, ext)))
        all_images.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if len(all_images) == 0:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(all_images)} images in {input_folder}")
    
    # Randomly select images
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    print(f"Selected {len(selected_images)} images for inference")
    
    # Measure model loading time
    print("Loading model...")
    model_load_start = time.time()
    
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    cfg.model.load_state_dict(state)
    default_height, default_width = cfg.model.encoder.eval_spatial_size
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            input_sizes = torch.tensor([[images.shape[-1], images.shape[-2]]], device=images.device)
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes, input_sizes)
            return outputs
    
    model = Model().to(device)
    model.eval()
    
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.3f} seconds")
    
    # Prepare transforms
    def get_transform(w, h):
        scaleX = default_width / w
        scaleY = default_height / h
        scale = min(scaleX, scaleY)
        
        new_H = int(scale * h)
        new_W = int(scale * w)
        
        val_h = (default_height - new_H) // 2
        val_w = (default_width - new_W) // 2
        
        return T.Compose([
            T.Resize((new_H, new_W)),
            T.Pad(padding=(val_w, val_h, val_w, val_h)),
            T.ToTensor(),
        ])
    
    # Inference
    total_inference_time = 0
    inference_times = []
    first_inference_time = 0
    detailed_results = []
    
    print("Starting inference...")
    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(selected_images, desc="Processing images")):
            try:
                # Load image
                im_pil = Image.open(img_path).convert('RGB')
                w, h = im_pil.size
                orig_size = torch.tensor([w, h])[None].to(device)
                
                # Transform image
                transform = get_transform(w, h)
                im_data = transform(im_pil)[None].to(device)
                
                # Measure inference time
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                output = model(im_data, orig_size)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                inference_time = time.time() - start_time
                
                # Special handling for first inference (includes model warmup)
                if idx == 0:
                    first_inference_time = model_load_time + inference_time
                
                total_inference_time += inference_time
                inference_times.append(inference_time)
                
                # Get predictions
                labels, boxes, coords, scores = output
                
                # Filter by threshold
                scr = scores[0]
                mask = scr > threshold
                filtered_scores = scores[0][mask].cpu().numpy().tolist()
                num_objects = len(filtered_scores)
                
                # Save detailed information
                img_info = {
                    'image_name': os.path.basename(img_path),
                    'num_objects_detected': num_objects,
                    'scores': filtered_scores,
                    'inference_time': inference_time
                }
                detailed_results.append(img_info)
                
                # Save annotated image
                output_filename = os.path.join(output_folder, os.path.basename(img_path))
                draw(im_pil, labels, boxes, coords, scores, output_filename, thrh=threshold)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Calculate and log statistics
    avg_inference_time = total_inference_time / len(selected_images)
    min_time = min(inference_times)
    max_time = max(inference_times)
    
    print("\n" + "="*50)
    print("Inference Statistics:")
    print(f"Total images processed: {len(selected_images)}")
    print(f"Model loading time: {model_load_time:.3f} seconds")
    print(f"First inference time (including model loading): {first_inference_time:.3f} seconds")
    print(f"Total inference time: {total_inference_time:.3f} seconds")
    print(f"Average inference time per image: {avg_inference_time:.3f} seconds")
    print(f"Min inference time: {min_time:.3f} seconds")
    print(f"Max inference time: {max_time:.3f} seconds")
    print(f"FPS: {1/avg_inference_time:.2f}")
    print(f"Output saved to: {output_folder}")
    print("="*50)
    
    # Save summary statistics
    log_path = os.path.join(output_folder, 'inference_summary.txt')
    with open(log_path, 'w') as f:
        f.write("Inference Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Total images processed: {len(selected_images)}\n")
        f.write(f"Model loading time: {model_load_time:.3f} seconds\n")
        f.write(f"First inference time (including model loading): {first_inference_time:.3f} seconds\n")
        f.write(f"Total inference time: {total_inference_time:.3f} seconds\n")
        f.write(f"Average inference time per image: {avg_inference_time:.3f} seconds\n")
        f.write(f"Min inference time: {min_time:.3f} seconds\n")
        f.write(f"Max inference time: {max_time:.3f} seconds\n")
        f.write(f"FPS: {1/avg_inference_time:.2f}\n")
    
    # Save detailed results for each image
    detailed_log_path = os.path.join(output_folder, 'detailed_results.txt')
    with open(detailed_log_path, 'w') as f:
        f.write("Detailed Results for Each Image\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Image Name':<40} {'Objects':<10} {'Inference Time':<15} {'Scores'}\n")
        f.write("-" * 80 + "\n")
        
        for result in detailed_results:
            img_name = result['image_name']
            num_objects = result['num_objects_detected']
            inf_time = result['inference_time']
            scores_str = ', '.join([f"{s:.3f}" for s in result['scores']])
            
            f.write(f"{img_name:<40} {num_objects:<10} {inf_time:<15.3f} {scores_str}\n")
    
    # Also save as JSON for easier parsing
    json_path = os.path.join(output_folder, 'detailed_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'summary': {
                'total_images': len(selected_images),
                'model_loading_time': model_load_time,
                'first_inference_time': first_inference_time,
                'average_inference_time': avg_inference_time,
                'min_inference_time': min_time,
                'max_inference_time': max_time,
                'fps': 1/avg_inference_time,
                'threshold': threshold
            },
            'detailed_results': detailed_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to:")
    print(f"  - Summary: {log_path}")
    print(f"  - Detailed TXT: {detailed_log_path}")
    print(f"  - Detailed JSON: {json_path}")


def main(args):
    """main
    """
    if hasattr(args, 'batch') and args.batch:
        # Batch inference mode
        batch_inference(
            config_path=args.config,
            checkpoint_path=args.resume,
            input_folder=args.input,
            output_folder=args.output,
            num_images=args.num_images,
            device=args.device,
            threshold=args.threshold,
            seed=args.seed
        )
    else:
        # Single image inference mode (original functionality)
        cfg = YAMLConfig(args.config, resume=args.resume)

        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
            
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu') 
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('Only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)
        default_height,default_width = cfg.model.encoder.eval_spatial_size

        class Model(nn.Module):
            def __init__(self, ) -> None:
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
                
            def forward(self, images, orig_target_sizes):
                input_sizes = torch.tensor([[images.shape[-1], images.shape[-2]]], device=images.device)
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes,input_sizes)
                return outputs

        model = Model().to(args.device)

        file_path = args.input

        im_pil = Image.open(file_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        scaleX = default_width / w
        scaleY = default_height / h

        scale = scaleX if scaleX<scaleY else scaleY

        new_H = int(scale*h)
        new_W = int(scale*w)

        val_h = (default_height - new_H)//2
        val_w = (default_width - new_W)//2

        transforms = T.Compose([
            T.Resize((new_H,new_W)),
            T.Pad(padding=(val_w,val_h,val_w,val_h)),
            T.ToTensor(),
        ])

        im_data = transforms(im_pil)[None].to(args.device)

        output = model(im_data, orig_size)
        torch.cuda.synchronize()
        labels, boxes, coords, scores = output

        draw(im_pil, labels, boxes, coords,scores,"results.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-i', '--input', type=str, required=True)
    
    # Add batch inference arguments
    parser.add_argument('--batch', action='store_true', help='Enable batch inference mode')
    parser.add_argument('-o', '--output', type=str, help='Output folder for batch inference')
    parser.add_argument('-n', '--num-images', type=int, default=100, help='Number of images to process')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Validate arguments for batch mode
    if args.batch and not args.output:
        parser.error("--output is required when using --batch mode")
    
    main(args)
