import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from vendor import EV_API
from contourformer_inference import ContourFormerInference


class SatellitePipeline:
    """
    Pipeline class that integrates satellite imagery acquisition with ContourFormer inference.
    """
    
    def __init__(self, config_path: str, model_checkpoint: str, device: str = 'cpu'):
        """
        Initialize the satellite imagery pipeline.
        
        Args:
            config_path: Path to the ContourFormer model configuration YAML file
            model_checkpoint: Path to the trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        print("Initializing Satellite Pipeline...")
        
        # Initialize the satellite imagery API
        self.imagery_api = EV_API()
        print("✓ Satellite imagery API initialized")
        
        # Initialize the ContourFormer inference model
        self.contourformer = ContourFormerInference(
            config_path=config_path,
            model_checkpoint=model_checkpoint,
            device=device
        )
        print("✓ ContourFormer inference model initialized")
        
        print("Satellite Pipeline ready!")
        
    def get_satellite_image(self, min_lat: float, min_lon: float, 
                          max_lat: float, max_lon: float) -> Image.Image:
        """
        Get satellite imagery for the given bounding box coordinates.
        
        Args:
            min_lat: Minimum latitude of the bounding box
            min_lon: Minimum longitude of the bounding box
            max_lat: Maximum latitude of the bounding box
            max_lon: Maximum longitude of the bounding box
            
        Returns:
            PIL Image of the satellite imagery
        """
        print(f"Fetching satellite imagery for bbox: ({min_lat}, {min_lon}) to ({max_lat}, {max_lon})")
        
        try:
            # Create bounding box tuple
            bbox = (min_lat, min_lon, max_lat, max_lon)
            
            # Use your EV_API to get the satellite image with bounding box
            image = self.imagery_api.get_image_by_bbox(bbox)
            
            if image is None:
                raise ValueError("Failed to retrieve satellite image from EV_API")
                
            # Ensure image is in RGB format
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
                
            print(f"✓ Retrieved satellite image of size: {image.size}")
            return image
            
        except Exception as e:
            print(f"✗ Error fetching satellite image: {e}")
            raise
    
    def process_bbox(self, min_lat: float, min_lon: float, 
                    max_lat: float, max_lon: float,
                    score_threshold: float = 0.5,
                    output_dir: str = "output",
                    save_original: bool = True,
                    base_filename: Optional[str] = None) -> Dict:
        """
        Complete pipeline: bbox coordinates -> satellite image -> ContourFormer -> annotated results.
        
        Args:
            min_lat: Minimum latitude of the bounding box
            min_lon: Minimum longitude of the bounding box
            max_lat: Maximum latitude of the bounding box
            max_lon: Maximum longitude of the bounding box
            score_threshold: Minimum confidence score for detections
            output_dir: Directory to save outputs
            save_original: Whether to save the original satellite image
            base_filename: Base filename for outputs (auto-generated if None)
            
        Returns:
            Dictionary containing all results and file paths
        """
        print(f"\n{'='*60}")
        print(f"Starting Satellite Pipeline")
        print(f"Bounding Box: ({min_lat:.6f}, {min_lon:.6f}) to ({max_lat:.6f}, {max_lon:.6f})")
        print(f"Score Threshold: {score_threshold}")
        print(f"{'='*60}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Generate base filename if not provided
        if base_filename is None:
            base_filename = f"satellite_{min_lat:.4f}_{min_lon:.4f}_to_{max_lat:.4f}_{max_lon:.4f}"
        
        try:
            # Step 1: Get satellite imagery
            print("\n[1/3] Acquiring satellite imagery...")
            satellite_image = self.get_satellite_image(min_lat, min_lon, max_lat, max_lon)
            
            # Save original image if requested
            original_path = None
            if save_original:
                original_path = output_path / f"{base_filename}_original.jpg"
                satellite_image.save(original_path, quality=95)
                print(f"✓ Original satellite image saved: {original_path}")
            
            # Step 2: Run ContourFormer inference
            print("\n[2/3] Running ContourFormer inference...")
            detection_results = self.contourformer.predict(
                image=satellite_image,
                score_threshold=score_threshold
            )
            
            num_detections = len(detection_results['labels'])
            print(f"✓ Found {num_detections} instances above threshold {score_threshold}")
            
            # Step 3: Create annotated visualization
            print("\n[3/3] Creating annotated visualization...")
            annotated_path = output_path / f"{base_filename}_annotated.jpg"
            annotated_image = self.contourformer.visualize_results(
                image=satellite_image,
                results=detection_results,
                output_path=str(annotated_path),
                show_boxes=True,
                show_masks=True,
                show_labels=True
            )
            
            # Step 4: Save results metadata
            print("\nSaving results metadata...")
            results_json = {
                'bbox': {
                    'min_lat': min_lat,
                    'min_lon': min_lon,
                    'max_lat': max_lat,
                    'max_lon': max_lon
                },
                'image_size': {
                    'width': satellite_image.size[0],
                    'height': satellite_image.size[1]
                },
                'model_config': {
                    'score_threshold': score_threshold,
                    'config_path': self.contourformer.config_path,
                    'checkpoint_path': self.contourformer.model_checkpoint
                },
                'num_detections': num_detections,
                'detections': []
            }
            
            # Add detection details
            for i, (label, box, score) in enumerate(zip(
                detection_results['labels'], 
                detection_results['boxes'], 
                detection_results['scores']
            )):
                detection = {
                    'id': i,
                    'class_id': int(label),
                    'confidence': float(score),
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3]),
                        'width': float(box[2] - box[0]),
                        'height': float(box[3] - box[1])
                    }
                }
                results_json['detections'].append(detection)
            
            # Save JSON results
            json_path = output_path / f"{base_filename}_results.json"
            with open(json_path, 'w') as f:
                json.dump(results_json, f, indent=2)
            print(f"✓ Results metadata saved: {json_path}")
            
            # Compile final results
            pipeline_results = {
                'success': True,
                'bbox': (min_lat, min_lon, max_lat, max_lon),
                'image_size': satellite_image.size,
                'num_detections': num_detections,
                'files': {
                    'original_image': str(original_path) if original_path else None,
                    'annotated_image': str(annotated_path),
                    'results_json': str(json_path)
                },
                'detection_results': detection_results,
                'results_summary': results_json
            }
            
            print(f"\n{'='*60}")
            print(f"✓ Pipeline completed successfully!")
            print(f"📁 Output directory: {output_path}")
            print(f"🖼️  Annotated image: {annotated_path.name}")
            print(f"📊 Results JSON: {json_path.name}")
            print(f"🎯 Detections found: {num_detections}")
            print(f"{'='*60}")
            
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            print(f"\n✗ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'bbox': (min_lat, min_lon, max_lat, max_lon),
                'files': {},
                'num_detections': 0
            }
    
    def batch_process(self, bbox_list: list, 
                     score_threshold: float = 0.5,
                     output_dir: str = "output") -> Dict:
        """
        Process multiple bounding boxes in batch.
        
        Args:
            bbox_list: List of tuples (min_lat, min_lon, max_lat, max_lon)
            score_threshold: Minimum confidence score for detections
            output_dir: Base output directory
            
        Returns:
            Dictionary containing batch processing results
        """
        print(f"\nStarting batch processing of {len(bbox_list)} bounding boxes...")
        
        batch_results = {
            'total_processed': len(bbox_list),
            'successful': 0,
            'failed': 0,
            'results': [],
            'summary': {
                'total_detections': 0,
                'avg_detections_per_image': 0
            }
        }
        
        for i, bbox in enumerate(bbox_list):
            min_lat, min_lon, max_lat, max_lon = bbox
            print(f"\nProcessing {i+1}/{len(bbox_list)}: {bbox}")
            
            # Create batch-specific output directory
            batch_output_dir = Path(output_dir) / f"batch_result_{i+1:03d}"
            
            try:
                result = self.process_bbox(
                    min_lat=min_lat,
                    min_lon=min_lon,
                    max_lat=max_lat,
                    max_lon=max_lon,
                    score_threshold=score_threshold,
                    output_dir=str(batch_output_dir),
                    base_filename=f"bbox_{i+1:03d}"
                )
                
                if result['success']:
                    batch_results['successful'] += 1
                    batch_results['summary']['total_detections'] += result['num_detections']
                else:
                    batch_results['failed'] += 1
                
                batch_results['results'].append(result)
                
            except Exception as e:
                print(f"✗ Failed to process bbox {i+1}: {e}")
                batch_results['failed'] += 1
                batch_results['results'].append({
                    'success': False,
                    'error': str(e),
                    'bbox': bbox
                })
        
        # Calculate summary statistics
        if batch_results['successful'] > 0:
            batch_results['summary']['avg_detections_per_image'] = \
                batch_results['summary']['total_detections'] / batch_results['successful']
        
        # Save batch summary
        summary_path = Path(output_dir) / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        print(f"\nBatch processing completed!")
        print(f"✓ Successful: {batch_results['successful']}")
        print(f"✗ Failed: {batch_results['failed']}")
        print(f"📊 Total detections: {batch_results['summary']['total_detections']}")
        print(f"📄 Batch summary saved: {summary_path}")
        
        return batch_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Satellite Imagery ContourFormer Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single bounding box
  python satellite_pipeline.py --config configs/contourformer_hgnetv2_b2_coco.yml \\
                                --checkpoint model.pth \\
                                --min-lat 37.7749 --min-lon -122.4194 \\
                                --max-lat 37.7849 --max-lon -122.4094
  
  # With custom parameters
  python satellite_pipeline.py --config configs/contourformer_hgnetv2_b2_coco.yml \\
                                --checkpoint model.pth \\
                                --min-lat 37.7749 --min-lon -122.4194 \\
                                --max-lat 37.7849 --max-lon -122.4094 \\
                                --score-threshold 0.7 --device cuda \\
                                --output-dir results --no-save-original
        """
    )
    
    # Required arguments
    parser.add_argument('--config', '-c', required=True,
                       help='Path to ContourFormer model configuration YAML file')
    parser.add_argument('--checkpoint', '-ckpt', required=True,
                       help='Path to trained model checkpoint')
    
    # Bounding box coordinates
    parser.add_argument('--min-lat', type=float, required=True,
                       help='Minimum latitude of the bounding box')
    parser.add_argument('--min-lon', type=float, required=True,
                       help='Minimum longitude of the bounding box')
    parser.add_argument('--max-lat', type=float, required=True,
                       help='Maximum latitude of the bounding box')
    parser.add_argument('--max-lon', type=float, required=True,
                       help='Maximum longitude of the bounding box')
    
    # Optional parameters
    parser.add_argument('--device', default='cpu',
                       help='Device for inference (cpu/cuda, default: cpu)')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                       help='Minimum confidence score threshold (default: 0.5)')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--base-filename', default=None,
                       help='Base filename for output files (auto-generated if not provided)')
    parser.add_argument('--no-save-original', action='store_true',
                       help='Do not save the original satellite image')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        print("Initializing Satellite Pipeline...")
        pipeline = SatellitePipeline(
            config_path=args.config,
            model_checkpoint=args.checkpoint,
            device=args.device
        )
        
        # Run pipeline
        results = pipeline.process_bbox(
            min_lat=args.min_lat,
            min_lon=args.min_lon,
            max_lat=args.max_lat,
            max_lon=args.max_lon,
            score_threshold=args.score_threshold,
            output_dir=args.output_dir,
            save_original=not args.no_save_original,
            base_filename=args.base_filename
        )
        
        if results['success']:
            print(f"\n