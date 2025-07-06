#!/usr/bin/env python3
"""
DINOv2 Multi-Label Classification Inference Script

This script performs inference on a COCO-formatted dataset using a trained DINOv2 model,
generates evaluation metrics, and optionally saves mismatched predictions with attention maps.

Usage:
    python dinov2_inference.py --input_json path/to/test.json --image_dir path/to/images 
                              --checkpoint path/to/model.pth --output_dir path/to/output
                              [--save_mismatched] [--threshold 0.5] [--batch_size 32]
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from datetime import datetime
import pandas as pd

# Import our modules (assuming they're in the same directory or installed)
from dataset import COCOMultiLabelDataset, create_transforms, collate_fn
from model import ProductionDINOv2MultiLabelClassifier, AttentionVisualizer
from utils import calculate_multilabel_metrics, print_detailed_metrics

class DINOv2Inference:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize the inference class
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.attention_visualizer = AttentionVisualizer()
        
        # Load model
        self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path):
        """Load the trained model from checkpoint"""
        print(f"Loading model from {checkpoint_path}...")
        
        try:
            self.model = ProductionDINOv2MultiLabelClassifier.from_lora_checkpoint(checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
            
            # Get number of classes
            self.num_classes = self.model.classifier.out_features
            print(f"Model has {self.num_classes} classes")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def run_inference(self, input_json, image_dir, output_dir, 
                     save_mismatched=False, threshold=0.5, batch_size=32):
        """
        Run complete inference pipeline
        
        Args:
            input_json: Path to COCO-formatted JSON file
            image_dir: Directory containing images
            output_dir: Directory to save results
            save_mismatched: Whether to save mismatched predictions with attention
            threshold: Threshold for binary classification
            batch_size: Batch size for inference
        """
        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        mismatched_dir = output_path / "mismatched_predictions"
        if save_mismatched:
            mismatched_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset
        print("Creating dataset...")
        dataset = COCOMultiLabelDataset(
            annotation_file=input_json,
            image_dir=image_dir,
            transform=create_transforms(is_training=False)
        )
        
        self.class_names = dataset.get_class_names()
        print(f"Dataset created with {len(dataset)} samples and {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        # Run inference
        print("Running inference...")
        results = self._inference_loop(dataloader, threshold, save_mismatched, mismatched_dir)
        
        # Generate and save reports
        print("Generating evaluation reports...")
        self._generate_reports(results, output_path, threshold)
        
        # Save detailed results
        self._save_detailed_results(results, output_path)
        
        print(f"✅ Inference completed. Results saved to {output_dir}")
        
        return results
    
    def _inference_loop(self, dataloader, threshold, save_mismatched, mismatched_dir):
        """Run inference loop and collect results"""
        all_predictions = []
        all_targets = []
        all_image_info = []
        mismatched_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Get predictions and attention if needed
                if save_mismatched:
                    logits, attention_weights = self.model(images, return_attention=True)
                else:
                    logits = self.model(images)
                    attention_weights = None
                
                predictions = torch.sigmoid(logits)
                binary_preds = (predictions > threshold).float()
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
                # Process each sample in the batch
                for i in range(images.size(0)):
                    sample_info = {
                        'image_id': batch['image_ids'][i],
                        'file_name': batch['file_names'][i],
                        'predictions': predictions[i].cpu(),
                        'targets': targets[i].cpu(),
                        'binary_predictions': binary_preds[i].cpu()
                    }
                    all_image_info.append(sample_info)
                    
                    # Check for mismatch and save if requested
                    if save_mismatched:
                        is_mismatch = not torch.equal(binary_preds[i], targets[i])
                        
                        if is_mismatch:
                            mismatched_count += 1
                            self._save_mismatched_sample(
                                images[i], 
                                attention_weights[i] if attention_weights is not None else None,
                                sample_info,
                                mismatched_dir,
                                batch['original_images'][i]
                            )
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx * images.size(0)} samples...")
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        print(f"Inference completed. Processed {len(all_image_info)} samples")
        if save_mismatched:
            print(f"Found {mismatched_count} mismatched predictions")
        
        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'image_info': all_image_info,
            'mismatched_count': mismatched_count if save_mismatched else None
        }
    
    def _save_mismatched_sample(self, image_tensor, attention_weights, sample_info, 
                               mismatched_dir, original_image):
        """Save mismatched sample with attention visualization"""
        try:
            file_name = sample_info['file_name']
            predictions = sample_info['predictions']
            targets = sample_info['targets']
            
            # Create filename
            base_name = Path(file_name).stem
            save_path = mismatched_dir / f"{base_name}_mismatch.png"
            
            # Create detailed visualization
            self._create_mismatch_visualization(
                image_tensor, attention_weights, predictions, targets, 
                save_path, file_name
            )
            
            # Save prediction scores as JSON
            scores_path = mismatched_dir / f"{base_name}_scores.json"
            scores_data = {
                'file_name': file_name,
                'image_id': sample_info['image_id'],
                'predictions': {
                    name: float(predictions[i]) 
                    for i, name in enumerate(self.class_names)
                },
                'targets': {
                    name: int(targets[i]) 
                    for i, name in enumerate(self.class_names)
                },
                'mismatched_classes': [
                    self.class_names[i] for i in range(len(self.class_names))
                    if sample_info['binary_predictions'][i] != targets[i]
                ]
            }
            
            with open(scores_path, 'w') as f:
                json.dump(scores_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save mismatched sample {file_name}: {e}")
    
    def _create_mismatch_visualization(self, image_tensor, attention_weights, 
                                     predictions, targets, save_path, file_name):
        """Create comprehensive visualization for mismatched predictions"""
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image_tensor * std + mean
        image = torch.clamp(image, 0, 1)
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_np)
        ax1.set_title(f'Original Image\n{file_name}', fontsize=10)
        ax1.axis('off')
        
        # Attention visualization if available
        if attention_weights is not None:
            try:
                attention_map = self.attention_visualizer.create_attention_heatmap(attention_weights)
                if attention_map is not None:
                    # Attention heatmap
                    ax2 = fig.add_subplot(gs[0, 1])
                    im = ax2.imshow(attention_map, cmap='hot')
                    ax2.set_title('Attention Map', fontsize=10)
                    ax2.axis('off')
                    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                    
                    # Attention overlay
                    ax3 = fig.add_subplot(gs[0, 2])
                    ax3.imshow(image_np)
                    ax3.imshow(attention_map, alpha=0.6, cmap='hot')
                    ax3.set_title('Attention Overlay', fontsize=10)
                    ax3.axis('off')
            except Exception as e:
                print(f"Warning: Could not create attention visualization: {e}")
        
        # Prediction vs Target comparison
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        # Create text summary
        summary_text = "Prediction vs Target Summary:\n\n"
        for i, class_name in enumerate(self.class_names):
            pred_score = predictions[i].item()
            target_val = int(targets[i].item())
            status = "✓" if (pred_score > 0.5) == target_val else "✗"
            summary_text += f"{status} {class_name[:15]:<15}: {pred_score:.3f} | {target_val}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        # Detailed bar charts
        # Predictions bar chart
        ax5 = fig.add_subplot(gs[1, :2])
        y_pos = np.arange(len(self.class_names))
        pred_scores = predictions.cpu().numpy()
        colors = ['green' if score > 0.5 else 'red' for score in pred_scores]
        
        bars = ax5.barh(y_pos, pred_scores, color=colors, alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([name[:20] for name in self.class_names], fontsize=8)
        ax5.set_xlabel('Prediction Score')
        ax5.set_title('Prediction Scores (Green: >0.5, Red: ≤0.5)')
        ax5.set_xlim(0, 1)
        ax5.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        
        # Add score labels on bars
        for bar, score in zip(bars, pred_scores):
            ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=7)
        
        # Ground truth
        ax6 = fig.add_subplot(gs[1, 2:])
        target_vals = targets.cpu().numpy()
        colors_gt = ['blue' if val == 1 else 'gray' for val in target_vals]
        
        bars_gt = ax6.barh(y_pos, target_vals, color=colors_gt, alpha=0.7)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels([name[:20] for name in self.class_names], fontsize=8)
        ax6.set_xlabel('Ground Truth')
        ax6.set_title('Ground Truth Labels (Blue: Positive, Gray: Negative)')
        ax6.set_xlim(0, 1)
        
        # Mismatch analysis
        ax7 = fig.add_subplot(gs[2, :])
        
        # Calculate mismatches
        binary_preds = (pred_scores > 0.5).astype(int)
        mismatches = binary_preds != target_vals
        mismatch_types = []
        
        for i in range(len(self.class_names)):
            if mismatches[i]:
                if target_vals[i] == 1:
                    mismatch_types.append(f"False Negative: {self.class_names[i]} (missed)")
                else:
                    mismatch_types.append(f"False Positive: {self.class_names[i]} (extra)")
        
        if mismatch_types:
            mismatch_text = "Mismatched Classifications:\n\n" + "\n".join(mismatch_types)
        else:
            mismatch_text = "No mismatches found (this shouldn't happen in mismatch visualization)"
        
        ax7.text(0.05, 0.95, mismatch_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
        ax7.axis('off')
        
        plt.suptitle(f'Mismatch Analysis: {file_name}', fontsize=14, fontweight='bold')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_reports(self, results, output_path, threshold):
        """Generate and save evaluation reports"""
        predictions = results['predictions']
        targets = results['targets']
        
        # Calculate comprehensive metrics
        metrics = calculate_multilabel_metrics(predictions, targets, self.class_names, threshold)
        
        # Print metrics
        print_detailed_metrics(metrics)
        
        # Save classification report
        binary_preds = (predictions > threshold).numpy().astype(int)
        targets_np = targets.numpy().astype(int)
        
        # Generate classification report for each class
        report_dict = {}
        for i, class_name in enumerate(self.class_names):
            try:
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, support = precision_recall_fscore_support(
                    targets_np[:, i], binary_preds[:, i], average='binary', zero_division=0
                )
                
                report_dict[class_name] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1-score': float(f1),
                    'support': int(support)
                }
            except:
                report_dict[class_name] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1-score': 0.0,
                    'support': 0
                }
        
        # Add overall metrics
        report_dict['overall'] = {
            'mAP': metrics['mAP'],
            'mAUC': metrics['mAUC'],
            'mF1': metrics['mF1'],
            'exact_match': metrics['exact_match'],
            'hamming_loss': metrics['hamming_loss'],
            'threshold_used': threshold
        }
        
        # Save classification report as JSON
        with open(output_path / 'classification_report.json', 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Save classification report as text
        with open(output_path / 'classification_report.txt', 'w') as f:
            f.write("MULTI-LABEL CLASSIFICATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Threshold used: {threshold}\n")
            f.write(f"Total samples: {len(predictions)}\n\n")
            
            f.write("Overall Metrics:\n")
            f.write(f"  Mean Average Precision (mAP): {metrics['mAP']:.4f}\n")
            f.write(f"  Mean AUC-ROC: {metrics['mAUC']:.4f}\n")
            f.write(f"  Mean F1-Score: {metrics['mF1']:.4f}\n")
            f.write(f"  Exact Match Ratio: {metrics['exact_match']:.4f}\n")
            f.write(f"  Hamming Loss: {metrics['hamming_loss']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-" * 75 + "\n")
            
            for class_name, metrics_dict in report_dict.items():
                if class_name != 'overall':
                    f.write(f"{class_name:<25} {metrics_dict['precision']:<10.4f} "
                           f"{metrics_dict['recall']:<10.4f} {metrics_dict['f1-score']:<10.4f} "
                           f"{metrics_dict['support']:<10}\n")
        
        # Generate and save confusion matrices
        self._save_confusion_matrices(binary_preds, targets_np, output_path)
        
        # Save metrics plots
        self._save_metrics_plots(metrics, output_path)
    
    def _save_confusion_matrices(self, predictions, targets, output_path):
        """Save confusion matrices for each class"""
        num_classes = len(self.class_names)
        
        # Calculate confusion matrices for each class
        cms = []
        for i in range(num_classes):
            cm = confusion_matrix(targets[:, i], predictions[:, i])
            cms.append(cm)
        
        # Plot all confusion matrices
        cols = 4
        rows = (num_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (class_name, cm) in enumerate(zip(self.class_names, cms)):
            row = i // cols
            col = i % cols
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[row, col])
            axes[row, col].set_title(f'{class_name}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(num_classes, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save individual confusion matrices as CSV
        cm_dir = output_path / 'confusion_matrices'
        cm_dir.mkdir(exist_ok=True)
        
        for i, (class_name, cm) in enumerate(zip(self.class_names, cms)):
            cm_df = pd.DataFrame(cm, 
                               index=['Actual_Negative', 'Actual_Positive'],
                               columns=['Pred_Negative', 'Pred_Positive'])
            cm_df.to_csv(cm_dir / f'{class_name}_confusion_matrix.csv')
    
    def _save_metrics_plots(self, metrics, output_path):
        """Save various metrics visualization plots"""
        # Per-class metrics bar plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        x_pos = np.arange(len(self.class_names))
        
        # Average Precision
        bars1 = axes[0].bar(x_pos, metrics['per_class_ap'], alpha=0.7, color='skyblue')
        axes[0].set_title('Average Precision by Class')
        axes[0].set_ylabel('Average Precision')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars1, metrics['per_class_ap']):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # AUC-ROC
        bars2 = axes[1].bar(x_pos, metrics['per_class_auc'], alpha=0.7, color='lightgreen')
        axes[1].set_title('AUC-ROC by Class')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        for bar, score in zip(bars2, metrics['per_class_auc']):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # F1 Score
        bars3 = axes[2].bar(x_pos, metrics['per_class_f1'], alpha=0.7, color='salmon')
        axes[2].set_title('F1-Score by Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        
        for bar, score in zip(bars3, metrics['per_class_f1']):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_detailed_results(self, results, output_path):
        """Save detailed results including all predictions and metadata"""
        predictions = results['predictions'].numpy()
        targets = results['targets'].numpy()
        image_info = results['image_info']
        
        # Create detailed results DataFrame
        detailed_data = []
        
        for i, info in enumerate(image_info):
            row = {
                'image_id': info['image_id'],
                'file_name': info['file_name'],
            }
            
            # Add predictions and targets for each class
            for j, class_name in enumerate(self.class_names):
                row[f'pred_{class_name}'] = float(predictions[i, j])
                row[f'target_{class_name}'] = int(targets[i, j])
                row[f'binary_pred_{class_name}'] = int(info['binary_predictions'][j])
            
            # Add summary metrics
            row['num_positive_targets'] = int(targets[i].sum())
            row['num_positive_predictions'] = int(info['binary_predictions'].sum())
            row['exact_match'] = int(torch.equal(info['binary_predictions'], info['targets']))
            
            detailed_data.append(row)
        
        # Save as CSV
        df = pd.DataFrame(detailed_data)
        df.to_csv(output_path / 'detailed_results.csv', index=False)
        
        # Save summary statistics
        summary_stats = {
            'total_samples': len(image_info),
            'mismatched_samples': results['mismatched_count'] if results['mismatched_count'] is not None else 'not_computed',
            'timestamp': datetime.now().isoformat(),
            'class_names': self.class_names,
            'model_info': {
                'num_classes': self.num_classes,
                'device': str(self.device)
            }
        }
        
        with open(output_path / 'inference_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"✅ Detailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='DINOv2 Multi-Label Classification Inference')
    
    # Required arguments
    parser.add_argument('--input_json', type=str, required=True,
                       help='Path to COCO-formatted JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save inference results')
    
    # Optional arguments
    parser.add_argument('--save_mismatched', action='store_true',
                       help='Save mismatched predictions with attention maps')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary classification (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on (default: cuda)')
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input JSON file not found: {args.input_json}")
    
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Create inference object
    print("Initializing DINOv2 inference...")
    inference = DINOv2Inference(args.checkpoint, args.device)
    
    # Run inference
    print(f"Starting inference on {args.input_json}...")
    results = inference.run_inference(
        input_json=args.input_json,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        save_mismatched=args.save_mismatched,
        threshold=args.threshold,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print(f"Total samples processed: {len(results['image_info'])}")
    if results['mismatched_count'] is not None:
        print(f"Mismatched predictions: {results['mismatched_count']}")
    print("Generated files:")
    print("  - classification_report.json")
    print("  - classification_report.txt") 
    print("  - confusion_matrices.png")
    print("  - per_class_metrics.png")
    print("  - detailed_results.csv")
    print("  - inference_summary.json")
    if args.save_mismatched:
        print("  - mismatched_predictions/ (folder with attention visualizations)")


if __name__ == "__main__":
    main()
