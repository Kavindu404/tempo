import json
import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
from pycocotools import mask as coco_mask
import cv2

class COCOMultiLabelDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None, min_bbox_area=100, 
                 noise_std=0.3, use_polygon_mask=True):
        """
        COCO dataset for multi-label classification with polygon masking
        
        Args:
            annotation_file: Path to COCO JSON file
            image_dir: Directory containing images
            transform: Image transforms
            min_bbox_area: Minimum bbox area to include
            noise_std: Standard deviation for Gaussian noise
            use_polygon_mask: Whether to use polygon masking
        """
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform
        self.min_bbox_area = min_bbox_area
        self.noise_std = noise_std
        self.use_polygon_mask = use_polygon_mask
        
        # Build mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Create category_id to index mapping
        self.category_ids = sorted(list(self.categories.keys()))
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.category_ids)}
        self.idx_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_idx.items()}
        
        # Create annotation samples (one per annotation, not per image)
        self.annotation_samples = self._create_annotation_samples()
        
        print(f"Created {len(self.annotation_samples)} annotation samples from {len(self.category_ids)} categories")
        self._print_dataset_stats()
    
    def _create_annotation_samples(self):
        """Create individual samples for each annotation"""
        annotation_samples = []
        
        for ann in self.coco_data['annotations']:
            # Check if image exists
            if ann['image_id'] not in self.images:
                continue
            
            # Check bbox area
            bbox = ann['bbox']
            area = bbox[2] * bbox[3]
            if area < self.min_bbox_area:
                continue
            
            # Check if category exists
            if ann['category_id'] not in self.categories:
                continue
            
            # Check if annotation has segmentation (for polygon masking)
            if self.use_polygon_mask and 'segmentation' not in ann:
                continue
            
            annotation_samples.append(ann)
        
        return annotation_samples
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        category_counts = defaultdict(int)
        
        for ann in self.annotation_samples:
            category_counts[ann['category_id']] += 1
        
        print(f"\nDataset Statistics:")
        print(f"Total annotation samples: {len(self.annotation_samples)}")
        
        print("\nCategory distribution:")
        for cat_id in self.category_ids:
            cat_name = self.categories[cat_id]['name']
            count = category_counts[cat_id]
            print(f"  {cat_name}: {count} samples")
    
    def _create_polygon_mask(self, image_shape, segmentation):
        """Create polygon mask from COCO segmentation"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        if isinstance(segmentation, list):
            # Polygon format
            for seg in segmentation:
                if len(seg) >= 6:  # At least 3 points (x,y pairs)
                    # Convert to polygon points
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], 1)
        else:
            # RLE format
            if isinstance(segmentation, dict):
                rle = segmentation
            else:
                rle = {'size': image_shape[:2], 'counts': segmentation}
            mask = coco_mask.decode(rle)
        
        return mask.astype(bool)
    
    def _create_masked_crop(self, image, annotation):
        """Create masked crop with Gaussian noise background"""
        image_np = np.array(image)
        
        if self.use_polygon_mask and 'segmentation' in annotation:
            # Create polygon mask
            mask = self._create_polygon_mask(image_np.shape, annotation['segmentation'])
            
            # Create Gaussian noise image with same shape
            noise_image = np.random.normal(
                loc=128,  # Gray background
                scale=self.noise_std * 255,
                size=image_np.shape
            ).astype(np.uint8)
            
            # Clip values to valid range
            noise_image = np.clip(noise_image, 0, 255)
            
            # Copy polygon area from original to noise image
            masked_image = noise_image.copy()
            masked_image[mask] = image_np[mask]
            
            # Convert back to PIL Image
            masked_image_pil = Image.fromarray(masked_image)
        else:
            # Fallback: use original image
            masked_image_pil = image
        
        # Crop bbox
        bbox = annotation['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        cropped_image = masked_image_pil.crop((x, y, x + w, y + h))
        
        return cropped_image
    
    def __len__(self):
        return len(self.annotation_samples)
    
    def __getitem__(self, idx):
        annotation = self.annotation_samples[idx]
        
        # Load image
        image_info = self.images[annotation['image_id']]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Create masked crop
        cropped_image = self._create_masked_crop(image, annotation)
        
        # Create single-label target (not multi-label since we have one annotation per sample)
        target = torch.zeros(len(self.category_ids), dtype=torch.float32)
        cat_idx = self.cat_id_to_idx[annotation['category_id']]
        target[cat_idx] = 1.0
        
        # Apply transforms
        original_cropped = cropped_image.copy()  # Keep for visualization
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        return {
            'image': cropped_image,
            'target': target,
            'image_id': annotation['image_id'],
            'annotation_id': annotation['id'],
            'category_id': annotation['category_id'],
            'bbox': annotation['bbox'],
            'original_crop': original_cropped,
            'file_name': image_info['file_name']
        }
    
    def get_class_names(self):
        """Get class names in model index order"""
        return [self.categories[cat_id]['name'] for cat_id in self.category_ids]
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        category_counts = defaultdict(int)
        
        for ann in self.annotation_samples:
            category_counts[ann['category_id']] += 1
        
        weights = []
        total_samples = len(self.annotation_samples)
        
        for cat_id in self.category_ids:
            count = category_counts[cat_id]
            weight = total_samples / (len(self.category_ids) * count) if count > 0 else 1.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler"""
        # Count samples per class
        category_counts = defaultdict(int)
        for ann in self.annotation_samples:
            category_counts[ann['category_id']] += 1
        
        # Calculate weight for each sample
        sample_weights = []
        total_samples = len(self.annotation_samples)
        
        for ann in self.annotation_samples:
            cat_id = ann['category_id']
            count = category_counts[cat_id]
            # Inverse frequency weighting
            weight = total_samples / count
            sample_weights.append(weight)
        
        return torch.FloatTensor(sample_weights)
    
    def create_weighted_sampler(self, replacement=True):
        """Create WeightedRandomSampler for balanced sampling"""
        sample_weights = self.get_sample_weights()
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.annotation_samples),
            replacement=replacement
        )
        
        return sampler

def create_transforms(image_size=224, is_training=True):
    """Create transforms for DINOv2"""
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def collate_fn(batch):
    """Custom collate function for the new dataset structure"""
    images = torch.stack([item['image'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    # Keep other information as lists
    image_ids = [item['image_id'] for item in batch]
    annotation_ids = [item['annotation_id'] for item in batch]
    category_ids = [item['category_id'] for item in batch]
    bboxes = [item['bbox'] for item in batch]
    original_crops = [item['original_crop'] for item in batch]
    file_names = [item['file_name'] for item in batch]
    
    return {
        'images': images,
        'targets': targets,
        'image_ids': image_ids,
        'annotation_ids': annotation_ids,
        'category_ids': category_ids,
        'bboxes': bboxes,
        'original_images': original_crops,  # Keep same key for compatibility
        'file_names': file_names
    }

# Utility functions for visualization and analysis
def visualize_polygon_masking(dataset, num_samples=8, save_path=None):
    """Visualize the polygon masking effect"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Get sample without transform for visualization
        dataset_no_transform = COCOMultiLabelDataset(
            dataset.annotation_file, dataset.image_dir, 
            transform=None, use_polygon_mask=False
        )
        
        # Original crop (without masking)
        original_sample = dataset_no_transform[idx]
        original_crop = original_sample['original_crop']
        
        # Masked crop
        masked_sample = dataset[idx]  # This has masking enabled
        masked_crop = masked_sample['original_crop']
        
        # Display original
        axes[0, i].imshow(original_crop)
        axes[0, i].set_title(f'Original\n{dataset.categories[masked_sample["category_id"]]["name"]}', 
                           fontsize=8)
        axes[0, i].axis('off')
        
        # Display masked
        axes[1, i].imshow(masked_crop)
        axes[1, i].set_title('Polygon Masked', fontsize=8)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Polygon Masking Effect', y=1.02, fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def analyze_class_distribution(dataset, save_path=None):
    """Analyze and visualize class distribution"""
    import matplotlib.pyplot as plt
    
    category_counts = defaultdict(int)
    for ann in dataset.annotation_samples:
        category_counts[ann['category_id']] += 1
    
    class_names = dataset.get_class_names()
    counts = [category_counts[cat_id] for cat_id in dataset.category_ids]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), counts, alpha=0.7)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print(f"Class distribution statistics:")
    print(f"  Total samples: {sum(counts)}")
    print(f"  Most frequent class: {class_names[np.argmax(counts)]} ({max(counts)} samples)")
    print(f"  Least frequent class: {class_names[np.argmin(counts)]} ({min(counts)} samples)")
    print(f"  Imbalance ratio: {max(counts)/min(counts):.2f}")

def create_balanced_dataloader(dataset, batch_size=32, num_workers=4, use_weighted_sampler=True):
    """Create dataloader with optional weighted sampling for class balance"""
    from torch.utils.data import DataLoader
    
    if use_weighted_sampler:
        sampler = dataset.create_weighted_sampler()
        shuffle = False  # Don't shuffle when using sampler
        print("Using WeightedRandomSampler for balanced sampling")
    else:
        sampler = None
        shuffle = True
        print("Using regular random sampling")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True if sampler else False
    )
    
    return dataloader

# Example usage and testing
def test_polygon_dataset():
    """Test the polygon masking dataset"""
    print("Testing polygon masking dataset...")
    
    # Create dataset (replace with your paths)
    dataset = COCOMultiLabelDataset(
        annotation_file='path/to/annotations.json',
        image_dir='path/to/images/',
        transform=create_transforms(is_training=False),
        use_polygon_mask=True,
        noise_std=0.3
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Category: {dataset.categories[sample['category_id']]['name']}")
    
    # Analyze class distribution
    analyze_class_distribution(dataset)
    
    # Visualize polygon masking
    visualize_polygon_masking(dataset, num_samples=4)
    
    # Test weighted sampler
    sampler = dataset.create_weighted_sampler()
    print(f"Weighted sampler created with {len(sampler)} samples")
    
    # Test dataloader
    dataloader = create_balanced_dataloader(dataset, batch_size=8, use_weighted_sampler=True)
    
    # Test one batch
    for batch in dataloader:
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Batch targets shape: {batch['targets'].shape}")
        print(f"Categories in batch: {batch['category_ids']}")
        break
    
    print("✅ Polygon dataset test completed successfully!")

if __name__ == "__main__":
    test_polygon_dataset()
