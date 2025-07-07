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
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

class COCOMultiLabelDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None, min_bbox_area=100, 
                 noise_std=0.3, use_polygon_mask=True, cache_dir=None, precompute_masks=True,
                 num_workers=8):
        """
        COCO dataset for multi-label classification with polygon masking
        
        Args:
            annotation_file: Path to COCO JSON file
            image_dir: Directory containing images
            transform: Image transforms
            min_bbox_area: Minimum bbox area to include
            noise_std: Standard deviation for Gaussian noise
            use_polygon_mask: Whether to use polygon masking
            cache_dir: Directory to cache preprocessed data
            precompute_masks: Whether to precompute all masks
            num_workers: Number of workers for parallel processing
        """
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform
        self.min_bbox_area = min_bbox_area
        self.noise_std = noise_std
        self.use_polygon_mask = use_polygon_mask
        self.num_workers = num_workers
        
        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.mask_cache_path = self.cache_dir / "mask_cache.pkl"
            self.crop_cache_dir = self.cache_dir / "crops"
            self.crop_cache_dir.mkdir(exist_ok=True)
        
        # Build mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Create category_id to index mapping
        self.category_ids = sorted(list(self.categories.keys()))
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.category_ids)}
        self.idx_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_idx.items()}
        
        # Create annotation samples
        self.annotation_samples = self._create_annotation_samples()
        
        # Cache for masks and crops
        self.mask_cache = {}
        self.crop_cache = {}
        
        # Load or precompute masks
        if self.use_polygon_mask:
            if precompute_masks:
                self._precompute_masks()
        
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
    
    def _precompute_masks(self):
        """Precompute all masks and save to cache"""
        print("Precomputing masks for faster training...")
        
        # Check if cache exists
        if self.cache_dir and self.mask_cache_path.exists():
            print("Loading masks from cache...")
            with open(self.mask_cache_path, 'rb') as f:
                self.mask_cache = pickle.load(f)
            print(f"Loaded {len(self.mask_cache)} cached masks")
            return
        
        # Precompute masks in parallel
        def process_annotation(ann):
            try:
                image_info = self.images[ann['image_id']]
                image_path = os.path.join(self.image_dir, image_info['file_name'])
                
                # Get image shape
                with Image.open(image_path) as img:
                    image_shape = (img.height, img.width, 3)
                
                # Create mask
                if 'segmentation' in ann:
                    mask = self._create_polygon_mask(image_shape, ann['segmentation'])
                    return ann['id'], mask
                else:
                    return ann['id'], None
            except Exception as e:
                print(f"Error processing annotation {ann['id']}: {e}")
                return ann['id'], None
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_annotation, self.annotation_samples))
        
        # Store results
        for ann_id, mask in results:
            if mask is not None:
                self.mask_cache[ann_id] = mask
        
        # Save cache
        if self.cache_dir:
            with open(self.mask_cache_path, 'wb') as f:
                pickle.dump(self.mask_cache, f)
            print(f"Cached {len(self.mask_cache)} masks")
    
    def _get_cached_crop_path(self, ann_id):
        """Get cache path for preprocessed crop"""
        if not self.cache_dir:
            return None
        return self.crop_cache_dir / f"crop_{ann_id}.jpg"
    
    def _create_polygon_mask_fast(self, image_shape, segmentation):
        """Optimized polygon mask creation"""
        if isinstance(segmentation, list):
            # Use OpenCV for faster polygon filling
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            for seg in segmentation:
                if len(seg) >= 6:
                    points = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [points], 1)
            return mask.astype(bool)
        else:
            # RLE format - fallback to original method
            if isinstance(segmentation, dict):
                rle = segmentation
            else:
                rle = {'size': image_shape[:2], 'counts': segmentation}
            mask = coco_mask.decode(rle)
            return mask.astype(bool)
    
    def _create_masked_crop_fast(self, image, annotation):
        """Optimized masked crop creation with caching"""
        ann_id = annotation['id']
        
        # Check if crop is cached
        cached_crop_path = self._get_cached_crop_path(ann_id)
        if cached_crop_path and cached_crop_path.exists():
            return Image.open(cached_crop_path).convert('RGB')
        
        # Get image as numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        if self.use_polygon_mask:
            # Get mask from cache or compute
            if ann_id in self.mask_cache:
                mask = self.mask_cache[ann_id]
            elif 'segmentation' in annotation:
                mask = self._create_polygon_mask_fast(image_np.shape, annotation['segmentation'])
            else:
                mask = None
            
            if mask is not None:
                # Create noise background more efficiently
                bbox = annotation['bbox']
                x, y, w, h = [int(v) for v in bbox]
                
                # Only create noise for the bbox region
                crop_shape = (h, w, 3)
                noise_crop = np.random.normal(
                    loc=128,
                    scale=self.noise_std * 255,
                    size=crop_shape
                ).astype(np.uint8)
                noise_crop = np.clip(noise_crop, 0, 255)
                
                # Extract crops
                image_crop = image_np[y:y+h, x:x+w]
                mask_crop = mask[y:y+h, x:x+w]
                
                # Apply mask
                masked_crop = noise_crop.copy()
                if mask_crop.any():  # Only if mask has positive pixels
                    masked_crop[mask_crop] = image_crop[mask_crop]
                
                cropped_image_pil = Image.fromarray(masked_crop)
            else:
                # Fallback: regular crop
                bbox = annotation['bbox']
                x, y, w, h = bbox
                cropped_image_pil = image.crop((x, y, x + w, y + h))
        else:
            # Regular crop without masking
            bbox = annotation['bbox']
            x, y, w, h = bbox
            cropped_image_pil = image.crop((x, y, x + w, y + h))
        
        # Cache the crop
        if cached_crop_path:
            cropped_image_pil.save(cached_crop_path, 'JPEG', quality=95)
        
        return cropped_image_pil
    
    def __len__(self):
        return len(self.annotation_samples)
    
    def __getitem__(self, idx):
        annotation = self.annotation_samples[idx]
        
        # Load image
        image_info = self.images[annotation['image_id']]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy black image
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Create masked crop (with caching)
        cropped_image = self._create_masked_crop_fast(image, annotation)
        
        # Create single-label target
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

def create_balanced_dataloader(dataset, batch_size=32, num_workers=8, use_weighted_sampler=True, 
                              pin_memory=True, persistent_workers=True):
    """Create optimized dataloader with optional weighted sampling for class balance"""
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
        drop_last=True if sampler else False,
        pin_memory=pin_memory,  # Faster GPU transfer
        persistent_workers=persistent_workers,  # Keep workers alive
        prefetch_factor=4,  # Prefetch more batches
    )
    
    return dataloader

# GPU-accelerated noise generation (optional)
class GPUNoiseGenerator:
    """GPU-accelerated noise generation for faster processing"""
    def __init__(self, device='cuda', batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.noise_cache = {}
    
    def generate_noise_batch(self, shapes, noise_std=0.3):
        """Generate noise for multiple crops at once"""
        noises = []
        for shape in shapes:
            h, w = shape[:2]
            key = (h, w)
            
            if key not in self.noise_cache:
                # Generate on GPU
                noise = torch.normal(
                    mean=128.0, 
                    std=noise_std * 255,
                    size=(self.batch_size, h, w, 3),
                    device=self.device,
                    dtype=torch.uint8
                )
                self.noise_cache[key] = noise
            
            # Get one from cache
            noise = self.noise_cache[key][0].cpu().numpy()
            noises.append(noise)
        
        return noises

# Fast dataset creation function
def create_fast_dataset(annotation_file, image_dir, cache_dir=None, 
                       precompute_masks=True, num_workers=8, **kwargs):
    """Create optimized dataset with caching"""
    dataset = COCOMultiLabelDataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        cache_dir=cache_dir,
        precompute_masks=precompute_masks,
        num_workers=num_workers,
        **kwargs
    )
    
    print(f"Dataset optimization tips:")
    print(f"  - Use cache_dir for faster subsequent runs")
    print(f"  - Set num_workers={min(num_workers, os.cpu_count())} in DataLoader")
    print(f"  - Enable pin_memory=True for GPU training")
    print(f"  - Use persistent_workers=True")
    
    return dataset

# Example usage and testing
def test_polygon_dataset():
    """Test the optimized polygon masking dataset"""
    print("Testing optimized polygon masking dataset...")
    
    # Create dataset with caching (replace with your paths)
    dataset = create_fast_dataset(
        annotation_file='path/to/annotations.json',
        image_dir='path/to/images/',
        cache_dir='./cache',  # Enable caching
        transform=create_transforms(is_training=False),
        use_polygon_mask=True,
        noise_std=0.3,
        precompute_masks=True,
        num_workers=8
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Category: {dataset.categories[sample['category_id']]['name']}")
    
    # Test optimized dataloader
    import time
    
    print("\nTesting dataloader speed...")
    dataloader = create_balanced_dataloader(
        dataset, 
        batch_size=32, 
        num_workers=8,
        use_weighted_sampler=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Time a few batches
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 5:  # Test 5 batches
            break
        print(f"Batch {i}: {batch['images'].shape}, Categories: {len(set(batch['category_ids']))}")
    
    elapsed = time.time() - start_time
    print(f"Processed 5 batches in {elapsed:.2f}s ({elapsed/5:.2f}s per batch)")
    
    print("✅ Optimized polygon dataset test completed!")

if __name__ == "__main__":
    test_polygon_dataset()
