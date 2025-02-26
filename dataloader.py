#!/usr/bin/env python3
"""
Custom dataset and dataloader for SAM 2.1 fine-tuning
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random

class SAMFineTuneDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, prompt_type='box'):
        """
        Dataset for SAM 2.1 fine-tuning.
        
        Args:
            data_dir: path to the data directory
            split: 'train' or 'val'
            transform: transforms to apply to images
            prompt_type: 'box' or 'point' for the prompt type
        """
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        self.transform = transform
        self.prompt_type = prompt_type
        
        # Load metadata
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} samples for {split} set")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, item['image_path'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.data_dir, item['mask_path'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)  # Convert to binary mask
        
        # Get prompt based on type
        if self.prompt_type == 'box':
            # Use the bbox as prompt
            x, y, w, h = item['bbox']
            prompt = np.array([x, y, x + w, y + h])  # x1, y1, x2, y2 format
        else:  # point prompt
            # Calculate centroid of the mask as a point prompt
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_center = np.mean(x_indices)
                y_center = np.mean(y_indices)
                prompt = np.array([x_center, y_center])
            else:
                # Fallback to center of bbox for empty masks
                x, y, w, h = item['bbox']
                prompt = np.array([x + w/2, y + h/2])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask, prompt=prompt)
            image = transformed['image']
            mask = transformed['mask']
            prompt = transformed['prompt']
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.from_numpy(prompt).float()
        
        return {
            'image': image,
            'mask': mask,
            'prompt': prompt,
            'category_id': item['category_id'],
            'image_path': image_path
        }

class SAMTransform:
    def __init__(self, image_size=1024, is_train=True):
        self.image_size = image_size
        self.is_train = is_train
    
    def __call__(self, image, mask, prompt):
        # Get original dimensions
        h, w = image.shape[:2]
        
        # Resize image and mask
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Scale prompt (bbox coordinates)
        scale_x, scale_y = self.image_size / w, self.image_size / h
        if len(prompt) == 4:  # bbox
            prompt = np.array([
                prompt[0] * scale_x, 
                prompt[1] * scale_y,
                prompt[2] * scale_x,
                prompt[3] * scale_y
            ])
        else:  # point
            prompt = np.array([prompt[0] * scale_x, prompt[1] * scale_y])
        
        # Apply data augmentation in training mode
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
                if len(prompt) == 4:  # bbox
                    prompt = np.array([
                        self.image_size - prompt[2],
                        prompt[1],
                        self.image_size - prompt[0],
                        prompt[3]
                    ])
                else:  # point
                    prompt = np.array([self.image_size - prompt[0], prompt[1]])
            
            # Random brightness, contrast, and saturation
            image = Image.fromarray(image)
            image = F.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = F.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = F.adjust_saturation(image, random.uniform(0.8, 1.2))
            image = np.array(image)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        return {
            'image': image,
            'mask': mask,
            'prompt': prompt
        }

def get_dataloader(data_dir, batch_size=8, num_workers=4, image_size=1024, prompt_type='box'):
    """Create train and validation dataloaders."""
    train_transform = SAMTransform(image_size=image_size, is_train=True)
    val_transform = SAMTransform(image_size=image_size, is_train=False)
    
    train_dataset = SAMFineTuneDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        prompt_type=prompt_type
    )
    
    val_dataset = SAMFineTuneDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        prompt_type=prompt_type
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataset
    data_dir = "prepared_dataset"
    train_loader, val_loader = get_dataloader(data_dir, batch_size=2)
    
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"Prompt shape: {batch['prompt'].shape}")
        print(f"Category ID: {batch['category_id']}")
        
        if i >= 2:  # Just check first few batches
            break
