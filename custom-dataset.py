import json
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, config, split='train', transform=None, tokenizer=None, shuffle_tokens=False):
        self.cfg = config
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.shuffle_tokens = shuffle_tokens
        
        # Determine image directory based on split
        if split == 'train':
            self.img_dir = self.cfg.TRAIN_IMAGES_DIR
        else:
            self.img_dir = self.cfg.VAL_IMAGES_DIR
        
        # Load annotations
        with open(self.cfg.ANNOTATIONS_PATH, 'r') as f:
            self.coco = json.load(f)
            
        # Filter images for the current split
        self.images = [img for img in self.coco['images'] if self._is_in_split(img['file_name'])]
        self.annotations = self.coco['annotations']
        
        # Create image_id to annotation mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def _is_in_split(self, file_name):
        """Determine if an image belongs to the current split based on file path"""
        if self.split == 'train':
            return os.path.exists(os.path.join(self.cfg.TRAIN_IMAGES_DIR, file_name))
        else:
            return os.path.exists(os.path.join(self.cfg.VAL_IMAGES_DIR, file_name))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        # Create empty mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
        corners_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
        
        # Collect all polygon coordinates
        all_coords = []
        
        for ann in anns:
            segmentation = ann['segmentation']
            for seg in segmentation:
                if len(seg) > 0:
                    # Convert to numpy array of shape (N, 2)
                    poly = np.array(seg).reshape(-1, 2)
                    
                    # Add polygon to mask
                    cv2.fillPoly(mask, [np.int32(poly)], 1)
                    
                    # Add corners to all_coords and corners_mask
                    for coord in poly:
                        all_coords.append(coord)
                        cv2.circle(corners_mask, (int(coord[0]), int(coord[1])), 0, 1, -1)
        
        all_coords = np.array(all_coords) if len(all_coords) > 0 else np.zeros((0, 2), dtype=np.float32)
        
        # Create permutation matrix for ground truth
        N = len(all_coords)
        perm_matrix = np.zeros((self.cfg.N_VERTICES, self.cfg.N_VERTICES), dtype=np.float32)
        
        if N > 0:
            # Fill permutation matrix based on polygon connectivity
            # This is a simplified version - you may need to adjust based on your data
            for i in range(N - 1):
                if i < self.cfg.N_VERTICES and i+1 < self.cfg.N_VERTICES:
                    perm_matrix[i, i+1] = 1.0
            if N-1 < self.cfg.N_VERTICES and 0 < self.cfg.N_VERTICES:
                perm_matrix[min(N-1, self.cfg.N_VERTICES-1), 0] = 1.0  # Close the loop
        
        # Apply transformations
        if self.transform:
            if len(all_coords) > 0:
                transformed = self.transform(
                    image=image, 
                    mask=mask,
                    mask1=corners_mask,
                    keypoints=all_coords
                )
                image = transformed['image']
                mask = transformed['mask']
                corners_mask = transformed['mask1']
                transformed_coords = transformed['keypoints']
            else:
                transformed = self.transform(
                    image=image, 
                    mask=mask,
                    mask1=corners_mask
                )
                image = transformed['image']
                mask = transformed['mask']
                corners_mask = transformed['mask1']
                transformed_coords = all_coords
        else:
            transformed_coords = all_coords
        
        # Tokenize coordinates
        if self.tokenizer and len(transformed_coords) > 0:
            coords_seq, rand_idxs = self.tokenizer(transformed_coords, shuffle=self.shuffle_tokens)
            coords_seq = torch.LongTensor(coords_seq)
            
            # Update permutation matrix according to shuffled indices
            if len(rand_idxs) > 0 and N > 0:
                perm_matrix_shuffled = np.zeros_like(perm_matrix)
                for i, i_new in enumerate(rand_idxs):
                    if i < self.cfg.N_VERTICES and i_new < self.cfg.N_VERTICES:
                        for j, j_new in enumerate(rand_idxs):
                            if j < self.cfg.N_VERTICES and j_new < self.cfg.N_VERTICES:
                                perm_matrix_shuffled[i_new, j_new] = perm_matrix[i, j]
                perm_matrix = perm_matrix_shuffled
        else:
            # Create a default sequence with just BOS and EOS tokens
            coords_seq = torch.LongTensor([self.tokenizer.BOS_code, self.tokenizer.EOS_code])
        
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        corners_mask = torch.from_numpy(corners_mask).unsqueeze(0).float()
        perm_matrix = torch.from_numpy(perm_matrix).float()
        
        return image, mask, corners_mask, coords_seq, perm_matrix

class CustomDatasetTest(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image