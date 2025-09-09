import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transform(image_size=(1024,1024), augment=True):
    if augment:
        return A.Compose([
            A.LongestMaxSize(max(image_size)),
            A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ], is_check_shapes=False)
    else:
        return A.Compose([
            A.LongestMaxSize(max(image_size)),
            A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(),
            ToTensorV2(),
        ], is_check_shapes=False)

class CocoSingleClassDataset(Dataset):
    def __init__(self, json_path, image_dir, image_size=(1024,1024), augment=True):
        super().__init__()
        self.coco = COCO(json_path)
        self.image_dir = image_dir
        self.ids = list(self.coco.imgs.keys())
        self.tfm = build_transform(image_size, augment)
        # Map all categories to "1" (single class)
        self.cat_id = 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        h, w = img_info["height"], img_info["width"]
        for ann in anns:
            m = self.coco.annToMask(ann)  # (H,W) {0,1}
            if m.sum() == 0:
                continue
            masks.append(m)
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x+bw, y+bh])
            labels.append(self.cat_id)
            areas.append(ann.get("area", float(bw*bh)))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(masks) == 0:
            masks = np.zeros((0, h, w), dtype=np.uint8)
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            masks = np.stack(masks, axis=0).astype(np.uint8)
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        transformed = self.tfm(image=image, masks=list(masks))
        timg = transformed["image"]                    # (3,H,W) float
        tmasks = transformed["masks"]                  # list of (H,W) uint8
        if len(tmasks) > 0:
            tmasks = np.stack(tmasks, axis=0)
        else:
            tmasks = np.zeros((0, timg.shape[1], timg.shape[2]), dtype=np.uint8)

        # Resize boxes according to transform scaling/padding: approximate via mask bbox after transform
        tboxes = []
        for m in tmasks:
            ys, xs = np.where(m > 0)
            if len(xs)==0 or len(ys)==0:
                tboxes.append([0,0,0,0])
                continue
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            tboxes.append([x0, y0, x1, y1])
        tboxes = np.array(tboxes, dtype=np.float32)
        tlabels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "image_id": torch.tensor(img_id, dtype=torch.int64),
            "boxes": torch.as_tensor(tboxes, dtype=torch.float32),  # xyxy
            "labels": tlabels,
            "masks": torch.as_tensor(tmasks, dtype=torch.uint8),
            "orig_size": torch.as_tensor([image.shape[0], image.shape[1]], dtype=torch.int64),
            "file_name": img_info["file_name"],
        }
        return timg, target

def collate_fn(batch):
    imgs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    imgs = torch.stack(imgs, dim=0)
    return imgs, tgts
