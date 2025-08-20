import os, random, math, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A

def _albumentations_train(img_size):
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0), ratio=(0.8, 1.25), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.CoarseDropout(max_holes=8, max_height=int(img_size*0.08), max_width=int(img_size*0.08),
                        min_holes=1, fill_value=0, p=0.15),
        A.Normalize()
    ], is_check_shapes=False)

def _albumentations_val(img_size):
    return A.Compose([
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize()
    ], is_check_shapes=False)

def _masks_to_boxes(masks: np.ndarray):
    # masks: [N,H,W] bool/uint8
    boxes = []
    for m in masks:
        ys, xs = np.where(m > 0)
        if len(xs)==0 or len(ys)==0:
            boxes.append([0,0,0,0])
        else:
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
            boxes.append([x0, y0, x1, y1])
    return np.array(boxes, dtype=np.float32)

class COCOMaskDataset(Dataset):
    """
    Returns:
        image: FloatTensor [3,H,W] in 0..1 after denorm later
        target: dict{
            'masks': BoolTensor [N,H,W],
            'labels': LongTensor [N],
            'boxes': FloatTensor [N,4] xyxy,
            'image_id': int,
            'file_name': str,
            'size': (H,W)
        }
    """
    def __init__(self, img_dir, ann_file, img_size=1024, is_train=True,
                 use_mosaic=True, mosaic_prob=0.5, use_mixup=True, mixup_prob=0.3, mixup_alpha=0.5):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.img_size = img_size
        self.is_train = is_train

        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.use_mixup = use_mixup
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

        self.tf_train = _albumentations_train(img_size)
        self.tf_val   = _albumentations_val(img_size)

    def __len__(self): return len(self.img_ids)

    # ---------- helpers ----------
    def _load_img_target(self, img_id):
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info['file_name'])
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        masks, labels = [], []
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            m = self.coco.annToMask(a).astype(np.uint8)
            if m.sum() == 0:  # skip empty
                continue
            masks.append(m)
            labels.append(int(a["category_id"]))

        masks = np.stack(masks, axis=0) if len(masks)>0 else np.zeros((0, H, W), dtype=np.uint8)
        labels = np.array(labels, dtype=np.int64) if len(labels)>0 else np.zeros((0,), dtype=np.int64)
        boxes  = _masks_to_boxes(masks) if len(masks)>0 else np.zeros((0,4), dtype=np.float32)
        return im, masks, labels, boxes, info['file_name'], H, W

    def _apply_tf(self, im, masks):
        # Albumentations expects list of masks
        tf = self.tf_train if self.is_train else self.tf_val
        mlist = [m for m in masks]
        out = tf(image=im, masks=mlist)
        im = out['image']
        masks = np.stack(out['masks'], axis=0) if len(out['masks'])>0 else np.zeros((0, self.img_size, self.img_size), dtype=np.uint8)
        return im, masks

    def _mosaic(self):
        ids = random.sample(self.img_ids, k=4)
        s = self.img_size
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        out_masks = []
        out_labels = []
        # 2x2 grid
        quadrants = [(0,0),(0,s//2),(s//2,0),(s//2,s//2)]
        for q,(y0,x0) in enumerate(quadrants):
            im, masks, labels, _, _, _, _ = self._load_img_target(ids[q])
            # resize to half
            im = cv2.resize(im, (s//2, s//2), interpolation=cv2.INTER_LINEAR)
            canvas[y0:y0+s//2, x0:x0+s//2] = im
            if len(masks)>0:
                ms = np.stack([cv2.resize(m.astype(np.uint8), (s//2, s//2), interpolation=cv2.INTER_NEAREST) for m in masks],0)
                # place on big canvas
                placed = np.zeros((ms.shape[0], s, s), dtype=np.uint8)
                placed[:, y0:y0+s//2, x0:x0+s//2] = ms
                out_masks.append(placed)
                out_labels.append(labels)
        if len(out_masks)>0:
            out_masks = np.concatenate(out_masks, axis=0)
            out_labels = np.concatenate(out_labels, axis=0)
        else:
            out_masks = np.zeros((0, s, s), dtype=np.uint8)
            out_labels = np.zeros((0,), dtype=np.int64)
        return canvas, out_masks, out_labels

    def _mixup(self, a_img, a_masks, a_labels):
        # sample a partner and overlay instances; image mix is linear blend (alpha)
        b_id = random.choice(self.img_ids)
        b_img, b_masks, b_labels, _, _, _, _ = self._load_img_target(b_id)
        # resize both to img_size via val tf (deterministic sizing)
        a_img, a_masks = self._apply_tf(a_img, a_masks)
        b_img, b_masks = self._apply_tf(b_img, b_masks)
        alpha = self.mixup_alpha
        img = (alpha * a_img + (1 - alpha) * b_img).astype(np.uint8)
        if len(b_masks)>0:
            masks = np.concatenate([a_masks, b_masks], axis=0)
            labels = np.concatenate([a_labels, b_labels], axis=0)
        else:
            masks, labels = a_masks, a_labels
        return img, masks, labels

    # ---------- main ----------
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        im, masks, labels, boxes, file_name, H, W = self._load_img_target(img_id)

        # Training-time composite augs
        if self.is_train and self.use_mosaic and random.random() < self.mosaic_prob:
            im, masks, labels = self._mosaic()

        if self.is_train and self.use_mixup and random.random() < self.mixup_prob:
            im, masks, labels = self._mixup(im, masks, labels)

        # Albumentations pipeline (resize/normalize/etc.)
        im, masks = self._apply_tf(im, masks)

        # Recompute boxes after geometric augs
        boxes = _masks_to_boxes(masks) if len(masks)>0 else np.zeros((0,4), dtype=np.float32)

        # Torchify
        image = torch.from_numpy(im.transpose(2,0,1)).float()  # [3,H,W]
        target = {
            "masks": torch.from_numpy(masks.astype(np.bool_)),
            "labels": torch.from_numpy(labels.astype(np.int64)),
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "image_id": int(img_id),
            "file_name": file_name,
            "size": (self.img_size, self.img_size)
        }
        return image, target

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
