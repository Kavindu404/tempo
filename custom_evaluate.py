import os
import time
import json
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

from functools import partial
import torch
from torchvision.utils import make_grid
import albumentations as A
from albumentations.pytorch import ToTensorV2

from custom_config import CFG
from tokenizer import Tokenizer
from utils import (
    seed_everything,
    load_checkpoint,
    test_generate,
    postprocess,
    permutations_to_polygons,
)
from models.model import (
    Encoder,
    Decoder,
    EncoderDecoder
)

from torch.utils.data import DataLoader
from datasets.dataset_custom_coco import CustomCocoDataset_val
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy


def bounding_box_from_points(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0]/2) * 2
    odd_locations = even_locations + 1
    X = np.take(points, even_locations.tolist())
    Y = np.take(points, odd_locations.tolist())
    bbox = [X.min(), Y.min(), X.max()-X.min(), Y.max()-Y.min()]
    bbox = [int(b) for b in bbox]
    return bbox


def single_annotation(image_id, poly):
    _result = {}
    _result["image_id"] = int(image_id)
    _result["category_id"] = 100 
    _result["score"] = 1
    _result["segmentation"] = poly
    _result["bbox"] = bounding_box_from_points(_result["segmentation"])
    return _result


def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length.
    """

    image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, idx_batch = [], [], [], [], [], []
    for image, mask, c_mask, seq, perm_mat, idx in batch:
        image_batch.append(image)
        mask_batch.append(mask)
        coords_mask_batch.append(c_mask)
        coords_seq_batch.append(seq)
        perm_matrix_batch.append(perm_mat)
        idx_batch.append(idx)

    coords_seq_batch = pad_sequence(
        coords_seq_batch,
        padding_value=pad_idx,
        batch_first=True
    )

    if max_len:
        pad = torch.ones(coords_seq_batch.size(0), max_len - coords_seq_batch.size(1)).fill_(pad_idx).long()
        coords_seq_batch = torch.cat([coords_seq_batch, pad], dim=1)

    image_batch = torch.stack(image_batch)
    mask_batch = torch.stack(mask_batch)
    coords_mask_batch = torch.stack(coords_mask_batch)
    perm_matrix_batch = torch.stack(perm_matrix_batch)
    idx_batch = torch.stack(idx_batch)
    return image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, idx_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()

    seed_everything(42)

    # Set checkpoint path
    if args.checkpoint:
        CHECKPOINT_PATH = args.checkpoint
    else:
        CHECKPOINT_PATH = f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/best_valid_metric.pth"
    
    # Set output directory
    OUTPUT_DIR = os.path.join("runs", CFG.EXPERIMENT_NAME, "evaluation")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set batch size
    BATCH_SIZE = args.batch_size

    valid_transforms = A.Compose(
        [
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    val_ds = CustomCocoDataset_val(
        cfg=CFG,
        dataset_dir=CFG.VAL_DATASET_DIR,
        annotations_file=CFG.VAL_ANNOTATIONS_FILE,
        transform=valid_transforms,
        tokenizer=tokenizer,
        shuffle_tokens=CFG.SHUFFLE_TOKENS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        collate_fn=partial(collate_fn, max_len=CFG.MAX_LEN, pad_idx=CFG.PAD_IDX),
        num_workers=2
    )

    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
    decoder = Decoder(
        cfg=CFG,
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6
    )
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
    model.to(CFG.DEVICE)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epochs_run']

    print(f"Model loaded from epoch: {epoch}")
    ckpt_desc = f"epoch_{epoch}"
    
    mean_iou_metric = BinaryJaccardIndex()
    mean_acc_metric = BinaryAccuracy()

    with torch.no_grad():
        cumulative_miou = []
        cumulative_macc = []
        speed = []
        predictions = []
        for i_batch, (x, y_mask, y_corner_mask, y, y_perm, idx) in enumerate(tqdm(val_loader)):
            all_coords = []
            all_confs = []
            t0 = time.time()
            batch_preds, batch_confs, perm_preds = test_generate(model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
            speed.append(time.time() - t0)
            vertex_coords, confs = postprocess(batch_preds, batch_confs, tokenizer)

            all_coords.extend(vertex_coords)
            all_confs.extend(confs)

            coords = []
            for i in range(len(all_coords)):
                if all_coords[i] is not None:
                    coord = torch.from_numpy(all_coords[i])
                else:
                    coord = torch.tensor([])

                padd = torch.ones((CFG.N_VERTICES - len(coord), 2)).fill_(CFG.PAD_IDX)
                coord = torch.cat([coord, padd], dim=0)
                coords.append(coord)
            batch_polygons = permutations_to_polygons(perm_preds, coords, out='torch')  # [0, 224]

            for ip, pp in enumerate(batch_polygons):
                for p in pp:
                    p = torch.fliplr(p)
                    p = p[p[:, 0] != CFG.PAD_IDX]
                    p = p * (CFG.IMG_SIZE / CFG.INPUT_WIDTH)
                    p = p.view(-1).tolist()
                    if len(p) > 0:
                        predictions.append(single_annotation(idx[ip], [p]))

            B, C, H, W = x.shape

            polygons_mask = np.zeros((B, 1, H, W))
            for b in range(len(batch_polygons)):
                for c in range(len(batch_polygons[b])):
                    poly = batch_polygons[b][c]
                    poly = poly[poly[:, 0] != CFG.PAD_IDX]
                    cnt = np.flip(np.int32(poly.cpu()), 1)
                    if len(cnt) > 0:
                        cv2.fillPoly(polygons_mask[b, 0], pts=[cnt], color=1.)
            polygons_mask = torch.from_numpy(polygons_mask)

            batch_miou = mean_iou_metric(polygons_mask, y_mask)
            batch_macc = mean_acc_metric(polygons_mask, y_mask)

            cumulative_miou.append(batch_miou)
            cumulative_macc.append(batch_macc)

            # Save visualizations
            pred_grid = make_grid(polygons_mask).permute(1, 2, 0)
            gt_grid = make_grid(y_mask).permute(1, 2, 0)
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(pred_grid)
            plt.title("Predicted Polygons")
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(gt_grid)
            plt.title("Ground Truth")
            plt.axis('off')

            os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)
            plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', f'batch_{i_batch}.png'))
            plt.close()

        # Save and print final metrics
        print("Average model speed: ", np.mean(speed) / BATCH_SIZE, " [s / image]")
        print(f"Average Mean IOU: {torch.tensor(cumulative_miou).nanmean()}")
        print(f"Average Mean Acc: {torch.tensor(cumulative_macc).nanmean()}")

        # Save predictions to disk
        with open(os.path.join(OUTPUT_DIR, f'predictions_{ckpt_desc}.json'), "w") as fp:
            fp.write(json.dumps(predictions))

        # Save metrics to disk
        with open(os.path.join(OUTPUT_DIR, f'metrics_{ckpt_desc}.txt'), 'w') as ff:
            print(f"Average model speed: {np.mean(speed) / BATCH_SIZE} [s / image]", file=ff)
            print(f"Average Mean IOU: {torch.tensor(cumulative_miou).nanmean()}", file=ff)
            print(f"Average Mean Acc: {torch.tensor(cumulative_macc).nanmean()}", file=ff)


if __name__ == "__main__":
    main()
