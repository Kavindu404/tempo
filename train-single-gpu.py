import os
from os import path as osp
import torch
from torch import nn
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from custom_config import CFG
from tokenizer import Tokenizer
from utils import seed_everything, load_checkpoint
from custom_loader import get_custom_loaders

from models.model import Encoder, Decoder, EncoderDecoder
from engine import train_eval

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


def main():
    seed_everything(42)
    os.makedirs(f"runs/{CFG.EXPERIMENT_NAME}/logs/tensorboard", exist_ok=True)
    os.makedirs(f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints", exist_ok=True)

    # Define tensorboard for logging
    writer = SummaryWriter(f"runs/{CFG.EXPERIMENT_NAME}/logs/tensorboard")
    attrs = vars(CFG)
    with open(f"runs/{CFG.EXPERIMENT_NAME}/config.txt", "w") as f:
        print("\n".join("%s: %s" % item for item in attrs.items()), file=f)

    # Define transformations
    train_transforms = A.Compose(
        [
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(p=0.5),
            A.ToGray(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )

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

    # Initialize tokenizer
    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    # Get data loaders
    train_loader, val_loader, test_loader = get_custom_loaders(
        CFG.TRAIN_DATASET_DIR,
        CFG.VAL_DATASET_DIR,
        CFG.TEST_IMAGES_DIR,
        CFG.TRAIN_ANNOTATIONS_FILE,
        CFG.VAL_ANNOTATIONS_FILE,
        tokenizer,
        CFG.MAX_LEN,
        tokenizer.PAD_code,
        CFG.SHUFFLE_TOKENS,
        CFG.BATCH_SIZE,
        train_transforms,
        valid_transforms,
        CFG.NUM_WORKERS,
        CFG.PIN_MEMORY,
    )

    # Initialize model
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

    # Setup loss functions
    weight = torch.ones(CFG.PAD_IDX + 1, device=CFG.DEVICE)
    weight[tokenizer.num_bins:tokenizer.BOS_code] = 0.0
    vertex_loss_fn = nn.CrossEntropyLoss(ignore_index=CFG.PAD_IDX, label_smoothing=CFG.LABEL_SMOOTHING, weight=weight)
    perm_loss_fn = nn.BCELoss()

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY, betas=(0.9, 0.95))

    num_training_steps = CFG.NUM_EPOCHS * (len(train_loader.dataset) // CFG.BATCH_SIZE)
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )

    # Load checkpoint if specified
    if CFG.LOAD_MODEL:
        checkpoint_name = osp.basename(osp.realpath(CFG.CHECKPOINT_PATH))
        start_epoch = load_checkpoint(
            torch.load(f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/{checkpoint_name}"),
            model,
            optimizer,
            lr_scheduler
        )
        CFG.START_EPOCH = start_epoch + 1

    # Start training
    print(f"Starting training with batch size {CFG.BATCH_SIZE} for {CFG.NUM_EPOCHS} epochs")
    train_eval(
        model,
        train_loader,
        val_loader,
        val_loader,  # Using val_loader as test_loader
        tokenizer,
        vertex_loss_fn,
        perm_loss_fn,
        optimizer,
        lr_scheduler=lr_scheduler,
        step='batch',
        writer=writer
    )


if __name__ == "__main__":
    main()
