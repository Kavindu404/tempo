import os
import torch
from torch import nn
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

# Import your custom config and dataset
from custom_config import CFG
from custom_dataset import CustomDataset, CustomDatasetTest
from tokenizer import Tokenizer
from utils import seed_everything, load_checkpoint, AverageMeter, get_lr, save_checkpoint, save_single_predictions_as_images
from models.model import Encoder, Decoder, EncoderDecoder
from torch.nn.utils.rnn import pad_sequence

# Collate function
def collate_fn(batch, max_len, pad_idx):
    image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch = [], [], [], [], []
    for image, mask, c_mask, seq, perm_mat in batch:
        image_batch.append(image)
        mask_batch.append(mask)
        coords_mask_batch.append(c_mask)
        coords_seq_batch.append(seq)
        perm_matrix_batch.append(perm_mat)

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
    return image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch

# Train one epoch function
def train_one_epoch(epoch, iter_idx, model, train_loader, optimizer, lr_scheduler, vertex_loss_fn, perm_loss_fn, writer):
    model.train()
    vertex_loss_fn.train()
    perm_loss_fn.train()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = tqdm(train_loader, total=len(train_loader))

    for x, y_mask, y_corner_mask, y, y_perm in loader:
        x = x.to(CFG.DEVICE, non_blocking=True)
        y = y.to(CFG.DEVICE, non_blocking=True)
        y_perm = y_perm.to(CFG.DEVICE, non_blocking=True)

        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        preds, perm_mat = model(x, y_input)

        if epoch < CFG.MILESTONE:
            vertex_loss_weight = CFG.vertex_loss_weight
            perm_loss_weight = 0.0
        else:
            vertex_loss_weight = CFG.vertex_loss_weight
            perm_loss_weight = CFG.perm_loss_weight

        vertex_loss = vertex_loss_weight*vertex_loss_fn(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
        perm_loss = perm_loss_weight*perm_loss_fn(perm_mat, y_perm)

        loss = vertex_loss + perm_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_meter.update(loss.item(), x.size(0))
        vertex_loss_meter.update(vertex_loss.item(), x.size(0))
        perm_loss_meter.update(perm_loss.item(), x.size(0))

        lr = get_lr(optimizer)
        loader.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")
        writer.add_scalar('Running_logs/Train_Loss', loss_meter.avg, iter_idx)
        writer.add_scalar('Running_logs/LR', lr, iter_idx)

        iter_idx += 1
    
    print(f"Total train loss: {loss_meter.avg}\n\n")
    loss_dict = {
        'total_loss': loss_meter.avg,
        'vertex_loss': vertex_loss_meter.avg,
        'perm_loss': perm_loss_meter.avg,
    }

    return loss_dict, iter_idx

# Validation function
def valid_one_epoch(epoch, model, valid_loader, vertex_loss_fn, perm_loss_fn):
    print(f"\nValidating...")
    model.eval()
    vertex_loss_fn.eval()
    perm_loss_fn.eval()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for x, y_mask, y_corner_mask, y, y_perm in loader:
            x = x.to(CFG.DEVICE, non_blocking=True)
            y = y.to(CFG.DEVICE, non_blocking=True)
            y_perm = y_perm.to(CFG.DEVICE, non_blocking=True)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            preds, perm_mat = model(x, y_input)

            if epoch < CFG.MILESTONE:
                vertex_loss_weight = CFG.vertex_loss_weight
                perm_loss_weight = 0.0
            else:
                vertex_loss_weight = CFG.vertex_loss_weight
                perm_loss_weight = CFG.perm_loss_weight
            vertex_loss = vertex_loss_weight*vertex_loss_fn(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
            perm_loss = perm_loss_weight*perm_loss_fn(perm_mat, y_perm)

            loss = vertex_loss + perm_loss

            loss_meter.update(loss.item(), x.size(0))
            vertex_loss_meter.update(vertex_loss.item(), x.size(0))
            perm_loss_meter.update(perm_loss.item(), x.size(0))

    loss_dict = {
        'total_loss': loss_meter.avg,
        'vertex_loss': vertex_loss_meter.avg,
        'perm_loss': perm_loss_meter.avg,
    }

    return loss_dict

def main():
    seed_everything(42)

    # Define tensorboard for logging
    writer = SummaryWriter(f"runs/{CFG.EXPERIMENT_NAME}/logs/tensorboard")
    os.makedirs(f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints", exist_ok=True)
    
    attrs = vars(CFG)
    with open(f"runs/{CFG.EXPERIMENT_NAME}/config.txt", "w") as f:
        print("\n".join("%s: %s" % item for item in attrs.items()), file=f)

    # Data augmentation transforms
    train_transforms = A.Compose(
        [
            A.Affine(rotate=[-360, 360], fit_output=True, p=0.8),
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.RandomRotate90(p=1.),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(),
            A.ToGray(p=0.4),
            A.GaussNoise(),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
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
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    if "debug" in CFG.EXPERIMENT_NAME:
        train_transforms = valid_transforms

    # Initialize tokenizer
    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    # Create datasets and dataloaders
    train_ds = CustomDataset(
        config=CFG,
        split='train',
        transform=train_transforms,
        tokenizer=tokenizer,
        shuffle_tokens=CFG.SHUFFLE_TOKENS
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        collate_fn=partial(collate_fn, max_len=CFG.MAX_LEN, pad_idx=CFG.PAD_IDX),
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        drop_last=True
    )

    valid_ds = CustomDataset(
        config=CFG,
        split='val',
        transform=valid_transforms,
        tokenizer=tokenizer,
        shuffle_tokens=CFG.SHUFFLE_TOKENS
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG.BATCH_SIZE,
        collate_fn=partial(collate_fn, max_len=CFG.MAX_LEN, pad_idx=CFG.PAD_IDX),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_ds = CustomDatasetTest(
        image_dir=CFG.VAL_IMAGES_DIR,
        transform=valid_transforms
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
    )

    # Initialize model components
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

    # Loss functions
    weight = torch.ones(CFG.PAD_IDX + 1, device=CFG.DEVICE)
    weight[tokenizer.num_bins:tokenizer.BOS_code] = 0.0
    vertex_loss_fn = nn.CrossEntropyLoss(ignore_index=CFG.PAD_IDX, label_smoothing=CFG.LABEL_SMOOTHING, weight=weight)
    perm_loss_fn = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY, betas=(0.9, 0.95))

    num_training_steps = CFG.NUM_EPOCHS * (len(train_loader.dataset) // CFG.BATCH_SIZE)
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )

    CFG.START_EPOCH = 0
    if CFG.LOAD_MODEL:
        checkpoint_name = os.path.basename(os.path.realpath(CFG.CHECKPOINT_PATH))
        start_epoch = load_checkpoint(
            torch.load(f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/{checkpoint_name}"),
            model,
            optimizer,
            lr_scheduler
        )
        CFG.START_EPOCH = start_epoch + 1

    # Training loop
    best_loss = float('inf')
    best_metric = float('-inf')
    iter_idx = CFG.START_EPOCH * len(train_loader)

    for epoch in range(CFG.START_EPOCH, CFG.NUM_EPOCHS):
        print(f"\n\nEPOCH: {epoch + 1}\n\n")

        # Train
        train_loss_dict, iter_idx = train_one_epoch(
            epoch,
            iter_idx,
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            vertex_loss_fn,
            perm_loss_fn,
            writer
        )

        writer.add_scalar('Train_Losses/Total_Loss', train_loss_dict['total_loss'], epoch)
        writer.add_scalar('Train_Losses/Vertex_Loss', train_loss_dict['vertex_loss'], epoch)
        writer.add_scalar('Train_Losses/Perm_Loss', train_loss_dict['perm_loss'], epoch)

        # Validate
        valid_loss_dict = valid_one_epoch(
            epoch,
            model,
            valid_loader,
            vertex_loss_fn,
            perm_loss_fn,
        )
        print(f"Valid loss: {valid_loss_dict['total_loss']:.3f}\n\n")

        # Save best validation loss epoch
        if valid_loss_dict['total_loss'] < best_loss and CFG.SAVE_BEST:
            best_loss = valid_loss_dict['total_loss']
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epochs_run": epoch,
                "loss": train_loss_dict["total_loss"]
            }
            save_checkpoint(
                checkpoint,
                folder=f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/",
                filename="best_valid_loss.pth"
            )
            print(f"Saved best val loss model.")

        # Save latest checkpoint every epoch
        if CFG.SAVE_LATEST:
            checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epochs_run": epoch,
                    "loss": train_loss_dict["total_loss"]
                }
            save_checkpoint(
                checkpoint,
                folder=f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/",
                filename="latest.pth"
            )

        # Save checkpoints periodically
        if (epoch + 1) % CFG.SAVE_EVERY == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epochs_run": epoch,
                "loss": train_loss_dict["total_loss"]
            }
            save_checkpoint(
                checkpoint,
                folder=f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/",
                filename=f"epoch_{epoch}.pth"
            )

        writer.add_scalar('Val_Losses/Total_Loss', valid_loss_dict['total_loss'], epoch)
        writer.add_scalar('Val_Losses/Vertex_Loss', valid_loss_dict['vertex_loss'], epoch)
        writer.add_scalar('Val_Losses/Perm_Loss', valid_loss_dict['perm_loss'], epoch)

        # Evaluate and save example outputs
        if (epoch + 1) % CFG.VAL_EVERY == 0:
            # Since we don't have a test_loader with ground truth, we'll use valid_loader
            val_metrics_dict = save_single_predictions_as_images(
                valid_loader,
                model,
                tokenizer,
                epoch,
                writer,
                folder=f"runs/{CFG.EXPERIMENT_NAME}/runtime_outputs/",
                device=CFG.DEVICE
            )
            
            # Save best validation metric epoch
            if val_metrics_dict["miou"] > best_metric and CFG.SAVE_BEST:
                best_metric = val_metrics_dict["miou"]
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epochs_run": epoch,
                    "loss": train_loss_dict["total_loss"]
                }
                save_checkpoint(
                    checkpoint,
                    folder=f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/",
                    filename="best_valid_metric.pth"
                )
                print(f"Saved best val metric model.")

if __name__ == "__main__":
    main()