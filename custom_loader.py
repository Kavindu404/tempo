from functools import partial
from torch.utils.data import DataLoader

from datasets.dataset_custom_coco import CustomCocoDataset, CustomCocoDataset_val, CustomCocoDatasetTest, collate_fn


def get_custom_loaders(
    train_dataset_dir,
    val_dataset_dir,
    test_images_dir,
    train_annotations_file,
    val_annotations_file,
    tokenizer,
    max_len,
    pad_idx,
    shuffle_tokens,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True
):
    train_ds = CustomCocoDataset(
        dataset_dir=train_dataset_dir,
        annotations_file=train_annotations_file,
        transform=train_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    valid_ds = CustomCocoDataset(
        dataset_dir=val_dataset_dir,
        annotations_file=val_annotations_file,
        transform=val_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_ds = CustomCocoDatasetTest(
        dataset_dir=test_images_dir,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader
