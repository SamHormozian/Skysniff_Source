import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from tqdm import tqdm
from dataset import GasLeakSegDataset  # Ensure dataset.py exists and works

# --- Configuration ---
BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_CLASSES = 3  # 0=background, 1=Gas Leak Day, 2=Gas Leak Night
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

# --- Albumentations-to-PyTorch Wrapper ---
class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask):
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()
        augmented = self.transform(image=image_np, mask=mask_np)
        return augmented['image'], augmented['mask']

# --- Define Transforms ---
train_transform = A.Compose([
    A.Resize(320, 480),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(320, 480),
    A.Normalize(),
    ToTensorV2()
])

# --- Training Function ---
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        masks = masks.long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

# --- Validation Function ---
def validate(model, loader, loss_fn):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            masks = masks.long().to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            running_loss += loss.item()

    return running_loss / len(loader)

# --- Main Training Loop ---
def main():
    full_dataset = GasLeakSegDataset(transforms=AlbumentationsTransform(train_transform))
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transforms = AlbumentationsTransform(val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = validate(model, val_loader, loss_fn)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "unet_gas_leak_segmentation.pth")
    print("✅ Model saved to 'unet_gas_leak_segmentation.pth'")


if __name__ == "__main__":
    main()