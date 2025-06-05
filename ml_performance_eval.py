import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --- Configuration ---
NUM_CLASSES = 3  # Background + Gas Leak Day + Gas Leak Night
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
MODEL_PATH = "unet_gas_leak_segmentation_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
IMAGE_DIR = "./classification_dataset/"
ANNOTATION_PATH = "./classification_dataset/annotations.json"
NUM_EXAMPLES_TO_SHOW = 10

# --- Color Map for Visualization ---
class_colors = {
    0: [0, 0, 0],         # Background - black
    1: [255, 0, 0],       # Gas Leak Day - red
    2: [0, 0, 255],       # Gas Leak Night - blue
}

def decode_segmap(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        color_mask[mask == cls] = class_colors[cls]
    return color_mask

# --- Define Transform ---
transform = A.Compose([
    A.Resize(320, 480),
    A.Normalize(),
    ToTensorV2()
])

# --- Load COCO Annotations ---
coco = COCO(ANNOTATION_PATH)
img_ids = list(coco.imgs.keys())

# --- Dummy Dataset Class for COCO ---
class CocoSegDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, coco, transform=None):
        self.image_dir = image_dir
        self.coco = coco
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = np.array(Image.open(image_path).convert("RGB"))

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for i, ann in enumerate(anns):
            category_id = ann["category_id"]
            # Map category_id to your class indices (e.g., 1 -> 1 for "Gas Leak Day", etc.)
            class_id = ann["category_id"]  # Adjust based on your dataset mapping
            mask_instance = self.coco.annToMask(ann)
            mask[mask_instance > 0] = class_id  # Overwrite with class ID

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

# --- Metric Functions ---
def compute_iou(preds, labels, num_classes):
    preds = preds.view(-1)
    labels = labels.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

def compute_dice(preds, labels, num_classes):
    preds = preds.view(-1)
    labels = labels.view(-1)
    dice_scores = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        total = pred_inds.sum().item() + target_inds.sum().item()
        if total == 0:
            dice_scores.append(float('nan'))
        else:
            dice_scores.append(2 * intersection / total)
    return dice_scores

# --- Main Evaluation Function ---
def evaluate():
    dataset = CocoSegDataset(IMAGE_DIR, coco, transform=transform)

    val_size = int(0.2 * len(dataset))
    _, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Build and load model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_ious = []
    all_dices = []
    all_correct = 0
    all_total = 0
    examples_shown = 0

    print("📊 Evaluating model performance...")

    fig, axes = plt.subplots(4 * NUM_EXAMPLES_TO_SHOW // 5, 5, figsize=(20, 4 * (NUM_EXAMPLES_TO_SHOW // 5)))
    axes = axes.flatten()

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.long().to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Metrics
            ious = compute_iou(preds, masks, NUM_CLASSES)
            dices = compute_dice(preds, masks, NUM_CLASSES)
            all_ious.extend([ious])
            all_dices.extend([dices])

            correct = (preds == masks).sum().item()
            total = masks.numel()
            all_correct += correct
            all_total += total

            # Show predictions
            if examples_shown < NUM_EXAMPLES_TO_SHOW:
                for i in range(images.size(0)):
                    if examples_shown >= NUM_EXAMPLES_TO_SHOW:
                        break

                    image = images[i].cpu().permute(1, 2, 0).numpy()
                    true_mask = masks[i].cpu().numpy()
                    pred_mask = preds[i].cpu().numpy()

                    # Denormalize image
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image = std * image + mean
                    image = np.clip(image, 0, 1)

                    true_color_mask = decode_segmap(true_mask)
                    pred_color_mask = decode_segmap(pred_mask)

                    ax = axes[examples_shown]
                    ax.imshow(image)
                    ax.set_title(f"Input {examples_shown+1}")
                    ax.axis("off")

                    ax = axes[examples_shown + 1]
                    ax.imshow(true_color_mask)
                    ax.set_title("True Mask")
                    ax.axis("off")

                    ax = axes[examples_shown + 2]
                    ax.imshow(pred_color_mask)
                    ax.set_title("Pred Mask")
                    ax.axis("off")

                    examples_shown += 3  # Each sample uses 3 subplots

        # Final metrics
        mean_ious = np.nanmean(np.array(all_ious), axis=0)
        mean_dices = np.nanmean(np.array(all_dices), axis=0)
        accuracy = all_correct / all_total
        error_rate = 1 - accuracy

        print("\n📈 Performance Report")
        print("-" * 40)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Error Rate: {error_rate:.4f}")
        print(f"Mean IoU: {np.nanmean(mean_ious):.4f}")
        print(f"Mean Dice: {np.nanmean(mean_dices):.4f}")
        print("-" * 40)
        print("Class-wise Metrics:")
        class_names = ["Background", "Gas Leak Day", "Gas Leak Night"]
        for i, name in enumerate(class_names):
            print(f"{name:15s} | IoU: {mean_ious[i]:.4f} | Dice: {mean_dices[i]:.4f}")

        # Remove unused subplots
        for j in range(examples_shown, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    evaluate()