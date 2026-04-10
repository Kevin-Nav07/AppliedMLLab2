from pathlib import Path
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


TRAIN_IMAGE_DIR = Path("data/splits/train_images")
TRAIN_MASK_DIR = Path("data/splits/train_masks")
VAL_IMAGE_DIR = Path("data/splits/validation_images")
VAL_MASK_DIR = Path("data/splits/validation_masks")

CHECKPOINT_DIR = Path("models/checkpoints")
OUTPUT_DIR = Path("outputs/metrics")

BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
IMAGE_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BuildingSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(image_dir.glob("*.png"))

        self.image_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        self.mask_resize = transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{image_path.stem}_mask.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)

        mask = self.mask_resize(mask)
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def create_model():
    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    return model


def dice_score_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (total + eps)
    return dice.mean().item()


def iou_score_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.set_grad_enabled(is_train):
        for images, masks in loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score_from_logits(outputs, masks)
            total_iou += iou_score_from_logits(outputs, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = BuildingSegmentationDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
    val_dataset = BuildingSegmentationDataset(VAL_IMAGE_DIR, VAL_MASK_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
    }

    best_val_dice = -1.0

    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_dice={train_dice:.4f} val_dice={val_dice:.4f} | "
            f"train_iou={train_iou:.4f} val_iou={val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_deeplabv3_building.pt")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "final_deeplabv3_building.pt")

    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.savefig(OUTPUT_DIR / "loss_curves.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_dice"], label="train_dice")
    plt.plot(epochs, history["val_dice"], label="val_dice")
    plt.plot(epochs, history["train_iou"], label="train_iou")
    plt.plot(epochs, history["val_iou"], label="val_iou")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Dice and IoU Curves")
    plt.savefig(OUTPUT_DIR / "metric_curves.png", bbox_inches="tight")
    plt.close()

    print("Training complete.")
    print(f"Best validation Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()