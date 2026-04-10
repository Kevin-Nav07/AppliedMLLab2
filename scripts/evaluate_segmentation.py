from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


TEST_IMAGE_DIR = Path("data/splits/test_images")
TEST_MASK_DIR = Path("data/splits/test_masks")
CHECKPOINT_PATH = Path("models/checkpoints/best_deeplabv3_building.pt")
OUTPUT_METRICS_PATH = Path("outputs/metrics/test_metrics.txt")

BATCH_SIZE = 2
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


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_score_from_logits(outputs, masks)
            total_iou += iou_score_from_logits(outputs, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    dataset = BuildingSegmentationDataset(TEST_IMAGE_DIR, TEST_MASK_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    criterion = nn.BCEWithLogitsLoss()

    test_loss, test_dice, test_iou = evaluate(model, loader, criterion)

    OUTPUT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Dice: {test_dice:.4f}\n")
        f.write(f"Test IoU: {test_iou:.4f}\n")

    print(f"Using device: {DEVICE}")
    print(f"Test samples: {len(dataset)}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU: {test_iou:.4f}")


if __name__ == "__main__":
    main()