from pathlib import Path
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


TEST_IMAGE_DIR = Path("data/splits/test_images")
TEST_MASK_DIR = Path("data/splits/test_masks")
CHECKPOINT_PATH = Path("models/checkpoints/best_deeplabv3_building.pt")
OUTPUT_DIR = Path("outputs/figures")

IMAGE_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 5


def create_model():
    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    return model


def load_pair(image_path):
    mask_path = TEST_MASK_DIR / f"{image_path.stem}_mask.png"

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_resized = image.resize(IMAGE_SIZE)
    mask_resized = mask.resize(IMAGE_SIZE)

    image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(DEVICE)
    mask_array = np.array(mask_resized)
    mask_array = (mask_array > 0).astype(np.uint8)

    return image_resized, mask_array, image_tensor


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = create_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    image_paths = sorted(TEST_IMAGE_DIR.glob("*.png"))
    selected = random.sample(image_paths, min(NUM_SAMPLES, len(image_paths)))

    for i, image_path in enumerate(selected, start=1):
        image_pil, gt_mask, image_tensor = load_pair(image_path)

        with torch.no_grad():
            logits = model(image_tensor)["out"]
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float().cpu().numpy()[0, 0]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image_pil)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"prediction_{i}.png", bbox_inches="tight")
        plt.close()

    print(f"Saved {len(selected)} prediction figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()