from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

image_path = sorted(Path("data/splits/train_images").glob("*.png"))[0]
mask_path = Path("data/splits/train_masks") / f"{image_path.stem}_mask.png"

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title("Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()