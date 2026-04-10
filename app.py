import io
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from flask import Flask, request, jsonify
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from dotenv import load_dotenv

load_dotenv()

IMAGE_SIZE = (256, 256)
PORT = int(os.getenv("PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "models/checkpoints/best_deeplabv3_building.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

model = None

image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def create_model():
    segmentation_model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    segmentation_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    return segmentation_model


def get_model():
    global model

    if model is None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

        temp_model = create_model().to(DEVICE)
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        temp_model.load_state_dict(state_dict)
        temp_model.eval()

        model = temp_model

    return model


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "task": "house-segmentation",
        "device": DEVICE,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "checkpoint_path": str(CHECKPOINT_PATH)
    })


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Send a file field named 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file."}), 400

    try:
        segmentation_model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Model loading failed: {str(e)}"}), 500

    original_size = image.size
    input_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    try:
        with torch.no_grad():
            logits = segmentation_model(input_tensor)["out"]
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float().cpu().numpy()[0, 0]
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(file.filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mask_image = (pred_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_image, mode="L")
    mask_filename = f"{base_name}_pred_mask_{timestamp}.png"
    mask_path = output_dir / mask_filename
    mask_pil.save(mask_path)

    resized_image = image.resize((pred_mask.shape[1], pred_mask.shape[0])).convert("RGB")
    image_array = np.array(resized_image).copy()

    overlay = image_array.copy()
    overlay[pred_mask > 0.5] = [255, 0, 0]

    blended = (0.6 * image_array + 0.4 * overlay).astype(np.uint8)
    overlay_pil = Image.fromarray(blended)

    overlay_filename = f"{base_name}_overlay_{timestamp}.png"
    overlay_path = output_dir / overlay_filename
    overlay_pil.save(overlay_path)

    house_pixels = int(pred_mask.sum())
    total_pixels = int(pred_mask.size)
    house_ratio = house_pixels / total_pixels

    return jsonify({
        "message": "Segmentation completed",
        "original_width": original_size[0],
        "original_height": original_size[1],
        "mask_width": int(pred_mask.shape[1]),
        "mask_height": int(pred_mask.shape[0]),
        "house_pixels": house_pixels,
        "total_pixels": total_pixels,
        "house_ratio": house_ratio,
        "saved_mask_path": str(mask_path),
        "saved_overlay_path": str(overlay_path)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=FLASK_DEBUG)