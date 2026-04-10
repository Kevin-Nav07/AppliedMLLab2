from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw


DATASET_NAME = "keremberke/satellite-building-segmentation"
DATASET_CONFIG = "full"

OUTPUT_ROOT = Path("data/splits")


def ensure_dirs():
    for split in ["train", "validation", "test"]:
        (OUTPUT_ROOT / f"{split}_images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / f"{split}_masks").mkdir(parents=True, exist_ok=True)


def draw_bbox(draw, bbox):
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    draw.rectangle([x, y, x2, y2], fill=255)


def draw_segmentation(draw, segmentation):
    """
    Handles COCO-style polygon segmentation.
    segmentation may be:
    - a flat list: [x1, y1, x2, y2, ...]
    - a list of flat lists: [[x1, y1, x2, y2, ...], [...]]
    """
    if not segmentation:
        return

    # if it's already one polygon list
    if isinstance(segmentation, list) and len(segmentation) > 0 and isinstance(segmentation[0], (int, float)):
        if len(segmentation) >= 6:
            points = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
            draw.polygon(points, fill=255)
        return

    # if it's a list of polygons
    if isinstance(segmentation, list):
        for poly in segmentation:
            if isinstance(poly, list) and len(poly) >= 6:
                points = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                draw.polygon(points, fill=255)


def build_mask(example):
    width = example["width"]
    height = example["height"]

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    objects = example["objects"]
    segmentations = objects.get("segmentation", [])
    bboxes = objects.get("bbox", [])

    num_objects = max(len(segmentations), len(bboxes))

    for i in range(num_objects):
        used_segmentation = False

        if i < len(segmentations):
            seg = segmentations[i]
            if seg:
                draw_segmentation(draw, seg)
                used_segmentation = True

        if not used_segmentation and i < len(bboxes):
            bbox = bboxes[i]
            if bbox:
                draw_bbox(draw, bbox)

    return mask


def save_split(dataset_split, split_name):
    image_dir = OUTPUT_ROOT / f"{split_name}_images"
    mask_dir = OUTPUT_ROOT / f"{split_name}_masks"

    for idx, example in enumerate(dataset_split):
        image = example["image"].convert("RGB")
        mask = build_mask(example)

        image_id = example["image_id"]
        stem = f"{image_id}"

        image_path = image_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}_mask.png"

        image.save(image_path)
        mask.save(mask_path)

        if idx % 250 == 0:
            print(f"[{split_name}] saved {idx}/{len(dataset_split)}")


def main():
    ensure_dirs()

    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG)

    save_split(ds["train"], "train")
    save_split(ds["validation"], "validation")
    save_split(ds["test"], "test")

    print("Done preparing dataset.")


if __name__ == "__main__":
    main()