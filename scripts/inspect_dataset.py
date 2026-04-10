from datasets import load_dataset

ds = load_dataset("keremberke/satellite-building-segmentation", name="full")

print(ds)
print("Train size:", len(ds["train"]))

example = ds["train"][0]
print("Keys:", example.keys())
print("Objects keys:", example["objects"].keys())
print("First bbox:", example["objects"]["bbox"][0] if len(example["objects"]["bbox"]) > 0 else None)
print("Image size:", example["image"].size)