"""
Prepare crack detection dataset
Uses Ultralytics crack-seg dataset (segmentation) for detection training
"""

import os
from pathlib import Path
import yaml
import zipfile
import urllib.request

def find_dataset_directory(data_dir):

    candidates = [
        data_dir / "crack-seg",
        data_dir / "datasets" / "crack-seg",
    ]
    for item in data_dir.rglob("images"):
        if item.is_dir():
            candidates.append(item.parent)

    for candidate in candidates:
        if candidate.exists() and (candidate / "images").exists():
            print(f"Found dataset at: {candidate}")
            return candidate

    return None

def download_crack_seg_dataset():

    print("Downloading..")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    dataset_dir = data_dir / "crack-seg"

    if dataset_dir.exists() and (dataset_dir / "images").exists():
        print(f"\nDataset already exists at: {dataset_dir}")
        verify_dataset(dataset_dir)
        return True

    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip"
    zip_path = data_dir / "crack-seg.zip"

    print(f"\nDownloading from: {url}")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nManual download:")
        print(f"1. Download: {url}")
        print(f"2. Extract to: {data_dir}/crack-seg/")
        return False

    # Extract
    print(f"\nExtracting to: {dataset_dir}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete!")

        zip_path.unlink()

    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

    # Find actual dataset directory (might be nested)
    actual_dataset_dir = find_dataset_directory(data_dir)
    if actual_dataset_dir is None:
        print("Could not find dataset structure")
        return False

    verify_dataset(actual_dataset_dir)
    create_data_yaml(actual_dataset_dir)
    create_readme(actual_dataset_dir)

    return True

def verify_dataset(dataset_dir):

    print(f"\nDataset Statistics:")

    splits = ['train', 'val', 'test']
    stats = {}

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.*")))
            lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            stats[split] = {'images': img_count, 'labels': lbl_count}
            print(f"   - {split.capitalize()}: {img_count} images, {lbl_count} labels")
        else:
            print(f"   - {split.capitalize()}: NOT FOUND")
            stats[split] = {'images': 0, 'labels': 0}

    total_images = sum(s['images'] for s in stats.values())
    total_labels = sum(s['labels'] for s in stats.values())

    print(f"\n   Total: {total_images} images, {total_labels} annotations")

    return stats

def create_data_yaml(dataset_dir):
    dataset_dir.mkdir(parents=True, exist_ok=True)

    yaml_content = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'crack'},
        'nc': 1  # number of classes
    }

    yaml_path = dataset_dir / "data.yaml"

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated: {yaml_path}")

    return yaml_path

def create_readme(dataset_dir):
    #Create data/README.md

    # Get stats
    train_count = len(list((dataset_dir / "images" / "train").glob("*.*")))
    val_count = len(list((dataset_dir / "images" / "val").glob("*.*")))
    test_count = len(list((dataset_dir / "images" / "test").glob("*.*")))

    readme_content = f"""# Crack Detection Dataset

## Source
**Ultralytics Crack-seg Dataset**
- URL: https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip
- Documentation: https://docs.ultralytics.com/datasets/segment/crack-seg/
- License: AGPL-3.0

## Dataset Type
Segmentation dataset (polygon annotations) used for detection training.
YOLOv8 automatically converts segmentation masks to bounding boxes during training.

## Dataset Structure
```
crack-seg/
├── data.yaml          # Dataset configuration
├── images/
│   ├── train/         # {train_count} training images
│   ├── val/           # {val_count} validation images
│   └── test/          # {test_count} test images
└── labels/
    ├── train/         # Segmentation polygons (YOLO format)
    ├── val/           # Segmentation polygons
    └── test/          # Segmentation polygons
```

## Statistics
- **Train:** {train_count} images
- **Validation:** {val_count} images
- **Test:** {test_count} images
- **Total:** {train_count + val_count + test_count} images

## Annotation Format
YOLO segmentation format (normalized polygon coordinates):
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Example from `{dataset_dir.name}`:
```
0 0.037 0.981 0.032 0.960 0.031 0.952 ...
```

YOLOv8 converts these polygons to bounding boxes automatically during detection training.

## Augmentations (Applied During Training)
To simulate drone flight conditions:
- Random resize/crop (scale=0.5)
- Brightness/contrast adjustment (hsv_v=0.4, hsv_s=0.7)
- Motion blur simulation
- JPEG compression artifacts
- Perspective tilt (0.001)
- Horizontal flip (50%)
- Mosaic augmentation
- Color jitter (hsv_h=0.015)

Configured in `train.py` using YOLOv8 built-in parameters.

## Classes
- `0`: crack (single class detection)

## Data Splits
Images are from different concrete surfaces and scenes to prevent leakage.
No overlap between train/val/test splits.

## Usage
```python
from ultralytics import YOLO

# Train detector (uses segmentation data automatically)
model = YOLO('yolov8n.pt')
model.train(data='data/crack-seg/data.yaml', task='detect')
```

## Notes
- Original dataset is for segmentation
- We use it for detection by letting YOLO convert masks to boxes
- This provides more precise annotations than manual bounding boxes
"""

    readme_path = Path("../README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"Created: {readme_path}")

def main():
    print("CRACK DETECTION DATASET PREPARATION")

    success = download_crack_seg_dataset()

    if success:
        print("Dataset prep done")

    else:
        print("\nDataset prep failed")
        print("Go manually and re-run, bro!!")

if __name__ == "__main__":
    main()