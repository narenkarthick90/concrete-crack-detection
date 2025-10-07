# Crack Detection Dataset

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
│   ├── train/         # 3717 training images
│   ├── val/           # 200 validation images
│   └── test/          # 112 test images
└── labels/
    ├── train/         # Segmentation polygons (YOLO format)
    ├── val/           # Segmentation polygons
    └── test/          # Segmentation polygons
```

## Statistics
- **Train:** 3717 images
- **Validation:** 200 images
- **Test:** 112 images
- **Total:** 4029 images

## Annotation Format
YOLO segmentation format (normalized polygon coordinates):
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Example from `data`:
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
