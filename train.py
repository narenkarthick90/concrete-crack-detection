from ultralytics import YOLO
import yaml
from pathlib import Path

def train():
    """Train YOLOv8 nano detector on segmentation data"""
    with open("models/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    print("Training YOLO Crack-Detector")

    print("\nNote: Using segmentation annotations for detection training")
    print("YOLOv8 automatically converts polygons to bounding boxes\n")

    model = YOLO('yolov8n.pt')


    results = model.train(
        data=cfg['data'],

        #For 'detect', not 'segment'
        task='detect',

        # Training params
        epochs=cfg['epochs'],
        batch=cfg['batch'],
        imgsz=cfg['imgsz'],


        patience=5,  #req: stop after 5 epochs of no improvement

        # Augmentations (for drone conditions)
        hsv_h=0.015,           # hue
        hsv_s=0.7,             # saturation
        hsv_v=0.4,             # brightness/contrast
        degrees=10.0,          # rotation
        translate=0.1,         # translation
        scale=0.5,             # random resize
        perspective=0.001,     # perspective tilt
        flipud=0.0,            # no vertical flip
        fliplr=0.5,            # horizontal flip
        mosaic=1.0,            # mosaic augmentation

        amp=True,              # FP16 mixed precision
        cache=True,


        project='results',
        name='crack_detector',
        exist_ok=True,

        optimizer='Adam',
        lr0=cfg.get('lr', 0.01),
        device=cfg.get('device', 0),
        workers=8,

        val=True,
        plots=True,            # save PR curve, confusion matrix
        save=True,
        save_period=5,
        verbose=True,
    )
    print("Training Done!")
    best_weights = Path("results/crack_detector/weights/best.pt")
    print(f"\nBest weights: {best_weights}")

    if best_weights.exists():
        size_mb = best_weights.stat().st_size / (1024 * 1024)
        status = 'Yes' if size_mb <= 25 else 'No'
        print(f"Model size: {size_mb:.2f} MB {status}")

    metrics = results.results_dict
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)

    print(f"\nDetection Metrics:")
    print(f"   - mAP@0.5: {map50:.4f} {'Yes, done' if map50 >= 0.65 else 'No, try again'} (target: ≥0.65)")
    print(f"   - mAP@0.5:0.95: {map50_95:.4f}")
    print(f"\nPlots saved to: results/crack_detector/")
    print(f"   - PR curve: results/crack_detector/PR_curve.png")
    print(f"   - Confusion matrix: results/crack_detector/confusion_matrix.png")
    print(f"   - Training curves: results/crack_detector/results.png")

    return results

if __name__ == "__main__":
    train()