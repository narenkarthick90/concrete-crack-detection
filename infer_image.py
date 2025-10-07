"""
Inference on single image
Usage: python infer_image.py --source path/to/image.jpg --view --save
"""

from ultralytics import YOLO
import argparse
from pathlib import Path


def main():
    #use parser (from Claude)
    parser = argparse.ArgumentParser(description='Run crack detection on single image')
    parser.add_argument('--source', type=str, required=True, help='Path to image')
    parser.add_argument('--view', action='store_true', help='Display result')
    parser.add_argument('--save', action='store_true', help='Save annotated image')
    parser.add_argument('--weights', type=str, default='results/crack_detector/weights/best.pt',
                        help='Model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')

    args = parser.parse_args()

    # Load model
    model = YOLO(args.weights)

    # Run inference
    results = model.predict(
        source=args.source,
        save=args.save,
        show=args.view,
        conf=args.conf,
        save_txt=True,
        save_conf=True,
        project='runs/detect',
        name='predict',
        exist_ok=True,
    )

    if args.save:
        print(f"Annotated image saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()