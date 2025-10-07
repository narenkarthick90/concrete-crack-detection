"""
python infer_folder.py --source path/to/folder --view --save --to_video 25
"""

from ultralytics import YOLO
import argparse
from pathlib import Path
import cv2
import os


def images_to_video(image_folder, output_path, fps=25):
    """Create MP4 from folder of annotated images"""

    images = sorted([img for img in os.listdir(image_folder)
                     if img.endswith(('.jpg', '.png', '.jpeg'))])

    if not images:
        print("No images found, upload and try again")
        return

    first_img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Video saved: {output_path}")

#use parser(from Claude)
def main():
    parser = argparse.ArgumentParser(description='Run crack detection on folder')
    parser.add_argument('--source', type=str, required=True, help='Path to folder')
    parser.add_argument('--view', action='store_true', help='Display results')
    parser.add_argument('--save', action='store_true', help='Save annotated images')
    parser.add_argument('--to_video', type=int, default=0,
                        help='Create video at specified FPS (e.g., 25)')
    parser.add_argument('--weights', type=str, default='results/crack_detector/weights/best.pt',
                        help='Model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')

    args = parser.parse_args()

    model = YOLO(args.weights)

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
        stream=True,  # efficient for large folders
    )

    for r in results:
        pass  # predictions already saved

    save_dir = Path('runs/detect/predict')
    print(f" Results saved to: {save_dir}")

    if args.to_video > 0:
        video_path = str(save_dir / 'output.mp4')
        images_to_video(str(save_dir), video_path, fps=args.to_video)


if __name__ == "__main__":
    main()