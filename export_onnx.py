"""
Requirement: static shape 1x3x720x1280
"""

from ultralytics import YOLO
from pathlib import Path


def export_onnx(weights='results/crack_detector/weights/best.pt'):
    #Exporting YOLO model to ONNX
    print("Exporting to ONNX")

    model = YOLO(weights)

    # Export to ONNX
    onnx_path = model.export(
        format='onnx',
        imgsz=[720, 1280],
        simplify=True,
        opset=12,
        dynamic=False,  # static shape for TensorRT(not so sure about it)
    )

    print(f"\nONNX export complete")
    print(f"Model: {onnx_path}")

    # Check the size
    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB {'Good' if size_mb <= 25 else 'Reduce Size'}")

    return onnx_path

if __name__ == "__main__":
    export_onnx()