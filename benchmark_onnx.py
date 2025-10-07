import onnxruntime as ort
import numpy as np
import psutil, time, os, argparse, glob
from PIL import Image

def preprocess_image(image_path, input_shape):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_shape[3], input_shape[2]))
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))  # HWC -> CHW
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

def benchmark_onnx(model_path, folder_path, warmup=5):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available_providers = ort.get_available_providers()
    provider = 'CUDAExecutionProvider' if 'CUDAExecutionProvider' in available_providers else 'CPUExecutionProvider'
    print(f"Using provider: {provider}")

    session = ort.InferenceSession(model_path, providers=[provider])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) +
                         glob.glob(os.path.join(folder_path, "*.png")))

    #if there's no image, then:
    if not image_paths:
        print("No images found in the folder.")
        return

    print(f"Found {len(image_paths)} images in {folder_path}")

    #Warmup
    sample = preprocess_image(image_paths[0], input_shape)
    for _ in range(warmup):
        _ = session.run(None, {input_name: sample})

    #Benchmark across folder
    start_time = time.time()
    for img_path in image_paths:
        img_data = preprocess_image(img_path, input_shape)
        _ = session.run(None, {input_name: img_data})
    end_time = time.time()

    total_time = (end_time - start_time) * 1000
    avg_time = total_time / len(image_paths)
    fps = 1000 / avg_time

    #Memory usage and model size (use psutil)
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)
    model_size = os.path.getsize(model_path) / (1024 * 1024)

    # Results
    print("\nBenchmark Results:")
    print(f"Processed {len(image_paths)} images")
    print(f"Average inference time: {avg_time:.2f} ms/image")
    print(f"FPS (approx): {fps:.2f}")
    print(f"Memory usage: {mem_usage:.2f} MB")
    print(f"Model size: {model_size:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="crack_detector.onnx", help="Path to ONNX model")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing images")
    args = parser.parse_args()

    benchmark_onnx(args.model, args.folder)
