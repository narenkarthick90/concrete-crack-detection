import cv2
import os

image_folder = "Artifacts"
video_name = "Artifacts/crack_detection.mp4"

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()
print(f"Video saved to {video_name}")
