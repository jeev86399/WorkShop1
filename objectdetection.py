# Install required libraries
!pip install ultralytics opencv-python matplotlib

# Imports
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Upload image
uploaded = files.upload()

# Read uploaded image
image_path = list(uploaded.keys())[0]
img = cv2.imread(image_path)

# Check if image loaded correctly
if img is None:
    raise ValueError("Image not loaded. Check the file path.")

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(8,8))
plt.imshow(img_rgb)
plt.axis('off')
plt.title("Original Image")
plt.show()

# ----------------------------
# Perform object detection
# ----------------------------
results = model(img_rgb)

# Draw bounding boxes
annotated_image = results[0].plot()

# Display annotated image
plt.figure(figsize=(10,10))
plt.imshow(annotated_image)
plt.axis('off')
plt.title("YOLOv8 Detection Result")
plt.show()
