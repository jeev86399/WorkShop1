pip install opencv-python numpy
!wget https://pjreddie.com/media/files/yolov3.weights
!wget wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
!wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
from google.colab import files
uploaded = files.upload()
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg.1')
with open('coco.names','r') as f:
  classes=f.read().strip().split("\n")
layer_names=net.getLayerNames()
output_layers=[layer_names[i-1] for i in net.getUnconnectedOutLayers()]
img=cv2.imread('WIN_20250521_12_50_16_Pro (1).jpg')
hieght,width,channels=img.shape
blob=cv2.dnn.blobFromImage(img, 1/255.0, (416,416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
outputs=net.forward(output_layers)
boxes=[]
conflidence=[]
class_ids = []
confidences = []
boxes = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  
            # Confidence threshold
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * hieght)
            w = int(detection[2] * width)
            h = int(detection[3] * hieght)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, color, 2)

# Display the output image
cv2_imshow(img)