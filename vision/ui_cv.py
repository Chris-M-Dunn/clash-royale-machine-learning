import os
import cv2
import torch
from ultralytics import YOLO

detection_model = YOLO("runs_cards/classify/train2/weights/best.pt")
image_file_path = 'cv_tests/classification/card10.png'
test_image = cv2.imread(image_file_path)
results = detection_model(test_image, conf=0.10, iou=0.5)
annotated = results[0].plot()
cv2.imshow("result", annotated)
cv2.waitKey(0)