import cv2
import torch
import mss
from ultralytics import YOLO
import numpy as np
import time

class ObjectDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        
    def run_object_detection(self, stop_event):
        time.sleep(5)
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            frame_number = 1

            while not stop_event.is_set():
                if frame_number % 3 != 0:
                    frame_number += 1
                    continue

                full_screen = np.array(sct.grab(monitor))
                full_screen = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2BGR)

                bots_vision = full_screen[100:1100, 930:1630]

                cv2.imshow("Bots vision", bots_vision)

                results = self.model(bots_vision, imgsz=1280, conf=0.25, iou=0.5, verbose=False)
                annotated = results[0].plot()

                """ for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    h = int(y2 - y1)
                    print("box height:", h) """

                cv2.imshow("With detection", annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

                # print(f"Processed frame {frame_number}\n")
                frame_number += 1 

        cv2.destroyAllWindows()