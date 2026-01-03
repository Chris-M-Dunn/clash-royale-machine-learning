import cv2
import torch
import mss
from ultralytics import YOLO
import numpy as np
import time

DETECTION_MODEL = YOLO("runs_units/detect/train2/weights/best.pt")

""" image_file_path = 'cv_tests/detection/test3.png'
test_image = cv2.imread(image_file_path)
results = DETECTION_MODEL(test_image, conf=0.10, iou=0.5)
annotated = results[0].plot()
cv2.imshow("result", annotated)
cv2.waitKey(0) """

""" time.sleep(5)
with mss.mss() as sct:
    monitor = sct.monitors[1]

    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        arena = frame[100:1100, 930:1630]

        cv2.imshow("full_screen", frame)
        cv2.imshow("arena_crop", arena)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows() """

if __name__ == "__main__":
    time.sleep(5)
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        frame_number = 1

        while True:
            if frame_number % 100 != 0:
                frame_number += 1
                continue

            full_screen = np.array(sct.grab(monitor))
            full_screen = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2BGR)

            bots_vision = full_screen[100:1100, 930:1630]

            cv2.imshow("Bots vision", bots_vision)

            results = DETECTION_MODEL(bots_vision, imgsz=1280, conf=0.25, iou=0.5)
            annotated = results[0].plot()

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                h = int(y2 - y1)
                print("box height:", h)

            cv2.imshow("With detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"Processed frame {frame_number}\n")
            frame_number += 1

    cv2.destroyAllWindows()