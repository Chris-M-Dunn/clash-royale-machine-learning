import cv2
import torch
import mss
from ultralytics import YOLO
import numpy as np
import time
import threading
from state import GameState

class ObjectDetector:
    def __init__(self, model_path: str, game_state: GameState, lock: threading.Lock):
        self.model = YOLO(model_path)
        self.game_state = game_state
        self.lock = lock
        
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

                results = self.model(bots_vision, imgsz=1280, conf=0.25, iou=0.5, verbose=False)

                detected_units = []
                for bounding_box in results[0].boxes:
                    class_id = int(bounding_box.cls[0])
                    unit_name = self.model.names[class_id]
                    detected_units.append(unit_name)

                with self.lock:
                    self.game_state.enemy_units_on_board = detected_units

                annotated = results[0].plot()
                cv2.imshow("With Annotations", annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

                frame_number += 1 

        cv2.destroyAllWindows()