from ultralytics import YOLO
from game_state import GameState
import threading
import numpy
import cv2
import mss

class DetectionModel:
    def __init__(self, model_path: str, game_state: GameState, lock: threading.Lock):
        self.model = YOLO(model_path)
        self.game_state = game_state
        self.lock = lock
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.frame_number = 0
        
    def detect_objects(self):
            self.frame_number += 1

            if self.frame_number % 3 != 0:
                return

            full_screen = numpy.array(self.sct.grab(self.monitor))
            full_screen = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2BGR)

            bots_vision = full_screen[100:1100, 930:1630]

            results = self.model(bots_vision, imgsz=1280, conf=0.25, iou=0.5, verbose=False)

            detected_enemy_units = []
            detected_enemy_buildings = []

            detected_ally_units = []
            detected_ally_buildings = []

            for bounding_box in results[0].boxes:
                class_id = int(bounding_box.cls[0])
                object_name = self.model.names[class_id]

                if object_name == "enemy_troop":
                    detected_enemy_units.append(object_name)

                if object_name == "enemy_king_tower" or object_name == "enemy_tower" or object_name == "enemy_building":
                    detected_enemy_buildings.append(object_name)

                if object_name == "ally_troop":
                    detected_ally_units.append(object_name)

                if object_name == "ally_king_tower" or object_name == "ally_tower" or object_name == "ally_building":
                    detected_ally_buildings.append(object_name)

            with self.lock:
                self.game_state.ally_units_on_board = detected_ally_units
                self.game_state.ally_buildings_on_board = detected_ally_buildings
                self.game_state.enemy_units_on_board = detected_enemy_units
                self.game_state.enemy_buildings_on_board = detected_enemy_buildings

            annotated = results[0].plot()
            cv2.imshow("With Annotations", annotated)