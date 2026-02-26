from game_state import GameState
from ultralytics import YOLO
import threading
import numpy
import cv2

# game_board = full_screen[100:1100, 930:1630]

class DetectionModel:
    def __init__(self, model_path: str, game_state: GameState, lock: threading.Lock):
        self.model = YOLO(model_path)
        self.game_state = game_state
        self.lock = lock
        self.frame_number = 0
        
    def map_object_to_game_grid(self, x, y, board_height, board_width):
        game_board_rows = 30
        game_board_columns = 18

        column = int((x / board_width) * game_board_columns)
        row = int((y / board_height) * game_board_rows)

        return row, column

    def detect_objects(self, image):
            self.frame_number += 1

            if self.frame_number % 3 != 0:
                return

            results = self.model(image, imgsz=1280, conf=0.50, iou=0.5, verbose=False)

            game_board_height, game_board_width, = image.shape[:2] # take only the height and width, ignore the color channels
            game_board_grid = numpy.zeros((30, 18), dtype=numpy.int8)

            detected_enemy_units = []
            detected_enemy_buildings = []

            detected_ally_units = []
            detected_ally_buildings = []

            for bounding_box in results[0].boxes:
                class_id = int(bounding_box.cls[0])
                object_name = self.model.names[class_id]

                x1, y1, x2, y2 = bounding_box.xyxy[0]
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                row, column = self.map_object_to_game_grid(x_center, y_center, game_board_height, game_board_width)

                if object_name == "ally_troop":
                    detected_ally_units.append(object_name)
                    game_board_grid[row, column] = 1

                if object_name == "ally_king_tower" or object_name == "ally_tower" or object_name == "ally_building":
                    detected_ally_buildings.append(object_name)
                    game_board_grid[row, column] = 2

                if object_name == "enemy_troop":
                    detected_enemy_units.append(object_name)
                    game_board_grid[row, column] = 3

                if object_name == "enemy_king_tower" or object_name == "enemy_tower" or object_name == "enemy_building":
                    detected_enemy_buildings.append(object_name)
                    game_board_grid[row, column] = 4

            with self.lock:
                self.game_state.ally_units_on_board = detected_ally_units
                self.game_state.ally_buildings_on_board = detected_ally_buildings
                self.game_state.enemy_units_on_board = detected_enemy_units
                self.game_state.enemy_buildings_on_board = detected_enemy_buildings
                self.game_state.game_board_grid = game_board_grid

            annotated = results[0].plot()
            cv2.imshow("With Annotations", annotated)