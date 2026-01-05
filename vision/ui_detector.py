from ultralytics import YOLO
from PIL import ImageGrab
import numpy as np
import cv2
import time
import threading
from state import GameState

class UIDetector:
    ELIXIR_BRIGHT_PINK = (240, 137, 244)

    OUTER_LEFT_CARD_BOX = (160, 1130, 310, 1330)
    INNER_LEFT_CARD_BOX = (305, 1130, 460, 1330)
    INNER_RIGHT_CARD_BOX = (450, 1130, 600, 1330)
    OUTER_RIGHT_CARD_BOX = (595, 1130, 745, 1330)

    CARD_LOCATIONS = {
        "outer_left_card": OUTER_LEFT_CARD_BOX,
        "inner_left_card": INNER_LEFT_CARD_BOX,
        "inner_right_card": INNER_RIGHT_CARD_BOX,
        "outer_right_card": OUTER_RIGHT_CARD_BOX
    }

    # (x, y) <-- elixir slot pixel coordinates
    ELIXIR_SLOTS = [
    (255, 1345),
    (290, 1345),
    (340, 1345),
    (390, 1345),
    (445, 1345),
    (500, 1345),
    (550, 1345),
    (605, 1345),
    (655, 1345),
    (705, 1345)
    ]

    def __init__(self, model_path: str, game_state: GameState, lock: threading.Lock):
        self.classification_model = YOLO(model_path)
        self.game_state = game_state
        self.lock = lock

    def get_cards_in_hand(self, cv_image):
        detected_cards = {}

        for name, (x1, y1, x2, y2) in self.CARD_LOCATIONS.items():
            cropped_image = cv_image[y1:y2, x1:x2]

            if cropped_image.size == 0:
                continue

            results = self.classification_model(cropped_image, verbose=False)
            
            probabilities = results[0].probs
            class_id = probabilities.top1

            class_name = self.classification_model.names[class_id]
            detected_cards[name] = class_name

        with self.lock:
            self.game_state.cards_in_hand = detected_cards

    @staticmethod
    def is_color_close(color, target, tolerance=80):
        for i in range(3):
            color_rgb_component = int(color[i])
            target_rgb_component = int(target[i])

            if abs(color_rgb_component - target_rgb_component) > tolerance:
                return False

        return True

    def is_elixir_slot_filled(self, frame, pixel_coordinates):
        pixel_color = frame[pixel_coordinates[1], pixel_coordinates[0]]

        if (self.is_color_close(pixel_color, self.ELIXIR_BRIGHT_PINK)):
            return True

        return False

    def get_elixir_count(self, cv_image):
        elixir = 0

        for pixel in self.ELIXIR_SLOTS:
            if self.is_elixir_slot_filled(cv_image, pixel):
                elixir += 1

        with self.lock:
            self.game_state.elixir_count = elixir

    def run_ui_detection(self, stop_event):
        time.sleep(2.3)

        while not stop_event.is_set():
            time.sleep(2.7)
            
            game_screen_box = (935, 0, 1695, 1380)
            screenshot = ImageGrab.grab(bbox=game_screen_box)

            cv_image = np.array(screenshot)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            self.get_cards_in_hand(cv_image)
            self.get_elixir_count(cv_image)