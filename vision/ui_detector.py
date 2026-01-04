from ultralytics import YOLO
from PIL import ImageGrab
import numpy as np
import cv2
import time
import pyautogui
import threading
from state import GameState

class UIDetector:
    ELIXIR_BRIGHT_PINK = (240, 137, 244)
    ELIXIR_DARK_PINK = (201, 31, 207)
    EMPTY_ELIXIR_SLOT = (4, 51, 119)
    PARTIALLY_FILLED_TOP_ELIXIR_SLOT = (54, 84, 157)
    PARTIALLY_FILLED_BOTTOM_ELIXIR_SLOT = (51, 72, 151)

    NEXT_CARD_BOX = (31, 1241, 97, 1326)
    OUTER_LEFT_CARD_BOX = (158, 1112, 290, 1282)
    INNER_LEFT_CARD_BOX = (295, 1112, 426, 1282)
    INNER_RIGHT_CARD_BOX = (432, 1112, 565, 1282)
    OUTER_RIGHT_CARD_BOX = (570, 1112, 701, 1282)

    CARD_LOCATIONS = {
        "next_card": NEXT_CARD_BOX,
        "outer_left_card": OUTER_LEFT_CARD_BOX,
        "inner_left_card": INNER_LEFT_CARD_BOX,
        "inner_right_card": INNER_RIGHT_CARD_BOX,
        "outer_right_card": OUTER_RIGHT_CARD_BOX
    }

    # ((top pixel x, top pixel y), (bottom pixel x, bottom pixel y)) <-- elixir slot
    ELIXIR_SLOTS = [
    ((236, 1339), (236, 1351)),
    ((275, 1339), (275, 1351)),
    ((329, 1339), (329, 1351)),
    ((379, 1339), (379, 1351)),
    ((429, 1339), (429, 1351)),
    ((487, 1339), (487, 1351)),
    ((541, 1339), (541, 1351)),
    ((594, 1339), (594, 1351)),
    ((652, 1339), (652, 1351)),
    ((706, 1339), (706, 1351)),
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
            confidence = float(probabilities.top1conf)

            class_name = self.classification_model.names[class_id]
            detected_cards[name] = (class_name, confidence)

        with self.lock:
            self.game_state.cards_in_hand = detected_cards

    @staticmethod
    def is_color_close(color, target, tolerance=50):
        for i in range(3):
            color_rgb_component = int(color[i])
            target_rgb_component = int(target[i])

            if abs(color_rgb_component - target_rgb_component) > tolerance:
                return False

        return True

    def is_elixir_slot_filled(self, frame, top_pixel, bottom_pixel):
        top_color = frame[top_pixel[1], top_pixel[0]]
        bottom_color = frame[bottom_pixel[1], bottom_pixel[0]]

        if (self.is_color_close(top_color, self.ELIXIR_BRIGHT_PINK) and self.is_color_close(bottom_color, self.ELIXIR_DARK_PINK)):
            return True

        if (self.is_color_close(top_color, self.PARTIALLY_FILLED_TOP_ELIXIR_SLOT) and self.is_color_close(bottom_color, self.PARTIALLY_FILLED_BOTTOM_ELIXIR_SLOT)):
            return False

        if self.is_color_close(top_color, self.EMPTY_ELIXIR_SLOT):
            return False

        return False

    def get_elixir_count(self, cv_image):
        elixir = 0

        for top_pixel, bottom_pixel in self.ELIXIR_SLOTS:
            if self.is_elixir_slot_filled(cv_image, top_pixel, bottom_pixel):
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