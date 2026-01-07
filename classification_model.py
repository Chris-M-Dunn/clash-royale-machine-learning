from ultralytics import YOLO
from game_state import GameState
from PIL import ImageGrab
import threading
import numpy
import time
import cv2

class ClassificationModel:
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

    def __init__(self, model_path: str, game_state: GameState, lock: threading.Lock):
        self.classification_model = YOLO(model_path)
        self.game_state = game_state
        self.lock = lock

    def get_cards_in_hand(self, cv_image):
        cards_in_hand = {}

        for card, (x1, y1, x2, y2) in self.CARD_LOCATIONS.items():
            cropped_image = cv_image[y1:y2, x1:x2]

            if cropped_image.size == 0:
                continue

            results = self.classification_model(cropped_image, verbose=False)
            
            probabilities = results[0].probs
            class_id = probabilities.top1

            card_name = self.classification_model.names[class_id]
            cards_in_hand[card] = card_name

        with self.lock:
            self.game_state.cards_in_hand = cards_in_hand

    def classify_cards(self):
            game_screen_box = (935, 0, 1695, 1380)
            screenshot = ImageGrab.grab(bbox=game_screen_box)

            cv_image = numpy.array(screenshot)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            self.get_cards_in_hand(cv_image)