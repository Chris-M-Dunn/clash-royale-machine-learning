from ultralytics import YOLO
from game_state import GameState
import threading

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

    def classify_cards(self, image):
        cards_in_hand = {}

        for card, (x1, y1, x2, y2) in self.CARD_LOCATIONS.items():
            cropped_image = image[y1:y2, x1:x2]

            if cropped_image.size == 0:
                continue

            results = self.classification_model(cropped_image, verbose=False)
            
            probabilities = results[0].probs
            class_id = probabilities.top1

            card_name = self.classification_model.names[class_id]
            cards_in_hand[card] = card_name

        with self.lock:
            self.game_state.cards_in_hand = cards_in_hand