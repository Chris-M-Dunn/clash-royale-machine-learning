# screen size: 2560x1440
# coordinates for game screen if in full screen: 900, 0, 1686, 1440
# single card icon: 126x158, plus 16 pixels between cards
# next card icon: 58x70

from ultralytics import YOLO
from PIL import ImageGrab
import numpy as np
import cv2
import time

next_card_coordinates = (42, 1295, 100, 1365)
outer_left_card_coordinates = (174, 1160, 300, 1318)
inner_right_card_coordinates = (458, 1160, 584, 1318)
inner_left_card_coordinates = (316, 1160, 442, 1318)
outer_right_card_coordinates = (600, 1160, 726, 1318)

card_locations = {
    "next_card": next_card_coordinates,
    "outer_left_card": outer_left_card_coordinates,
    "inner_left_card": inner_left_card_coordinates,
    "inner_right_card": inner_right_card_coordinates,
    "outer_right_card": outer_right_card_coordinates
}

classification_model = YOLO("runs_cards/classify/train2/weights/best.pt")

while True:
    time.sleep(10)
    game_screen_coordinates = (900, 0, 1686, 1440)
    screenshot = ImageGrab.grab(bbox=game_screen_coordinates)

    cv_image = np.array(screenshot)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    detected_cards = {}

    for name, (x1, y1, x2, y2) in card_locations.items():
        cropped_image = cv_image[y1:y2, x1:x2]

        if cropped_image.size == 0:
            continue

        results = classification_model(cropped_image)

        probabilities = results[0].probs
        class_id = probabilities.top1
        confidence = probabilities.top1conf

        class_name = classification_model.names[class_id]
        detected_cards[name] = (class_name, float(confidence))

    print(detected_cards)