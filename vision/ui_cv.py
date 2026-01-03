# screen size: 2560x1440
# coordinates for game screen if in full screen: 900, 0, 1686, 1440
# single card icon: 126x158, plus 16 pixels between cards
# next card icon: 58x70

from ultralytics import YOLO
from PIL import ImageGrab
import numpy as np
import cv2
import time
import pyautogui

ELIXIR_BRIGHT_PINK = (240, 137, 244)
ELIXIR_DARK_PINK = (201, 31, 207)
EMPTY_ELIXIR_SLOT = (4, 51, 119)
PARTIALLY_FILLED_TOP_ELIXIR_SLOT = (54, 84, 157)
PARTIALLY_FILLED_BOTTOM_ELIXIR_SLOT = (51, 72, 151)

X_AXIS_PADDING = 10
Y_AXIS_PADDING = 20

NEXT_CARD_BOX = (42-X_AXIS_PADDING, 1295, 100, 1365+Y_AXIS_PADDING)
OUTER_LEFT_CARD_BOX = (174-X_AXIS_PADDING, 1160, 300, 1318+Y_AXIS_PADDING)
INNER_LEFT_CARD_BOX = (316-X_AXIS_PADDING, 1160, 442, 1318+Y_AXIS_PADDING)
INNER_RIGHT_CARD_BOX = (458-X_AXIS_PADDING, 1160, 584, 1318+Y_AXIS_PADDING)
OUTER_RIGHT_CARD_BOX = (600-X_AXIS_PADDING, 1160, 726, 1318+Y_AXIS_PADDING)

CARD_LOCATIONS = {
    "next_card": NEXT_CARD_BOX,
    "outer_left_card": OUTER_LEFT_CARD_BOX,
    "inner_left_card": INNER_LEFT_CARD_BOX,
    "inner_right_card": INNER_RIGHT_CARD_BOX,
    "outer_right_card": OUTER_RIGHT_CARD_BOX
}

# ((top pixel x, top pixel y), (bottom pixel x, bottom pixel y)) <-- elixir slot
ELIXIR_SLOTS = [
    ((244, 1397), (244, 1410)),
    ((284, 1397), (284, 1410)),
    ((340, 1397), (340, 1410)),
    ((392, 1397), (392, 1410)),
    ((444, 1397), (444, 1410)),
    ((504, 1397), (504, 1410)),
    ((560, 1397), (560, 1410)),
    ((614, 1397), (614, 1410)),
    ((674, 1397), (674, 1410)),
    ((730, 1397), (730, 1410)),
]

CLASSIFICATION_MODEL = YOLO("runs_cards/classify/train2/weights/best.pt")

def get_cards_in_hand():
    detected_cards = {}

    for name, (x1, y1, x2, y2) in CARD_LOCATIONS.items():
        cropped_image = cv_image[y1:y2, x1:x2]

        if cropped_image.size == 0:
            continue

        results = CLASSIFICATION_MODEL(cropped_image, verbose=False)

        probabilities = results[0].probs
        class_id = probabilities.top1
        confidence = probabilities.top1conf

        class_name = CLASSIFICATION_MODEL.names[class_id]
        detected_cards[name] = (class_name, float(confidence))

    print("\nCards in hand:")
    for card_slot, (card_name, _) in detected_cards.items():
        print(f"{card_slot}: {card_name}")

def is_color_close(color, target, tolerance=50):
    for i in range(3):
        color_rgb_component = int(color[i])
        target_rgb_component = int(target[i])

        if abs(color_rgb_component - target_rgb_component) > tolerance:
            return False

    return True

def is_elixir_slot_filled(frame, top_pixel, bottom_pixel):
    top_color = frame[top_pixel[1], top_pixel[0]]
    bottom_color = frame[bottom_pixel[1], bottom_pixel[0]]

    if (is_color_close(top_color, ELIXIR_BRIGHT_PINK) and is_color_close(bottom_color, ELIXIR_DARK_PINK)):
        return True

    if (is_color_close(top_color, PARTIALLY_FILLED_TOP_ELIXIR_SLOT) and is_color_close(bottom_color, PARTIALLY_FILLED_BOTTOM_ELIXIR_SLOT)):
        return False

    if is_color_close(top_color, EMPTY_ELIXIR_SLOT):
        return False

    return False

def get_elixir_count():
    elixir = 0

    for top_pixel, bottom_pixel in ELIXIR_SLOTS:
        if is_elixir_slot_filled(cv_image, top_pixel, bottom_pixel):
            elixir += 1

    print(f"Elixir count: {elixir}")

if __name__ == "__main__": 
    while True:
        time.sleep(5)
        game_screen_box = (900, 0, 1686, 1440)
        screenshot = ImageGrab.grab(bbox=game_screen_box)

        cv_image = np.array(screenshot)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        get_cards_in_hand()
        get_elixir_count()