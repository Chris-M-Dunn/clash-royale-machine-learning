from PIL import ImageGrab
import cv2
import numpy as np
import time
import pyautogui

if __name__ == "__main__":
    # NEXT_CARD_BOX = (30, 1275, 100, 1380)
    """ OUTER_LEFT_CARD_BOX = (160, 1130, 310, 1330)
    INNER_LEFT_CARD_BOX = (305, 1130, 460, 1330)
    INNER_RIGHT_CARD_BOX = (450, 1130, 600, 1330)
    OUTER_RIGHT_CARD_BOX = (595, 1130, 745, 1330)

    time.sleep(3)

    game_screen_box = (935, 0, 1695, 1380)
    game_screen = ImageGrab.grab(bbox=game_screen_box)

    cv_image = np.array(game_screen)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    cropped_next_card = cv_image[OUTER_LEFT_CARD_BOX[1]:OUTER_LEFT_CARD_BOX[3], OUTER_LEFT_CARD_BOX[0]:OUTER_LEFT_CARD_BOX[2]]

    cv2.rectangle(cv_image, (30, 1275), (100, 1375), (0,0,255), 2)
    cv2.imshow("Cropped area", cropped_next_card)

    print(cv_image.shape)
    print(OUTER_LEFT_CARD_BOX)

    cv2.waitKey(0)
    cv2.destroyAllWindows() """

    ELIXIR_SLOTS = [
    ((255, 1345), (255, 1355)),
    ((290, 1345), (290, 1355)),
    ((340, 1345), (340, 1355)),
    ((390, 1345), (390, 1355)),
    ((445, 1345), (445, 1355)),
    ((500, 1345), (500, 1355)),
    ((550, 1345), (550, 1355)),
    ((605, 1345), (605, 1355)),
    ((655, 1345), (655, 1355)),
    ((705, 1345), (705, 1355)),
    ]

    time.sleep(3)
    screenshot = ImageGrab.grab(bbox=(935, 0, 1695, 1380))
    cv_image = np.array(screenshot)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # Draw circles or lines for each slot
    for top, bottom in ELIXIR_SLOTS:
        x1, y1 = top
        x2, y2 = bottom
        # Draw a small circle at top and bottom
        cv2.circle(cv_image, (x1, y1), 3, (0, 0, 255), -1)  # red dot
        cv2.circle(cv_image, (x2, y2), 3, (0, 255, 0), -1)  # green dot
        # Draw a line connecting them
        cv2.line(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 1)  # blue line

    # Show the image
    cv2.imshow("Elixir Slots", cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()