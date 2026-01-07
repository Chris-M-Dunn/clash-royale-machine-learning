from classification_model import ClassificationModel
from detection_model import DetectionModel
from elixir_manager import ElixirManager
from game_state import GameState
from PIL import ImageGrab
import threading
import numpy
import time
import cv2
import os

# possibly use mss in the future for better performance

def clear_terminal():
    if os.name == "nt":
        os.system("cls")

def main():
    game_state = GameState()
    lock = threading.Lock()
    stop_event = threading.Event()

    card_classifier = ClassificationModel("classification_runs/classify/train2/weights/best.pt", game_state, lock)
    object_detector = DetectionModel("detection_runs/detect/train2/weights/best.pt", game_state, lock)
    elixir_manager = ElixirManager(game_state, lock)

    try:
        time.sleep(3)

        last_elixir_update = 0
        ELIXIR_INTERVAL = 2.7

        game_screen_box = (935, 0, 1695, 1380)

        while not stop_event.is_set():
            now = time.time()

            game_screen = ImageGrab.grab(bbox=game_screen_box)
            converted_game_screen = numpy.array(game_screen)
            converted_game_screen = cv2.cvtColor(converted_game_screen, cv2.COLOR_RGB2BGR)
            game_board = converted_game_screen[100:1100, 50:715]

            object_detector.detect_objects(game_board)
            card_classifier.classify_cards(converted_game_screen)

            if now - last_elixir_update >= ELIXIR_INTERVAL:
                elixir_manager.get_elixir_count(converted_game_screen)
                last_elixir_update = now

            with lock:
                clear_terminal()
                game_state.print_game_state()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                
            # run at 20fps
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nTerminating...")
        stop_event.set()

    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == "__main__":
    main()