from classification_model import ClassificationModel
from detection_model import DetectionModel
from elixir_manager import ElixirManager
from game_state import GameState
from PIL import ImageGrab
import threading
import numpy
import time
import cv2
import mss
import os

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

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        try:
            time.sleep(3)

            last_ui_update = 0
            ELIXIR_INTERVAL = 2.7

            while not stop_event.is_set():
                now = time.time()

                game_screen_box = (935, 0, 1695, 1380)
                screenshot = ImageGrab.grab(bbox=game_screen_box)
                cv_image = numpy.array(screenshot)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)

                object_detector.detect_objects()
                card_classifier.classify_cards()

                if now - last_ui_update >= ELIXIR_INTERVAL:
                    elixir_manager.get_elixir_count(cv_image)
                    last_ui_update = now
                    
                with lock:
                    clear_terminal()
                    game_state.print_game_state()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nTerminating...")
            stop_event.set()

        cv2.destroyAllWindows()
        print("Program terminated.")

if __name__ == "__main__":
    main()