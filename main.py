from vision import ObjectDetector, UIDetector
from state import GameState
import threading
import time
import os
import sys

def clear_terminal():
    if os.name == "nt":
        os.system("cls")

if __name__ == "__main__":
    game_state = GameState()
    lock = threading.Lock()
    stop_event = threading.Event()
    
    ui_detector = UIDetector("runs_cards/classify/train2/weights/best.pt", game_state, lock)
    object_detector = ObjectDetector("runs_units/detect/train2/weights/best.pt", game_state, lock)

    t1 = threading.Thread(target=object_detector.run_object_detection, args=(stop_event,), daemon=True)
    t2 = threading.Thread(target=ui_detector.run_ui_detection, args=(stop_event,), daemon=True)

    t1.start()
    t2.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)

            with lock: 
                clear_terminal()

                print("\n----- GAME STATE -----")

                print(f"\nCards in Hand:")
                for card_slot, card in game_state.cards_in_hand.items():
                    print(f"  - {card_slot}: {card}")

                print(f"\nElixir Count: {game_state.elixir_count}")

                print(f"\nEnemy Units on Board:")
                for unit in game_state.enemy_units_on_board:
                    print(f"  - {unit}")

    except KeyboardInterrupt:
        print("\nTerminating...")
        stop_event.set()

    t1.join()
    t2.join()

    print("Program terminated.")