from vision import ObjectDetector, UIDetector
import threading
import time

if __name__ == "__main__":
    stop_event = threading.Event()
    ui_detector = UIDetector("runs_cards/classify/train2/weights/best.pt")
    object_detector = ObjectDetector("runs_units/detect/train2/weights/best.pt")

    t1 = threading.Thread(
        target=object_detector.run_object_detection,
        args=(stop_event),
        daemon=True
    )

    t2 = threading.Thread(
        target=ui_detector.run_ui_detection,
        args=(stop_event),
        daemon=True
    )

    t1.start()
    t2.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nTerminating...")
        stop_event.set()

    t1.join()
    t2.join()
    print("Program terminated.")