from game_state import GameState
import color_checker
import threading

class ElixirManager():
    ELIXIR_PINK = (240, 137, 244)

    # (x, y) <-- elixir slot pixel coordinates
    ELIXIR_SLOTS = [
        (255, 1345),
        (290, 1345),
        (340, 1345),
        (390, 1345),
        (445, 1345),
        (500, 1345),
        (550, 1345),
        (605, 1345),
        (655, 1345),
        (705, 1345)
    ]

    def __init__(self, game_state: GameState, lock: threading.Lock):
        self.color_checker = color_checker.ColorChecker()
        self.game_state = game_state
        self.lock = lock

    def get_elixir_count(self, image):
        elixir = 0

        for pixel in self.ELIXIR_SLOTS:
            pixel_color = image[pixel[1], pixel[0]]

            if self.color_checker.is_color_close(pixel_color, self.ELIXIR_PINK):
                elixir += 1

        with self.lock:
            self.game_state.elixir_count = elixir