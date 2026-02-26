from dataclasses import dataclass, field
import numpy

@dataclass
class GameState:
    elixir_count: int = 0
    cards_in_hand: dict = field(default_factory=dict)
    ally_units_on_board: list = field(default_factory=list)
    ally_buildings_on_board: list = field(default_factory=list)
    enemy_units_on_board: list = field(default_factory=list)
    enemy_buildings_on_board: list = field(default_factory=list)
    game_board_grid: numpy.ndarray = field(default_factory=lambda: numpy.zeros((30, 18), dtype=numpy.int8))

    def print_game_state(self):
        print("\n----- GAME STATE -----")

        print(f"\nCards in Hand:")
        for card_slot, card in self.cards_in_hand.items():
            print(f"  - {card_slot}: {card}")

        print(f"\nElixir Count: {self.elixir_count}")

        print(f"\nAlly Units on Board:")
        for unit in self.ally_units_on_board:
            print(f"  - {unit}")

        print(f"\nAlly Buildings on Board:")
        for building in self.ally_buildings_on_board:
             print(f"  - {building}")

        print(f"\nEnemy Units on Board:")
        for unit in self.enemy_units_on_board:
            print(f"  - {unit}")

        print(f"\nEnemy Buildings on Board:")
        for building in self.enemy_buildings_on_board:
            print(f"  - {building}")

        print(f"\nGame Board Grid (0=empty, 1=ally unit, 2=ally building, 3=enemy unit, 4=enemy building):")
        print(self.game_board_grid)