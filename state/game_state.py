from dataclasses import dataclass, field

@dataclass
class GameState:
    cards_in_hand: dict = field(default_factory=dict)
    enemy_units_on_board: list = field(default_factory=list)
    elixir_count: int = 0