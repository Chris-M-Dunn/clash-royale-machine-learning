from dataclasses import dataclass, field

@dataclass
class GameState:
    elixir_count: int = 0
    cards_in_hand: dict = field(default_factory=dict)
    ally_units_on_board: list = field(default_factory=list)
    ally_buildings_on_board: list = field(default_factory=list)
    enemy_units_on_board: list = field(default_factory=list)
    enemy_buildings_on_board: list = field(default_factory=list)