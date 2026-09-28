from typing import Any, Protocol

import numpy as np

from snake import Direction


class StateEncoder(Protocol):
    name: str
    input_size: int

    def encode(self, state: dict[str, Any]) -> np.ndarray: ...


class CompactStateEncoder:
    name = "compact"
    input_size = 11

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        head = state["snake_head"]
        food = state["food"]
        direction = state["direction"]

        food_left = food is not None and food.x < head.x
        food_right = food is not None and food.x > head.x
        food_up = food is not None and food.y < head.y
        food_down = food is not None and food.y > head.y

        return np.array(
            [
                state["danger_forward"],
                state["danger_right"],
                state["danger_left"],
                direction == Direction.RIGHT,
                direction == Direction.LEFT,
                direction == Direction.UP,
                direction == Direction.DOWN,
                food_right,
                food_left,
                food_up,
                food_down,
            ],
            dtype=np.float32,
        )
