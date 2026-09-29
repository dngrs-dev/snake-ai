from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from snake import Direction

DEFAULT_ENCODER_NAME = "compact"


class StateEncoder(Protocol):
    name: str
    input_size: int

    def encode(self, state: dict[str, Any]) -> np.ndarray: ...

    def get_config(self) -> dict[str, Any]: ...


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

    def get_config(self) -> dict[str, Any]:
        return {}


EncoderFactory = Callable[..., StateEncoder]


_ENCODER_FACTORIES: dict[str, EncoderFactory] = {
    CompactStateEncoder.name: CompactStateEncoder,
}


def available_encoder_names() -> tuple[str, ...]:
    return tuple(sorted(_ENCODER_FACTORIES))


def create_encoder(
    name: str,
    config: dict[str, Any] | None = None,
) -> StateEncoder:
    try:
        factory = _ENCODER_FACTORIES[name]
    except KeyError as error:
        available = ", ".join(available_encoder_names())

        raise ValueError(
            f"Unknown encoder {name!r}. Available encoders: {available}."
        ) from error

    return factory(**(config or {}))
