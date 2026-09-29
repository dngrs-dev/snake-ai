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


class OrderedGridStateEncoder:
    name = "ordered-grid"

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("Encoder width and height must be positive.")

        self.width = width
        self.height = height
        self.input_size = width * height * 2 + 4

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        body = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )
        food = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )

        snake_body = state["snake_body"]
        snake_length = len(snake_body)

        for index, point in enumerate(snake_body):
            self._validate_point(point.x, point.y)

            body[point.y, point.x] = (snake_length - index) / snake_length

        food_point = state["food"]

        if food_point is not None:
            self._validate_point(food_point.x, food_point.y)
            food[food_point.y, food_point.x] = 1.0

        direction = state["direction"]

        direction_values = np.array(
            [
                direction == Direction.RIGHT,
                direction == Direction.LEFT,
                direction == Direction.UP,
                direction == Direction.DOWN,
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            (
                body.reshape(-1),
                food.reshape(-1),
                direction_values,
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
        }

    def _validate_point(
        self,
        x: int,
        y: int,
    ) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(
                f"Point ({x}, {y}) is outside the encoder board of size {self.width}x{self.height}."
            )


EncoderFactory = Callable[..., StateEncoder]


_ENCODER_FACTORIES: dict[str, EncoderFactory] = {
    CompactStateEncoder.name: CompactStateEncoder,
    OrderedGridStateEncoder.name: OrderedGridStateEncoder,
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
