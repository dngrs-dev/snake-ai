import random
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any


class Reward(IntEnum):
    EAT_FOOD = 100
    COLLISION = -100
    STEP = 0
    WIN = 1000


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


class Direction(Enum):
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    def turn_right(self) -> "Direction":
        clockwise = (
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        )
        index = clockwise.index(self)
        return clockwise[(index + 1) % len(clockwise)]

    def turn_left(self) -> "Direction":
        clockwise = (
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        )
        index = clockwise.index(self)
        return clockwise[(index - 1) % len(clockwise)]


class Action(IntEnum):
    FORWARD = 0
    RIGHT = 1
    LEFT = 2


class Snake:
    def __init__(self, initial_position: Point):
        self.body: list[Point] = [initial_position]
        self.direction = Direction.RIGHT

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def tail(self) -> Point:
        return self.body[-1]

    @property
    def length(self) -> int:
        return len(self.body)

    def advance(self, new_head: Point, grow: bool = False):
        self.body.insert(0, new_head)
        if not grow:
            self.body.pop()


class Game:
    def __init__(
        self,
        width: int,
        height: int,
        seed: int | None = None,
        max_steps_since_food: int | None = None,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        if max_steps_since_food is not None and max_steps_since_food <= 0:
            raise ValueError("max_steps_since_food must be positive")

        self.width = width
        self.height = height
        self.board_area = width * height

        self.rng = random.Random(seed)

        self.snake: Snake
        self.food: Point | None
        self.score: int
        self.steps: int
        self.steps_since_food: int
        self.max_steps_since_food: int = (
            max_steps_since_food
            if max_steps_since_food is not None
            else self.board_area * 2
        )
        self.terminated: bool
        self.truncated: bool
        self.won: bool

        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)

        initial_position = Point(self.width // 2, self.height // 2)
        self.snake = Snake(initial_position)
        self.food = None
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.terminated = False
        self.truncated = False
        self.won = False

        if not self._place_food():
            self.terminated = True
            self.won = True

        return self._get_state()

    def step(self, action: int | Action) -> tuple[dict[str, Any], Reward, bool, bool]:
        if self.terminated or self.truncated:
            raise RuntimeError(
                "Game is terminated or truncated. Please reset the game."
            )

        try:
            action = Action(action)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid action: {action}") from e

        self.snake.direction = self._direction_after_action(action)
        next_head = self._next_head(self.snake.direction)

        will_eat = self.food is not None and next_head == self.food
        self.steps += 1

        if self._would_hit_wall(next_head) or self._would_hit_body(
            next_head, growing=will_eat
        ):
            self.terminated = True
            reward = Reward.COLLISION
            return self._get_state(), reward, self.terminated, self.truncated

        self.snake.advance(next_head, grow=will_eat)

        if will_eat:
            self.score += 1
            self.steps_since_food = 0
            reward = Reward.EAT_FOOD

            if self.snake.length == self.board_area:
                self.terminated = True
                self.won = True
                self.food = None
                reward = Reward.WIN
            else:
                self._place_food()
        else:
            self.steps_since_food += 1
            reward = Reward.STEP

            if self.steps_since_food >= self.max_steps_since_food:
                self.truncated = True

        return self._get_state(), reward, self.terminated, self.truncated

    def _direction_after_action(self, action: Action) -> Direction:
        if action == Action.FORWARD:
            return self.snake.direction
        elif action == Action.RIGHT:
            return self.snake.direction.turn_right()
        elif action == Action.LEFT:
            return self.snake.direction.turn_left()
        else:
            raise ValueError(f"Invalid action: {action}")

    def _next_head(self, direction: Direction) -> Point:
        return Point(
            self.snake.head.x + direction.dx,
            self.snake.head.y + direction.dy,
        )

    def _would_hit_wall(self, point: Point) -> bool:
        return (
            point.x < 0
            or point.x >= self.width
            or point.y < 0
            or point.y >= self.height
        )

    def _would_hit_body(self, point: Point, growing: bool) -> bool:
        if growing:
            occupied = self.snake.body
        else:
            occupied = self.snake.body[:-1]  # Exclude tail if not growing

        return point in occupied

    def _place_food(self) -> bool:
        occupied = set(self.snake.body)
        empty_cells: list[Point] = []

        for y in range(self.height):
            for x in range(self.width):
                candidate = Point(x, y)

                if candidate not in occupied:
                    empty_cells.append(candidate)

        if not empty_cells:
            self.food = None
            return False

        self.food = self.rng.choice(empty_cells)
        return True

    def _danger(self, direction: Direction) -> bool:
        next_head = self._next_head(direction)
        will_eat = self.food is not None and next_head == self.food
        return self._would_hit_wall(next_head) or self._would_hit_body(
            next_head, growing=will_eat
        )

    def _get_state(self) -> dict[str, Any]:
        return {
            "snake_head": self.snake.head,
            "food": self.food,
            "direction": self.snake.direction,
            "snake_body": tuple(self.snake.body),
            "danger_forward": self._danger(self.snake.direction),
            "danger_right": self._danger(self.snake.direction.turn_right()),
            "danger_left": self._danger(self.snake.direction.turn_left()),
        }
