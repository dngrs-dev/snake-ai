import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from snake import Action, Direction


@dataclass
class Experience:
    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


class DQNAgent:
    def __init__(
        self,
        seed: int | None = None,
        memory_size: int = 100_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.rng = random.Random(seed)

        self.memory: deque[Experience] = deque(maxlen=memory_size)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.games_played = 0

    def encode_state(self, state: dict[str, Any]) -> np.ndarray:
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

    def choose_action(self, state: dict[str, Any]) -> Action:
        encoded_state = self.encode_state(state)

        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(Action))

        return self._choose_best_action(encoded_state)

    def _choose_best_action(self, state: np.ndarray) -> Action:
        return self.rng.choice(list(Action))

    def remember(
        self,
        state: dict[str, Any],
        action: Action,
        reward: float,
        next_state: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        experience = Experience(
            state=self.encode_state(state),
            action=action,
            reward=float(reward),
            next_state=self.encode_state(next_state),
            terminated=terminated,
            truncated=truncated,
        )

        self.memory.append(experience)

    def sample_batch(self, batch_size: int) -> list[Experience]:
        if batch_size > len(self.memory):
            raise ValueError(
                f"Cannot sample {batch_size} experiences "
                f"from memory containing {len(self.memory)} experiences."
            )

        return self.rng.sample(list(self.memory), batch_size)

    def end_episode(self) -> None:
        self.games_played += 1

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def learn(
        self,
        state: dict[str, Any],
        action: Action,
        reward: float,
        next_state: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.remember(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
        )
