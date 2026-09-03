import random
from typing import Any

from snake import Action


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_action(self, state: dict[str, Any]) -> Action:
        return self.rng.choice(list(Action))

    def learn(
        self,
        state: dict[str, Any],
        action: Action,
        reward: int,
        next_state: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ):
        pass
