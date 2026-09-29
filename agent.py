import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from encoder import (
    DEFAULT_ENCODER_NAME,
    CompactStateEncoder,
    StateEncoder,
)
from model import DQN
from snake import Action


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
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_frequency: int = 1000,
        encoder: StateEncoder | None = None,
    ):
        self.rng = random.Random(seed)
        self.encoder = encoder if encoder is not None else CompactStateEncoder()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.memory: deque[Experience] = deque(maxlen=memory_size)

        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.games_played = 0
        self.training_steps = 0

        self.target_update_frequency = target_update_frequency

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_model = DQN(
            input_size=self.encoder.input_size,
        ).to(self.device)

        self.target_model = DQN(
            input_size=self.encoder.input_size,
        ).to(self.device)

        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()
        self.last_loss: float | None = None

    def choose_action(self, state: dict[str, Any]) -> Action:
        encoded_state = self.encoder.encode(state)

        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(Action))

        return self._choose_best_action(encoded_state)

    def _choose_best_action(self, state: np.ndarray) -> Action:
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.policy_model.eval()

        with torch.no_grad():
            q_values = self.policy_model(state_tensor)

        action_index = int(torch.argmax(q_values, dim=1).item())

        return Action(action_index)

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
            state=self.encoder.encode(state),
            action=action,
            reward=float(reward),
            next_state=self.encoder.encode(next_state),
            terminated=terminated,
            truncated=truncated,
        )

        self.memory.append(experience)

    def sample_batch(
        self,
        batch_size: int,
    ) -> list[Experience]:
        if batch_size > len(self.memory):
            raise ValueError(
                f"Cannot sample {batch_size} experiences "
                f"from memory containing {len(self.memory)} experiences."
            )

        return self.rng.sample(
            list(self.memory),
            batch_size,
        )

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

        if len(self.memory) < self.batch_size:
            return

        batch = self.sample_batch(self.batch_size)

        self.last_loss = self._train_batch(batch)

        self.training_steps += 1

        if self.training_steps % self.target_update_frequency == 0:
            self._update_target_model()

    def _train_batch(
        self,
        batch: list[Experience],
    ) -> float:
        states = torch.tensor(
            np.array([experience.state for experience in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            [int(experience.action) for experience in batch],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            [experience.reward for experience in batch],
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            np.array([experience.next_state for experience in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        terminated = torch.tensor(
            [experience.terminated for experience in batch],
            dtype=torch.float32,
            device=self.device,
        )

        self.policy_model.train()

        all_q_values = self.policy_model(states)

        current_q_values = all_q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            # Select the best action.
            next_policy_q_values = self.policy_model(next_states)

            next_actions = next_policy_q_values.argmax(
                dim=1,
                keepdim=True,
            )

            # Evaluate that action.
            next_target_q_values = self.target_model(next_states)

            selected_next_q_values = next_target_q_values.gather(
                1, next_actions
            ).squeeze(1)

            target_q_values = rewards + self.gamma * selected_next_q_values * (
                1 - terminated
            )

        loss = self.loss_function(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()

        loss.backward()

        # Protect against large updates.
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        return loss.item()

    def _update_target_model(self) -> None:
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def save(self, path: str | Path) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "checkpoint_version": 1,
            "encoder": {
                "name": self.encoder.name,
                "config": self.encoder.get_config(),
                "input_size": self.encoder.input_size,
            },
            "policy_model": self.policy_model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "games_played": self.games_played,
            "training_steps": self.training_steps,
        }

        torch.save(checkpoint, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self._validate_checkpoint_encoder(checkpoint)

        self.policy_model.load_state_dict(checkpoint["policy_model"])

        self.target_model.load_state_dict(checkpoint["target_model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = checkpoint["epsilon"]
        self.games_played = checkpoint["games_played"]
        self.training_steps = checkpoint["training_steps"]

        self.target_model.eval()

    def _validate_checkpoint_encoder(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        encoder_metadata = checkpoint.get("encoder")

        if encoder_metadata is None:
            if self.encoder.name != DEFAULT_ENCODER_NAME:
                raise ValueError(
                    "Checkpoint does not contain encoder metadata, but the agent uses a non-default encoder."
                )

            return

        checkpoint_name = encoder_metadata["name"]
        checkpoint_config = encoder_metadata.get("config", {})
        checkpoint_input_size = encoder_metadata["input_size"]

        if checkpoint_name != self.encoder.name:
            raise ValueError(
                f"Checkpoint uses encoder {checkpoint_name}, but the agent uses {self.encoder.name}."
            )

        if checkpoint_config != self.encoder.get_config():
            raise ValueError(
                f"Checkpoint encoder configuration does not match the agent: {checkpoint_config} != {self.encoder.get_config()}."
            )

        if checkpoint_input_size != self.encoder.input_size:
            raise ValueError(
                f"Checkpoint encoder input size does not match the agent: {checkpoint_input_size} != {self.encoder.input_size}."
            )
