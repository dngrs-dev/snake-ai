from pathlib import Path

import torch

from agent import DQNAgent
from encoder import DEFAULT_ENCODER_NAME, StateEncoder, create_encoder

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


def find_latest_checkpoint(
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.is_dir():
        return None

    checkpoints = [path for path in checkpoint_dir.glob("*.pt") if path.is_file()]

    if not checkpoints:
        return None

    return max(
        checkpoints,
        key=lambda path: path.stat().st_mtime_ns,
    )


def resolve_checkpoint(
    checkpoint: str | Path | None = None,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path | None:
    if checkpoint is not None:
        checkpoint = Path(checkpoint)

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

        return checkpoint

    return find_latest_checkpoint(checkpoint_dir)


def read_checkpoint_encoder(
    checkpoint_path: str | Path,
) -> StateEncoder:
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    encoder_metadata = checkpoint.get("encoder")

    if encoder_metadata is None:
        return create_encoder(DEFAULT_ENCODER_NAME)

    encoder_name = encoder_metadata["name"]
    encoder_config = encoder_metadata.get("config", {})

    encoder = create_encoder(
        encoder_name,
        config=encoder_config,
    )

    checkpoint_input_size = encoder_metadata["input_size"]

    if checkpoint_input_size != encoder.input_size:
        raise ValueError(
            f"Checkpoint declares encoder input size {checkpoint_input_size}, but reconstructed encoder {encoder_name} has input size {encoder.input_size}."
        )

    return encoder


def load_configured_agent_checkpoint(
    checkpoint: str | Path | None = None,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    *,
    required: bool = True,
    seed: int | None = None,
) -> tuple[DQNAgent, Path] | None:
    checkpoint_path = resolve_checkpoint(
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
    )

    if checkpoint_path is None:
        if required:
            raise FileNotFoundError(
                f"No .pt checkpoints found in {Path(checkpoint_dir)}"
            )

        return None

    encoder = read_checkpoint_encoder(checkpoint_path)

    agent = DQNAgent(
        seed=seed,
        encoder=encoder,
    )

    agent.load(checkpoint_path)

    return agent, checkpoint_path


def load_agent_checkpoint(
    agent: "DQNAgent",
    checkpoint: str | Path | None = None,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    *,
    required: bool = True,
) -> Path | None:
    checkpoint_path = resolve_checkpoint(
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
    )

    if checkpoint_path is None:
        if required:
            raise FileNotFoundError(
                f"No .pt checkpoints found in {Path(checkpoint_dir)}"
            )

        return None

    agent.load(checkpoint_path)

    return checkpoint_path


def save_agent_checkpoint(
    agent: "DQNAgent",
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"snake_{agent.games_played}.pt"
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")

    agent.save(temporary_path)
    temporary_path.replace(checkpoint_path)

    return checkpoint_path
