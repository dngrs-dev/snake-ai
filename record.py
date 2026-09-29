import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray

from checkpoint_utils import load_configured_agent_checkpoint
from renderer import GameRenderer
from snake import Game


class VideoWriter(Protocol):
    def append_data(
        self,
        image: NDArray[np.uint8],
    ) -> None: ...


def positive_int(value: str) -> int:
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")

    return number


def nonnegative_float(value: str) -> float:
    number = float(value)

    if number < 0:
        raise argparse.ArgumentTypeError("Value cannot be negative.")

    return number


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record games played by a trained Snake agent."
    )

    parser.add_argument(
        "--games",
        type=positive_int,
        default=5,
        help="Number of complete games to record (default: 5).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Checkpoint to load. If omitted, the latest checkpoint "
            "from --checkpoint-dir is used."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing checkpoints (default: checkpoints).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 file. A timestamped name is used if omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Game seed used for reproducible recordings (default: 123).",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=8,
        help="Video and game speed in frames per second (default: 8).",
    )
    parser.add_argument(
        "--cell-size",
        type=positive_int,
        default=40,
        help="Rendered cell size in pixels (default: 40).",
    )
    parser.add_argument(
        "--pause",
        type=nonnegative_float,
        default=1.0,
        help="Seconds to hold the final frame of each game (default: 1).",
    )

    return parser.parse_args()


def make_output_path(
    requested_output: Path | None,
    checkpoint_path: Path,
    games: int,
) -> Path:
    if requested_output is not None:
        output_path = requested_output
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")

        output_path = Path("recordings") / (
            f"{checkpoint_path.stem}_{games}_games_{timestamp}.mp4"
        )

    if output_path.suffix.lower() != ".mp4":
        raise ValueError("The output file must have the .mp4 extension.")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


def append_current_frame(
    writer: VideoWriter,
    renderer: GameRenderer,
) -> None:
    writer.append_data(renderer.capture_frame())


def main() -> None:
    arguments = parse_arguments()

    game = Game(
        width=10,
        height=10,
        seed=arguments.seed,
    )

    loaded = load_configured_agent_checkpoint(
        checkpoint=arguments.checkpoint,
        checkpoint_dir=arguments.checkpoint_dir,
        required=True,
    )

    if loaded is None:
        raise RuntimeError("A checkpoint is required for recording.")

    agent, checkpoint_path = loaded

    # Evaluation must not contain random exploration.
    agent.epsilon = 0.0
    agent.policy_model.eval()

    output_path = make_output_path(
        requested_output=arguments.output,
        checkpoint_path=checkpoint_path,
        games=arguments.games,
    )

    renderer = GameRenderer(
        game=game,
        cell_size=arguments.cell_size,
        fps=arguments.fps,
    )

    completed_games = 0
    running = True

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Encoder: {agent.encoder.name}")
    print(f"Recording: {output_path}")
    print(f"Games requested: {arguments.games}")

    try:
        with imageio.get_writer(
            str(output_path),
            fps=arguments.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as raw_writer:
            writer = cast(VideoWriter, raw_writer)
            for game_number in range(1, arguments.games + 1):
                state = game.reset()

                # Record the initial state before the first action.
                renderer.render()
                append_current_frame(writer, renderer)

                while not game.terminated and not game.truncated:
                    running = renderer.process_events()

                    if not running:
                        break

                    action = agent.choose_action(state)

                    (
                        state,
                        _reward,
                        _terminated,
                        _truncated,
                    ) = game.step(action)

                    renderer.render()
                    append_current_frame(writer, renderer)

                if not running:
                    break

                completed_games += 1

                print(
                    f"Game {game_number}/{arguments.games} | "
                    f"Score: {game.score} | "
                    f"Won: {game.won} | "
                    f"Terminated: {game.terminated} | "
                    f"Truncated: {game.truncated}"
                )

                # Repeating the last frame creates a visible pause in the
                # resulting video without recording desktop content.
                pause_frames = round(arguments.pause * arguments.fps)

                final_frame = renderer.capture_frame()

                for _ in range(pause_frames):
                    writer.append_data(final_frame)

    finally:
        renderer.close()

    print(f"Recorded games: {completed_games}")
    print(f"Video saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
