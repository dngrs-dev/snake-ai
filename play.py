import argparse
from pathlib import Path

from checkpoint_utils import load_configured_agent_checkpoint
from renderer import GameRenderer
from snake import Game


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained Snake agent play.")

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
        "--seed",
        type=int,
        default=None,
        help="Optional game seed.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Rendering speed (default: 8).",
    )

    arguments = parser.parse_args()

    if arguments.fps <= 0:
        parser.error("--fps must be greater than zero")

    return arguments


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
        raise RuntimeError("A checkpoint is required for playback.")

    agent, checkpoint_path = loaded

    # Disable exploration during evaluation.
    agent.epsilon = 0.0
    agent.policy_model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Encoder: {agent.encoder.name}")
    print(f"Training episodes: {agent.games_played}")
    print(f"Training steps: {agent.training_steps}")

    renderer = GameRenderer(
        game=game,
        cell_size=40,
        fps=arguments.fps,
    )

    try:
        running = True

        while running:
            state = game.reset()

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

            if not running:
                break

            print(
                f"Score: {game.score} | "
                f"Won: {game.won} | "
                f"Terminated: {game.terminated} | "
                f"Truncated: {game.truncated}"
            )

    finally:
        renderer.close()


if __name__ == "__main__":
    main()
