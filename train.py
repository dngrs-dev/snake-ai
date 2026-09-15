import argparse
from pathlib import Path

from agent import DQNAgent
from snake import Game


def positive_int(value: str) -> int:
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")

    return number


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=5_000,
        help="Number of additional episodes to train (default: 5000).",
    )
    parser.add_argument(
        "--save-every",
        type=positive_int,
        default=500,
        help="Save after this many episodes in the current run (default: 500).",
    )
    parser.add_argument(
        "--log-every",
        type=positive_int,
        default=100,
        help="Print statistics every N episodes (default: 100).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing checkpoints (default: checkpoints).",
    )

    resume_group = parser.add_mutually_exclusive_group()

    resume_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Resume from a specific checkpoint.",
    )
    resume_group.add_argument(
        "--force",
        action="store_true",
        help="Start from episode 0 and ignore existing checkpoints.",
    )

    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None

    checkpoints = [path for path in checkpoint_dir.glob("*.pt") if path.is_file()]

    if not checkpoints:
        return None

    return max(
        checkpoints,
        key=lambda path: path.stat().st_mtime_ns,
    )


def load_checkpoint(
    agent: DQNAgent,
    arguments: argparse.Namespace,
) -> Path | None:
    if arguments.force:
        print("Forced fresh start. Existing checkpoints will be ignored.")
        return None

    if arguments.checkpoint is not None:
        checkpoint = arguments.checkpoint

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    else:
        checkpoint = find_latest_checkpoint(arguments.checkpoint_dir)

    if checkpoint is None:
        print("No checkpoint found. Starting from episode 0.")
        return None

    agent.load(checkpoint)

    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Completed episodes: {agent.games_played}")
    print(f"Training steps: {agent.training_steps}")
    print(f"Epsilon: {agent.epsilon:.6f}")

    return checkpoint


def save_checkpoint(
    agent: DQNAgent,
    checkpoint_dir: Path,
) -> Path:
    checkpoint_path = checkpoint_dir / f"snake_{agent.games_played}.pt"
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")

    agent.save(temporary_path)
    temporary_path.replace(checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")

    return checkpoint_path


def train_episode(game: Game, agent: DQNAgent) -> int:
    state = game.reset()

    while True:
        action = agent.choose_action(state)

        (
            next_state,
            reward,
            terminated,
            truncated,
        ) = game.step(action)

        agent.learn(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
        )

        state = next_state

        if terminated or truncated:
            break

    agent.end_episode()

    return game.score


def main() -> None:
    arguments = parse_arguments()

    game = Game(
        width=10,
        height=10,
        seed=81,
    )

    agent = DQNAgent(seed=37)

    load_checkpoint(agent, arguments)

    starting_episode = agent.games_played
    ending_episode = starting_episode + arguments.episodes

    print(f"Training episodes {starting_episode + 1} through {ending_episode}.")

    scores: list[int] = []
    last_saved_episode: int | None = None

    for run_episode in range(1, arguments.episodes + 1):
        score = train_episode(game, agent)
        scores.append(score)

        if run_episode % arguments.log_every == 0:
            recent_scores = scores[-arguments.log_every :]
            average_score = sum(recent_scores) / len(recent_scores)

            loss_text = "None" if agent.last_loss is None else f"{agent.last_loss:.6f}"

            print(
                f"Episode: {agent.games_played:>6} | "
                f"Run: {run_episode:>6}/{arguments.episodes} | "
                f"Score: {score:>3} | "
                f"Average: {average_score:>7.2f} | "
                f"Run best: {max(scores):>3} | "
                f"Epsilon: {agent.epsilon:.5f} | "
                f"Loss: {loss_text} | "
                f"Memory: {len(agent.memory)}"
            )

        if run_episode % arguments.save_every == 0:
            save_checkpoint(agent, arguments.checkpoint_dir)
            last_saved_episode = agent.games_played

    # Save final checkpoint
    if last_saved_episode != agent.games_played:
        save_checkpoint(agent, arguments.checkpoint_dir)

    print(f"Training complete. Total completed episodes: {agent.games_played}")


if __name__ == "__main__":
    main()
