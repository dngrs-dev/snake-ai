import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent import DQNAgent
from checkpoint_utils import load_configured_agent_checkpoint
from snake import Game

DEFAULT_BOARD_WIDTH = 10
DEFAULT_BOARD_HEIGHT = 10


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    score: int
    steps: int
    won: bool
    collision: bool
    truncated: bool


def positive_int(value: str) -> int:
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")

    return number


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Snake agent over fixed game seeds."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Checkpoint to evaluate. If omitted, the latest checkpoint from --checkpoint-dir is used."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing checkpoints (default: checkpoints).",
    )
    parser.add_argument(
        "--games",
        type=positive_int,
        default=100,
        help="Number of games to evaluate (default: 100).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Seed used for the first evaluation game (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        default=None,
        help=(
            "Board width. Defaults to the encoder width when available, otherwise 10."
        ),
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        default=None,
        help=(
            "Board height. Defaults to the encoder height when available, otherwise 10."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a JSON evaluation report.",
    )

    return parser.parse_args()


def resolve_board_dimensions(
    agent: DQNAgent,
    requested_width: int | None,
    requested_height: int | None,
) -> tuple[int, int]:
    encoder_config = agent.encoder.get_config()

    encoder_width = encoder_config.get("width")
    encoder_height = encoder_config.get("height")

    width = (
        requested_width
        if requested_width is not None
        else encoder_width or DEFAULT_BOARD_WIDTH
    )
    height = (
        requested_height
        if requested_height is not None
        else encoder_height or DEFAULT_BOARD_HEIGHT
    )

    if encoder_width is not None and width != encoder_width:
        raise ValueError(
            f"Requested board width {width} does not match encoder width {encoder_width}."
        )

    if encoder_height is not None and height != encoder_height:
        raise ValueError(
            f"Requested board height {height} does not match encoder height {encoder_height}."
        )

    return width, height


def evaluate_policy(
    agent: DQNAgent,
    games: int,
    seed_start: int,
    width: int,
    height: int,
) -> list[EpisodeResult]:
    if games <= 0:
        raise ValueError("Number of evaluation games must be positive.")

    results: list[EpisodeResult] = []

    game = Game(
        width=width,
        height=height,
    )

    for game_index in range(games):
        seed = seed_start + game_index
        state = game.reset(seed=seed)

        while not game.terminated and not game.truncated:
            action = agent.choose_action(state)

            (
                state,
                _reward,
                _terminated,
                _truncated,
            ) = game.step(action)

        results.append(
            EpisodeResult(
                seed=seed,
                score=game.score,
                steps=game.steps,
                won=game.won,
                collision=game.terminated and not game.won,
                truncated=game.truncated,
            )
        )

    return results


def percentile(
    values: list[int],
    probability: float,
) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty collection.")

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile probability must be between 0 and 1.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index

    lower = ordered[lower_index]
    upper = ordered[upper_index]

    return lower + (upper - lower) * fraction


def summarize_results(
    results: list[EpisodeResult],
) -> dict[str, int | float]:
    if not results:
        raise ValueError("Cannot summarize empty evaluation results.")

    scores = [result.score for result in results]
    steps = [result.steps for result in results]

    games = len(results)
    wins = sum(result.won for result in results)
    collisions = sum(result.collision for result in results)
    truncations = sum(result.truncated for result in results)

    return {
        "games": games,
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "score_standard_deviation": statistics.pstdev(scores),
        "score_percentile_10": percentile(scores, 0.10),
        "score_percentile_90": percentile(scores, 0.90),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "mean_steps": statistics.fmean(steps),
        "wins": wins,
        "win_rate": wins / games,
        "collisions": collisions,
        "collision_rate": collisions / games,
        "truncations": truncations,
        "truncation_rate": truncations / games,
    }


def print_summary(
    checkpoint_path: Path,
    agent: DQNAgent,
    results: list[EpisodeResult],
    summary: dict[str, int | float],
    width: int,
    height: int,
) -> None:
    first_seed = results[0].seed
    last_seed = results[-1].seed

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Encoder: {agent.encoder.name}")
    print(f"Board: {width}x{height}")
    print(f"Games: {summary['games']}")
    print(f"Seeds: {first_seed} through {last_seed}")
    print()
    print("Scores")
    print(f"  Mean: {summary['mean_score']:.3f}")
    print(f"  Median: {summary['median_score']:.3f}")
    print(f"  Standard deviation: {summary['score_standard_deviation']:.3f}")
    print(f"  10th percentile: {summary['score_percentile_10']:.3f}")
    print(f"  90th percentile: {summary['score_percentile_90']:.3f}")
    print(f"  Minimum: {summary['minimum_score']}")
    print(f"  Maximum: {summary['maximum_score']}")
    print(f"  Mean steps: {summary['mean_steps']:.3f}")
    print()
    print("Outcomes")
    print(f"  Wins: {summary['wins']} ({summary['win_rate']:.1%})")
    print(f"  Collisions: {summary['collisions']} ({summary['collision_rate']:.1%})")
    print(f"  Truncations: {summary['truncations']} ({summary['truncation_rate']:.1%})")


def save_report(
    output_path: Path,
    checkpoint_path: Path,
    agent: DQNAgent,
    results: list[EpisodeResult],
    summary: dict[str, int | float],
    width: int,
    height: int,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Evaluation report already exists: {output_path}")

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "encoder": {
            "name": agent.encoder.name,
            "config": agent.encoder.get_config(),
            "input_size": agent.encoder.input_size,
        },
        "environment": {
            "width": width,
            "height": height,
        },
        "summary": summary,
        "episodes": [asdict(result) for result in results],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()

    loaded = load_configured_agent_checkpoint(
        checkpoint=arguments.checkpoint,
        checkpoint_dir=arguments.checkpoint_dir,
        required=True,
    )

    if loaded is None:
        raise RuntimeError("A checkpoint is required for evaluation.")

    agent, checkpoint_path = loaded

    agent.epsilon = 0.0
    agent.policy_model.eval()

    width, height = resolve_board_dimensions(
        agent=agent,
        requested_width=arguments.width,
        requested_height=arguments.height,
    )

    results = evaluate_policy(
        agent=agent,
        games=arguments.games,
        seed_start=arguments.seed_start,
        width=width,
        height=height,
    )

    summary = summarize_results(results)

    print_summary(
        checkpoint_path=checkpoint_path,
        agent=agent,
        results=results,
        summary=summary,
        width=width,
        height=height,
    )

    if arguments.output is not None:
        save_report(
            output_path=arguments.output,
            checkpoint_path=checkpoint_path,
            agent=agent,
            results=results,
            summary=summary,
            width=width,
            height=height,
        )

        print()
        print(f"Report saved to: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
