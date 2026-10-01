import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent import DQNAgent
from checkpoint_utils import (
    find_latest_checkpoint,
    load_configured_agent_checkpoint,
)
from evaluate import (
    EpisodeResult,
    evaluate_policy,
    positive_int,
    resolve_board_dimensions,
    summarize_results,
)


@dataclass(frozen=True)
class ComparisonResult:
    checkpoint_path: Path
    encoder_name: str
    encoder_config: dict[str, Any]
    input_size: int
    training_episodes: int
    episode_results: list[EpisodeResult]
    summary: dict[str, int | float]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and compare Snake checkpoints over identical game seeds."
        )
    )

    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        metavar="CHECKPOINT_OR_DIRECTORY",
        help=(
            "Checkpoint file, checkpoint directory, or root directory whose "
            "immediate child directories contain checkpoints."
        ),
    )
    parser.add_argument(
        "--games",
        type=positive_int,
        default=100,
        help="Number of games per checkpoint (default: 100).",
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
            "Board width. Defaults to the encoder width when available, "
            "otherwise 10."
        ),
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        default=None,
        help=(
            "Board height. Defaults to the encoder height when available, "
            "otherwise 10."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a combined JSON comparison report.",
    )

    return parser.parse_args()


def checkpoints_from_directory(
    directory: Path,
) -> list[Path]:
    checkpoints: list[Path] = []

    direct_checkpoint = find_latest_checkpoint(directory)

    if direct_checkpoint is not None:
        checkpoints.append(direct_checkpoint)

    child_directories = sorted(
        (path for path in directory.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )

    for child_directory in child_directories:
        child_checkpoint = find_latest_checkpoint(child_directory)

        if child_checkpoint is not None:
            checkpoints.append(child_checkpoint)

    return checkpoints


def discover_checkpoints(
    sources: list[Path],
) -> list[Path]:
    discovered: list[Path] = []
    seen_paths: set[Path] = set()

    for source in sources:
        if source.is_file():
            if source.suffix.lower() != ".pt":
                raise ValueError(
                    f"Checkpoint file must have the .pt extension: {source}"
                )

            source_checkpoints = [source]

        elif source.is_dir():
            source_checkpoints = checkpoints_from_directory(source)

            if not source_checkpoints:
                raise FileNotFoundError(
                    f"No checkpoints found in directory or its immediate "
                    f"child directories: {source}"
                )

        else:
            raise FileNotFoundError(f"Checkpoint source does not exist: {source}")

        for checkpoint_path in source_checkpoints:
            resolved_path = checkpoint_path.resolve()

            if resolved_path in seen_paths:
                continue

            seen_paths.add(resolved_path)
            discovered.append(checkpoint_path)

    return discovered


def validate_checkpoint_count(
    checkpoint_paths: list[Path],
) -> None:
    if len(checkpoint_paths) < 2:
        raise ValueError("At least two checkpoints are required for comparison.")


def checkpoint_name(
    checkpoint_path: Path,
) -> str:
    parent_name = checkpoint_path.parent.name

    if parent_name:
        return f"{parent_name}/{checkpoint_path.stem}"

    return checkpoint_path.stem


def load_policies(
    checkpoint_paths: list[Path],
) -> list[tuple[Path, DQNAgent]]:
    loaded_policies: list[tuple[Path, DQNAgent]] = []

    for checkpoint_path in checkpoint_paths:
        print(f"Loading {checkpoint_name(checkpoint_path)}: " f"{checkpoint_path}")

        loaded = load_configured_agent_checkpoint(
            checkpoint=checkpoint_path,
            required=True,
        )

        if loaded is None:
            raise RuntimeError(f"Could not load checkpoint: {checkpoint_path}")

        agent, resolved_checkpoint_path = loaded

        # Comparison must use greedy policy actions only.
        agent.epsilon = 0.0
        agent.policy_model.eval()

        loaded_policies.append(
            (
                resolved_checkpoint_path,
                agent,
            )
        )

    return loaded_policies


def resolve_shared_board_dimensions(
    loaded_policies: list[tuple[Path, DQNAgent]],
    requested_width: int | None,
    requested_height: int | None,
) -> tuple[int, int]:
    dimensions = {
        resolve_board_dimensions(
            agent=agent,
            requested_width=requested_width,
            requested_height=requested_height,
        )
        for _checkpoint_path, agent in loaded_policies
    }

    if len(dimensions) != 1:
        formatted_dimensions = ", ".join(
            f"{width}x{height}" for width, height in sorted(dimensions)
        )

        raise ValueError(
            "All compared checkpoints must use the same board dimensions. "
            f"Resolved dimensions: {formatted_dimensions}."
        )

    return dimensions.pop()


def run_comparison(
    loaded_policies: list[tuple[Path, DQNAgent]],
    games: int,
    seed_start: int,
    width: int,
    height: int,
) -> list[ComparisonResult]:
    comparisons: list[ComparisonResult] = []

    for checkpoint_path, agent in loaded_policies:
        name = checkpoint_name(checkpoint_path)

        print(f"Evaluating {name} " f"with encoder {agent.encoder.name}...")

        episode_results = evaluate_policy(
            agent=agent,
            games=games,
            seed_start=seed_start,
            width=width,
            height=height,
        )

        comparisons.append(
            ComparisonResult(
                checkpoint_path=checkpoint_path,
                encoder_name=agent.encoder.name,
                encoder_config=agent.encoder.get_config(),
                input_size=agent.encoder.input_size,
                training_episodes=agent.games_played,
                episode_results=episode_results,
                summary=summarize_results(episode_results),
            )
        )

    return comparisons


def format_table(
    headers: list[str],
    rows: list[list[str]],
    left_aligned_columns: set[int] | None = None,
) -> str:
    if not rows:
        raise ValueError("Cannot format a table without rows.")

    if any(len(row) != len(headers) for row in rows):
        raise ValueError(
            "Every table row must have the same number of values " "as the header."
        )

    left_aligned_columns = left_aligned_columns or set()

    widths = [
        max(
            len(headers[column_index]),
            *(len(row[column_index]) for row in rows),
        )
        for column_index in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        values: list[str] = []

        for index, value in enumerate(row):
            if index in left_aligned_columns:
                values.append(value.ljust(widths[index]))
            else:
                values.append(value.rjust(widths[index]))

        return " | ".join(values)

    separator = "-+-".join("-" * width for width in widths)

    return "\n".join(
        [
            format_row(headers),
            separator,
            *(format_row(row) for row in rows),
        ]
    )


def print_comparison_table(
    comparisons: list[ComparisonResult],
    games: int,
    seed_start: int,
    width: int,
    height: int,
) -> None:
    last_seed = seed_start + games - 1

    print()
    print(f"Board: {width}x{height}")
    print(f"Games per checkpoint: {games}")
    print(f"Seeds: {seed_start} through {last_seed}")
    print()

    headers = [
        "Checkpoint",
        "Encoder",
        "Train",
        "Mean",
        "Median",
        "Std",
        "P10",
        "P90",
        "Min",
        "Max",
        "Collision",
        "Timeout",
        "Win",
    ]

    rows: list[list[str]] = []

    for comparison in comparisons:
        summary = comparison.summary

        rows.append(
            [
                checkpoint_name(comparison.checkpoint_path),
                comparison.encoder_name,
                str(comparison.training_episodes),
                f"{summary['mean_score']:.2f}",
                f"{summary['median_score']:.2f}",
                f"{summary['score_standard_deviation']:.2f}",
                f"{summary['score_percentile_10']:.2f}",
                f"{summary['score_percentile_90']:.2f}",
                str(summary["minimum_score"]),
                str(summary["maximum_score"]),
                f"{summary['collision_rate']:.1%}",
                f"{summary['truncation_rate']:.1%}",
                f"{summary['win_rate']:.1%}",
            ]
        )

    print(
        format_table(
            headers=headers,
            rows=rows,
            left_aligned_columns={0, 1},
        )
    )


def save_comparison_report(
    output_path: Path,
    comparisons: list[ComparisonResult],
    games: int,
    seed_start: int,
    width: int,
    height: int,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Comparison report already exists: {output_path}")

    report: dict[str, Any] = {
        "environment": {
            "width": width,
            "height": height,
        },
        "evaluation": {
            "games_per_checkpoint": games,
            "seed_start": seed_start,
            "seed_end": seed_start + games - 1,
        },
        "policies": [
            {
                "checkpoint": str(comparison.checkpoint_path.resolve()),
                "encoder": {
                    "name": comparison.encoder_name,
                    "config": comparison.encoder_config,
                    "input_size": comparison.input_size,
                },
                "training_episodes": comparison.training_episodes,
                "summary": comparison.summary,
                "episodes": [asdict(result) for result in comparison.episode_results],
            }
            for comparison in comparisons
        ],
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

    checkpoint_paths = discover_checkpoints(arguments.sources)

    validate_checkpoint_count(checkpoint_paths)

    loaded_policies = load_policies(checkpoint_paths)

    width, height = resolve_shared_board_dimensions(
        loaded_policies=loaded_policies,
        requested_width=arguments.width,
        requested_height=arguments.height,
    )

    comparisons = run_comparison(
        loaded_policies=loaded_policies,
        games=arguments.games,
        seed_start=arguments.seed_start,
        width=width,
        height=height,
    )

    print_comparison_table(
        comparisons=comparisons,
        games=arguments.games,
        seed_start=arguments.seed_start,
        width=width,
        height=height,
    )

    if arguments.output is not None:
        save_comparison_report(
            output_path=arguments.output,
            comparisons=comparisons,
            games=arguments.games,
            seed_start=arguments.seed_start,
            width=width,
            height=height,
        )

        print()
        print(f"Report saved to: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
