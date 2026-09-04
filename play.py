from agent import DQNAgent
from renderer import GameRenderer
from snake import Game

CHECKPOINT = "checkpoints/snake_500.pt"


def main():
    game = Game(
        width=10,
        height=10,
        seed=123,
    )

    agent = DQNAgent()

    agent.load(CHECKPOINT)

    agent.epsilon = 0.0

    renderer = GameRenderer(
        game=game,
        cell_size=40,
        fps=8,
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
                    next_state,
                    _,
                    _,
                    _,
                ) = game.step(action)

                state = next_state

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
