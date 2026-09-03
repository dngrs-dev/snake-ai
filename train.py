from agent import DQNAgent
from snake import Game

EPISODES = 10_000


def main():
    game = Game(
        width=10,
        height=10,
        seed=81,
    )

    agent = DQNAgent(seed=37)

    scores: list[int] = []

    collisions = 0
    truncations = 0
    wins = 0

    for episode in range(EPISODES):
        state = game.reset()

        while True:
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated = game.step(action)

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

        scores.append(game.score)

        if game.won:
            wins += 1
        elif truncated:
            truncations += 1
        elif terminated:
            collisions += 1

        if (episode + 1) % 1000 == 0:
            recent_scores = scores[-1000:]

            print(
                f"Episode: {episode + 1:>5} | "
                f"Average score: {sum(recent_scores) / len(recent_scores)} | "
                f"Best: {max(scores)}"
            )

    print()
    print("Finished")
    print(f"Episodes: {EPISODES}")
    print(f"Avg score: {sum(scores) / len(scores)}")
    print(f"Best score: {max(scores)}")
    print(f"Collisions: {collisions}")
    print(f"Truncations: {truncations}")
    print(f"Wins: {wins}")


if __name__ == "__main__":
    main()
