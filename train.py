from agent import DQNAgent
from snake import Game

EPISODES = 5_000


def main():
    game = Game(
        width=10,
        height=10,
        seed=81,
    )

    agent = DQNAgent(seed=37)

    scores: list[int] = []

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

        if (episode + 1) % 100 == 0:
            recent_scores = scores[-100:]

            print(
                f"Episode: {episode + 1:>5} | "
                f"Score: {game.score:>3} | "
                f"Average score: {sum(recent_scores) / len(recent_scores)} | "
                f"Best: {max(scores)} | "
                f"Epsilon: {agent.epsilon} | "
                f"Memory: {len(agent.memory)} | "
            )


if __name__ == "__main__":
    main()
