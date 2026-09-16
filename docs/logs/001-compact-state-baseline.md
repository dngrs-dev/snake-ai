# Experiment 001: Compact-State DQN Baseline

## Status

The agent learns to move toward food and avoid immediate collisions. However,
when the snake becomes longer, it often enters a cycle or surrounds an area in
a way that closes its only exit. It cannot reliably plan a safe route around
its own body.

## Behavior

[Watch the recordings](../../recordings/001)

The recording was produced automatically with exploration disabled, so every
action was selected greedily by the loaded policy network.

Checkpoint information:

- Training episodes: 5,000
- Evaluation seed: 123

## Experimental setup

### Environment

- Board: 10 x 10 cells
- Actions: move forward, turn right, or turn left
- Episode timeout: 200 steps without eating
- Food reward: +100
- Collision reward: -100
- Normal step reward: 0
- Board-completion reward: +1000

### Observation

The policy receives an 11-element binary observation:

1. Danger forward
2. Danger right
3. Danger left
4. Moving right
5. Moving left
6. Moving up
7. Moving down
8. Food to the right
9. Food to the left
10. Food above
11. Food below

### Agent

- Algorithm: Double DQN
- Network: 11 -> 256 -> 256 -> 3
- Hidden activation: ReLU
- Replay-buffer capacity: 100,000 transitions
- Batch size: 64
- Discount factor: 0.99
- Learning rate: 0.001
- Epsilon: 1.0 initially, decaying to 0.05
- Epsilon decay: 0.995 per completed episode
- Target-network update interval: 1,000 training steps
- Loss: Smooth L1
- Gradient clipping: 10.0

## Result

This approach does not reliably solve the game.

The agent is capable of collecting food and avoiding hazards that are one move
away. Its decisions become unreliable when survival depends on the shape of
the complete snake body or on the amount of space that will remain reachable
after several moves.

The video is qualitative evidence of this behavior. Quantitative evaluation
over a fixed collection of seeds has not yet been implemented, so no average
score, collision rate, or timeout rate is claimed here.

## Why the approach fails

### Partial observability

Different board configurations can produce exactly the same 11 input values.
For example, a turn can have no immediate danger in two states while leading
to an enclosed region in only one of them. Because the policy receives the
same observation, it cannot learn a different action for each situation.

### Only immediate danger is represented

The danger features answer whether the next cell causes a collision. They do
not describe corridors, enclosed regions, the location of the tail, or the
amount of space reachable after a move.

### Sparse feedback for cycles

A normal step has zero reward, and reaching the step limit also produces no
explicit negative reward. The learning target masks collision termination but
continues to bootstrap timeout transitions. Therefore, repeated movement is
not clearly identified as undesirable unless it eventually causes a
collision.

### No memory

The feed-forward network uses only the current observation. It has no history
from which to recognize that it is repeating an earlier route.

## Conclusion

The compact observation is sufficient for learning basic food-seeking and
one-step collision avoidance, but it is insufficient for planning around a
long body.
