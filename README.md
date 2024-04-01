# Smart Traffic Control

This repository contains the source code and documentation for a smart traffic control system.

The system utilizes reinforcement-learning algorithms and game-theory concepts to optimize traffic flow and improve efficiency on road networks.

We used [OpenAI's gym](https://www.gymlibrary.dev/) as our main platform, combining it with [pyRDDLGym](https://github.com/ataitler/pyRDDLGym) for easy, pythonic, environment creation.

## Our Algorithm

As most algorithms try to solve traffic control problems using a centralized algorithm, we wanted to try a more scalable solution, in a de-centralized algorithm: Each junction in the network is an agent, learning his optimal policy based on his own DQN algorithm, using a hard-updated target-net and a replay-memory buffer.

But that was done a few times already. Our improvements to the algorithm came from the understanding that the agents can and should collaborate.

The cooperation between the agents is modeled in 3 ways:

1. A shared reward function - each agent is aware of his neighbors' reward.
2. A shared state - the agents share their state with their neighbors.
3. Game theory - modeling our environment as a cooperative, non-zero-sum multi-player game.

## Contributors

1. Alon Ben-Arieh
2. David Marder
3. Ayal Taitler
