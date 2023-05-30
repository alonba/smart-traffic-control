MEMORY_SIZE = 10 ** 5   # The size (number of transitions) of the replay memory buffer # TODO raise to 10^6
BATCH_SIZE = 64         # The number of transitions sampled from the replay buffer
GAMMA = 0.95             # The discount factor as mentioned in the previous section
EXPLORE_CHANCE = 0.1    # Exploration probability  # TODO analyse performance with exploration = 0
TAU = 0.005             # The update rate of the target network
LR = 1e-4               # The learning rate of the optimizer
REWARD_DOWNSCALE = 1    # Down-scaling the reward by this factor, for stabler training process  #TODO later - calculate the downscaling factor
EPISODES_NUM = 25        # Number of episodes in the simulation
UPDATES = 100           # Number of updates taken every episode
IS_SOFT = False         # Hard or Soft update
HARD_UPDATE_N = 10