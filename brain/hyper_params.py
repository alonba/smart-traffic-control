# Agent
BATCH_SIZE = 64         # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.8             # GAMMA is the discount factor as mentioned in the previous section
EXPLORE_CHANCE = 0.1    # Exploration probability
TAU = 0.005             # TAU is the update rate of the target network
LR = 1e-4               # LR is the learning rate of the optimizer

# Net
REWARD_DOWNSCALE = 100

# Main
EPISODES_NUM = 3
UPDATES = 100
IS_SOFT = False