LEARN = True                            # Learn or Analyze   # TODO After training the smart_net, analyze mode produces the same reward every episode, as training is disabled. Maybe i misunderstood the explore=0 instruction.

EPISODES_NUM = 10                       # Number of episodes in the simulation
UPDATES = 100                           # Number of updates taken every episode

MEMORY_SIZE = 10 ** 5                   # The size (number of transitions) of the replay memory buffer # TODO raise to 10^6. Although - a big buffer gives causes overuse -> overwighting of early samples/steps
BATCH_SIZE = 64                         # The number of transitions sampled from the replay buffer

GAMMA = 0.95                            # The discount factor as mentioned in the previous section
EXPLORE_CHANCE = 0.1 if LEARN else 0    # Exploration probability
TAU = 0.005                             # The update rate of the target network
LR = 1e-4                               # The learning rate of the optimizer

REWARD_DOWNSCALE = 1                    # Down-scaling the reward by this factor, for stabler training process  #TODO later - calculate the downscaling factor

IS_SOFT = False                         # Hard or Soft update
HARD_UPDATE_N = 10                      # Hard update every N episodes

NEIGHBRS_WEIGHT = 0.2