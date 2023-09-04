from enum import Enum
class Observability(Enum):
    QUEUE = 'q'
    NUMBER_OF_CARS = 'Nc'

ALLOW_MPS = False                       # Is allowed to use Apple's MPS
VERBOSE = True                          # Print progression logs or not

GRID_SIZE = '1x3'                       # The grid topology: rows x columns
EPISODES_NUM = 10000                    # Number of episodes in the simulation
UPDATES = 100                           # Number of updates taken every episode

MEMORY_SIZE = 10 ** 5                   # The size (number of transitions) of the replay memory buffer # TODO raise to 10^6. Although - a big buffer causes overuse -> overweighting of early samples COULD BE SOLVED BY INITIATING THE BUFFER WITH DATA
BATCH_SIZE = 64                         # The number of transitions sampled from the replay buffer

GAMMA = 0.95                            # The discount factor as mentioned in the previous section
EXPLORE_CHANCE = 0.1                    # Exploration probability

LR = 1e-4                               # The learning rate of the optimizer
NET_WIDTH = 64                          # The width of the neural network

TAU = 0.005                             # The update rate of the target network
IS_SOFT = False                         # Hard or Soft update
HARD_UPDATE_N = 12                      # Hard update every N episodes

IS_PRE_PROCESS_PHASE_TO_CYCLIC = True   # Should we perform the cyclic transformation to the phase state

IS_STATE_USE_Q = True                   # Should the state for an agent use the number of cars in the queues of the agent.
IS_STATE_USE_NC = False                 # Should the state for an agent use the number of cars on links coming toward the agent
REWARD_TYPE = Observability.QUEUE       # Wether to use number of cars on link (Nc) or in queue (q) for reward calculation.
REWARD_DOWNSCALE = 1                    # Down-scaling the reward by this factor, for stabler training process  #TODO later - calculate the downscaling factor
NEIGHBORS_WEIGHT = 0                    # The weight an agent gives to his neighbors' rewards. Sums all neighbors weights to 1 for each agent. Meaning - if an agent has 2 neighbors, beta=0.5. if he has 3->beta=0.3333
SHARE_STATE = True                      # Whether to share the neighbor's state or not.
STACKELBERG = SHARE_STATE and False     # Whether we're playing a Stackelberg game.

LSTM = SHARE_STATE and True             # Whether we want to use an LSTM NN as an encoder for the neighbor's shared state.
K_STEPS_BACK = 25                       # How many steps back do we feed the LSTM with
HIDDEN_DIM = 32                         # LSTM layer output size
EMBEDDING_DIM = 16                      # The size of the embedding space created by the LSTM