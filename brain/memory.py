import random
import brain.hyper_params as hpam
from collections import namedtuple, deque


EMPTY_NEXT_STATE = ''
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, agent_state, agent_action, agent_reward, is_last_step):
        """
        Save a transition
        """
        # Set the next state of the last step's Transition
        if self.memory:   # Avoid doing that when the memory is empty
            self.memory[-1] = self.memory[-1]._replace(next_state=agent_state)
        
        # Set the current step's Transition
        if not is_last_step:   # Don't submit the last step, as we will not have a next_step to add to it.
            trans = Transition(agent_state, agent_action, EMPTY_NEXT_STATE, agent_reward)
            self.memory.append(trans)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)