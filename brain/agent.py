import gym

from pyRDDLGym.Policies.Agents import BaseAgent

class SmartAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def sample_action(self):
        # TODO: DQN
        random_action = self.action_space.sample()

        # Transform the actions to a dictionary key-value format.
        action = {}
        for sample in random_action:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = random_action[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = random_action[sample]
        return action