import gym

from pyRDDLGym.Policies.Agents import BaseAgent

class SmartAgent(BaseAgent):
    """
    A smart agent is a single traffic light.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def tranform_to_dict(self, actions):
        """
        Transform the actions to a dictionary key-value format.
        """
        action = {}
        for sample in actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = actions[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = actions[sample]
        return action        
        
    def sample_action(self):
        # TODO: DQN
        random_action = self.action_space.sample()
        action = self.tranform_to_dict(random_action)

        return action