import torch
# import gym
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.dqn import DQN

class SmartAgent(BaseAgent):
    """
    A smart agent is a single traffic light.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
        n_observations = len(observation_space.spaces)
        n_actions = len(action_space)
        # Our policy net works a little different than the one in the CartPole example, as they need to choose between 2 different actions,
        # Where I need to choose only between advance or not. 
        # 2 options how to do that:
        # have 1 output, and if it's 0 then do not advance and if it's 1 - advance
        # Have 2 outputs - 1 for advance and 1 for stay. 
        self.policy_net = DQN(n_observations, n_actions)
        
    def filter_agent_state_from_full_state(self, state):
        agent_state = {}
        for k,v in self.observation_space.items():
            agent_state[k] = state[k]
        return agent_state
        
    def sample_action(self, state):
        """
        Infer from DQN (policy net)
        The output of the DQN is a number between 0 (stay) and 1 (advance).
        """
        ADVANCE = 1
        STAY = 0
        THRESH = 0.5
        
        # Convert state values to pyTorch tensor
        state_vals_tensor = torch.Tensor(list(state.values()))
        
        # Infer from policy net
        with torch.no_grad():
            net_output = self.policy_net(state_vals_tensor).item()
            
        # Choose action according to policy output
        should_advance = ADVANCE if net_output > THRESH else STAY
        
        # Wrap action in a dictionary, with action name as key
        # TODO find a better way to extract the action name
        action_name = [s for s in self.action_space][0]
        action = {action_name: should_advance}
        return action
    
    def train(self, state, reward):
        """
        Receives the state and reward of the agent and trains the its policy net.
        """
        