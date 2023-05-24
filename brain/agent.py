import torch
import random
import math
from gym.spaces import Dict
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.dqn import DQN
from brain.memory import ReplayMemory, Transition

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


class SmartAgent(BaseAgent):
    """
    A smart agent is a single traffic light.
    """
    def __init__(self, intersection_name: str, net_action_space, net_state):
        self.name = intersection_name
        self.action_space = Dict(SmartAgent.filter_agent_actions_from_net_actions(self.name, net_action_space))
        self.observation_space = Dict(SmartAgent.filter_agent_obs_from_net_state(self.name, net_state))
        
        n_observations = len(self.observation_space.spaces)
        n_actions = len(self.action_space)
        
        # Init networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(10**4)
        self.steps_done = 0
        self.criterion = torch.nn.SmoothL1Loss()
    
    @staticmethod
    def get_observations_from_state(state):
        obs_set = {'signal', 'signal_t', 'q'}
        not_obs = 'virtual-q'
        
        observations = {}
        for k,v in state.items():
            for obs in obs_set:
                if obs in k and not_obs not in k:
                    observations[k] = v

        # TODO filter the q___
        return observations
    
    @staticmethod
    def filter_agent_obs_from_net_state(agent_name: str, net_state):
        """
        First, get only the signal, signal_t, and q from the complete state 
        Then, get just the observations relevant to the specific intersection 
        """
        observations = SmartAgent.get_observations_from_state(net_state)
        
        agent_obs = {}
        for k,v in observations.items():
            if agent_name in k:
                agent_obs[k] = v
        return agent_obs

    @staticmethod
    def filter_agent_actions_from_net_actions(agent_name: str, actions):
        agent_actions = {}
        for k,v in actions.items():
            if agent_name in k:
                agent_actions[k] = v
        return agent_actions

    def filter_agent_reward_from_full_reward(self, reward):
        # TODO filter agent reward from full reward
        return reward
    
    @staticmethod
    def dict_vals_to_tensor(dict):
        """
        Creates and returns a pyTorch tensor made from the dictionary values given.
        """
        return torch.Tensor(list(dict.values()))
    
    @staticmethod
    def ordered_dict_to_dict(order_dict):
        dict = {}
        for k,v in order_dict.items():
            dict[k] = v
        return dict
    
    def tuple_of_dicts_to_tensor(self, tuple_of_dicts, output_type):
        vals_list = [list(d.values()) for d in tuple_of_dicts]
        return torch.tensor(vals_list, device=self.device, dtype=output_type)

        
    def sample_action(self, state):
        """
        Infer from DQN (policy net)
        The output of the DQN is a number between 0 (stay) and 1 (advance).
        """
        
        sample = random.random()
        eps_thresh = EPS_END + ((EPS_START - EPS_END) * math.exp(-1 * self.steps_done / EPS_DECAY))
        self.steps_done += 1
        if sample > eps_thresh:
            # Use the policy net recommendation
            with torch.no_grad():
                state_tensor = self.dict_vals_to_tensor(state)
                net_output = self.policy_net(state_tensor)
                chosen_action_index = net_output.argmax().item()
                
                # TODO find a better way to extract the action name
                action_name = [s for s in self.action_space][0]
                action = {action_name: chosen_action_index}
                return action

        # Explore a random action
        return self.ordered_dict_to_dict(self.action_space.sample())
    
    def train_policy_net(self):
        """
        Trains the policy net using data from the replay memory
        Credit to:
        https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/9da0471a9eeb2351a488cd4b44fc6bbf/reinforcement_q_learning.ipynb#scrollTo=UumN5HdU_EeE
        """
        
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = self.tuple_of_dicts_to_tensor(batch.state, output_type=torch.float32)
        action_batch = self.tuple_of_dicts_to_tensor(batch.action, output_type=torch.int64)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = self.tuple_of_dicts_to_tensor(batch.next_state, output_type=torch.float32)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for next_states are computed based on the "older" target_net
        # Selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have the expected state value
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def train_target_net(self):
        """
        Soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)