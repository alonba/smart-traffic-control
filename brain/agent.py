import re
import torch
import random
from gym.spaces import Dict
import pandas as pd
from collections import OrderedDict
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.dqn import DQN
from brain.memory import ReplayMemory, Transition
import brain.hyper_params as hpam

class SmartAgent(BaseAgent):
    """
    A smart agent is a single traffic light.
    """
    def __init__(self, name: str, net_action_space: Dict, net_state: Dict) -> None:
        self.name = name
        self.action_space = Dict(SmartAgent.filter_agent_dict_from_net_dict(self.name, net_action_space))
        self.neighbors = self.get_neighbors(net_state)
        self.observation_space = Dict(self.filter_agent_and_neighbor_obs_from_net_state(net_state))

        n_observations = len(self.observation_space.spaces)
        n_actions = len(self.action_space)
        
        # Init networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=hpam.LR)
        self.memory = ReplayMemory(hpam.MEMORY_SIZE)
        self.criterion = torch.nn.SmoothL1Loss()
    
    @staticmethod
    def filter_agent_dict_from_net_dict(agent_name: str, net_dict) -> dict:
        agent_dict = {}
        for k,v in net_dict.items():
            if agent_name in k:
                agent_dict[k] = v
        return agent_dict

    def filter_neighbors_obs_from_net_state(self, net_state) -> dict:
        """
        Gets the smart_net state, and returns the observation space of the neighbors of the agent.
        # TODO improve implementation. not efficient.
        """
        to_agent_regex = f"^q___l-.\d+-.\d+__l-.\d+-{self.name}"
        signal_obs = 'signal'

        neighbors_state = {}
        for neighbor in self.neighbors:
            original_neighbor_state = SmartAgent.filter_agent_obs_from_net_state(neighbor, net_state)
            new_neighbor_state = original_neighbor_state.copy()
            for k in original_neighbor_state.keys():
                if not re.search(to_agent_regex, k) and signal_obs not in k:
                    del new_neighbor_state[k]
            neighbors_state.update(new_neighbor_state)

        return neighbors_state

    def filter_agent_and_neighbor_obs_from_net_state(self, net_state) -> dict:
        """
        Gets the smart_net state, and returns the obs space for the agent.
        The obs space of the agent also includes the neighbor's hand-picked observations.
        """
        agent_state = SmartAgent.filter_agent_obs_from_net_state(self.name, net_state)
        neighbors_state = self.filter_neighbors_obs_from_net_state(net_state)
        agent_state.update(neighbors_state)
        return agent_state

    @staticmethod
    def filter_agent_obs_from_net_state(agent_name: str, net_state) -> dict:
        """
        Returns only the 'signal', 'signal_t' and 'q' fluents relevant to the specific agent.
        The q fluents relevant are queues incoming to intersection, and going from it.
        """
        signal_obs = 'signal'
        q_regex = f"^q___l-.\d+-{agent_name}__l-{agent_name}-.\d+"
        
        observations = {}
        for k,v in net_state.items():
            if signal_obs in k and agent_name in k:    # If signal / signal_t AND relevant for agent.
                observations[k] = v
            elif re.search(q_regex, k):       # If queue
                prefix, from_1, to_1, from_2, to_2 = k.split('-')
                if from_1 != to_2:
                    observations[k] = v

        return observations
        
    # def calculate_agent_reward_from_state(self, state: dict) -> float:
    #     """
    #     Gets the net state, and calculates the reward for a specific agent.
    #     The reward is the sum of the Nc in the 4 lanes coming in towards an intersection.
    #     """
    #     reward = 0
    #     cars_number_regex = f"Nc___l-.\d+-{self.name}"
    #     for k,v in state.items():
    #         if re.search(cars_number_regex, k):
    #             reward -= v
    #     return reward
    
    def calculate_self_reward_from_Nc(self, cars_on_links: pd.DataFrame) -> float:
        """
        Gets a pandas DF with the number of cars on each link.
        Sums the number of cars entering the junction, and returns it with a minus sign.
        """
        reward = -cars_on_links[cars_on_links['to'] == self.name]['Nc'].sum()
        return reward
    
    def calculate_self_reward_from_q(self, cars_on_queues: pd.DataFrame) -> float:
        """
        Gets a pandas DF with the number of cars on each link.
        Sums the number of cars entering the junction, and returns it with a minus sign.
        """
        reward = -cars_on_queues[cars_on_queues['pivot'] == self.name]['q'].sum()
        return reward
    
    def calculate_neighbors_reward(self, cars_on_queues: pd.DataFrame) -> float:
        """
        Gets a pandas DF with the number of cars on each queue.
        For every one of our agent's neighbors, sum up the number of cars ro to our agent.
        """
        reward = 0
        for neighbor in self.neighbors:
            going_to_agent_mask = cars_on_queues['to'] == self.name
            going_from_neighbor_mask = cars_on_queues['pivot'] == neighbor
            reward -= cars_on_queues[going_to_agent_mask * going_from_neighbor_mask]['q'].sum()
        return reward
    
    def dict_vals_to_tensor(self, d: dict) -> torch.tensor:
        """
        Creates and returns a pyTorch tensor made from the dictionary values given.
        """
        return torch.tensor(list(d.values()), device=self.device).float()
    
    @staticmethod
    def ordered_dict_to_dict(order_dict: OrderedDict) -> dict:
        dict = {}
        for k,v in order_dict.items():
            dict[k] = v
        return dict
    
    def tuple_of_dicts_to_tensor(self, tuple_of_dicts: tuple, output_type: torch.dtype) -> torch.tensor:
        vals_list = [list(d.values()) for d in tuple_of_dicts]
        return torch.tensor(vals_list, device=self.device, dtype=output_type)

    def sample_action(self, state: dict) -> dict:
        """
        Infer from DQN (policy net), or explore the possible actions, with a pre-set probability.
        """
        # TODO - Optimization. First N episodes use only exploration. Than reduce(gradually, or instantly) to 0.1
        sample = random.random()
        if sample > hpam.EXPLORE_CHANCE:
            # Use the policy net recommendation
            with torch.no_grad():
                state_tensor = self.dict_vals_to_tensor(state)
                net_output = self.policy_net(state_tensor)
                chosen_action_index = net_output.argmax().item()
                
                action_name = [s for s in self.action_space][0]
                action = {action_name: chosen_action_index}
                return action

        # Explore a random action
        return self.ordered_dict_to_dict(self.action_space.sample())
    
    def train_policy_net(self) -> float:
        """
        Trains the policy net using data from the replay memory.
        Credit to:
        https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/9da0471a9eeb2351a488cd4b44fc6bbf/reinforcement_q_learning.ipynb#scrollTo=UumN5HdU_EeE
        
        Returns the training loss.
        """
        
        if len(self.memory) < hpam.BATCH_SIZE:
            return
        transitions = self.memory.sample(hpam.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        state_batch = self.tuple_of_dicts_to_tensor(batch.state, output_type=torch.float32)
        action_batch = self.tuple_of_dicts_to_tensor(batch.action, output_type=torch.int64)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = self.tuple_of_dicts_to_tensor(batch.next_state, output_type=torch.float32)
        
        # Compute Q(s_t, a) - the policy_net computes Q(s_t).
        # Then, we ues gather() to select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        # That is the reward the policy net expects to receive by choosing these actions with these states.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for next_states are computed based on the "older" target_net
        # Selecting their best reward with max(1)[0].
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * hpam.GAMMA) + reward_batch
        
        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
        
    def train_target_net_soft(self) -> None:
        """
        Soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*hpam.TAU + target_net_state_dict[key]*(1-hpam.TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        
    def train_target_net_hard(self) -> None:
        """
        Perform a hard update to the target network.
        """
        policy_net_state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(policy_net_state_dict)
        
    def get_neighbors(self, state: Dict) -> list:
        """
        Extract the intercsetion's neighbours from the state.
        """
        neighbors = []
        neighbors_regex = f"Nc___l-i\d+-{self.name}"
        for k,v in state.items():
            if re.search(neighbors_regex, k):
                neighbor_name = k.split('-')[1]
                neighbors.append(neighbor_name)
        return neighbors
