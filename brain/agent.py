import re
import torch
import random
from gym.spaces import Dict, Box
import numpy as np
import pandas as pd
from collections import OrderedDict
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.dqn import DQN
from brain.memory import ReplayMemory, Transition
import brain.hyper_params as hpam

class SmartAgent(BaseAgent):
    """
    A smart agent is a single traffic light.
    # TODO make the agent inherit the smart_net object (or at least just receive the smart_net as attribute)
    """
    def __init__(self, name: str, net_action_space: Dict, net_state: Dict, neighbors_weight: float, leadership: dict, phases: dict, turns_on_red: pd.DataFrame) -> None:
        self.name = name
        self.is_leader = leadership[self.name]
        self.net_phases = phases
        self.turns_on_red = turns_on_red
        self.action_space = Dict(SmartAgent.filter_agent_dict_from_net_dict(self.name, net_action_space))
        self.neighbors = self.get_neighbors(net_state, leadership)
        self.neighbors_weight = (neighbors_weight / len(self.neighbors)) if (len(self.neighbors) > 0) else 0
        self.raw_obs_space = self.filter_agent_and_neighbors_obs_space_from_net_obs_space(net_state, net_action_space)
        self.processed_obs_space = self.process_obs_space()

        n_obs = len(self.processed_obs_space)
        n_actions = len(self.action_space)
        
        # Init networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(n_obs, n_actions).to(self.device)
        self.target_net = DQN(n_obs, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=hpam.LR)
        self.memory = ReplayMemory(hpam.MEMORY_SIZE)
        self.criterion = torch.nn.SmoothL1Loss()
    
    @staticmethod
    def filter_agent_dict_from_net_dict(agent_name, net_dict) -> dict:
        agent_dict = {}
        for k,v in net_dict.items():
            if agent_name in k:
                agent_dict[k] = v
        return agent_dict

    def filter_and_process_agent_state(self, net_state: dict):
        agent_state = self.filter_agent_state_from_net_state(net_state)
        processed_agent_state = self.process_state(agent_state)
        return processed_agent_state

    def filter_agent_state_from_net_state(self, net_state: dict) -> dict:
        """
        Gets the net state. Extracts from it just the state fluents relevant for the agent.
        """
        agent_state = {}
        for k in self.raw_obs_space.keys():
            agent_state[k] = net_state[k]
            
        return agent_state

    @staticmethod
    def filter_agent_obs_space_from_net_obs_space(agent_name, net_obs_space, turns_on_red) -> dict:
        """
        Returns only the 'signal', 'signal_t' and 'q' fluents relevant to the specific agent.
        The q fluents relevant are queues incoming to intersection, and going from it.
        """
        signal_obs = 'signal'
        q_regex = f"^q___l-.\d+-{agent_name}__l-{agent_name}-.\d+"
        
        agent_turns_on_red = turns_on_red[turns_on_red['pivot'] == agent_name]
        
        observations = {}
        for k,v in net_obs_space.items():
            if signal_obs in k and agent_name in k:    # If signal / signal_t AND relevant for agent.
                observations[k] = v
            elif re.search(q_regex, k):       # If queue
                prefix, from_1, to_1, from_2, to_2 = k.split('-')
                is_turn_on_red = ((agent_turns_on_red['from'] == from_1) * (agent_turns_on_red['to'] == to_2)).any()
                if (from_1 != to_2) and not is_turn_on_red:
                    observations[k] = v

        return observations
    
    def filter_neighbors_obs_space_from_net_obs_space(self, net_obs_space: Dict) -> dict:
        """
        Gets the smart_net observation space, and returns the observation space of the neighbors of the agent.
        """
        to_agent_regex = f"^q___l-.\d+-.\d+__l-.\d+-{self.name}"
        # TODO create a global string for signals
        signal_obs = 'signal'

        neighbors_obs_space = {}
        for neighbor in self.neighbors:
            original_neighbor_obs_space = SmartAgent.filter_agent_obs_space_from_net_obs_space(neighbor, net_obs_space, self.turns_on_red)
            new_neighbor_obs_space = original_neighbor_obs_space.copy()
            for k in original_neighbor_obs_space.keys():
                if not re.search(to_agent_regex, k) and signal_obs not in k:
                    del new_neighbor_obs_space[k]
            neighbors_obs_space.update(new_neighbor_obs_space)

        return neighbors_obs_space
    
    def filter_agent_and_neighbors_obs_space_from_net_obs_space(self, net_obs_space: Dict, net_action_space: Dict) -> dict:
        """
        Gets the smart_net observation space, and returns the observation space of the agent.
        The observation space of the agent also includes the neighbor's hand-picked observations.
        """
        # TODO have 3 different attributes for the agent - self_space, neighbor_space, stackelberg_space
        # and have a variable that points to the three of them that we can use when we need the complete space
        agent_obs_space = SmartAgent.filter_agent_obs_space_from_net_obs_space(self.name, net_obs_space, self.turns_on_red)
        if hpam.SHARE_STATE:
            neighbors_obs_space = self.filter_neighbors_obs_space_from_net_obs_space(net_obs_space)
            agent_obs_space.update(neighbors_obs_space)
        if hpam.STACKELBERG:
            leaders_neighbors_net_outputs_obs_space = self.get_neighboring_leaders_net_outputs_obs_space(net_action_space)
            agent_obs_space.update(leaders_neighbors_net_outputs_obs_space)
        return agent_obs_space
    
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

    def sample_action(self, state: dict):
        """
        Gets the state, and returns:
            1. A dictionary of all policy-net inferred actions
            2. A dict of the policy nets outputs
        Infer from DQN (policy net), or explore the possible actions, with a pre-set probability.
        Also return the raw probabilities produced by the policy net.
        """
        
        # Get the policy net recommendation
        with torch.no_grad():
            state_tensor = self.dict_vals_to_tensor(state)
            net_output = self.policy_net(state_tensor)
            
            # TODO move to independent function
            net_output_dict = {}

            net_output_dict[f'{self.name}_net_output_diff'] = net_output[0].item() - net_output[1].item()
            
            chosen_action_index = net_output.argmax().item()
            
            action_name = [s for s in self.action_space][0]
            inferred_action = {action_name: chosen_action_index}
        
        # TODO - Optimization. First N episodes use only exploration. Than reduce(gradually, or instantly) to 0.1
        if random.random() <= hpam.EXPLORE_CHANCE:
            # Explore a random action
            random_action = self.ordered_dict_to_dict(self.action_space.sample())
            return net_output_dict, random_action

        return net_output_dict, inferred_action
        
    
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
        
    def get_neighbors(self, state: Dict, leadership: dict) -> dict:
        """
        Extract the intersection's neighbors from the state.
        Associates each neighbor with his leadership status.
        TODO make it a pandas df instead of dict
        """
        neighbors = {}
        neighbors_regex = f"Nc___l-i\d+-{self.name}"
        for k,v in state.items():
            if re.search(neighbors_regex, k):
                neighbor_name = k.split('-')[1]
                neighbors[neighbor_name] = leadership[neighbor_name]
        return neighbors

    def get_neighboring_leaders_net_outputs_obs_space(self, net_action_space: Dict) -> dict:
        """
        Checks if the neighbors are leaders. If so, return their net outputs as a part of the observation space.
        """
        net_output_obs_space = {}
        if not self.is_leader:
            for neighbor_name, is_neighbor_a_leader in self.neighbors.items():
                if is_neighbor_a_leader:
                    neighbor_action_space = SmartAgent.filter_agent_dict_from_net_dict(neighbor_name, net_action_space)
                    for action in neighbor_action_space.keys():
                        neighbor_net_output = {(neighbor_name + '_net_output_diff'): Box(-np.inf, np.inf)}
                    net_output_obs_space.update(neighbor_net_output)
                    
        return net_output_obs_space

    def process_obs_space(self) -> dict:
        """
        Process the raw observation space to create better features for the NN
        Returns the processed observation space
        
        TODO create a robust mechanism for pre-processing the observation space and state.
        One option is to create a tuple that holds (keys_to_delete, new_keys, key_to_use_for_processing(is it the same ones as keys_to_delete?), process_function that takes the arguments from the keys_to_use and outputs the processed data)
        Such that when the state process step comes, we just need to systematically iterate over all this tuples, and send the required data to the relevant funcs, and receive a new, processed, state.
        """
        new_obs_space = self.discrete_cyclic_to_sin_and_cos(self.raw_obs_space, is_obs_space=True)
        new_obs_space = self.sum_neighbor_donations(new_obs_space, is_obs_space=True)
        return new_obs_space

    def process_state(self, agent_state: dict) -> dict:
        """
        Process the raw state to create better features for the NN
        Returns the processed state
        """
        new_state = self.discrete_cyclic_to_sin_and_cos(agent_state, is_obs_space=False)
        new_state = self.sum_neighbor_donations(new_state, is_obs_space=False)
        return new_state
    
    def discrete_cyclic_to_sin_and_cos(self, raw_state: dict, is_obs_space: bool) -> dict:
        """
        This function takes a raw state / raw obs_space
        Returns the cyclic transformation of the signal phase as a part of a new state/obs_space.
        """
        new_state = raw_state.copy()
        for k in raw_state.keys():
            cyclic_dict = {}
            if 'signal__' in k:
                agent_of_signal = k[-2:]
                sin_name = f'signal_sin_{agent_of_signal}'
                cos_name = f'signal_cos_{agent_of_signal}'
                if is_obs_space:
                    cyclic_dict[sin_name] = cyclic_dict[cos_name] = Box(-1, 1)
                else:
                    number_of_phases = self.net_phases[agent_of_signal]
                    cyclic_dict[sin_name] = np.sin(raw_state[k] * 2 * np.pi / number_of_phases)
                    cyclic_dict[cos_name] = np.cos(raw_state[k] * 2 * np.pi / number_of_phases)
                del new_state[k]
            new_state.update(cyclic_dict)
            
        return new_state
    
    def sum_neighbor_donations(self, raw_state: dict, is_obs_space: bool) -> dict:
        """
        Takes a raw state/obs_space and returns the summed neighbor donations
        """
        neighbors_sum = {}
        for neighbor in self.neighbors.keys():
            if is_obs_space:
                neighbors_sum[f'sum_{neighbor}'] = Box(0, np.inf)
            else:
                neighbors_sum[f'sum_{neighbor}'] = 0
        
        new_state = raw_state.copy()
        for k in raw_state.keys():
            if 'q__' in k:
                broken_q = k.split('-')
                pivot = broken_q[3]
                if pivot != self.name:
                    if not is_obs_space:
                        neighbors_sum[f'sum_{pivot}'] += raw_state[k]
                    del new_state[k]
        new_state.update(neighbors_sum)
            
        return new_state
    