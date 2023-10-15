import re
import torch
import random
from gym.spaces import Dict, Box
import numpy as np
import pandas as pd
from collections import OrderedDict, deque
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.dqn import DQN
from brain.memory import ReplayMemory, Transition
import brain.hyper_params as hpam

# TODO Reduce as much as possible GPU(or MPS)-CPU communications. It takes a lot of time.

# TODO replace all dictionaries with pandas DF

class SmartAgent(BaseAgent):
    """
    A smart agent is a single traffic light.
    # TODO make the agent inherit the smart_net object (or at least just receive the smart_net as attribute)
    """
    def __init__(self, name: str, net_action_space: Dict, net_state: Dict, neighbors_weight: float, leadership: dict, num_of_phases_per_agent: dict, turns_on_red: pd.DataFrame, phases_greens: pd.DataFrame, steps_back: int) -> None:
        self.name = name
        self.is_leader = leadership[self.name]
        self.net_num_of_phases_per_agent = num_of_phases_per_agent
        self.turns_on_red = turns_on_red
        self.net_phases_greens = phases_greens
        self.action_space = Dict(SmartAgent.filter_agent_dict_from_net_dict(self.name, net_action_space))
        self.neighbors = self.get_neighbors(net_state, leadership)
        self.neighbors_weight = (neighbors_weight / len(self.neighbors)) if (len(self.neighbors) > 0) else 0
        self.steps_back = steps_back
        
        self.filter_agent_and_neighbors_obs_space_from_net_obs_space(net_state, net_action_space)
        self.process_obs_space()
        
        if hpam.LSTM:
            self.init_neighbors_shared_data_memory()
        
        n_own_obs = len(self.proc_agent_obs_space)
        n_neighbor_obs = self.get_lengths_of_neighbors_obs_spaces()
        n_actions = len(self.action_space)
        
        # Init networks
        self.set_device()
        self.policy_net = DQN(n_own_obs, n_neighbor_obs, n_actions, self.neighbors, device=self.device)
        self.target_net = DQN(n_own_obs, n_neighbor_obs, n_actions, self.neighbors, device=self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=hpam.LR)
        self.memory = ReplayMemory(hpam.MEMORY_SIZE)
        self.criterion = torch.nn.SmoothL1Loss()
    
    def get_lengths_of_neighbors_obs_spaces(self):
        lengths = {}
        sum = 0
        for neighbor, neighbor_obs_space in self.proc_neighbors_obs_space.items():
            lengths[neighbor] = len(neighbor_obs_space)
            sum += lengths[neighbor]
        lengths['sum'] = sum
        return lengths
    
    def init_neighbors_shared_data_memory(self) -> None:
        shared_data_memory = {}
        for neighbor in self.neighbors:
            shared_data_memory[neighbor] = deque(maxlen=self.steps_back)
        
        self.neighbors_shared_data_memory = shared_data_memory
    
    def set_device(self):
        if torch.cuda.is_available():
            device = 'cuda'
        # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        elif torch.backends.mps.is_available() and hpam.ALLOW_MPS:
            device = 'mps'
        else:
            device = 'cpu'
        self.device = torch.device(device)
    
    @staticmethod
    def filter_agent_dict_from_net_dict(agent_name, net_dict) -> dict:
        agent_dict = {}
        for k,v in net_dict.items():
            if agent_name in k:
                agent_dict[k] = v
        return agent_dict

    def filter_and_process_agent_state(self, net_state: dict):
        """
        This function get the net state, and filters the agent state from it.
        The agent state is split into the self_agent_state, the shared neighbors_state and the Stackelberg state (shared net outputs)
        The state is being split so we know which part of it goes into the LSTM,
        to which LSTM, and which goes to the other net inputs.
        """
        agent_state = SmartAgent.filter_agent_state_from_net_state(self.agent_obs_space, net_state)
        all_states = {'own': agent_state}
        if hpam.SHARE_STATE:
            neighbors_state = SmartAgent.filter_agent_state_from_net_state(self.neighbors_obs_space, net_state)
            all_states['neighbors'] = self.split_neighbors_state(neighbors_state)
        processed_agent_state = self.process_state(all_states)
        state_tensors = self.state_to_tensors(processed_agent_state)
        return state_tensors

    @staticmethod
    def filter_agent_state_from_net_state(obs_space: dict, net_state: dict) -> dict:
        """
        Gets the net state.
        From it, it extracts the state values corresponding to the keys in the given observation space.
        """
        agent_state = {}
        for k in obs_space.keys():
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
        nc_regex = f"Nc___l-.\d+-{agent_name}"
        agent_turns_on_red = turns_on_red[turns_on_red['pivot'] == agent_name]
        
        observations = {}
        for k,v in net_obs_space.items():
            if signal_obs in k and agent_name in k:    # If signal / signal_t AND relevant for agent.
                observations[k] = v
            elif re.search(q_regex, k) and hpam.IS_STATE_USE_Q:       # If queue
                prefix, from_1, to_1, from_2, to_2 = k.split('-')
                is_turn_on_red = ((agent_turns_on_red['from'] == from_1) * (agent_turns_on_red['to'] == to_2)).any()
                if (from_1 != to_2) and not is_turn_on_red:
                    observations[k] = v
            elif re.search(nc_regex, k) and hpam.IS_STATE_USE_NC:       # If number of cars
                prefix, from_1, to_1 = k.split('-')
                if to_1 == agent_name:
                    observations[k] = v

        return observations
    
    def filter_neighbors_obs_space_from_net_obs_space(self, net_obs_space: Dict) -> dict:
        """
        Gets the smart_net observation space, and returns the observation space of the neighbors of the agent.
        TODO Create a different dictionary for each neighbor
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
    
    def filter_agent_and_neighbors_obs_space_from_net_obs_space(self, net_obs_space: Dict, net_action_space: Dict) -> None:
        """
        Gets the smart_net observation space, and returns the observation space of the agent.
        The observation space of the agent also includes the neighbor's hand-picked observations.
        """
        # TODO wrap all raw obs spaces in one attribute "raw_obs_space" and all processed obs space under "proc_obs_space"
        self.agent_obs_space = SmartAgent.filter_agent_obs_space_from_net_obs_space(self.name, net_obs_space, self.turns_on_red)
        self.neighbors_obs_space = {}
        self.raw_obs_space = self.agent_obs_space.copy()
        if hpam.SHARE_STATE:
            self.neighbors_obs_space = self.filter_neighbors_obs_space_from_net_obs_space(net_obs_space)
        if hpam.STACKELBERG:
            stackelberg_obs_space = self.get_neighboring_leaders_net_outputs_obs_space(net_action_space)
            self.neighbors_obs_space.update(stackelberg_obs_space)
            
        self.raw_obs_space.update(self.neighbors_obs_space)
            
    
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
    
    def state_to_tensors(self, state):
        state_tensors = {'own': None, 'neighbors': {}}
        state_tensors['own'] = self.dict_vals_to_tensor(state['own'])
        if hpam.SHARE_STATE:
            for neighbor, neighbor_state in state['neighbors'].items():
                state_tensors['neighbors'][neighbor] = self.dict_vals_to_tensor(state['neighbors'][neighbor])
            
        return state_tensors
    
    def dict_vals_to_tensor(self, d: dict) -> torch.Tensor:
        """
        Creates and returns a pyTorch tensor made from the dictionary values given.
        """
        return torch.tensor(list(d.values()), device=self.device, dtype=torch.float)
    
    @staticmethod
    def ordered_dict_to_dict(order_dict: OrderedDict) -> dict:
        dict = {}
        for k,v in order_dict.items():
            dict[k] = v
        return dict
    
    def tuple_of_dicts_to_tensor(self, tuple_of_dicts: tuple, output_type: torch.dtype) -> torch.Tensor:
        vals_list = [list(d.values()) for d in tuple_of_dicts]
        return torch.tensor(vals_list, device=self.device, dtype=output_type)

    def tuple_of_dicts_of_tensors_to_dict_of_tensor(self, tuple_of_dicts_of_tensors) -> torch.Tensor:
        # Init the returned tensor
        result = {}
        for neighbor in self.neighbors.keys():
            result[neighbor] = ''
        
        # Pivot the data to dict of tensor
        for dict_of_tensors in tuple_of_dicts_of_tensors:
            for neighbor, state in dict_of_tensors.items():
                if hpam.LSTM:
                    state = state.view(1, self.steps_back, -1)                  
                if result[neighbor] == '':
                    result[neighbor] = state
                else:
                    result[neighbor] = torch.vstack([result[neighbor], state])
                    
        return result
    
    def split_neighbors_state(self, state: dict) -> dict:
        """
        Gets the neighbors state and split it to pandas DataFrames for each neighbor. 
        """
        neighbors_state_series = pd.Series(data=state.values(), index=state.keys())
        
        broken_state = {}
        for neighbor in self.neighbors.keys():
            broken_state[neighbor] = neighbors_state_series[neighbors_state_series.index.str.contains(neighbor)]
            
        return broken_state
    
    # def merge_divided_state_to_tensor(self, state: dict) -> torch.Tensor:
    #     """
    #     Takes the divided (own, neighbors) state and convert to a pyTorch tensor,
    #     such that all of the agent's own states are first,
    #     and the neighbors states are then ordered in an ascending way, by their names.
    #     TODO Don't split the neighbor's state inside, but receive it as an argument.
    #     """
    #     state_tensor = self.dict_vals_to_tensor(state['own'])
        
    #     neighbors_state = self.split_neighbors_state(state['neighbors'])
    #     for neighbor_state in neighbors_state.values():
    #         state_tensor = torch.cat([state_tensor, torch.tensor(neighbor_state.values, device=self.device, dtype=torch.float)])

    #     return state_tensor
    
    # def merge_divided_state_tensor(self, state: dict) -> torch.Tensor:
    #     """
    #     Gets the state, divided into 'own' and 'neighbors', while the 'neighbors' is also divided into individual neighbors.
    #     Merges all this tensors and returns the merged big tensor.
    #     """
    #     merged_tensor = 
        
    def sample_action(self, state: dict):
        """
        Gets the state, and returns:
            1. A dictionary of all policy-net inferred actions
            2. A dict of the policy nets outputs
        Infer from DQN (policy net), or explore the possible actions, with a pre-set probability.
        Also return the raw probabilities produced by the policy net.
        """
        # state_tensor = self.merge_divided_state_to_tensor(state)
                
        # Get the policy net recommendation
        with torch.no_grad():
            net_output = self.policy_net(state['own'], state['neighbors'])
            
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
        
        own_state_batch = torch.stack(batch.state_own)
        neighbors_state_batch = self.tuple_of_dicts_of_tensors_to_dict_of_tensor(batch.state_neighbors)
        action_batch = self.tuple_of_dicts_to_tensor(batch.action, output_type=torch.int64)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
        own_next_state_batch = torch.stack(batch.next_state_own)
        neighbors_next_state_batch = self.tuple_of_dicts_of_tensors_to_dict_of_tensor(batch.next_state_neighbors)
        
        # Compute Q(s_t, a) - the policy_net computes Q(s_t).
        # Then, we ues gather() to select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        # That is the reward the policy net expects to receive by choosing these actions with these states.
        state_action_values = self.policy_net(own_state_batch, neighbors_state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for next_states are computed based on the "older" target_net
        # Selecting their best reward with max(1)[0].
        with torch.no_grad():
            next_state_values = self.target_net(own_next_state_batch, neighbors_next_state_batch).max(1)[0]
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

    def process_obs_space(self) -> None:
        """
        Process the raw observation space to create better features for the NN
        Returns the processed observation space
        
        TODO create a robust mechanism for pre-processing the observation space and state.
        One option is to create a tuple that holds (keys_to_delete, new_keys, key_to_use_for_processing(is it the same ones as keys_to_delete?), process_function that takes the arguments from the keys_to_use and outputs the processed data)
        Such that when the state process step comes, we just need to systematically iterate over all this tuples, and send the required data to the relevant funcs, and receive a new, processed, state.
        """
        # new_obs_space = self.raw_obs_space.copy()
        self.proc_agent_obs_space = self.agent_obs_space.copy()
        self.proc_neighbors_obs_space = self.neighbors_obs_space.copy()
        
        if hpam.IS_PRE_PROCESS_PHASE_TO_CYCLIC:
            self.proc_agent_obs_space = self.discrete_cyclic_to_sin_and_cos(self.proc_agent_obs_space, is_obs_space=True)
            self.proc_neighbors_obs_space = self.discrete_cyclic_to_sin_and_cos(self.proc_neighbors_obs_space, is_obs_space=True)
        if hpam.SHARE_STATE:
            self.proc_neighbors_obs_space = self.sum_neighbor_donations(self.proc_neighbors_obs_space, is_obs_space=True)
            self.proc_neighbors_obs_space = self.split_neighbors_state(self.proc_neighbors_obs_space)
        self.proc_agent_obs_space = self.sum_green_queues_per_phase(self.proc_agent_obs_space, is_obs_space=True)

    def process_state(self, state: dict) -> dict:
        """
        Process the raw state to create better features for the NN
        Returns the processed state
        """
        if hpam.IS_PRE_PROCESS_PHASE_TO_CYCLIC:
            state['own'] = self.discrete_cyclic_to_sin_and_cos(state['own'], is_obs_space=False)
            if hpam.SHARE_STATE:
                for neighbor in state['neighbors'].keys():
                    state['neighbors'][neighbor] = self.discrete_cyclic_to_sin_and_cos(state['neighbors'][neighbor].to_dict(), is_obs_space=False)
        if hpam.SHARE_STATE:
            for neighbor in state['neighbors'].keys():
                state['neighbors'][neighbor] = self.sum_neighbor_donations(state['neighbors'][neighbor].to_dict(), is_obs_space=False, neighbor=neighbor)
        state['own'] = self.sum_green_queues_per_phase(state['own'], is_obs_space=False)
        return state
    
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
                    number_of_phases = self.net_num_of_phases_per_agent[agent_of_signal]
                    cyclic_dict[sin_name] = np.sin(raw_state[k] * 2 * np.pi / number_of_phases)
                    cyclic_dict[cos_name] = np.cos(raw_state[k] * 2 * np.pi / number_of_phases)
                del new_state[k]
            new_state.update(cyclic_dict)
            
        return new_state
    
    def sum_neighbor_donations(self, raw_state: dict, is_obs_space: bool, neighbor: str = None) -> dict:
        """
        Takes a raw state/obs_space and returns the summed neighbor donations
        """
        neighbors_sum = {}
        if is_obs_space:
            for neighbor in self.neighbors.keys():
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
                        if neighbor == pivot:
                            neighbors_sum[f'sum_{pivot}'] += raw_state[k]
                    del new_state[k]
        new_state.update(neighbors_sum)
            
        return new_state
    
    def sum_green_queues_per_phase(self, raw_state: dict, is_obs_space: bool) -> dict:
        """
        Instead of feeding the DQNs with raw data about the number of cars in each queue,
        We will try to make its life easier by summing up the queues relevant to each phase.
        """
        agent_phases_greens = self.net_phases_greens[self.net_phases_greens['pivot'] == self.name]
        new_state = raw_state.copy()
        phases = agent_phases_greens['phase'].unique()
        for phase in phases:
            turns_of_phase = agent_phases_greens[agent_phases_greens['phase'] == phase]
            sum = 0    # Used only on state, not on obs_space
            for index, turn in turns_of_phase.iterrows():
                turn_str = f'q___l-{turn["from"]}-{turn["pivot"]}__l-{turn["pivot"]}-{turn["to"]}'
                if not is_obs_space:
                    sum += new_state[turn_str]
                del new_state[turn_str]
            new_state[phase] = Box(0, np.inf) if is_obs_space else sum
                
        return new_state
    
    def shared_state_memory_to_tensor(self) -> dict:
        """
        Returns a dict with all the neighbor's shared data, as a pyTorch tensor
        """
        dict_of_tensors = {}
        for neighbor in self.neighbors.keys():
            neighbor_shared_data_tensor = torch.stack(list(self.neighbors_shared_data_memory[neighbor]))
            dict_of_tensors[neighbor] = neighbor_shared_data_tensor
        return dict_of_tensors
    
    def store_neighbors_shared_memory(self, state: dict) -> int:
        """
        stores the neighbor's shared memory.
        returns the memory size
        """
        for neighbor in self.neighbors.keys():
            self.neighbors_shared_data_memory[neighbor].append(state['neighbors'][neighbor])
        memory_size = len(self.neighbors_shared_data_memory[neighbor])
        return memory_size
