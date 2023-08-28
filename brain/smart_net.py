import re
import pandas as pd
from gym.spaces  import Dict
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.agent import SmartAgent
import brain.hyper_params as hpam

class SmartNet(BaseAgent):
    def __init__(self, nodes_num: int, net_obs_space: Dict, net_action_space: Dict, neighbors_weight: float, turns_on_red: pd.DataFrame, phases_greens: pd.DataFrame) -> None:
        self.size = nodes_num
        self.leadership = self.get_leadership()
        self.obs_space = net_obs_space
        self.num_of_phases_per_agent = self.get_num_of_phases_per_agent()
        self.turns_on_red = turns_on_red
        self.phases_greens = phases_greens
        agents = {}
        for i in range(nodes_num):
            agent_name = f"i{i}"
            agent = SmartAgent(agent_name, net_action_space, net_obs_space, neighbors_weight, self.leadership, self.num_of_phases_per_agent, turns_on_red, phases_greens)
            agents[agent_name] = agent
        self.agents = agents
        
    def sample_action(self, state: dict):
        """
        Get the actions for each node (agent).
        Return a dictionary with all the chosen actions.
        """
        actions = {}
        nets_outputs = {}

        # Get the leader's policy nets outputs and actions
        for agent_name, is_leader in self.leadership.items():
            if is_leader:
                agent_state = self.agents[agent_name].filter_and_process_agent_state(state)
                net_output, action = self.agents[agent_name].sample_action(agent_state)
                actions.update(action)
                nets_outputs.update(net_output)
                
        # TODO Create a function for that, so the code isn't duplicated
        
        # Use the leader's net_outputs to get the follower's chosen actions.
        state_with_net_outputs = {**state, **nets_outputs}
        for agent_name, is_leader in self.leadership.items():
            if not is_leader:
                agent_state = self.agents[agent_name].filter_and_process_agent_state(state_with_net_outputs)
                _, action = self.agents[agent_name].sample_action(agent_state)
                actions.update(action)
                
        return state_with_net_outputs, actions
    
    def train(self, episode: int, hard_upd_n: int) -> pd.Series:
        """
        Train the policy and target nets of the agents.
        Train the target net according to the method chosen - soft or hard update.
        Returns the training losses of the policy nets for all the agents.
        """
        losses = {}
        for agent in self.agents.values():
            # Train the policy net
            losses[agent.name] = agent.train_policy_net()
            
            # Train the target net
            if hpam.IS_SOFT:
                agent.train_target_net_soft()
            elif episode % hard_upd_n == 0:   # Hard update once every N updates
                agent.train_target_net_hard()
        
        return pd.Series(losses)
    
    
    def remember(self, net_state: dict, net_action: dict, rewards: pd.Series, is_last_step: bool) -> None:
        """
        Add the transition to the replay buffer.
        """
        for agent in self.agents.values():
            agent_state = agent.filter_and_process_agent_state(net_state)
            
            # Add data to the neighbors shared data memory
            if hpam.LSTM:
                memory_size = agent.store_neighbors_shared_memory(agent_state)
                agent_state['neighbors'] = agent.shared_state_memory_to_tensor()

            # Add data to the replay buffer
            # TODO create a function for this section
            if (hpam.LSTM and (memory_size == hpam.K_STEPS_BACK)) or (not hpam.LSTM):
                agent_action = SmartAgent.filter_agent_dict_from_net_dict(agent.name, net_action)
                agent_reward = rewards.loc[agent.name]
                agent.memory.push(agent_state['own'], agent_state['neighbors'], agent_action, agent_reward, is_last_step)
            
    @staticmethod
    def get_cars_on_links(state: dict) -> pd.DataFrame:
        """
        Gets the state, and returns a pandas DataFrame containing the number of cars on each link.
        """
        cars_number_list = []
        cars_number_regex = f"Nc___l-.\d+-i\d+"
        for k,v in state.items():
            if re.search(cars_number_regex, k):
                broken = k.split('-')
                cars_number_list.append({'from': broken[1], 'to': broken[2], 'Nc': v})
        return pd.DataFrame(cars_number_list)

    def get_cars_on_queues(self, state: dict) -> pd.DataFrame:
        """
        Gets the state, and returns a pandas DataFrame containing the number of cars on each queue.
        """
        cars_in_q_list = []
        for j in range(self.size):
            cars_in_q_regex = f"q___l-.\d+-i{j}__l-i{j}-.\d+"
            for k,v in state.items():
                if re.search(cars_in_q_regex, k):
                    broken = k.split('-')
                    data = {
                        'from': broken[1],
                        'pivot': broken[3],
                        'to': broken[4],
                        'q': v
                    }
                    if data['to'] != data['from']:
                        cars_in_q_list.append(data)
        return pd.DataFrame(cars_in_q_list)
    
    def compute_rewards_from_state(self, state: dict) -> pd.DataFrame:
        """
        Gets a state, returns a df with the calculated rewards per agent.
        """
        # Extract data from state
        cars_on_queues = self.get_cars_on_queues(state)
        cars_on_links = SmartNet.get_cars_on_links(state)
        
        # Calculate agents' self rewards
        self_rewards_Nc = {}
        self_rewards_q = {}
        for agent in self.agents.values():
            self_rewards_Nc[agent.name] = agent.calculate_self_reward_from_Nc(cars_on_links)
            self_rewards_q[agent.name] = agent.calculate_self_reward_from_q(cars_on_queues)
        self_rewards_Nc = pd.Series(self_rewards_Nc, name='self_Nc')/hpam.REWARD_DOWNSCALE
        self_rewards_q = pd.Series(self_rewards_q, name='self_q')/hpam.REWARD_DOWNSCALE
        
        # Calculate a weighted sum of self and neighboring rewards
        if hpam.REWARD_TYPE == 'q':
            weighted_rewards = self_rewards_q.copy().rename('weighted')
        else:    # hpam.REWARD_TYPE = 'Nc'
            weighted_rewards = self_rewards_Nc.copy().rename('weighted')
        for agent in self.agents.values():
            neighbors_reward = agent.calculate_neighbors_reward(cars_on_queues)
            weighted_rewards.loc[agent.name] += agent.neighbors_weight * neighbors_reward
        rewards = pd.concat([self_rewards_Nc, self_rewards_q, weighted_rewards], axis=1)
        return rewards
    
    def get_leadership(self) -> dict:
        """
        Returns a dictionary holding the role (leader / follower) of each agent in the smart net.
        TODO make it a pandas df instead of dict
        """
        leadership = {}
        for i in range(self.size):
            leadership[f'i{i}'] = False if i%2 == 0 else (True and hpam.STACKELBERG)
            
        return leadership
    
    def get_num_of_phases_per_agent(self) -> dict:
        """
        Extract each agent's num of phases from the net's observation space
        # TODO take this data from the phases df 
        """
        phases = {}
        for k,v in self.obs_space.items():
            if 'signal___' in k:
                agent_name = k[-2:]
                phases[agent_name] = v.n
                
        return phases