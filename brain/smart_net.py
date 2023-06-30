import re
import pandas as pd
from gym.spaces  import Dict
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.agent import SmartAgent
import brain.hyper_params as hpam

class SmartNet(BaseAgent):
    def __init__(self, nodes_num: int, net_obs_space: Dict, net_action_space: Dict, neighbors_weight: float) -> None:
        self.size = nodes_num
        agents = {}
        for i in range(nodes_num):
            agent_name = f"i{i}"
            agent = SmartAgent(agent_name, net_action_space, net_obs_space, neighbors_weight)
            agents[agent_name] = agent
        self.agents = agents
        
    def sample_action(self, state: dict) -> dict:
        """
        Get the actions for each node (agent).
        Return a dictionary with all the chosen actions.
        """
        actions = {}
        for agent in self.agents.values():
            agent_state = agent.filter_agent_state_from_net_state(state)
            actions.update(agent.sample_action(agent_state))
            # actions |= agent.sample_action(agent_state)
        
        return actions
    
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
                
    def remember(self, net_state: dict, net_action: dict, net_next_state: dict, rewards: pd.Series) -> None:
        for agent in self.agents.values():
            agent_state = agent.filter_agent_state_from_net_state(net_state)
            agent_action = agent.filter_agent_dict_from_net_dict(net_action)
            agent_next_state = agent.filter_agent_state_from_net_state(net_next_state)
            agent_reward = rewards.loc[agent.name]
            agent.memory.push(agent_state, agent_action, agent_next_state, agent_reward)
            
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
        else:
            weighted_rewards = self_rewards_Nc.copy().rename('weighted')
        for agent in self.agents.values():
            neighbors_reward = agent.calculate_neighbors_reward(cars_on_queues)
            weighted_rewards.loc[agent.name] += agent.neighbors_weight * neighbors_reward
        rewards = pd.concat([self_rewards_Nc, self_rewards_q, weighted_rewards], axis=1)
        return rewards