import pandas as pd
from gym.spaces  import Dict
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.agent import SmartAgent
import brain.hyper_params as hpam

class SmartNet(BaseAgent):
    def __init__(self, nodes_num: int, net_obs_space: Dict, net_action_space: Dict) -> None:
        agents = []
        for i in range(nodes_num):
            agent = SmartAgent(f"i{i}", net_action_space, net_obs_space)
            agents.append(agent)
        self.agents = agents
        
    def sample_action(self, state: dict) -> dict:
        """
        Get the actions for each node (agent).
        Return a dictionary with all the chosen actions.
        """
        actions = {}
        for agent in self.agents:
            agent_obs = SmartAgent.filter_agent_obs_from_net_state(agent.name, state)
            actions.update(agent.sample_action(agent_obs))
            # actions |= agent.sample_action(agent_obs)
        
        return actions
    
    def train(self, episode: int, hard_upd_n: int) -> pd.Series:
        """
        Train the policy and target nets of the agents.
        Train the target net according to the method chosen - soft or hard update.
        Returns the training losses of the policy nets for all the agents.
        """
        losses = {}
        for agent in self.agents:
            # Train the policy net
            losses[agent.name] = agent.train_policy_net()
            
            # Train the target net
            if hpam.IS_SOFT:
                agent.train_target_net_soft()
            elif episode % hard_upd_n == 0:   # Hard update once every N updates
                agent.train_target_net_hard()
        
        return pd.Series(losses)
                
    def remember(self, net_state: dict, net_action: dict, net_next_state: dict, rewards: pd.Series) -> None:
        for agent in self.agents:
            agent_obs = SmartAgent.filter_agent_obs_from_net_state(agent.name, net_state)
            agent_action = SmartAgent.filter_agent_dict_from_net_dict(agent.name, net_action)
            agent_next_obs = SmartAgent.filter_agent_obs_from_net_state(agent.name, net_next_state)
            agent_reward = rewards.loc[agent.name]
            agent.memory.push(agent_obs, agent_action, agent_next_obs, agent_reward)
            
    def compute_rewards_from_state(self, state: dict) -> pd.Series:
        """
        Gets a state, returns a df with the calculated rewards per agent.
        """
        # Calculate self rewards for all the agents
        self_rewards = {}
        for agent in self.agents:
            self_rewards[agent.name] = agent.calculate_agent_reward_from_state(state)
        self_rewards = pd.Series(self_rewards)/hpam.REWARD_DOWNSCALE
        
        # Calculate a weighted sum of self and neighboring rewards
        weighted_rewards = self_rewards.copy()
        for agent in self.agents:
            for neighbr in agent.neighbrs:
                weighted_rewards.loc[agent.name] += hpam.NEIGHBRS_WEIGHT * self_rewards.loc[neighbr]
                
        return weighted_rewards