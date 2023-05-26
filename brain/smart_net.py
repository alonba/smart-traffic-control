import pandas as pd
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.agent import SmartAgent

# TODO add types to functions
class SmartNet(BaseAgent):
    def __init__(self, nodes_num, net_obs_space, net_action_space):
        self.action_space = net_action_space
        
        agents = []
        for i in range(nodes_num):
            agent = SmartAgent(f"i{i}", net_action_space, net_obs_space)
            agents.append(agent)
            
        self.agents = agents
        
    def sample_action(self, state):
        """
        Get the actions for each node (agent).
        Return a dictionary with all the chosen actions.
        """
        actions = {}
        for agent in self.agents:
            agent_obs = SmartAgent.filter_agent_obs_from_net_state(agent.name, state)
            actions |= agent.sample_action(agent_obs)
        
        return actions
    
    def train(self):
        for agent in self.agents:
            agent.train_policy_net()
            agent.train_target_net()
            
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
        # TODO test
        rewards = {}
        for agent in self.agents:
            rewards[agent.name] = agent.calculate_agent_reward_from_state(state)
        return pd.Series(rewards)