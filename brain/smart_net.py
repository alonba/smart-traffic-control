from gym.spaces import Dict
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.agent import SmartAgent

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
            
    def remember(self, net_state, net_action, net_next_state, reward):
        for agent in self.agents:
            agent_obs = SmartAgent.filter_agent_obs_from_net_state(agent.name, net_state)
            agent_action = SmartAgent.filter_agent_actions_from_net_actions(agent.name, net_action)
            agent_next_obs = SmartAgent.filter_agent_obs_from_net_state(agent.name, net_next_state)
            agent_reward = agent.filter_agent_reward_from_full_reward(reward)
            agent.memory.push(agent_obs, agent_action, agent_next_obs, agent_reward)