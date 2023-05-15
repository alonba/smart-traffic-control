import gym

from gym.spaces import Dict
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.agent import SmartAgent

class SmartNet(BaseAgent):
    def __init__(self, action_space, nodes_num):
        self.action_space = action_space
        
        agents = []
        for i in range(nodes_num):
            # Get the actions of the specific agent
            agent_action_space = Dict()
            for k,v in action_space.items():
                if f"i{i}" in k:
                    agent_action_space[k] = v
            
            # Init the agent
            agent = SmartAgent(agent_action_space)
            agents.append(agent)
            
        self.agents = agents

    def sample_action(self):
        """
        Get the actions for each node (agent).
        Return a dictionary with all the chosen actions.
        """
        actions = {}
        for agent in self.agents:
            actions |= agent.sample_action()
        
        return actions