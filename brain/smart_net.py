from gym.spaces import Dict
from pyRDDLGym.Policies.Agents import BaseAgent
from brain.agent import SmartAgent

class SmartNet(BaseAgent):
    def __init__(self, nodes_num, net_obs_space, action_space):
        self.action_space = action_space
        
        agents = {}
        for i in range(nodes_num):
            intersection_name = f"i{i}"
            
            # Get the actions of the specific agent
            # TODO use function
            agent_action_space = Dict()
            for k,v in action_space.items():
                if intersection_name in k:
                    agent_action_space[k] = v

            # Init the agent
            agent = SmartAgent(intersection_name, agent_action_space, net_obs_space)
            agents[intersection_name] = agent
            
        self.agents = agents

    def sample_action(self, state):
        """
        Get the actions for each node (agent).
        Return a dictionary with all the chosen actions.
        """
        actions = {}
        for intersection_name, agent in self.agents.items():
            agent_obs = SmartAgent.filter_agent_obs_from_net_state(agent.name, state)
            actions |= agent.sample_action(agent_obs)
        
        return actions
    
    def train(self, state, reward):
        """
        Gets the full state of the whole net.
        Distribute the state and rewards down to each agent to learn.
        """
        for intersection_name, agent in self.agents.items():
            agent.train_policy_net()
            agent.train_target_net()
            
    def remember(self, net_state, action, net_next_state, reward):
        for intersection_name, agent in self.agents.items():
            agent_obs = SmartAgent.filter_agent_obs_from_net_state(agent.name, net_state)
            agent_action = agent.filter_agent_action_from_full_action(action)
            agent_next_obs = SmartAgent.filter_agent_obs_from_net_state(agent.name, net_next_state)
            agent_reward = agent.filter_agent_reward_from_full_reward(reward)
            agent.memory.push(agent_obs, agent_action, agent_next_obs, agent_reward)