import datetime
import pickle
import pandas as pd

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from brain.smart_net import SmartNet

output_path = "output/"
EPISODES_NUM = 20
UPDATES = 100

# Init problem
domain = "problem/domain.rddl"
instance = "problem/instance0.rddl"   # 2 nodes grid
env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)
num_of_nodes_in_grid = len(env.model.objects['intersection'])

# Init agents (a net holds agents, one for each node)
smart_net = SmartNet(nodes_num=num_of_nodes_in_grid, net_obs_space=env.observation_space, net_action_space=env.action_space)

# Set visualizer
viz = ExampleManager.GetEnvInfo('Traffic').get_visualizer()
env.set_visualizer(viz)

def print_step_number(step):
    print(f'step = {step}')
    
def print_step_status(step, state, action, next_state, reward):
    print()
    print(f'step = {step}')
    print(f'state      = {state}')
    print(f'action     = {action}')
    print(f'next state = {next_state}')
    print(f'reward     = {reward}')
    
def now() -> str:
    return datetime.datetime.now().strftime("%Y-%b-%d_%H%M")

def plot_and_save_rewards_per_episode(reward_list: list):
    reward_series = pd.Series(reward_list)
    reward_series.plot().get_figure().savefig(f'{output_path}{now()}_reward')
    reward_series.plot()
    
def save_to_pickle(data, filename):
    filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
if __name__=="__main__":
    reward_list = []
    for episode in range(EPISODES_NUM):
        # Initialize env
        total_reward = 0
        state = env.reset()
        
        # Run simulation
        for step in range(env.horizon):
            # Visualize
            # env.render()
            
            # Select action
            action = smart_net.sample_action(state)
            
            # Make a step
            print_step_number(step)
            next_state, centralized_reward, done, info = env.step(action)
            
            # Calculate rewards
            # TODO Why is the centralized reward different than summed computed rewards
            # If so - replace the centralized_reward with rewards.sum()
            rewards = smart_net.compute_rewards_from_state(next_state)
            total_reward += centralized_reward
            
            # Store the transition in memory
            smart_net.remember(state, action, next_state, rewards)
            
            # Progress to the next step
            state = next_state
            
        # Train the policies networks
        print("Finished episode. Starting training phase.")
        for update in range(UPDATES):
            print(f"Update = {update}")
            smart_net.train()
            
        # Finish episode
        reward_list.append(total_reward)
        print(f"Episode ended with reward {total_reward}")
        env.close()
    
    # Plot and save rewards
    plot_and_save_rewards_per_episode(reward_list)
    save_to_pickle(smart_net, f'output/{now()}_smart_net')
    
    end_of_file = True