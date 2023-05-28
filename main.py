import os
import datetime
import pickle
import pandas as pd

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from brain.smart_net import SmartNet

start_time = datetime.datetime.now()
EPISODES_NUM = 2
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

def print_progress(step: int, skip: int, name: str) -> None:
    real_step = step + 1
    if real_step % skip == 0:
        print(f'{name} = {real_step}')

def strfdelta(tdelta: datetime.timedelta, fmt: str) -> str:
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)
    
def now() -> str:
    return datetime.datetime.now().strftime("%b-%d-%H%M")

def elapsed_time(start_time: datetime.datetime) -> str:
    elapsed = datetime.datetime.now() - start_time
    return strfdelta(elapsed, "{hours}H-{minutes}M-{seconds}S")
    
def save_to_pickle(data, filename: str) -> None:
    filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_from_pickle(filename: str):
    filename = str(filename) + '.pickle'
    with open(filename, 'rb') as handle:
        res = pickle.load(handle)
    return res

def save_rewards_per_episode_plot(rewards: list, path: str) -> None:
    """
    Saves 2 files -
        1. Rewards list as pickle
        2. PNG image of the figure
    """
    save_to_pickle(rewards, path + '/rewards_list')
    
    reward_series = pd.Series(rewards)
    ax = reward_series.plot(xlabel='Episode', ylabel='Reward', title='Total reward per episode')
    ax.get_figure().savefig(path + '/reward_per_episode', bbox_inches='tight')
    
    
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
            print_progress(step, 50, 'Step')
            next_state, centralized_reward, done, info = env.step(action)
            
            # Calculate rewards
            # TODO Why is the centralized reward different than summed computed rewards
            rewards = smart_net.compute_rewards_from_state(next_state)
            total_reward += rewards.sum()
            
            # Store the transition in memory
            smart_net.remember(state, action, next_state, rewards)
            
            # Progress to the next step
            state = next_state
            
        # Train the policies networks
        for update in range(UPDATES):
            print_progress(update, 20, 'Update')
            smart_net.train()
            
        # Finish episode
        reward_list.append(total_reward)
        print(f"Episode {episode + 1} ended with reward {total_reward}")
        env.close()
    
    # Plot and save rewards
    output_dir = "output/" + now() + '_ET-' + elapsed_time(start_time)
    os.mkdir(output_dir)
    save_rewards_per_episode_plot(reward_list, output_dir)
    save_to_pickle(smart_net, output_dir + '/smart_net')
    
    end_of_file = True