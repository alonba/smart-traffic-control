import os
import argparse
import datetime
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
import brain.auxiliary as aux
from brain.smart_net import SmartNet
from brain.smart_writer import SmartWriter
import brain.hyper_params as hpam

start_time = datetime.datetime.now()

# Search hpams by bash script
parser = argparse.ArgumentParser(description='Run smart traffic control simulation')
parser.add_argument('-hu', '--hard_update', type=int, default=hpam.HARD_UPDATE_N, help='Hard update once every N episodes')
parser.add_argument('-w', '--neighbors_weight', type=float, default=hpam.NEIGHBORS_WEIGHT, help='Weight of neighbors rewards')
# parser.add_argument('-e', '--explore', type=float, default=hpam.EXPLORE_CHANCE, help='Exploration probability')
args = parser.parse_args()

# Init problem
domain = "problem/domain.rddl"
instance = f"problem/{hpam.GRID_SIZE}.rddl"
env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)
num_of_nodes_in_grid = len(env.model.objects['intersection'])

# Init agents (a net holds agents, one for each node)
# if hpam.LEARN:
smart_net = SmartNet(
    nodes_num=num_of_nodes_in_grid, 
    net_obs_space=env.observation_space, 
    net_action_space=env.action_space, 
    neighbors_weight=args.neighbors_weight
    )
# else:   # Use pre-trained net for performance analysis
#     smart_net_name = 'May30_17-41_ET-0H-1M-56S'
#     smart_net = aux.load_from_pickle(f'output/{smart_net_name}/smart_net')

# Set visualizer
viz = ExampleManager.GetEnvInfo('Traffic').get_visualizer()
env.set_visualizer(viz)

# Initialize the SummaryWriter for TensorBoard. Its output will be written to ./runs/
# if hpam.LEARN:
run_name = f'{aux.now()}_{hpam.GRID_SIZE}_Reward-{hpam.REWARD_TYPE}_Explore{hpam.EXPLORE_CHANCE}_Beta{args.neighbors_weight}_Hard{args.hard_update}'
# else:
#     run_name = f'Analyze_{smart_net_name}'
writer = SmartWriter(run_name)

 
if __name__=="__main__":
    reward_list = []
    for episode in range(hpam.EPISODES_NUM):
        # Initialize env
        total_losses = 0
        total_rewards_q = 0
        total_rewards_Nc = 0
        state = env.reset()
        
        # Run simulation
        for step in range(env.horizon):
            # Visualize
            # env.render()
            
            # Select action
            action = smart_net.sample_action(state)
            
            # Make a step
            aux.print_progress(step, 50, 'Step')
            next_state, centralized_reward, done, info = env.step(action)
            
            # Calculate rewards
            # TODO Why is the centralized reward different than summed computed rewards
            rewards = smart_net.compute_rewards_from_state(next_state)
            total_rewards_q += rewards['self_q']
            total_rewards_Nc += rewards['self_Nc']
            
            # Store the transition in memory
            # if hpam.LEARN:
            smart_net.remember(state, action, next_state, rewards['weighted'])
            
            # Progress to the next step
            state = next_state
            
        # Train the networks
        # if hpam.LEARN:
        for update in range(hpam.UPDATES):
            aux.print_progress(update, 25, 'Update')
            losses = smart_net.train(episode, args.hard_update)
            total_losses += losses
            
        # Finish episode
        # if hpam.LEARN:
        writer.weight_histograms(smart_net, episode)
        writer.rewards_or_losses(smart_net, 'Loss', total_losses, episode)
        writer.rewards_or_losses(smart_net, 'Reward-q', total_rewards_q, episode)
        writer.rewards_or_losses(smart_net, 'Reward-Nc', total_rewards_Nc, episode)
        episode_total_reward = total_rewards_q.sum()
        reward_list.append(episode_total_reward)
        print(f"Episode {episode + 1} ended with reward {episode_total_reward}")
        env.close()
    
    # Save rewards vs episode plot, and pickle smart_net.
    output_dir = "output/" + run_name + '_ET-' + aux.elapsed_time(start_time)
    os.mkdir(output_dir)
    aux.save_rewards_per_episode_plot(reward_list, output_dir)
    aux.save_to_pickle(smart_net, output_dir + '/smart_net')
    
    # Save graphs of models to TensorBoard
    # if hpam.LEARN:
    writer.graphs(smart_net, state)
    
    end_of_file = True