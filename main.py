import os
import datetime
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
import brain.auxiliary as aux
from brain.smart_net import SmartNet
from brain.smart_writer import SmartWriter
import brain.hyper_params as hpam

start_time = datetime.datetime.now()

# Init problem
domain = "problem/domain.rddl"
instance = "problem/1x2.rddl"
env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)
num_of_nodes_in_grid = len(env.model.objects['intersection'])

# Init agents (a net holds agents, one for each node)
smart_net = SmartNet(nodes_num=num_of_nodes_in_grid, net_obs_space=env.observation_space, net_action_space=env.action_space)

# Set visualizer
viz = ExampleManager.GetEnvInfo('Traffic').get_visualizer()
env.set_visualizer(viz)

# Initialize the SummaryWriter for TensorBoard. Its output will be written to ./runs/
run_name = f'{aux.now()}_Gamma{hpam.GAMMA}_Explore{hpam.EXPLORE_CHANCE}_Hard{hpam.HARD_UPDATE_N}'
writer = SmartWriter(run_name)

 
if __name__=="__main__":
    reward_list = []
    for episode in range(hpam.EPISODES_NUM):
        # Initialize env
        total_losses = 0
        total_rewards = 0
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
            total_rewards += rewards
            
            # Store the transition in memory
            smart_net.remember(state, action, next_state, rewards)
            
            # Progress to the next step
            state = next_state
            
        # Train the networks
        for update in range(hpam.UPDATES):
            aux.print_progress(update, 20, 'Update')
            losses = smart_net.train(episode)
            total_losses += losses
            
        # Finish episode
        writer.rewards_or_losses(smart_net, 'Loss', total_losses, episode)
        writer.rewards_or_losses(smart_net, 'Reward', total_rewards, episode)
        writer.weight_histograms(smart_net, episode)
        episode_total_reward = total_rewards.sum()
        reward_list.append(episode_total_reward)
        print(f"Episode {episode + 1} ended with reward {episode_total_reward}")
        env.close()
    
    # Plot and save rewards
    output_dir = "output/" + aux.now() + '_ET-' + aux.elapsed_time(start_time)
    os.mkdir(output_dir)
    aux.save_rewards_per_episode_plot(reward_list, output_dir)
    aux.save_to_pickle(smart_net, output_dir + '/smart_net')
    
    # Save graphs of models to TensorBoard
    writer.graphs(smart_net, state)
    
    end_of_file = True