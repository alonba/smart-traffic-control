import os
import argparse
import datetime
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
import brain.auxiliary as aux
from brain.smart_net import SmartNet
from brain.smart_writer import SmartWriter
import brain.hyper_params as hpam
import brain.env_process as env_process

# TODO type checking + all funcs to have input and output types
# TODO check that it works with 1 agent

start_time = datetime.datetime.now()

# Search hyper params by bash script
parser = argparse.ArgumentParser(description='Run smart traffic control simulation')
parser.add_argument('-hu', '--hard_update', type=int, default=hpam.HARD_UPDATE_N, help='Hard update once every N episodes')
parser.add_argument('-w', '--neighbors_weight', type=float, default=hpam.NEIGHBORS_WEIGHT, help='Weight of neighbors rewards')
parser.add_argument('-k', '--steps_back', type=int, default=hpam.K_STEPS_BACK, help='How many steps back does the LSTM can use for training.')
args = parser.parse_args()

# Init problem
domain = "problem/domain.rddl"
instance = f"problem/{hpam.GRID_SIZE}.rddl"
env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)
num_of_nodes_in_grid = len(env.model.objects['intersection'])

# Set visualizer
viz = ExampleManager.GetEnvInfo('Traffic').get_visualizer()
env.set_visualizer(viz)

# Extract data from env
turns_on_red = env_process.get_turns_on_red(env)
phases_greens = env_process.get_green_turns_per_phase(env)

# Init agents (a net holds agents, one for each node)
smart_net = SmartNet(
    nodes_num=num_of_nodes_in_grid, 
    net_obs_space=env.observation_space, 
    net_action_space=env.action_space, 
    neighbors_weight=args.neighbors_weight,
    turns_on_red = turns_on_red,
    phases_greens = phases_greens,
    steps_back=args.steps_back
    )

# Initialize the SummaryWriter for TensorBoard. Its output will be written to ./runs/
# TODO if hpam.LSTM is false, we get LSTM-False-False-False. Change implementation so we get LSTM-False
run_name = f'\
{aux.now()}_\
{hpam.GRID_SIZE}_\
Reward-{hpam.REWARD_TYPE.value}_\
Explore{hpam.EXPLORE_CHANCE}_\
Beta{args.neighbors_weight}_\
StateShare-{hpam.SHARE_STATE}_\
Stackelberg-{hpam.STACKELBERG}_\
OwnEmbedder-{hpam.IS_OWN_AGENT_STATE_EMBEDDING}_\
LSTM-{hpam.LSTM and args.steps_back}-{hpam.LSTM and hpam.HIDDEN_DIM}-{hpam.LSTM and hpam.EMBEDDING_DIM}_\
StateQ-{hpam.IS_STATE_USE_Q}_\
StateNc-{hpam.IS_STATE_USE_NC}_\
Cyclic-{hpam.IS_PRE_PROCESS_PHASE_TO_CYCLIC}\
'
writer = SmartWriter(run_name)

 
if __name__ == "__main__":
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
            state_with_net_outputs, action = smart_net.sample_action(state)
            
            # Make a step
            aux.print_progress(step, 100, 'Step')
            next_state, centralized_reward, done, info = env.step(action)
            
            # Calculate rewards
            # TODO Why is the centralized reward different than summed computed rewards
            rewards = smart_net.compute_rewards_from_state(next_state)
            total_rewards_q += rewards['self_q']
            total_rewards_Nc += rewards['self_Nc']
            
            # Store the transition in memory
            is_last_step = True if step == (env.horizon - 1) else False
            smart_net.remember(state_with_net_outputs, action, rewards['weighted'], is_last_step)
            
            # Progress to the next step
            state = next_state
            
        # Train the networks
        for update in range(hpam.UPDATES):
            aux.print_progress(update, 50, 'Update')
            losses = smart_net.train(episode, args.hard_update)
            total_losses += losses
            
        # Finish episode
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
    writer.graphs(smart_net, state_with_net_outputs)
    
    print("Job ended successfully!")
    end_of_file = True