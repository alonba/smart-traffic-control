from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from brain.smart_net import SmartNet


# Init problem
domain = "problem/domain.rddl"
instance = "problem/instance0.rddl"   # 2 nodes grid
env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)
num_of_nodes_in_grid = len(env.model.objects['intersection'])

# Init agents (a net holds agents, one for each node)
smart_net = SmartNet(nodes_num=num_of_nodes_in_grid, observation_space=env.observation_space, action_space=env.action_space)

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
    
if __name__=="__main__":
    total_reward = 0
    reward = 0
    state = env.reset()
    for step in range(env.horizon):
        env.render()
        
        # Select action
        action = smart_net.sample_action(state)
        
        # Make a step
        print_step_number(step)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Store the transition in memory
        smart_net.remember(state, action, next_state, reward)
        
        # Progress to the next step
        state = next_state
        
        # Train the policies networks
        smart_net.train(state, reward)
        
        if done:
            break
        
    print(f"episode ended with reward {total_reward}")
    env.close()