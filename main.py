from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from brain.smart_net import SmartNet


# Init problem
domain = "problem/domain.rddl"
instance = "problem/instance0.rddl"   # 2 nodes grid
env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)
num_of_nodes_in_grid = len(env.model.objects['intersection'])

# Init agents (a net holds agents, one for each node)
smart_net = SmartNet(action_space=env.action_space, nodes_num=num_of_nodes_in_grid)

# Set visualizer
viz = ExampleManager.GetEnvInfo('Traffic').get_visualizer()
env.set_visualizer(viz)

# TODO init replay memory 
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/9da0471a9eeb2351a488cd4b44fc6bbf/reinforcement_q_learning.ipynb#scrollTo=UumN5HdU_EeE

if __name__=="__main__":
    total_reward = 0
    state = env.reset()
    for step in range(env.horizon):
        env.render()
        
        # Training
        smart_net.train(state, reward)
        
        # Inference
        action = smart_net.sample_action()
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        # print()
        print(f'step = {step}')
        # print(f'state      = {state})
        # print(f'action     = {action})
        # print(f'next state = {next_state})
        # print(f'reward     = {reward})
        state = next_state
        if done:
            break
    print("episode ended with reward {}".format(total_reward))
    env.close()