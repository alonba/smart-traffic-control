from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent


# Init problem
domain = "problem/domain.rddl"
instance = "problem/instance0.rddl"   # 2 nodes grid
env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)

# Init agent
agent = RandomAgent(action_space=env.action_space, num_actions=env.numConcurrentActions)

# Set visualizer
viz = ExampleManager.GetEnvInfo('Traffic').get_visualizer()
env.set_visualizer(viz)


if __name__=="__main__":
    total_reward = 0
    state = env.reset()
    for step in range(env.horizon):
        env.render()
        action = agent.sample_action()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print()
        print('step       = {}'.format(step))
        print('state      = {}'.format(state))
        print('action     = {}'.format(action))
        print('next state = {}'.format(next_state))
        print('reward     = {}'.format(reward))
        state = next_state
        if done:
            break
    print("episode ended with reward {}".format(total_reward))
    env.close()