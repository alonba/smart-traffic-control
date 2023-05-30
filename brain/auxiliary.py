import sys
import pickle
import datetime
import pandas as pd

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
    return datetime.datetime.now().strftime("%b%d_%H-%M")

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
    
def get_command_line_args():
    # Access the command-line arguments
    arguments = sys.argv[1:]

    # Print the arguments
    for arg in arguments:
        print(arg)