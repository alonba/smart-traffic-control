import pickle
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import brain.hyper_params as hpam

def print_progress(step: int, skip: int, name: str) -> None:
    if hpam.VERBOSE:
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

def plot_10000_runs():
    df_0_1 = pd.read_csv('output/10000/Jul01_10-55_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-False.csv')
    df_0_2 = pd.read_csv('output/10000/Jul01_22-57_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-False.csv')
    df_0_3 = pd.read_csv('output/10000/Jul02_10-50_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-False.csv')
    avg_0 = (df_0_1['Value'] + df_0_2['Value'] + df_0_3['Value']) / 3
    
    df_05_1 = pd.read_csv('output/10000/Jul02_22-39_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-False.csv')
    df_05_2 = pd.read_csv('output/10000/Jul03_10-23_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-False.csv')
    df_05_3 = pd.read_csv('output/10000/Jul03_22-14_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-False.csv')
    avg_05 = (df_05_1['Value'] + df_05_2['Value'] + df_05_3['Value']) / 3

    df_1_1 = pd.read_csv('output/10000/Jul04_10-06_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-False.csv')
    df_1_2 = pd.read_csv('output/10000/Jul04_21-54_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-False.csv')
    df_1_3 = pd.read_csv('output/10000/Jul05_09-40_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-False.csv')
    avg_1 = (df_1_1['Value'] + df_1_2['Value'] + df_1_3['Value']) / 3

    df_15_1 = pd.read_csv('output/10000/Jul05_21-23_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-False.csv')
    df_15_2 = pd.read_csv('output/10000/Jul06_09-10_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-False.csv')
    df_15_3 = pd.read_csv('output/10000/Jul06_21-03_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-False.csv')
    avg_15 = (df_15_1['Value'] + df_15_2['Value'] + df_15_3['Value']) / 3
    
    x_axis = avg_0.index * 10
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(x_axis, df_0_1['Value'], label='0-1')
    ax1.plot(x_axis, df_0_2['Value'], label='0-2')
    ax1.plot(x_axis, df_0_3['Value'], label='0-3')
    ax1.plot(x_axis, avg_0, label='0-avg')
    ax1.legend(loc='lower right')
    ax1.set_title('Beta=0')
    
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(x_axis, df_05_1['Value'], label='0.5-1')
    ax2.plot(x_axis, df_05_2['Value'], label='0.5-2')
    ax2.plot(x_axis, df_05_3['Value'], label='0.5-3')
    ax2.plot(x_axis, avg_05, label='0.5-avg')
    ax2.legend(loc='lower right')
    ax2.set_title('Beta=0.5')
    
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(x_axis, df_1_1['Value'], label='1-1')
    ax3.plot(x_axis, df_1_2['Value'], label='1-2')
    ax3.plot(x_axis, df_1_3['Value'], label='1-3')
    ax3.plot(x_axis, avg_1, label='1-avg')
    ax3.legend(loc='lower right')
    ax3.set_title('Beta=1')
    
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(x_axis, df_15_1['Value'], label='1.5-1')
    ax4.plot(x_axis, df_15_2['Value'], label='1.5-2')
    ax4.plot(x_axis, df_15_3['Value'], label='1.5-3')
    ax4.plot(x_axis, avg_15, label='1.5-avg')
    ax4.legend(loc='lower right')
    ax4.set_title('Beta=1.5')
    
    ax5 = fig.add_subplot(3, 1, 3)
    ax5.plot(x_axis, avg_0, label='0-avg')
    ax5.plot(x_axis, avg_05, label='0.5-avg')
    ax5.plot(x_axis, avg_1, label='1-avg')
    ax5.plot(x_axis, avg_15, label='1.5-avg')
    ax5.legend(loc='lower right')
    ax5.set_title('Averages')
    
    fig.suptitle('Reward per episode, Share state = False', fontsize=20)
    fig.tight_layout()
    fig.savefig('output/10000/1x3')

def plot_6000_runs():
    df_0_1 = pd.read_csv('output/6000/Jul08_09-24_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-True.csv')
    df_0_2 = pd.read_csv('output/6000/Jul09_14-04_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-True.csv')
    df_0_3 = pd.read_csv('output/6000/Jul10_18-51_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-True.csv')
    avg_0 = (df_0_1['Value'] + df_0_2['Value'] + df_0_3['Value']) / 3
    
    df_05_1 = pd.read_csv('output/6000/Jul08_16-38_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-True.csv')
    df_05_2 = pd.read_csv('output/6000/Jul09_21-15_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-True.csv')
    df_05_3 = pd.read_csv('output/6000/Jul11_02-02_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-True.csv')
    avg_05 = (df_05_1['Value'] + df_05_2['Value'] + df_05_3['Value']) / 3

    df_1_1 = pd.read_csv('output/6000/Jul08_23-45_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-True.csv')
    df_1_2 = pd.read_csv('output/6000/Jul10_04-23_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-True.csv')
    df_1_3 = pd.read_csv('output/6000/Jul11_09-12_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-True.csv')
    avg_1 = (df_1_1['Value'] + df_1_2['Value'] + df_1_3['Value']) / 3

    df_15_1 = pd.read_csv('output/6000/Jul09_06-52_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-True.csv')
    df_15_2 = pd.read_csv('output/6000/Jul10_11-39_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-True.csv')
    df_15_3 = pd.read_csv('output/6000/Jul11_16-22_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-True.csv')
    avg_15 = (df_15_1['Value'] + df_15_2['Value'] + df_15_3['Value']) / 3
    
    x_axis = avg_0.index * 6
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(x_axis, df_0_1['Value'], label='0-1')
    ax1.plot(x_axis, df_0_2['Value'], label='0-2')
    ax1.plot(x_axis, df_0_3['Value'], label='0-3')
    ax1.plot(x_axis, avg_0, label='0-avg')
    ax1.legend(loc='lower right')
    ax1.set_title('Beta=0')
    
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(x_axis, df_05_1['Value'], label='0.5-1')
    ax2.plot(x_axis, df_05_2['Value'], label='0.5-2')
    ax2.plot(x_axis, df_05_3['Value'], label='0.5-3')
    ax2.plot(x_axis, avg_05, label='0.5-avg')
    ax2.legend(loc='lower right')
    ax2.set_title('Beta=0.5')
    
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(x_axis, df_1_1['Value'], label='1-1')
    ax3.plot(x_axis, df_1_2['Value'], label='1-2')
    ax3.plot(x_axis, df_1_3['Value'], label='1-3')
    ax3.plot(x_axis, avg_1, label='1-avg')
    ax3.legend(loc='lower right')
    ax3.set_title('Beta=1')
    
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(x_axis, df_15_1['Value'], label='1.5-1')
    ax4.plot(x_axis, df_15_2['Value'], label='1.5-2')
    ax4.plot(x_axis, df_15_3['Value'], label='1.5-3')
    ax4.plot(x_axis, avg_15, label='15-avg')
    ax4.legend(loc='lower right')
    ax4.set_title('Beta=1.5')
    
    ax5 = fig.add_subplot(3, 1, 3)
    ax5.plot(x_axis, avg_0, label='0-avg')
    ax5.plot(x_axis, avg_05, label='0.5-avg')
    ax5.plot(x_axis, avg_1, label='1-avg')
    ax5.plot(x_axis, avg_15, label='1.5-avg')
    ax5.legend(loc='lower right')
    ax5.set_title('Averages')
    
    fig.suptitle('Reward per episode, Share state = True', fontsize=20)
    fig.tight_layout()
    fig.savefig('output/6000/1x3')
    
def plot_avgs_share_state_true_vs_false():
    df_0_1_6000 = pd.read_csv('output/6000/Jul08_09-24_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-True.csv')
    df_0_2_6000 = pd.read_csv('output/6000/Jul09_14-04_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-True.csv')
    df_0_3_6000 = pd.read_csv('output/6000/Jul10_18-51_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-True.csv')
    avg_0_6000 = (df_0_1_6000['Value'] + df_0_2_6000['Value'] + df_0_3_6000['Value']) / 3
    
    df_05_1_6000 = pd.read_csv('output/6000/Jul08_16-38_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-True.csv')
    df_05_2_6000 = pd.read_csv('output/6000/Jul09_21-15_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-True.csv')
    df_05_3_6000 = pd.read_csv('output/6000/Jul11_02-02_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-True.csv')
    avg_05_6000 = (df_05_1_6000['Value'] + df_05_2_6000['Value'] + df_05_3_6000['Value']) / 3

    df_1_1_6000 = pd.read_csv('output/6000/Jul08_23-45_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-True.csv')
    df_1_2_6000 = pd.read_csv('output/6000/Jul10_04-23_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-True.csv')
    df_1_3_6000 = pd.read_csv('output/6000/Jul11_09-12_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-True.csv')
    avg_1_6000 = (df_1_1_6000['Value'] + df_1_2_6000['Value'] + df_1_3_6000['Value']) / 3

    df_15_1_6000 = pd.read_csv('output/6000/Jul09_06-52_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-True.csv')
    df_15_2_6000 = pd.read_csv('output/6000/Jul10_11-39_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-True.csv')
    df_15_3_6000 = pd.read_csv('output/6000/Jul11_16-22_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-True.csv')
    avg_15_6000 = (df_15_1_6000['Value'] + df_15_2_6000['Value'] + df_15_3_6000['Value']) / 3

    df_0_1_10000 = pd.read_csv('output/10000/Jul01_10-55_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-False.csv')
    df_0_2_10000 = pd.read_csv('output/10000/Jul01_22-57_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-False.csv')
    df_0_3_10000 = pd.read_csv('output/10000/Jul02_10-50_1x3_Reward-q_Explore0.1_Beta0.0_StateShare-False.csv')
    avg_0_10000 = (df_0_1_10000['Value'] + df_0_2_10000['Value'] + df_0_3_10000['Value']) / 3
    
    df_05_1_10000 = pd.read_csv('output/10000/Jul02_22-39_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-False.csv')
    df_05_2_10000 = pd.read_csv('output/10000/Jul03_10-23_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-False.csv')
    df_05_3_10000 = pd.read_csv('output/10000/Jul03_22-14_1x3_Reward-q_Explore0.1_Beta0.5_StateShare-False.csv')
    avg_05_10000 = (df_05_1_10000['Value'] + df_05_2_10000['Value'] + df_05_3_10000['Value']) / 3

    df_1_1_10000 = pd.read_csv('output/10000/Jul04_10-06_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-False.csv')
    df_1_2_10000 = pd.read_csv('output/10000/Jul04_21-54_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-False.csv')
    df_1_3_10000 = pd.read_csv('output/10000/Jul05_09-40_1x3_Reward-q_Explore0.1_Beta1.0_StateShare-False.csv')
    avg_1_10000 = (df_1_1_10000['Value'] + df_1_2_10000['Value'] + df_1_3_10000['Value']) / 3

    df_15_1_10000 = pd.read_csv('output/10000/Jul05_21-23_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-False.csv')
    df_15_2_10000 = pd.read_csv('output/10000/Jul06_09-10_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-False.csv')
    df_15_3_10000 = pd.read_csv('output/10000/Jul06_21-03_1x3_Reward-q_Explore0.1_Beta1.5_StateShare-False.csv')
    avg_15_10000 = (df_15_1_10000['Value'] + df_15_2_10000['Value'] + df_15_3_10000['Value']) / 3

    x_axis = avg_0_10000.index * 10
    fig, axs = plt.subplots(2,1)
    axs[0].plot(x_axis, avg_0_6000, label='0-avg-True')
    axs[0].plot(x_axis, avg_05_6000, label='0.5-avg-True')
    axs[0].plot(x_axis, avg_1_6000, label='1-avg-True')
    axs[0].plot(x_axis, avg_15_6000, label='1.5-avg-True')
    axs[0].plot(x_axis, avg_0_10000, label='0-avg-False')
    axs[0].plot(x_axis, avg_05_10000, label='0.5-avg-False')
    axs[0].plot(x_axis, avg_1_10000, label='1-avg-False')
    axs[0].plot(x_axis, avg_15_10000, label='1.5-avg-False')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Share state - true vs false')
    
    true_avg = (avg_0_6000 + avg_05_6000 + avg_1_6000 + avg_15_6000) / 4
    false_avg = (avg_0_10000 + avg_05_10000 + avg_1_10000 + avg_15_10000) / 4
    axs[1].plot(x_axis, true_avg, label='Share state - True')
    axs[1].plot(x_axis, false_avg, label='Share state - False')
    axs[1].legend(loc='lower right')
    axs[1].set_title('Share state - true vs false - Averages')
        
    fig.tight_layout()
    fig.savefig('output/true_vs_false_share_state/1x3-avg')
    
if __name__=="__main__":
    # plot_6000_runs()
    # plot_10000_runs()
    plot_avgs_share_state_true_vs_false()