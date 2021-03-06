import matplotlib.pyplot as plt
import numpy as np

def plot_training_data(game="Seaquest", ds=[1,4,20,60], file_ends="rewards_every_episode.dat", x_axis="frame_number"):
    map_2_index = {"time_steps": 1, "frame_number": 2, "episode_number": 3, "exec_time": 4}
    legends = []
    for d in ds:
        data = np.loadtxt(f'data/{game}/d={d}_{file_ends}')
        # data = [len(rewards), time_steps, frame_number, episode_number, exec_time, episode_reward]
        if d != 1:
            plt.scatter(data[:,map_2_index[x_axis]], data[:,5], s=0.1)
        else:
            plt.scatter(data[:,map_2_index[x_axis]], data[:,4], s=0.1)
        legends.append(f'd={d}')
        # plt.show()
    plt.legend(legends)
    # plt.xlim([0,1000])
    plt.title(f'Episode reward vs {x_axis}')
    plt.ylabel('Episode reward')
    plt.xlabel(f'{x_axis}')
    plt.show()

def multi_plots_training_data(game="Seaquest", ds=[1,4,20,60], file_ends="rewards_every_episode.dat", x_axis="frame_number"):
    map_2_index = {"time_steps": 1, "frame_number": 2, "episode_number": 3, "exec_time": 4}
    legends = []
    fig, axs = plt.subplots(2, 2)
    for i,d in enumerate(ds):
        data = np.loadtxt(f'data/{game}/d={d}_{file_ends}')
        # data = [len(rewards), time_steps, frame_number, episode_number, exec_time, episode_reward]
        if d!= 1:
            axs[i//2, i%2].plot(data[:,map_2_index[x_axis]], data[:,5])
        else:
            axs[i//2, i%2].plot(data[:,map_2_index[x_axis]], data[:,4])
        axs[i//2, i%2].set_title(f'd={d}')
        # plt.show()
    for ax in axs.flat:
        ax.set(xlabel=f'{x_axis}', ylabel='Episode reward')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()


def plot_test_data(game="Seaquest", ds=[1,4,20,60], file_ends="rewards_eval_every_episode.dat", x_axis="frame_number"):
    map_2_index = {"time_steps": 0, "frame_number": 1, "episode_number": 2, "exec_time": 3}
    legends = []
    for d in ds:
        data = np.loadtxt(f'data/{game}/d={d}_{file_ends}')
        # data = [time_steps, frame_number, episode_number, exec_time, episode_reward]
        plt.plot(data[:,map_2_index[x_axis]], data[:,4])
        legends.append(f'd={d}')
        # plt.show()
    plt.legend(legends)
    plt.xlim([0,5e7])
    plt.title(f'Episode reward vs {x_axis}')
    plt.ylabel('Episode reward')
    plt.xlabel(f'{x_axis}')
    plt.show()

if __name__ == "__main__":
    plot_test_data(x_axis="time_steps", ds=[1,4,20,60])
    # plot_training_data(x_axis="frame_number")