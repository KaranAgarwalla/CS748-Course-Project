import argparse
import os
import numpy as np

def patch(reward_file):
    if not os.path.exists(reward_file):
        raise FileNotFoundError(f'Following file not found: {reward_file}')
    
    reward_data = np.loadtxt(reward_file)
    if reward_data.shape[1] != 5:
        raise ValueError(f'Files are already patched!')
    reward_data = np.delete(reward_data, 0, 1)
    np.savetxt(reward_file, reward_data, fmt='%d %d %d %.2f')

if __name__ == '__main__':
    # Setup Parser
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", help="Path to patch PATH/'GAME'-'FRAMESKIP'/GAMMA-'GAMMA'/run_'RUN_ID'/")
    
    args = parser.parse_args()
    PATH = args.path
            
    reward_per_01 = os.path.join(PATH, 'rewards_every_episode.dat')
    reward_per_10 = os.path.join(PATH, 'rewards_every_10_episodes.dat')

    if not os.path.exists(reward_per_01):
        raise FileNotFoundError("No files found to patch")

    patch(reward_per_01)
    patch(reward_per_10)
    print("Patched")
