import argparse
import os
import pickle
import numpy as np
import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def clean(reward_file, time_step, frame_number, episode_number):
    if not os.path.exists(reward_file):
        raise FileNotFoundError(f'''Following file not found: {reward_file}; 
            Files may be in inconsistent state: Restart from timestep 0''')
    
    reward_data = np.loadtxt(reward_file)
    mask = reward_data[:, 0] <= time_step
    assert(np.all(reward_data[mask][:, 1] <= frame_number))
    assert(np.all(reward_data[mask][:, 2] <= episode_number))
    reward_data = reward_data[mask,...]
    np.savetxt(reward_file, reward_data, fmt='%d %d %d %.2f')

if __name__ == '__main__':
    # Setup Parser
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", help="Path to cleanup PATH/'GAME'-'FRAMESKIP'/GAMMA-'GAMMA'/run_'RUN_ID'/")
    
    args = parser.parse_args()
    PATH = args.path
    if not os.path.exists(os.path.join(PATH, 'train_time_step.p')):
        raise ValueError("No files found: Delete the folder and restart training")

    time_step = pickle.load(open(os.path.join(PATH, 'train_time_step.p'), 'rb'))
    print(f'Last Time Step:{time_step}')
    frame_number = pickle.load(open(os.path.join(PATH, 'train_frame_number.p'), 'rb'))
    print(f'Last Frame Number:{frame_number}')
    episode_number = pickle.load(open(os.path.join(PATH, 'train_episode_number.p'), 'rb'))
    print(f'Last Episode Number:{episode_number}')

    reward_per_01 = os.path.join(PATH, 'rewards_every_episode.dat')
    reward_per_10 = os.path.join(PATH, 'rewards_every_10_episodes.dat')
    reward_eval_01= os.path.join(PATH, 'rewards_eval_every_episodes.dat')
    reward_eval   = os.path.join(PATH, 'rewards_eval.dat')

    print("Cleaning reward files")
    clean(reward_per_01, time_step, frame_number, episode_number)
    clean(reward_per_10, time_step, frame_number, episode_number)
    clean(reward_eval_01, time_step, frame_number, episode_number)
    clean(reward_eval, time_step, frame_number, episode_number)

    # ### Clean checkpoint files
    print("Updating Checkpoint")
    tf.train.update_checkpoint_state(PATH, os.path.join(PATH, f'my_model-{time_step}'), [os.path.join(PATH, f'my_model-{time_step}')]) 
    
    ### Delete unnecessary models
    print("Deleting models")
    rex = {}
    rex[0] = re.compile(r'my_model-([\d]*)\.meta')
    rex[1] = re.compile(r'my_model-([\d]*)\.data-00000-of-00001')
    rex[2] = re.compile(r'my_model-([\d]*)\.index')

    files = []
    for fd in os.listdir(PATH):
        for key, rexpr in rex.items():
            if rexpr.findall(fd) and rexpr.findall(fd)[0].isdigit() and int(rexpr.findall(fd)[0]) > time_step:
                files.append(os.path.join(PATH, fd))
                break
    
    for fd in files:
        os.remove(fd)
    
    print("Cleaning Done!")
