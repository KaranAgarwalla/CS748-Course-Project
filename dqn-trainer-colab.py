# %%writefile dqn-train.py
# To run this file, use normal command in the cell below like 
# python3 dqn-train.py --game PongDeterminstic --frameskip 1 --train

#Train a DQN Agent to play a specific game

#CONSTANTS
TRAIN           = None           # Boolean value indicating whether the model is to be trained or tested
SAVE            = None           # Boolean value indicating whether models and results need to be saved
GAME            = None           # Name of game
ENV_NAME        = None           # Name of the environment in ALE
ENV_FRAME_SHAPE = [210, 160, 3]  # Shape of frames in the environment
FRAME_SKIP      = None           # Count of frame-skip value; FRAME_SKIP = 1 means no frame skipping 

#CONTROL PARAMETERS
MAX_EPISODE_LENGTH = 72000       # Equivalent of 20 minutes of gameplay at 60 frames per second
SAVE_FREQUENCY = 100000          # Model saved after every SAVE_FREQUENCY timesteps
EVAL_FREQUENCY = 800000          # Number of time_steps between evaluations
EVAL_STEPS = None                # Number of frames for one evaluation
NETW_UPDATE_FREQ = None          # Number of time_steps between updating the target network. 
                                 # set to min(10000*FRAME_SKIP, 160000)
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = 200000# Number of completely random timesteps, 
                                 # before the agent starts learning
MAX_STEPS = 50000000             # Total number of frames the agent sees
TRAIN_STEPS = 50000000           # Total number of frames that the agent sees in current iterations
MEMORY_SIZE = 500000             # Number of transitions stored in the replay memory
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an 
                                 # evaluation episode
UPDATE_FREQ = None               # Every four actions a gradient descend step is performed: set to max(FRAME_SKIP, 16)
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output 
                                 # has the shape (1,1,1024) which is split into two streams. Both 
                                 # the advantage stream and value stream have the shape 
                                 # (1,1,512).
TARGET_LEARNING_RATE = 0.00001   # Learning rate of target network
LEARNING_RATE = 0.00025          # Set to 0.00025 for quicker results. 
BS = 32                          # Batch size
AGENT_HISTORY_LENGTH = 4         # Number of frames stacked together to create a state
FRACTION_GPU = 0.95              # If running multiple instances on same GPU, reduce it to 0.4 else 1

# OBJECT VARIABLES
MAIN_DQN        = None
TARGET_DQN      = None
init            = None
saver           = None
MAIN_DQN_VARS   = None
atari           = None
TARGET_DQN_VARS = None

# PATH VARIABLES
PATH            = None
SUMMARIES       = None
RUNID           = None

import os
import argparse
import random
import gym
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
import numpy as np
import imageio
from skimage.transform import resize
import warnings
import pickle

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = FRACTION_GPU

class FrameProcessor:
    """Resizes and converts RGB Atari frames to grayscale"""
    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.placeholder(shape=ENV_FRAME_SHAPE, dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize_images(self.processed, [self.frame_height, self.frame_width], 
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
    def __call__(self, session, frame):
        """
        Args:
            session: A Tensorflow session object
            frame: A ENV_FRAME_SHAPE frame of an Atari game in RGB
        Returns:
            A processed (frame_height, frame_width, 1) frame in grayscale
        """
        return session.run(self.processed, feed_dict={self.frame:frame})

class DQN(object):
    """Implements a Deep Q Network"""
    
    # pylint: disable=too-many-instance-attributes
    
    def __init__(self, n_actions, hidden=HIDDEN, learning_rate=TARGET_LEARNING_RATE,
                 frame_height=84, frame_width=84, agent_history_length=AGENT_HISTORY_LENGTH):
        """
        Args:
            n_actions: Integer, number of possible actions
            hidden: Integer, Number of filters in the final convolutional layer. 
                    This is different from the DeepMind implementation
            learning_rate: Float, Learning rate for the Adam optimizer
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        
        self.input = tf.placeholder(shape=[None, self.frame_height, 
                                           self.frame_width, self.agent_history_length], 
                                    dtype=tf.float32)
        # Normalizing the input
        self.inputscaled = self.input/255
        
        # Convolutional layers
        self.conv1 = tf.layers.conv2d(
            inputs=self.inputscaled, filters=32, kernel_size=[8, 8], strides=4,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2, 
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1, 
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')
        self.conv4 = tf.layers.conv2d(
            inputs=self.conv3, filters=self.hidden, kernel_size=[7, 7], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')
        
        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.split(self.conv4, 2, 3)
        self.valuestream = tf.layers.flatten(self.valuestream)
        self.advantagestream = tf.layers.flatten(self.advantagestream)
        self.advantage = tf.layers.dense(
            inputs=self.advantagestream, units=self.n_actions,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name="advantage")
        self.value = tf.layers.dense(
            inputs=self.valuestream, units=1, 
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value')
        
        # Combining value and advantage into Q-values as described above
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.best_action = tf.argmax(self.q_values, 1)
        
        # The next lines perform the parameter update. This will be explained in detail later.
        # targetQ according to Bellman equation: 
        # Q = r + gamma*max Q', calculated in the function learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # Action that was performed
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)), axis=1)
        
        # Parameter updates
        self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

class ExplorationExploitationScheduler(object):
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    def __init__(self, DQN, n_actions, eps_initial=1, eps_final=0.1, eps_final_step=0.01, 
                 eps_evaluation=0.0, eps_annealing_steps=4000000, 
                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE, max_steps=MAX_STEPS):
        """
        Args:
            DQN: A DQN object
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first 
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after 
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the 
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during 
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_step = eps_final_step
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_steps = eps_annealing_steps
        self.replay_memory_start_size = replay_memory_start_size
        self.max_steps = max_steps
        
        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_steps
        self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_step)/(self.max_steps - self.eps_annealing_steps - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_step - self.slope_2*self.max_steps
        
        self.DQN = DQN

    def get_action(self, session, time_step, state, evaluation=False):
        """
        Args:
            session: A tensorflow session object
            time_step: Integer, number of the current time_step
            state: A (84, 84, 4) sequence of frames of an Atari game in grayscale
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions - 1 determining the action the agent perfoms next
        """
        if evaluation:
            eps = self.eps_evaluation
        elif time_step < self.replay_memory_start_size:
            eps = self.eps_initial
        elif time_step >= self.replay_memory_start_size and time_step < self.replay_memory_start_size + self.eps_annealing_steps:
            eps = self.slope*time_step + self.intercept
        elif time_step >= self.replay_memory_start_size + self.eps_annealing_steps and time_step < self.max_steps:
            eps = self.slope_2*time_step + self.intercept_2
        else:
            eps = self.eps_final_step

        if random.random() < eps:
            return random.randint(0, self.n_actions)
        return session.run(self.DQN.best_action, feed_dict={self.DQN.input:[state]})[0]  

class ReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, size=MEMORY_SIZE, frame_height=84, frame_width=84, 
                 agent_history_length=AGENT_HISTORY_LENGTH, batch_size=BS):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        
        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length, 
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length, 
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        
    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1 
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
             
    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError(f'Index must be min {self.agent_history_length-1}')
        return self.frames[index-self.agent_history_length+1:index+1, ...]
        
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index
            
    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
        
        self._get_valid_indices()
            
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        
        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]
    
    def load(self, path):
        """
            Loads the Replay Memory State Variables from path
        """
        self.count          = pickle.load(open(PATH+'replay_count.p'), 'rb')
        self.current        = pickle.load(open(PATH+'replay_current.p'), 'rb')
        self.actions        = np.load(PATH+'replay_actions.npy')
        self.rewards        = np.load(PATH+'replay_rewards.npy')
        self.frames         = np.load(PATH+'replay_frames.npy')
        self.terminal_flags = np.load(PATH+'replay_terminal_flags.npy')
    
    def save(self, path):
        """
            Saves the Replay Memory State Variables to path
        """
        pickle.dump(self.count, open(PATH+'/replay_count.p'), 'wb')
        pickle.dump(self.current, open(PATH+'/replay_current.p'), 'wb')
        np.save(PATH+'replay_actions.npy', self.actions)
        np.save(PATH+'replay_rewards.npy', self.rewards)
        np.save(PATH+'replay_frames.npy', self.frames)
        np.save(PATH+'replay_terminal_flags.npy', self.terminal_flags)

def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
    """
    Args:
        session: A tensorflow sesson object
        replay_memory: A ReplayMemory object
        main_dqn: A DQN object
        target_dqn: A DQN object
        batch_size: Integer, Batch size
        gamma: Float, discount factor for the Bellman equation
    Returns:
        loss: The loss of the minibatch, for tensorboard
    Draws a minibatch from the replay memory, calculates the 
    target Q-value that the prediction Q-value is regressed to. 
    Then a parameter update is performed on the main DQN.
    """
    # Draw a minibatch from the replay memory
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()    
    # The main network estimates which action is best (in the next 
    # state s', new_states is passed!) 
    # for every transition in the minibatch
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!) 
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that 
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q * (1-terminal_flags))
    # Gradient descend step to update the parameters of the main network
    loss, _ = session.run([main_dqn.loss, main_dqn.update], 
                          feed_dict={main_dqn.input:states, 
                                     main_dqn.target_q:target_q, 
                                     main_dqn.action:actions})
    return loss

class TargetNetworkUpdater(object):
    """Copies the parameters of the main DQN to the target DQN"""
    def __init__(self, main_dqn_vars, target_dqn_vars):
        """
        Args:
            main_dqn_vars: A list of tensorflow variables belonging to the main DQN network
            target_dqn_vars: A list of tensorflow variables belonging to the target DQN network
        """
        self.main_dqn_vars = main_dqn_vars
        self.target_dqn_vars = target_dqn_vars

    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops
            
    def __call__(self, sess):
        """
        Args:
            sess: A Tensorflow session object
        Assigns the values of the parameters of the main network to the 
        parameters of the target network
        """
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)

def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    for idx, frame_idx in enumerate(frames_for_gif): 
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), 
                                     preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{path}/ATARI_frame_{frame_number}_reward_{reward}.gif', 
                    frames_for_gif, duration=1/30)

class Atari(object):
    """Wrapper for the environment provided by gym"""
    def __init__(self, envName, no_op_steps=NO_OP_STEPS, agent_history_length=AGENT_HISTORY_LENGTH, frameskip=FRAME_SKIP):
        self.env = gym.make(envName, frameskip=frameskip)
        self.process_frame = FrameProcessor()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length
        self.frameskip = frameskip

    def reset(self, sess, evaluation=False):
        """
        Args:
            sess: A Tensorflow session object
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to 
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True # Set to true so that the agent starts 
                                  # with a 'FIRE' action when evaluating
        if evaluation:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1) # Action 'Fire'
        processed_frame = self.process_frame(sess, frame)   # (★★★)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)
        
        return terminal_life_lost

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)  # (5★)
            
        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']
        
        processed_new_frame = self.process_frame(sess, new_frame)   # (6★)
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2) # (6★)   
        self.state = new_state
        
        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame

def clip_reward(reward):
    return np.sign(reward)  

def train(LOAD, PATH):
    """Contains the training and evaluation loops"""

    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)
    if LOAD:
        my_replay_memory.load(PATH)

    update_networks = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)
    explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n, 
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
        max_steps=MAX_STEPS)

    reward_per_01 = PATH + '/rewards_every_episode.dat'
    reward_per_10 = PATH + '/rewards_every_10_episodes.dat'
    reward_eval_01= PATH + '/rewards_eval_every_episodes.dat'
    reward_eval   = PATH + '/rewards_eval.dat'

    with tf.Session(config=config) as sess:

        time_step = 0
        episode_number = 0
        frame_number = 0
        rewards = []
        
        if LOAD:
            ### Load the values
            checkpoint_file = tf.train.latest_checkpoint(PATH)
            loader = tf.train.import_meta_graph(checkpoint_file + '.meta')
            loader.restore(sess, checkpoint_file)
            time_step = pickle.load(open(PATH+'train_time_step.p'), 'rb')
            episode_number = pickle.load(open(PATH+'train_episode_number.p'), 'rb')
            frame_number = pickle.load(open(PATH+'train_frame_number.p'), 'rb')
            rewards = pickle.load(open(PATH+'train_rewards.p'), 'rb')
        else:
            sess.run(init)
        
        if time_step >= TRAIN_STEPS:
            raise ValueError("Agent already trained upto this time_step")
        while time_step < TRAIN_STEPS:
            
            ########################
            ####### Training #######
            ########################

            epoch_steps = 0
            while epoch_steps < EVAL_FREQUENCY:
                terminal_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                episode_iter = 0
                while episode_iter < MAX_EPISODE_LENGTH:
                    episode_iter += FRAME_SKIP # (4★)
                    action = explore_exploit_sched.get_action(sess, time_step, atari.state) # (5★)
                    processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)  
                    time_step += FRAME_SKIP
                    frame_number += 1
                    epoch_steps += FRAME_SKIP
                    episode_reward_sum += reward
                    
                    # Clip the reward
                    clipped_reward = clip_reward(reward)
                    
                    # (7★) Store transition in the replay memory
                    my_replay_memory.add_experience(action=action, 
                                                    frame=processed_new_frame[:, :, 0],
                                                    reward=clipped_reward, 
                                                    terminal=terminal_life_lost)   
                    
                    ## Perform Gradient Descent
                    if time_step % UPDATE_FREQ == 0 and time_step > REPLAY_MEMORY_START_SIZE:
                        loss = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                     BS, gamma = DISCOUNT_FACTOR) # (8★)

                    ## Update the Target Network
                    if time_step % NETW_UPDATE_FREQ == 0 and time_step > REPLAY_MEMORY_START_SIZE:
                        update_networks(sess) # (9★)
                    
                    ## Save the network parameters
                    if time_step % SAVE_FREQUENCY == 0:
                        saver.save(sess, PATH+'/my_model', global_step=time_step)
        
                    if terminal:
                        terminal = False
                        break

                episode_number += 1
                rewards.append(episode_reward_sum)
                
                with open(reward_per_01, 'a') as f:
                    print(len(rewards), time_step, frame_number, episode_number, episode_reward_sum, file = f)
                
                # Output the progress:
                if len(rewards) % 10 == 0:
                    print(len(rewards), time_step, np.mean(rewards[-100:]))
                    with open(reward_per_10, 'a') as f:
                        print(len(rewards), time_step, frame_number, episode_number,
                            np.mean(rewards[-10:]), file=f)
            
            ########################
            ###### Evaluation ######
            ########################
            terminal = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluate_frame_number = 0
            
            for _ in range(EVAL_STEPS):
                if terminal:
                    terminal_life_lost = atari.reset(sess, evaluation=True)
                    episode_reward_sum = 0
                    terminal = False
               
                # Fire (action 1), when a life was lost or the game just started, 
                # so that the agent does not stand around doing nothing. When playing 
                # with other environments, you might want to change this...
                action = 1 if terminal_life_lost else explore_exploit_sched.get_action(sess, time_step,
                                                                                       atari.state, 
                                                                                       evaluation=True)
                
                processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action) ### A seperate Atari
                evaluate_frame_number += 1
                episode_reward_sum += reward

                if gif: 
                    frames_for_gif.append(new_frame)
                if terminal:
                    with open(reward_eval_01, 'a') as f:
                        print(time_step, frame_number, episode_number, episode_reward_sum, file = f)
                    gif = False # Save only the first game of the evaluation as a gif
                    break
            
            ## Append the rewards
            eval_rewards.append(episode_reward_sum)
            print("Evaluation score:\n", np.mean(eval_rewards))       
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], PATH)
            except IndexError:
                print("No evaluation game finished")
            
            frames_for_gif = []
            with open(reward_eval, 'a') as f:
                print(time_step, frame_number, episode_number, np.mean(eval_rewards), file=f)
        
        if LOAD:
            saver.save(sess, PATH+'/my_model', global_step=time_step)
            pickle.dump(time_step, open(PATH+'/train_time_step.p'), 'wb')
            pickle.dump(episode_number, open(PATH+'/train_episode_number.p'), 'wb')
            pickle.dump(frame_number, open(PATH+'/train_frame_number.p'), 'wb')
            pickle.dump(rewards, open(PATH+'/train_rewards.p'), 'wb')
    
    if LOAD:
        my_replay_memory.save(PATH)

def eval_model(frameskip, time_step, meta_graph_path, checkpoint_path):
    '''
        frameskip: frameskip parameter
        meta_graph_path: path to meta-file (e.g.: '/home/DQN/Enduro-20/run_1/my_model-30000000.meta')
        checkpoint_path: path to checkpoint-file (e.g.: '/home/DQN/Enduro-20/run_1/my_model-30000000') 
    '''
    gif_path = "/home/karan1agarwalla/GIF/"
    os.makedirs(gif_path, exist_ok=True)

    explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n, 
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
        max_steps=MAX_STEPS)
    
    with tf.Session(config=config) as sess:

        ### Restore Model
        saver = tf.train.import_meta_graph(meta_graph_path)
        saver.restore(sess, checkpoint_path)

        frames_for_gif = []
        terminal_life_lost = atari.reset(sess, evaluation = True)
        episode_reward_sum = 0
        while len(frames_for_gif) < EVAL_STEPS:
            # atari.env.render()
            action = 1 if terminal_life_lost else explore_exploit_sched.get_action(sess, 0, atari.state,  
                                                                                evaluation = True)
            
            processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
            episode_reward_sum += reward
            frames_for_gif.append(new_frame)
            if terminal == True:
                break
        
        # atari.env.close()
        print("The total reward is {}".format(episode_reward_sum))
        print("Creating gif...")
        generate_gif(time_step, frames_for_gif, episode_reward_sum, gif_path)
        print(f'Gif created, check the folder /home/karan1agarwalla/GIFS/{GAME}_{FRAME_SKIP}_{time_step}')

if __name__ == '__main__':
    # Setup Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default = "Pong", help="Name of Atari Game")
    parser.add_argument("--version", default = 4, type = int, help="Version")
    parser.add_argument("--frameskip", default = 1, type = int, help="frameskip value")

    parser.add_argument("--train", action='store_true', help='Train vs Test')
    parser.add_argument("--save", action='store_true', help='Save Models and Results')
    parser.add_argument("--load", action='store_true', help="Load Model in last run_id in PATH")

    parser.add_argument("--eval_steps", type = int, help="Number of evaluation steps")
    parser.add_argument("--netw_update_freq", type = int, help="Frequency of swapping main and target network")
    parser.add_argument("--update_freq", type = int, help="Number of actions before gradient descent")
    parser.add_argument("--memory_size", type = int, default = 1000000, help="Size of replay memory: Default 0.5 million")
    parser.add_argument("--max_steps", type = int, default = 50000000, help="Total number of frames an agent sees")
    parser.add_argument("--train_steps", type = int, default = 50000000, help="Trained upto TRAIN_STEPS")
    parser.add_argument("--time_step", type = int, help="TIME_STEP corresponding to evaluation of model")
    parser.add_argument("--path", help="Path to store models and values: PATH/'GAME'-'FRAMESKIP'/run_'RUN_ID'/")

    random.seed(0)
    args = parser.parse_args()
    tf.reset_default_graph()

    GAME        = args.game
    ENV_NAME    = f'{args.game}Deterministic-v{args.version}'
    FRAME_SKIP  = args.frameskip

    TRAIN       = args.train
    SAVE        = args.save
    atari       = Atari(ENV_NAME, no_op_steps=NO_OP_STEPS, frameskip=FRAME_SKIP)
        
    # main DQN and target DQN networks:
    with tf.variable_scope('mainDQN'):
        MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)
    with tf.variable_scope('targetDQN'):
        TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN, TARGET_LEARNING_RATE)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=100000)    

    MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
    TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

    # update frequencies of the target and main networks
    if args.netw_update_freq:
        NETW_UPDATE_FREQ = args.netw_update_freq
    else:
        NETW_UPDATE_FREQ = min(10000*FRAME_SKIP, 160000)
    
    if args.update_freq:
        UPDATE_FREQ = args.update_freq
    else:
        UPDATE_FREQ = max(FRAME_SKIP, 16)
    
    if args.eval_steps:
        EVAL_STEPS = args.eval_steps
    else:
        EVAL_STEPS  = int(MAX_EPISODE_LENGTH/FRAME_SKIP)

    MEMORY_SIZE = args.memory_size
    MAX_STEPS   = args.max_steps
    TRAIN_STEPS = args.train_steps
    if args.train:
        ### Need to save and load the model
        if args.path:
            PATH = args.path+f'/{GAME}-{FRAME_SKIP}'
        else:
            PATH = f'/content/drive/MyDrive/DQN-Train/{GAME}-{FRAME_SKIP}' 
        ### Fetch RUNID
        RUNID = 1
        while os.path.exists(PATH + '/run_' + str(RUNID)):
            RUNID += 1
        
        if args.load:
            RUNID -= 1

        RUNID     = '/run_' + str(RUNID)
        PATH      = PATH + RUNID
        os.makedirs(PATH, exist_ok=True)
        print(f'The env {ENV_NAME} has the following {atari.env.action_space.n} \
        actions: {atari.env.unwrapped.get_action_meanings()}')
        train(args.load, PATH)
    
    else:
        if args.time_step:
            print(f'Proceeding with Evaluation of Model with time step {args.time_step}')
            META_PATH   = args.path + f'/my_model-{args.time_step}.meta'
            CHECKPOINT  = args.path + f'/my_model-{args.time_step}'
            eval_model(FRAME_SKIP, args.time_step, META_PATH, CHECKPOINT)
        else:
            raise ValueError("Evaluation Model Not Specified")   