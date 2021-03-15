# Frame-Skipping in Reinforcement Learning

Query to train and save models till 24 million time-steps in 3 steps:
- python3 dqn-trainer-colab.py --game Pong --frameskip 1 --update_freq 4 --train --memory_size 800000 --max_steps 50000000 --train_steps 8000000 --path /content/drive/MyDrive/DQN-Train
- python3 dqn-trainer-colab.py --game Pong --frameskip 1 --update_freq 4 --train --save --memory_size 800000 --max_steps 50000000 --train_steps 16000000 --path /content/drive/MyDrive/DQN-Train
- python3 dqn-trainer-colab.py --game Pong --frameskip 1 --update_freq 4 --train --save --memory_size 800000 --max_steps 50000000 --train_steps 24000000 --path /content/drive/MyDrive/DQN-Train