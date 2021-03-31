# Frame-Skipping in Reinforcement Learning

Patch Code:
- Made changes to storing of training rewards. len(rewards) removed for consistency across files.
- Run python3 patch.py --path PATH/'GAME'-'FRAMESKIP'/run_'RUN_ID'/ to modify the files for each such instance
- Thereafter use dqn-trainer-colab-patch.py for proper training

Training and saving of models in batches of 5 million steps (default values):
- !python3 dqn-trainer-colab.py --game Enduro --frameskip 4 --train --save --train_steps 5000000 --gamma 0.99 --path DQN-Train
- !python3 dqn-trainer-colab.py --game Enduro --frameskip 4 --train --save --load --train_steps 10000000 --gamma 0.99 --path DQN-Train
- !python3 dqn-trainer-colab.py --game Enduro --frameskip 4 --train --save --load --train_steps 15000000 --gamma 0.99 --path DQN-Train  

On completion of training look for the last line as "All files saved!"

In case training stops midway in Colab due to any reason
- If string "Training done till *" is not yet printed, use cleaner.py to clean the directory to continue from the last run
- Otherwise, delete the directory and make a fresh start from state 1