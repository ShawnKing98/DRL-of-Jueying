import tensorflow as tf
import numpy as np
from my_utilities import *


"""

main

Use this to train the quadruped

"""





if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """

    from my_ppo import ppo
    from os import path
    import env_config
    from spinup.exercises.common import print_result
    import gym
    import os
    import pandas as pd
    import psutil
    import time

    env_fn = env_config.Anymal
    on_policy = False
    if not on_policy:
        fpath = '/home/shawn/Documents/DRL/outcmaes_backup/2020.4.21/xrecentbest.dat'
        read_target_trajectory(fpath)
    logdir = path.join(path.dirname(__file__),"experiments/%s"%(time.ctime()))
    ppo(env_fn=env_fn, ac_kwargs=dict(policy=mlp_gaussian_policy, hidden_sizes=(256, 128,), activation=tf.tanh),
        steps_per_epoch=4000, epochs=2000, logger_kwargs=dict(output_dir=logdir), target_kl=0.05, on_policy=on_policy)

    # # Get scores from last five epochs to evaluate success.
    # data = pd.read_table(os.path.join(logdir,'progress.txt'))
    # last_scores = data['AverageEpRet'][-5:]
    #
    # # Your implementation is probably correct if the agent has a score >500,
    # # or if it reaches the top possible score of 1000, in the last five epochs.
    # correct = np.mean(last_scores) > 500 or np.max(last_scores)==1e3
    # print_result(correct)