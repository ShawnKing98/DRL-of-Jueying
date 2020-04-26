import tensorflow as tf
import numpy as np
from my_utilities import *

def init_learning(env_fn, seed=0):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, *obs_dim))
    a_ph = tf.placeholder(dtype=tf.float32, shape=(None, *act_dim))
    hidden_size = ()

    pi = mlp_gaussian_policy(x_ph, a_ph, hidden_sizes, activation, output_activation, action_space)

