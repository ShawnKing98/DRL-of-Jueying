import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.spaces import Box, Discrete
from tensorflow.python.ops import init_ops



EPS = 1e-8
FALL_JOINT_POSITIONS = [- 0.2, 0.7, -1.2,
                        0.2, 0.7, -1.2,
                        - 0.2, 0.7, 1.2,
                        0.2, 0.7, 1.2]

def gaussian_likelihood(x, mu, log_std):
    """
    Calculate the likelihood of a specific x in a Gaussian distribution N(mu, std)
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]

    Returns:
        Tensor with shape [batch]
    """
    EPS = 1e-8
    likelihoods = -0.5*(tf.reduce_sum(((x-mu)/tf.exp(log_std)+EPS)**2+2*log_std+np.log(2*np.pi), axis=1))

    return likelihoods


def my_mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, weight_list=None, bias_list=None):
    """
    Builds a multi-layer perceptron in Tensorflow.

    Args:
        x: Input tensor.

        hidden_sizes: Tuple, list, or other iterable giving the number of units
            for each hidden layer of the MLP.

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer.

        weight_list: The list of initial weight matrix of mlp

        bias_list: The list of initial bias vector of mlp

    Returns:
        A TF symbol for the output of an MLP that takes x as an input.

    """
    for i in range(len(hidden_sizes[0:-1])):
        layer = hidden_sizes[i]
        if weight_list is None:
            weight = None
        else:
            weight = tf.constant_initializer(weight_list[i])
        if bias_list is None:
            bias = init_ops.zeros_initializer()
        else:
            bias = tf.constant_initializer(bias_list[i])
        x = tf.layers.dense(inputs=x, units=layer, activation=activation, kernel_initializer=weight, bias_initializer=bias)

    if weight_list is None:
        weight = None
    else:
        weight = tf.constant_initializer(weight_list[-1])
    if bias_list is None:
        bias = init_ops.zeros_initializer()
    else:
        bias = tf.constant_initializer(bias_list[-1])
    y = tf.layers.dense(inputs=x, units=hidden_sizes[-1], activation=output_activation, kernel_initializer=weight, bias_initializer=bias)

    return y


def my_mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space, weight_list=None, bias_list=None, log_std=None):
    """
    Builds symbols to sample actions and compute log-probs of actions.

    Special instructions: Make log_std a tf variable with the same shape as
    the action vector, independent of x, initialized to [-0.5, -0.5, ..., -0.5].

    Args:
        x: Input tensor of states. Shape [batch, obs_dim].

        a: Input tensor of actions. Shape [batch, act_dim].

        hidden_sizes: Sizes of hidden layers for action network MLP.

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer (action layer).

        action_space: A gym.spaces object describing the action space of the
            environment this agent will interact with.

    Returns:
        pi: A symbol for sampling stochastic actions from a Gaussian
            distribution.

        logp: A symbol for computing log-likelihoods of actions from a Gaussian
            distribution.

        logp_pi: A symbol for computing log-likelihoods of actions in pi from a
            Gaussian distribution.

    """
    act_dim = a.shape.as_list()[1]
    mu = my_mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation, weight_list=weight_list, bias_list=bias_list)
    if log_std is None:
        log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    else:
        log_std = tf.get_variable(name='log_std', initializer=tf.constant(log_std))
    pi = tf.random_normal(shape=tf.shape(mu), mean=mu, stddev=tf.exp(log_std))
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


def my_mlp_actor_critic(x, a, hidden_sizes=(64, 64), activation=tf.tanh,
                        output_activation=None, policy=None, action_space=None,
                        pi_weight_list=None, pi_bias_list=None, log_std=None,
                        v_weight_list=None, v_bias_list=None,):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = my_mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        raise Exception("Oops, the actor-critic of discrete env has not been developed!~")
    #     policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space, pi_weight_list, pi_bias_list, log_std)
    with tf.variable_scope('v'):
        v = tf.squeeze(my_mlp(x, list(hidden_sizes) + [1], activation, None, v_weight_list, v_bias_list), axis=1)

    return pi, logp, logp_pi, v


def logisticKernal(x):

    """
    Use this kernel function to converts a tracking error to a bounded reward, reference from ETH paper.
    """

    return -1/(np.exp(x)+2+np.exp(-x))


def read_target_trajectory(fpath):
    xrecentbest = open(fpath, 'r')
    xrecentbestList = xrecentbest.readlines()
    tmpList = xrecentbestList.copy()
    for i in range(1, len(tmpList)):
        xrecentbestList[i - 1] = [eval(x) for x in tmpList[i].split()]
    xrecentbestList.pop(-1)
    costList = [x[4] for x in xrecentbestList]
    xbest_index = costList.index(min(costList))
    # xbest_index = -1
    xbest = xrecentbestList[xbest_index]
    xbest = xbest[5:]

    actionTime = 3
    timeLine = [i * 0.2 for i in range(1, int(actionTime / 0.2 + 1))]
    # trajectory = np.concatenate([HandTuningTrajectory(i) for i in timeLine])
    trajectory = np.array(xbest)
    global timePoint, tmpTrajectory
    timePoint = np.append(0, timeLine)
    tmpTrajectory = np.append(FALL_JOINT_POSITIONS, trajectory).reshape(-1, 12)
    xrecentbest.close()

    return


def get_action_from_target_policy(t):
    # use this to get the action from a CMAES data file. t represents the time in simulated world(float type).
    # print(timePoint)
    # print(tmpTrajectory)
    jointTarget = time_interp(t, timePoint, tmpTrajectory)

    return jointTarget


def time_interp(t, time, data):

    '''
    use this to implement the interp during a time window. len(time) == column(data)
    data must be numpy array
    '''

    assert len(time) == data.shape[0], "Oops, the length of time must be equal to the columns of data!"
    if len(time) == 0:
        return None
    return np.array([np.interp(t, time, data.T[i]) for i in range(data.shape[1])])


def plot_curriculum_factor():
    x = [i for i in range(2000)]
    plt.plot(x, [0.3 ** (0.997 ** i) for i in x])

    return
