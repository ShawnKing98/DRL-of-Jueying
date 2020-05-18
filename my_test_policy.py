import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
from env_config import SIMULATIONFREQUENCY


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value
        saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x) > 8]
        itr = '%d' % max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % itr

    # load the get_action function
    get_action = load_tf_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state['env']
        print("Oh yes, load environment succeed")
        print(env)
    except:
        print("Oh no, load environment failed")
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save' + itr)
    print('\n\nLoading from %s.\n\n' % fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    env.__init__("GUI")
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    a = get_action(o)

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        for i in range(25):
            o, r, d, _ = env.step(a)
            time.sleep(1/SIMULATIONFREQUENCY)
        # if env.t >2:
        #     input("hhh")
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    fpath = '/home/shawn/Documents/DRL/experiments/Fri May 15 22:59:55 2020'
    itr = 'last'
    deterministic = False
    # epi_len = None
    epi_len = 60
    episodes = 30
    render = True
    env, get_action = load_policy_and_env(fpath,
                                          itr,
                                          deterministic)
    run_policy(env, get_action, epi_len, episodes, render)
