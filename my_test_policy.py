import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
from env_config import SIMULATIONFREQUENCY
import numpy as np
import pybullet as p
from matplotlib import pyplot as plt


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

def run_Hand(env):
    from jueying_ff import HandTuningTrajectory, jointTargetsAsFunctionOfTime
    actionTime = 3
    timeLine = [i * 0.2 for i in range(1, int(actionTime / 0.2 + 1))]
    trajectory = np.concatenate([HandTuningTrajectory(i) for i in timeLine])
    env.reset()
    # input("Press any key to start\n")
    observation = env.reset()
    maxTorqueList1 = []
    maxVelocityList1 = []
    timeList = []
    for n in range(2 * 3 * SIMULATIONFREQUENCY):
        action = jointTargetsAsFunctionOfTime(env.t, timeLine, trajectory)
        observation, reward, done, measurement = env.step(action, addNoise=False)
        tttorque1 = np.array(measurement['torque'])
        maxTorqueList1.append(np.abs(tttorque1).max())
        # Max velocity
        maxVelocityList1.append(max(np.abs(measurement["velocity"])))
        timeList.append(env.t)
        # time.sleep(1.0 / SIMULATIONFREQUENCY)  # Normal speed is 1.0, here's to convenient observe
    if measurement['baseOrientationVector'].dot([0, 0, -1]) > np.cos(0.125 * np.pi)\
            and np.alltrue([measurement['position'][i] < -1 for i in [1, 4, 7, 10]]):
        success = True
    else:
        success = False
    return maxVelocityList1, maxTorqueList1, success

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, seed=None):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    success_num = 0
    Handlog = {'maxVelocity': np.array([]), 'maxTorque': np.array([])}
    CMAESlog = {'maxVelocity': np.array([]), 'maxTorque': np.array([])}
    DRLlog = {'maxVelocity': np.array([]), 'maxTorque': np.array([]), 'successNum': 0}
    tmpMaxVelocity = np.array([])
    tmpMaxTorque = np.array([])
    env.__init__("GUI", seed=seed)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    a = get_action(o)

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)
        a = get_action(o)
        for i in range(25):
            o, r, d, o_dict = env.step(a)
            time.sleep(1/SIMULATIONFREQUENCY)
        tmpMaxTorque = np.append(tmpMaxTorque, np.abs(o_dict['torque']).max())
        tmpMaxVelocity = np.append(tmpMaxVelocity, np.abs(o_dict['velocity']).max())
            # time.sleep(1/SIMULATIONFREQUENCY)
        # if env.t >2:
        #     input("hhh")
        ep_ret += r
        ep_len += 1

        # d = False
        # if d or (ep_len == max_ep_len):
        if ep_len == max_ep_len:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            satisfy = d
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            # satisfy = input("Is it satisfying? y or n:\n")
            if satisfy:
                print("done!")
                success_num += 1
                if len(DRLlog['maxVelocity']) != 0:
                    DRLlog['maxVelocity'] += tmpMaxVelocity
                    DRLlog['maxVelocity'] /= success_num
                    DRLlog['maxTorque'] += tmpMaxTorque
                    DRLlog['maxTorque'] /= success_num
                else:
                    DRLlog['maxVelocity'] = tmpMaxVelocity
                    DRLlog['maxTorque'] = tmpMaxTorque
                # tmpMaxVelocity, tmpMaxTorque, success = run_Hand(env)
                # if len(Handlog['maxVelocity']) != 0:
                #     Handlog['maxVelocity'] += tmpMaxVelocity
                #     Handlog['maxVelocity'] /= success_num
                #     Handlog['maxTorque'] += tmpMaxTorque
                #     Handlog['maxTorque'] /= success_num
                # else:
                #     Handlog['maxVelocity'] = tmpMaxVelocity
                #     Handlog['maxTorque'] = tmpMaxTorque
                # tmpMaxVelocity, tmpMaxTorque = run_CMAES()
                # if len(Handlog['maxVelocity']) != 0:
                #     CMAESlog['maxVelocity'] += tmpMaxVelocity
                #     CMAESlog['maxVelocity'] /= success_num
                #     CMAESlog['maxTorque'] += tmpMaxTorque
                #     CMAESlog['maxTorque'] /= success_num
                # else:
                #     CMAESlog['maxVelocity'] = tmpMaxVelocity
                #     CMAESlog['maxTorque'] = tmpMaxTorque
            tmpMaxVelocity = np.array([])
            tmpMaxTorque = np.array([])
            n += 1
    DRLlog['successNum'] = success_num

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    return DRLlog, Handlog, CMAESlog



if __name__ == '__main__':
    fpath = '/home/shawn/Documents/DRL/experiments/Thu May 21 15:16:48 2020'
    itr = 'last'
    deterministic = False
    # epi_len = None
    epi_len = 120
    episodes = 100
    render = True
    env, get_action = load_policy_and_env(fpath,
                                          itr,
                                          deterministic)
    DRLlog, Handlog, CMAESlog = run_policy(env, get_action, epi_len, episodes, render, seed=66)
    timeList = [(i+1)/20 for i in range(120)]
    plt.plot(timeList, DRLlog['maxTorque'])
























