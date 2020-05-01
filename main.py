import tensorflow as tf
import numpy as np
from my_utilities import *


"""

main

Use this to train the quadruped

"""





if __name__ == '__main__':

    from my_init_learning import init_learning
    from continue_ppo import ppo
    from os import path
    import env_config
    import time

    env_fn = env_config.Anymal

    on_policy = False
    if not on_policy:
        target_path = "/home/shawn/Documents/DRL/outcmaes_backup/2020.4.21/xrecentbest.dat"
        read_target_trajectory(target_path)

    # Change load_path to None if you want to train from the beginning
    load_path = "/home/shawn/Documents/DRL/experiments/Fri May  1 21:51:43 2020"
    # load_path = None
    if load_path is not None:
        fname = path.join(load_path, 'tf1_save')
        sess = tf.Session()
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            fname
        )
        graph = tf.get_default_graph()
        pi_weight_list = [sess.run(tf.global_variables("pi/dense/kernel:0")[0]),
                          sess.run(tf.global_variables("pi/dense_1/kernel:0")[0]),
                          sess.run(tf.global_variables("pi/dense_2/kernel:0")[0])]
        pi_bias_list = [sess.run(tf.global_variables("pi/dense/bias:0")[0]),
                        sess.run(tf.global_variables("pi/dense_1/bias:0")[0]),
                        sess.run(tf.global_variables("pi/dense_2/bias:0")[0])]
        v_weight_list = [sess.run(tf.global_variables("v/dense/kernel:0")[0]),
                         sess.run(tf.global_variables("v/dense_1/kernel:0")[0]),
                         sess.run(tf.global_variables("v/dense_2/kernel:0")[0])]
        v_bias_list = [sess.run(tf.global_variables("v/dense/bias:0")[0]),
                       sess.run(tf.global_variables("v/dense_1/bias:0")[0]),
                       sess.run(tf.global_variables("v/dense_2/bias:0")[0])]
        log_std = sess.run(tf.global_variables("pi/log_std:0")[0])
        tf.reset_default_graph()
        sess.close()
    else:
        [pi_weight_list, pi_bias_list, log_std, v_weight_list, v_bias_list] = 5 * [None]

    # log_std = -0.5 * np.ones(12, dtype=np.float32)    # reset std to jump out of local maximum

    logdir = path.join(path.dirname(__file__), "experiments/%s"%(time.ctime()))
    # ppo(env_fn=env_fn,
    #     GUI=False,
    #     actor_critic=my_mlp_actor_critic,
    #     ac_kwargs=dict(
    #         hidden_sizes=(256, 128,),
    #         activation=tf.tanh,
    #         pi_weight_list=pi_weight_list,
    #         pi_bias_list=pi_bias_list,
    #         log_std=log_std,
    #         v_weight_list=v_weight_list,
    #         v_bias_list=v_bias_list
    #     ),
    #     steps_per_epoch=4000,
    #     epochs=3000,
    #     logger_kwargs=dict(output_dir=logdir),
    #     max_ep_len=4000,
    #     target_kl=0.05,
    #     on_policy=on_policy,
    #     load_path=load_path)
    init_learning(env_fn=env_fn,
        GUI=False,
        actor_critic=my_mlp_actor_critic,
        ac_kwargs=dict(
            hidden_sizes=(256, 128,),
            activation=tf.tanh,
            pi_weight_list=pi_weight_list,
            pi_bias_list=pi_bias_list,
            log_std=log_std,
            v_weight_list=v_weight_list,
            v_bias_list=v_bias_list
        ),
        steps_per_epoch=4000,
        epochs=6000,
        logger_kwargs=dict(output_dir=logdir),
        max_ep_len=4000,
        target_kl=1.0,
        on_policy=on_policy,
        prev_epochs=2000)
