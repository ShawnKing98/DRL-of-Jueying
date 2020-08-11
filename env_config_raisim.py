import os
import numpy as np
import raisimpy as raisim
import raisimpy_gym.reward_logger   # raisim.gui.reward_logger is defined here
from raisimpy_gym.raisim_gym_env import RaisimGymEnv, keyboard_interrupt
from raisimpy_gym.vis_setup_callback import setup_callback
from my_utilities import logisticKernal, time_interp

fall_base_position = [0., 0., 1.]
fall_base_orientation = [1.6, 0., 0., 1.]
fall_joint_angles = [-0.2, 0.7, -1.2,
                    0.2, 0.7, -1.2,
                    -0.2, 0.7, 1.2,
                    0.2, 0.7, 1.2]

ANYMAL_RESOURCE_DIRECTORY = os.path.dirname(__file__)


def get_rotation_matrix_from_quaternion(q):
    """
    Get rotation matrix from the given quaternion.

    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]

    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    w, x, y, z = q
    rot = np.array([[2 * (w**2 + x**2) - 1, 2 * (x*y - w*z), 2 * (x*z + w*y)],
                    [2 * (x*y + w*z), 2 * (w**2 + y**2) - 1, 2*(y*z - w*x)],
                    [2 * (x*z - w*y), 2 * (y*z + w*x), 2 * (w**2 + z**2) - 1]])
    return rot


class JueyingEnv(RaisimGymEnv):
    # config is a dict containing some configuration of the RL environment, such as reward coefficients
    def __init__(self, config={}, resource_directory=ANYMAL_RESOURCE_DIRECTORY, visualizable=True):
        super(JueyingEnv, self).__init__(config, resource_directory, visualizable)

        # set variables
        self.distribution = lambda: np.random.normal(0.0, 0.2)  # UNKNOWN use

        # add objects
        self.robot = self.world.add_articulated_system(self.resource_directory + "/JueyingURDF/JueyingMiniV2.urdf")
        self.robot.set_control_mode(raisim.ControlMode.PD_PLUS_FEEDFORWARD_TORQUE)
        self.ground = self.world.add_ground()
        self.world.set_erp(0., 0.)  # UNKNOWN use

        # get robot data (gc = generalized coordinates, gv = generalized velocities)
        self.gc_dim = self.robot.get_generalized_coordinate_dim()
        self.gv_dim = self.robot.get_dof()
        self.num_joints = 12

        # initialize containers
        self.gc, self.gc_init = np.zeros(self.gc_dim), np.zeros(self.gc_dim)
        self.gv, self.gv_init = np.zeros(self.gv_dim), np.zeros(self.gv_dim)
        self.torques = np.zeros(self.gv_dim)
        self.p_targets, self.v_targets = np.zeros(self.gc_dim), np.zeros(self.gv_dim)
        self.p_target12 = np.zeros(self.num_joints)

        # this is the nominal configuration of Jueying
        self.gc_init = np.array([*fall_base_position, *fall_base_orientation, *fall_joint_angles])

        # set PD gains
        self.joint_p_gains, self.joint_d_gains = np.zeros(self.gv_dim), np.zeros(self.gv_dim)
        self.joint_p_gains[-self.num_joints:] = np.array(4 * [200., 200., 300.])  # the last 12 elements of p_gains are joints
        self.joint_d_gains[-self.num_joints:] = np.array(4 * [2., 2., 3.])
        self.robot.set_pd_gains(self.joint_p_gains, self.joint_d_gains)  # only effective in PD plus feedforward torque
        self.robot.set_generalized_forces(np.zeros(self.gv_dim))  # feedforward forces, not the actual forces

        # MUST BE DONE FOR ALL ENVIRONMENTS
        self.ob_dim = 45  # convention described on top
        self.action_dim = self.num_joints

        # action and observation scaling (interesting)
        self.action_mean = np.zeros(self.action_dim)
        self.action_std = 1.0 * np.ones(self.action_dim)

        self.ob_mean = np.array([
            *np.zeros(self.num_joints),     # joint angles
            *np.zeros(self.num_joints),     # joint velocity
            0.0, 0.0, 0.0,                  # base orientation vector
            *np.zeros(6),                   # body linear/angular velocity
            *np.zeros(12)                   # joint acceleration
        ])

        self.ob_std = np.array([
            *np.ones(self.num_joints),          # joint angles
            *np.ones(self.num_joints) * 10.0,   # joint velocity
            0.7, 0.7, 0.7,                      # base orientation vector
            *np.ones(3) * 2.0,                  # body linear velocity
            *np.ones(3) * 4.0,                  # body angular velocity
            *np.ones(12) * 1.0                  # joint acceleration
        ])

        # reward coefficients
        self.curriculum_factor = float(self.config['environment']['curriculumFactor'])
        self.base_orientation_coeff = float(self.config['environment']['baseOrientationCoeff'] * self.simulation_dt)
        self.joint_position_coeff_HAA = float(self.config['environment']['jointPositionCoeffHAA'] * self.simulation_dt)
        self.joint_position_coeff_HFE = float(self.config['environment']['jointPositionCoeffHFE'] * self.simulation_dt)
        self.joint_position_coeff_KFE = float(self.config['environment']['jointPositionCoeffKFE'] * self.simulation_dt)
        self.joint_velocity_coeff = float(self.config['environment']['jointVelocityCoeff'] * self.simulation_dt)
        self.joint_acceleration_coeff = float(self.config['environment']['jointAccelerationCoeff'] * self.simulation_dt)
        self.joint_torque_coeff = float(self.config['environment']['jointTorqueCoeff'] * self.simulation_dt)
        self.contact_slip_coeff = float(self.config['environment']['contactSlipCoeff'] * self.simulation_dt)
        self.body_contact_impulse_coeff = float(self.config['environment']['bodyContactImpulseCoeff'] * self.simulation_dt)
        self.internal_contact_coeff = float(self.config['environment']['internalContactCoeff'] * self.simulation_dt)
        self.smoothness_coeff = float(self.config['environment']['smoothnessCoeff'] * self.simulation_dt)

        # reward logger defined in reward_logger.py
        raisim.gui.reward_logger.init(["baseOrientationReward",
                                        "jointPositionReward",
                                        "jointVelocityReward",
                                        "jointAccelerationReward",
                                        "jointTorqueReward",
                                        "contactSlipReward",
                                        "bodyContactImpulseReward",
                                        "internalContactReward",
                                        "smoothnessReward",
                                        "totalReward"])

        # indices of links that should not make contact with ground
        # self.foot_indices = set([self.robot.get_body_index(name)
        #                          for name in ['LF_SHANK', 'RF_SHANK', 'LH_SHANK', 'RH_SHANK']])

        # visualize
        if self.visualizable:
            vis = raisim.OgreVis.get()

            # these methods must be called before initApp
            vis.set_world(self.world)
            vis.set_window_size(1280, 720)
            vis.set_default_callbacks()
            vis.set_setup_callback(setup_callback)
            vis.set_anti_aliasing(2)

            # starts visualizer thread
            vis.init_app()

            self.robot_visual = vis.create_graphical_object(self.robot, name="Jueying")
            vis.create_graphical_object(self.ground, dimension=20, name="floor", material="checkerboard_green")
            self.desired_fps = 60.
            vis.set_desired_fps(self.desired_fps)

        # define other variables
        self.base_orientation_reward = 0.
        self.joint_position_reward = 0.
        self.joint_velocity_reward = 0.
        self.joint_acceleration_reward = 0.
        self.joint_torque_reward = 0.
        self.contact_slip_reward = 0.
        self.body_contact_impulse_reward = 0.
        self.internal_contact_reward = 0.
        self.smoothness_reward = 0.
        self.total_reward = 0.
        self.done = False
        self.ob_double, self.ob_scaled = np.zeros(self.ob_dim), np.zeros(self.ob_dim)
        self.body_linear_vel, self.body_angular_vel = np.zeros(3), np.zeros(3)
        self.history_length = 0.03

        # define extra info
        self.extra_info["last_action"] = np.nan * np.ones(12)
        self.extra_info["joint_history"] = dict()
        self.extra_info["joint_history"]["time"] = np.array([])
        self.extra_info["joint_history"]["position"] = np.array([])
        self.extra_info["joint_history"]["velocity"] = np.array([])
        self.extra_info["joint_history"]["position_error"] = np.array([])

    # UNKNOWN use
    def init(self):
        pass

    def reset(self):
        self.robot.set_states(self.gc_init, self.gv_init)
        if self.visualizable:
            raisim.gui.reward_logger.clean()

            # reset camera
            vis = raisim.OgreVis.get()
            vis.select(self.robot_visual[0], False)
            vis.get_camera_man().set_yaw_pitch_dist(3.14, -1.3, 3, track=True)

        self.robot.set_control_mode(raisim.ControlMode.FORCE_AND_TORQUE)
        for i in range(int(1./self.simulation_dt)):
            self.world.integrate()
            vis_decimation = int(1. / (self.desired_fps * self.simulation_dt) + 1.e-10)  # frame dt / simulation dt
            if self.visualizable and self.visualize_this_step and (self.visualization_counter % vis_decimation == 0):
                vis = raisim.OgreVis.get()
                vis.render_one_frame()
            self.visualization_counter += 1
        self.robot.set_control_mode(raisim.ControlMode.PD_PLUS_FEEDFORWARD_TORQUE)

        self.extra_info["joint_history"]["velocity"] = np.array([self.gv_init[-self.num_joints:]])
        self.update_observation()
        self.extra_info["last_action"] = (self.gc_init[-self.num_joints:] - self.action_mean) / self.action_std
        self.extra_info["joint_history"]["time"] = np.array([0.])
        self.extra_info["joint_history"]["position"] = np.array([self.gc_init[-self.num_joints:]])
        self.extra_info["joint_history"]["position_error"] = np.array([np.zeros(12)])


        return self.ob_scaled

    @keyboard_interrupt
    def step(self, action, addNoise=False):  # step for 1 action
        # action scaling
        self.p_target12 = np.array(action, dtype=np.float64)
        self.p_target12 *= self.action_std
        self.p_target12 += self.action_mean
        self.p_targets[-self.num_joints:] = self.p_target12

        # set actions
        self.robot.set_pd_targets(self.p_targets, self.v_targets)

        loop_count = int(self.control_dt / self.simulation_dt + 1.e-10)
        vis_decimation = int(1. / (self.desired_fps * self.simulation_dt) + 1.e-10)  # frame dt / simulation dt

        # update world
        for i in range(loop_count):
            self.world.integrate()

            if self.visualizable and self.visualize_this_step and (self.visualization_counter % vis_decimation == 0):
                vis = raisim.OgreVis.get()
                vis.render_one_frame()

            self.visualization_counter += 1

        # update observation
        self.update_observation()

        # update if episode is over or not
        self.is_terminal_state()

        # update reward
        self.update_reward()

        # update extra info
        self.update_extra_info(action)

        # visualization
        if self.visualize_this_step:
            raisim.gui.reward_logger.log("baseOrientationReward", self.base_orientation_reward)
            raisim.gui.reward_logger.log("jointPositionReward", self.joint_position_reward)
            raisim.gui.reward_logger.log("jointVelocityReward", self.joint_velocity_reward)
            raisim.gui.reward_logger.log("jointAccelerationReward", self.joint_acceleration_reward)
            raisim.gui.reward_logger.log("jointTorqueReward", self.joint_torque_reward)
            raisim.gui.reward_logger.log("contactSlipReward", self.contact_slip_reward)
            raisim.gui.reward_logger.log("bodyContactImpulseReward", self.body_contact_impulse_reward)
            raisim.gui.reward_logger.log("internalContactReward", self.internal_contact_reward)
            raisim.gui.reward_logger.log("smoothnessReward", self.smoothness_reward)
            raisim.gui.reward_logger.log("totalReward", self.total_reward)

            # reset camera
            # vis = raisim.OgreVis.get()
            #
            # vis.select(self.robot_visual[0], False)
            # vis.get_camera_man().set_yaw_pitch_dist(3.14, -1.3, 3, track=True)

        return self.ob_scaled, self.total_reward, self.done, self.extra_info

    def update_observation(self):
        self.gc, self.gv = self.robot.get_states()
        self.ob_double, self.ob_scaled = np.zeros(self.ob_dim), np.zeros(self.ob_dim)

        # joint angles
        self.ob_double[0:12] = self.gc[-12:]

        # joint velocity
        self.ob_double[12:24] = self.gv[-12:]

        # base orientation vector
        quat = self.gc[3:7]
        rot = get_rotation_matrix_from_quaternion(quat)
        self.ob_double[24:27] = -rot[:, 2]  # numpy is row-major while Eigen is column-major

        # body linear/angular velocity
        self.body_linear_vel = rot.dot(self.gv[:3])
        self.body_angular_vel = rot.dot(self.gv[3:6])
        self.ob_double[27:30] = self.body_linear_vel
        self.ob_double[30:33] = self.body_angular_vel

        # joint acceleration
        self.ob_double[33:45] = (self.gv[-12:] - self.extra_info["joint_history"]["velocity"][-1]) / self.control_dt

        # scale the observation
        self.ob_scaled = np.asarray((self.ob_double - self.ob_mean) / self.ob_std, dtype=np.float)

        return self.ob_scaled

    def observe(self):
        return self.ob_scaled

    def update_reward(self):
        # TO BE REALIZED!!!!!!!!!!!!!!!!!!

        # self.torque_reward = self.torque_reward_coeff * np.linalg.norm(self.robot.get_generalized_forces())**2
        # self.forward_vel_reward = self.forward_vel_reward_coeff * self.body_linear_vel[0]
        # self.total_reward = self.torque_reward + self.forward_vel_reward
        #
        # if self.done:
        #     self.total_reward += self.terminal_reward_coeff

        return 0.

    def update_extra_info(self, action):
        self.extra_info["last_action"] = np.array(action)
        self.extra_info["joint_history"]["time"] = np.append(self.extra_info["joint_history"]["time"],
                                                             self.world.get_world_time())
        self.extra_info["joint_history"]["position"] = np.append(self.extra_info["joint_history"]["position"],
                                                                 self.gc[-12:])
        self.extra_info["joint_history"]["velocity"] = np.append(self.extra_info["joint_history"]["velocity"],
                                                                 self.gv[-12:])
        self.extra_info["joint_history"]["position_error"] = np.append(self.extra_info["joint_history"]["position_error"],
                                                                       self.p_target12 - self.gc[-12:])
        while self.extra_info["joint_history"]["time"][-1] - self.extra_info["joint_history"]["time"][0] > self.history_length: # Pop queue
            self.extra_info["joint_history"]["time"] = self.extra_info["joint_history"]["time"][1:]
            self.extra_info["joint_history"]["position"] = self.extra_info["joint_history"]["position"][1:]
            self.extra_info["joint_history"]["velocity"] = self.extra_info["joint_history"]["velocity"][1:]
            self.extra_info["joint_history"]["position_error"] = self.extra_info["joint_history"]["position_error"][1:]

        return self.extra_info

    def is_terminal_state(self):
        # if the body is flipped over and the joints are at the right place, the episode is over
        ##### TO BE REALIZED!!!!! ############
        self.done = False

        return self.done

    def curriculum_update(self):
        pass

    def set_seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def close(self):
        if not self.already_closed:
            if self.visualizable:
                vis = raisim.OgreVis.get()
                vis.close_app()
            self.already_closed = True

    def __del__(self):
        self.close()


# Quick test
if __name__ == '__main__':
    from itertools import count
    from my_utilities import HandTuningTrajectory_raisim
    config = {"environment": dict()}
    config['environment']['curriculumFactor'] = 0.3
    config['environment']['baseOrientationCoeff'] = 6.
    config['environment']['jointPositionCoeffHAA'] = 6.
    config['environment']['jointPositionCoeffHFE'] = 7.
    config['environment']['jointPositionCoeffKFE'] = 7.
    config['environment']['jointVelocityCoeff'] = 0.2
    config['environment']['jointAccelerationCoeff'] = 5e-7
    config['environment']['jointTorqueCoeff'] = 0.0005
    config['environment']['contactSlipCoeff'] = 6.
    config['environment']['bodyContactImpulseCoeff'] = 6.
    config['environment']['internalContactCoeff'] = 6.
    config['environment']['smoothnessCoeff'] = 0.0025

    env = JueyingEnv(config=config, visualizable=True)

    env.reset()
    t = 0.
    while True:
        while t <= 3.0:
            action = HandTuningTrajectory_raisim(t)
            obs, reward, done, info = env.step(action)
            print("reward: {}".format(reward))
            t += env.control_dt
        env.reset()
        t = 0.

    # for t in count():
    #     action = np.random.normal(0.0, 0.01, size=12)
    #     obs, reward, done, info = env.step(action)
    #     print("reward: {}".format(reward))
    #
    #     # if t == 200:
    #     #     print("End")
    #     #     break
