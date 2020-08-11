import sys
import gym
from gym import spaces
import pybullet as p
from os import path as osp
import numpy as np
from enum import IntEnum, unique
from my_utilities import logisticKernal, time_interp


@unique
class Joints(IntEnum):
    LF_HAA = 3
    LF_HFE = 4
    LF_KFE = 5
    RF_HAA = 7
    RF_HFE = 8
    RF_KFE = 9
    LH_HAA = 11
    LH_HFE = 12
    LH_KFE = 13
    RH_HAA = 15
    RH_HFE = 16
    RH_KFE = 17

@unique
class FootLinks(IntEnum):
    LF_FOOT = 7
    RF_FOOT = 11
    LH_FOOT =15
    RH_FOOT = 19

SIMULATIONFREQUENCY = 500
PGAIN_ORIGIN = np.array(4 * [200, 200, 300])
DGAIN_ORIGIN = np.array(4 * [2, 2, 3])
PGAIN_NOISE = 0
DGAIN_NOISE = 0
DAMPING = 2.0
MAXTORQUE = 30.0

DEFAULT_JOINT_POSITIONS = [0.1, 0.7, -1.2,
                           -0.1, 0.7, -1.2,
                           0.1, 0.7, 1.2,
                           -0.1, 0.7, 1.2]

FALL_JOINT_POSITIONS = [- 0.2, 0.7, -1.2,
                        0.2, 0.7, -1.2,
                        - 0.2, 0.7, 1.2,
                        0.2, 0.7, 1.2]
FALL_BASE_ORIENTATION = np.array([1.6, 0.0, 0.0, 1.0])
FALL_BASE_ORIENTATION /= np.linalg.norm(FALL_BASE_ORIENTATION)
# FALL_BASE_POSITION = [0.0, 0.0, 0.22]
FALL_BASE_POSITION = [0.0, 0.0, 1.0]

REST_JOINT_POSITIONS = [-0.05, 1.45, -2.65,
                        0.05, 1.45, -2.65,
                        -0.05, 1.45, 2.65,
                        0.05, 1.45, 2.65]

EAGLE_JOINT_POSITIONS = [0.0, -1.6, 0.0,
                         0.0, -1.6, 0.0,
                         0.0, -1.6, 0.0,
                         0.0, -1.6, 0.0]


class Anymal(gym.Env):
    def __init__(self, connection="GUI", prev_epoch=0, seed=None):
        np.random.seed(seed)
        if connection == "GUI":
            self.pybullet = p.connect(p.GUI)
        else:
            self.pybullet = p.connect(p.DIRECT)

        p.setTimeStep(1.0 / SIMULATIONFREQUENCY)
        p.setGravity(0, 0, -9.81)
        self.ground = p.loadURDF(
            "/home/shawn/Documents/ANYmal/assets/plane.urdf",
            [0, 0, 0]
        )
        self.anymal = p.loadURDF(
            "/home/shawn/Documents/DRL/JueyingProURDF/urdf/jueying.urdf",
            [0.0, 0.0, 1.0]
        )

        self.action_space = spaces.Box(
            low=-2 * np.pi * np.ones(12),
            high=2 * np.pi * np.ones(12)
        )

        self.actionTime = 3.0
        self.t = 0.0
        self.epoch = prev_epoch  # The number of DRL epochs that have been done
        self.lastTime = 0.0 # Used to add noise
        self.torqueNoise = 0.0
        self.positionNoise = 0.0

        maxPosition = np.pi * np.ones(12)
        maxVelocity = 2 * np.ones(12)
        maxPositionHistory = np.pi * np.ones(24)
        maxVelocityHistory = 2 * np.ones(24)
        maxBasePositionAndOrientation = np.ones(7)
        maxBasePosition = np.ones(3)
        maxBaseOrientationVector = np.ones(3)
        maxBaseVelocity = 1 * np.ones(3)
        maxBaseAngularVelocity = 2 * np.pi * np.ones(3)
        maxAcceleration = np.inf * np.ones(12)
        observationUpperBound = np.concatenate([
            maxPosition,
            maxVelocity,
            maxBasePositionAndOrientation,
            maxBaseOrientationVector,
            maxBaseVelocity,
            maxBaseAngularVelocity,
            maxAcceleration
        ])
        stateUpperBound = np.concatenate([
            maxBaseOrientationVector,
            maxBaseVelocity,
            maxBaseAngularVelocity,
            maxPosition,
            maxVelocity,
            maxPositionHistory,
            maxVelocityHistory,
            maxPosition
        ])
        # self.observation_space = spaces.Box(
        #     low=-observationUpperBound,
        #     high=observationUpperBound
        # )
        self.observation_space = spaces.Box(
            low=-stateUpperBound,
            high=stateUpperBound
        )

        self.history_buffer = dict()  # used to storage the joint state history and the last action
        self.history_buffer['last_action'] = np.nan * np.ones(12)
        self.history_buffer['joint_state'] = {'time': [], 'position': [], 'velocity': [], 'position_error': []}
        self.buffer_time_length = 0.03

        p.setJointMotorControlArray(
            self.anymal,
            [joint.value for joint in Joints],
            p.POSITION_CONTROL,
            targetPositions=np.zeros(12),
            forces=np.zeros(12)
        )

    def reset(self):
        # keep randomly resetting the base until it's not upside-down
        while True:
            FALL_BASE_ORIENTATION_tmp = FALL_BASE_ORIENTATION + np.random.uniform(-0.5, 0.5, 4)
            # FALL_BASE_ORIENTATION_tmp = np.random.uniform(-1, 1, 4)
            FALL_BASE_ORIENTATION_tmp /= np.linalg.norm(FALL_BASE_ORIENTATION_tmp)
            p.resetBasePositionAndOrientation(
                self.anymal,
                FALL_BASE_POSITION,
                FALL_BASE_ORIENTATION_tmp
            )
            observation, observation_dict = self._getObservation()
            if observation_dict['baseOrientationVector'].dot([0, 0, -1]) > 0:   # Means the robot isn't upside-down
                break

        self.t = 0.0
        self.lastTime = 0.0

        # randomly reset the joints
        jointNum = 0
        for joint in Joints:
            FALL_JOINT_POSITIONS_tmp = FALL_JOINT_POSITIONS + np.random.uniform(-0.7, 0.7, 12)
            positionTarget = FALL_JOINT_POSITIONS_tmp[jointNum]
            p.resetJointState(self.anymal, joint.value, positionTarget, 0.0)
            jointNum += 1

        # run shortly to make the robot look more naturally
        for _ in range(600):
            p.stepSimulation()
            # self.step(FALL_JOINT_POSITIONS)
            # time.sleep(10.0 / SIMULATIONFREQUENCY)  # To observe
        allJoints = [j.value for j in Joints]
        jointStates = p.getJointStates(self.anymal, allJoints)
        position = np.array([js[0] for js in jointStates])
        velocity = np.array([js[1] for js in jointStates])

        # reset history buffer
        self.t = 0.0
        self.history_buffer['last_action'] = position
        self.history_buffer['joint_state']['time'] = [0.0]
        self.history_buffer['joint_state']['position'] = [position]
        self.history_buffer['joint_state']['velocity'] = [velocity]
        self.history_buffer['joint_state']['position_error'] = [np.zeros(12)]

        state = self.getState()
        return state

    def step(self, action, addNoise=False):
        _, measurement = self._getObservation()
        if addNoise:
            if self.t - self.lastTime >= 0.2:
                self.lastTime += 0.2
                # self.torqueNoise = np.random.uniform(-6,6)
                self.positionNoise = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, 12)
            PGAIN = PGAIN_ORIGIN + PGAIN_NOISE
            DGAIN = DGAIN_ORIGIN + DGAIN_NOISE
        else:
            # self.torqueNoise = 0.0
            self.positionNoise = 0.0
            PGAIN = PGAIN_ORIGIN
            DGAIN = DGAIN_ORIGIN

        PD_torque = PGAIN * (np.array(action) + self.positionNoise - measurement["position"])
        PD_torque -= DGAIN * measurement["velocity"]
        PD_torque = np.clip(PD_torque, -MAXTORQUE, MAXTORQUE)
        PD_torque -= DAMPING * measurement["velocity"]

        p.setJointMotorControlArray(
            self.anymal,
            [j.value for j in Joints],
            p.TORQUE_CONTROL,
            forces=PD_torque + self.torqueNoise
        )
        p.stepSimulation()

        observation, observationAsDict = self._getObservation()
        observationAsDict['torque'] = PD_torque + self.torqueNoise
        reward = -self.calculateCost(PD_torque=PD_torque)

        self.t += 1.0 / SIMULATIONFREQUENCY
        # self.history_buffer['last_action'] = action
        self.history_buffer['joint_state']['time'].append(self.t)
        self.history_buffer['joint_state']['position'].append(observationAsDict['position'])
        self.history_buffer['joint_state']['velocity'].append(observationAsDict['velocity'])
        self.history_buffer['joint_state']['position_error'].append(observationAsDict['position'] - action)
        while self.history_buffer['joint_state']['time'][-1] - self.history_buffer['joint_state']['time'][0] > self.buffer_time_length: # Pop queue
            self.history_buffer['joint_state']['time'] = self.history_buffer['joint_state']['time'][1:]
            self.history_buffer['joint_state']['position'] = self.history_buffer['joint_state']['position'][1:]
            self.history_buffer['joint_state']['velocity'] = self.history_buffer['joint_state']['velocity'][1:]
            self.history_buffer['joint_state']['position_error'] = self.history_buffer['joint_state']['position_error'][1:]
        state = self.getState()

        # if (self.t > self.actionTime) and np.linalg.norm(observationAsDict['baseAngularVelocity']) < 5e-4:
        # if observationAsDict['baseOrientationVector'].dot([0,0,-1]) > np.cos(0.125*np.pi) and np.linalg.norm(observationAsDict['baseAngularVelocity']) < 5e-4:

        if observationAsDict['baseOrientationVector'].dot([0, 0, -1]) > np.cos(0.125 * np.pi)\
                and np.alltrue([observationAsDict['position'][i] < -1 for i in [1, 4, 7, 10]])\
                and self.t > 1.5:
            done = True
        else:
            done = False
        # done = False
        return state, reward, done, observationAsDict


    def render(self, mode="rgb"):
        # print("rendering now")
        pass

    def close(self):
        p.disconnect()

    def _getObservation(self):
        allJoints = [j.value for j in Joints]
        jointStates = p.getJointStates(self.anymal, allJoints)
        position = np.array([js[0] for js in jointStates])
        velocity = np.array([js[1] for js in jointStates])
        # global lastVelocity
        # if self.t == 0.0:
        #     lastVelocity = velocity
        if len(self.history_buffer['joint_state']['velocity']) == 0:
            acceleration = np.zeros(12)
        else:
            lastVelocity = self.history_buffer['joint_state']['velocity'][-1]
            acceleration = (velocity - lastVelocity) * SIMULATIONFREQUENCY
        # lastVelocity = velocity
        basePosition, baseOrientation = p.getBasePositionAndOrientation(self.anymal)
        baseOrientation_Matrix = np.array(p.getMatrixFromQuaternion(baseOrientation)).reshape(3, 3)
        baseOrientationVector = np.matmul(baseOrientation_Matrix, [0, 0, -1])
        baseVelocity, baseAngularVelocity = p.getBaseVelocity(self.anymal)
        observationAsArray = np.concatenate([
            position,
            velocity,
            basePosition,
            baseOrientation,
            baseOrientationVector,
            baseVelocity,
            baseAngularVelocity,
            acceleration
        ])
        observationAsDict = {
            "position": position,
            "velocity": velocity,
            "basePosition": basePosition,
            "baseOrientation": baseOrientation,
            "baseOrientationVector": baseOrientationVector,
            "baseVelocity": baseVelocity,
            "baseAngularVelocity": baseAngularVelocity,
            "acceleration": acceleration
        }
        return observationAsArray, observationAsDict

    def calculateCost(self, PD_torque):
        observation, observationAsDict = self._getObservation()

        # Curriculum factor, reference from ETH paper
        kc = 0.3 ** (0.995 ** self.epoch)

        # Base orientation cost
        c_o = 6 / SIMULATIONFREQUENCY
        baseOrientationVector = observationAsDict['baseOrientationVector']
        baseOrientationCost = c_o * (np.linalg.norm([0, 0, -1] - baseOrientationVector)) ** 2

        # Joint position cost
        c_HAA = 6 / SIMULATIONFREQUENCY
        c_HFE = 7 / SIMULATIONFREQUENCY
        c_KFE = 7 / SIMULATIONFREQUENCY
        if baseOrientationVector.dot([0, 0, -1]) < np.cos(0.25 * np.pi):  # which means it hasn't recovered
            jointPositionCost = 0
        else:
            jointPositionCost = 0
            for i in [0, 3, 6, 9]:  # HAA cost
                jointPositionCost += kc * c_HAA * logisticKernal(
                    EAGLE_JOINT_POSITIONS[i] - observationAsDict['position'][i])
            for i in [1, 4, 7, 10]:  # HFE cost
                jointPositionCost += kc * c_HFE * logisticKernal(
                    EAGLE_JOINT_POSITIONS[i] - observationAsDict['position'][i])
            for i in [2, 5, 8, 11]:  # KFE cost
                jointPositionCost += kc * c_KFE * logisticKernal(
                    EAGLE_JOINT_POSITIONS[i] - observationAsDict['position'][i])

        # Joint velocity cost
        c_jv = 0.2 / SIMULATIONFREQUENCY
        c_jvmax = 8
        if np.linalg.norm(observationAsDict['velocity'], 1) < c_jvmax:
            jointVelocityCost = 0
        else:
            jointVelocityCost = kc * c_jv * (np.linalg.norm(observationAsDict['velocity'])) ** 2

        # Joint acceleration cost
        c_ja = 5e-7 / SIMULATIONFREQUENCY
        jointAccelerationCost = kc * c_ja * (np.linalg.norm(observationAsDict['acceleration'])) ** 2

        # Torque cost
        c_t = 0.0005 / SIMULATIONFREQUENCY
        torqueCost = kc * c_t * (np.linalg.norm(PD_torque) ** 2)

        # Contact slip cost
        c_cv = 6 / SIMULATIONFREQUENCY
        contactPoints = p.getContactPoints(self.anymal)
        contactVelocity = []
        for point in contactPoints:
            body_A = point[1]
            body_B = point[2]
            link_A = point[3]
            link_B = point[4]
            if link_A == -1:
                link_A = 0
            if link_B == -1:
                link_B = 0
            if body_A == 0:
                v_A = np.zeros(3)
            else:
                v_A = np.array(p.getLinkState(body_A, link_A, 1)[6])
                # except:
                #     print('error!!!')
                #     print('Body_A:', body_A)
                #     print('Link_A:', link_A)
            if body_B == 0:
                v_B = np.zeros(3)
            else:
                v_B = p.getLinkState(body_B, link_B, 1)[6]
            v_rel = v_A - v_B
            contactVelocity.append(np.linalg.norm(v_rel) ** 2)
        if len(contactVelocity) == 0:
            averageContactVelocity = 0
        else:
            averageContactVelocity = np.mean(contactVelocity)
        contactSlipCost = kc * c_cv * averageContactVelocity

        # Body contact impulse cost
        c_cimp = 6 / SIMULATIONFREQUENCY
        contactImpulse = []
        for point in contactPoints:
            if point[2] == 0 and point[3] in [link.value for link in FootLinks]:  # Contact between feet and ground
                continue
            contactImpulse.append(np.linalg.norm([point[9]], 1))
        if len(contactImpulse) == 0:
            averageContactImpulse = 0
        else:
            averageContactImpulse = np.mean(contactImpulse)
        contactImpulseCost = kc * c_cimp * averageContactImpulse

        # Internal contact cost
        c_cint = 6 / SIMULATIONFREQUENCY
        num_int = 0
        for point in contactPoints:
            if point[1] == self.anymal and point[2] == self.anymal:
                num_int += 1
        internalContactCost = kc * c_cint * num_int

        # Smoothness cost
        c_s = 0.0025 / SIMULATIONFREQUENCY
        smoothnessCost = kc * c_s * (np.linalg.norm(self.history_buffer['last_action'] - PD_torque)) ** 2

        if 2.50 < self.t < 2.50:
            print("time:", self.t)
            print("baseOrientationCost:", baseOrientationCost)
            print("jointPositionCost:", jointPositionCost)
            print("jointVelocityCost:", jointVelocityCost)
            print("jointAccelerationCost:", jointAccelerationCost)
            print("contactSlipCost:", contactSlipCost)
            print("contactImpulseCost:", contactImpulseCost)
            print("internalContactCost:", internalContactCost)
            print("torqueCost:", torqueCost)
            print("smoothnessCost:", smoothnessCost)
        return baseOrientationCost + jointPositionCost + jointVelocityCost + jointAccelerationCost + contactSlipCost +\
               contactImpulseCost + internalContactCost + torqueCost + smoothnessCost

    def setEpoch(self, epoch):
        self.epoch = epoch

    def addEpoch(self):
        self.epoch = self.epoch + 1

    def getState(self):
        o_dict = self._getObservation()[1]
        base_orientation = o_dict['baseOrientationVector']
        base_v = o_dict['baseVelocity']
        base_w = o_dict['baseAngularVelocity']
        joint_position = o_dict['position']
        joint_velocity = o_dict['velocity']
        joint_position_history_1 = time_interp(self.t - 0.01,
                                               self.history_buffer['joint_state']['time'],
                                               np.array(self.history_buffer['joint_state']['position']))
        joint_position_history_2 = time_interp(self.t - 0.02,
                                               self.history_buffer['joint_state']['time'],
                                               np.array(self.history_buffer['joint_state']['position']))
        joint_position_history = np.append(joint_position_history_1, joint_position_history_2)
        joint_velocity_history_1 = time_interp(self.t - 0.01,
                                               self.history_buffer['joint_state']['time'],
                                               np.array(self.history_buffer['joint_state']['velocity']))
        joint_velocity_history_2 = time_interp(self.t - 0.02,
                                               self.history_buffer['joint_state']['time'],
                                               np.array(self.history_buffer['joint_state']['velocity']))
        joint_velocity_history = np.append(joint_velocity_history_1, joint_velocity_history_2)
        joint_position_error_history_1 = time_interp(self.t - 0.01,
                                               self.history_buffer['joint_state']['time'],
                                               np.array(self.history_buffer['joint_state']['position_error']))
        joint_position_error_history_2 = time_interp(self.t - 0.02,
                                               self.history_buffer['joint_state']['time'],
                                               np.array(self.history_buffer['joint_state']['position_error']))
        joint_position_error_history = np.append(joint_position_error_history_1, joint_position_error_history_2)
        last_a = self.history_buffer['last_action']

        # return np.concatenate([base_orientation, base_v, base_w, joint_position, joint_velocity, joint_position_history,
        #                        joint_velocity_history, last_a])
        return np.concatenate([base_orientation, base_v, base_w, joint_position, joint_velocity, joint_position_error_history,
                               joint_velocity_history, last_a])


if __name__ == "__main__":
    env = Anymal("GUI")
    # input("Press any key to reset.\n")

    for i in range(10):
        o = env.reset()
        # for i in range(500):
        #     env.step(np.zeros(12))
            # print(env.t)
        o, o_dict = env._getObservation()
        print(o_dict['position'])
        input("Press any key to quit.\n")
    env.close()
