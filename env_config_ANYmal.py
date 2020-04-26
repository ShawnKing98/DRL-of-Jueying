import sys
import gym
from gym import spaces
import pybullet as p
from os import path as osp
import numpy as np
from enum import IntEnum, unique
from my_utilities import logisticKernal


@unique
class Joints(IntEnum):
    LF_HAA = 1
    LF_HFE = 2
    LF_KFE = 3
    RF_HAA = 7
    RF_HFE = 8
    RF_KFE = 9
    LH_HAA = 13
    LH_HFE = 14
    LH_KFE = 15
    RH_HAA = 19
    RH_HFE = 20
    RH_KFE = 21

SIMULATIONFREQUENCY = 500
PGAIN_ORIGIN = 65
DGAIN_ORIGIN = 0.3
PGAIN_NOISE = 0
DGAIN_NOISE = 0
DAMPING = 2.0
MAXTORQUE = 25.0

DEFAULT_JOINT_POSITIONS = [0.1, 0.7, -1.2,
                           -0.1, 0.7, -1.2,
                           0.1, -0.7, 1.2,
                           -0.1, -0.7, 1.2]

FALL_JOINT_POSITIONS = [- 0.2, 0.7, -1.2,
                        0.2, 0.7, -1.2,
                        - 0.2, -0.7, 1.2,
                        0.2, -0.7, 1.2]
FALL_BASE_ORIENTATION = np.array([1.6, 0.0, 0.0, 1.0])
FALL_BASE_ORIENTATION /= np.linalg.norm(FALL_BASE_ORIENTATION)
FALL_BASE_POSITION = [0.0, 0.0, 0.22]

REST_JOINT_POSITIONS = [-0.05, 1.45, -2.65,
                        0.05, 1.45, -2.65,
                        -0.05, -1.45, 2.65,
                        0.05, -1.45, 2.65]

EAGLE_JOINT_POSITIONS = [0.0, -1.6, 0.0,
                         0.0, -1.6, 0.0,
                         0.0, 1.6, 0.0,
                         0.0, 1.6, 0.0]


class Anymal(gym.Env):
    def __init__(self, connection="GUI"):
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
            "/home/shawn/Documents/ANYmal/anymal_bedi_urdf/anymal.urdf",
            [0.0, 0.0, 1.0]
        )

        self.action_space = spaces.Box(
            low=-2 * np.pi * np.ones(12),
            high=2 * np.pi * np.ones(12)
        )

        self.actionTime = 3.0
        self.t = 0.0
        self.epoch = 0  # The number of DRL epochs that have been done

        maxPosition = np.pi * np.ones(12)
        maxVelocity = 2 * np.ones(12)
        maxBasePositionAndOrientation = np.ones(7)
        maxBaseVelocity = 1 * np.ones(3)
        maxBaseAngularVelocity = 2 * np.pi * np.ones(3)
        maxAcceleration = np.inf * np.ones(12)
        observationUpperBound = np.concatenate([
            maxPosition,
            maxVelocity,
            maxBasePositionAndOrientation,
            maxBaseVelocity,
            maxBaseAngularVelocity,
            maxAcceleration
        ])
        self.observation_space = spaces.Box(
            low=-observationUpperBound,
            high=observationUpperBound
        )

        p.setJointMotorControlArray(
            self.anymal,
            [joint.value for joint in Joints],
            p.POSITION_CONTROL,
            targetPositions=np.zeros(12),
            forces=np.zeros(12)
        )

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.anymal,
            FALL_BASE_POSITION,
            FALL_BASE_ORIENTATION
        )
        self.t = 0.0

        jointNum = 0
        for joint in Joints:
            positionTarget = FALL_JOINT_POSITIONS[jointNum]
            p.resetJointState(self.anymal, joint.value, positionTarget, 0.0)
            jointNum += 1
        for _ in range(100):
            self.step(FALL_JOINT_POSITIONS)
            # time.sleep(10.0 / SIMULATIONFREQUENCY)  # To observe
        observation, _ = self._getObservation()
        self.t = 0.0
        return observation

    def step(self, action, addNoise=False):
        _, measurement = self._getObservation()
        if addNoise:
            global lastTime, torqueNoise
            if self.t == 0.0:
                lastTime = 0
                torqueNoise = np.random.uniform(-6,6)
            if self.t - lastTime >= 0.1:
                lastTime += 0.1
                torqueNoise = np.random.uniform(-6,6)
            PGAIN = PGAIN_ORIGIN + PGAIN_NOISE
            DGAIN = DGAIN_ORIGIN + DGAIN_NOISE
        else:
            torqueNoise = 0.0
            PGAIN = PGAIN_ORIGIN
            DGAIN = DGAIN_ORIGIN

        PD_torque = PGAIN * (action - measurement["position"])
        PD_torque -= DGAIN * measurement["velocity"]
        PD_torque = np.clip(PD_torque, -MAXTORQUE, MAXTORQUE)
        PD_torque -= DAMPING * measurement["velocity"]

        p.setJointMotorControlArray(
            self.anymal,
            [j.value for j in Joints],
            p.TORQUE_CONTROL,
            forces=PD_torque + torqueNoise
        )
        p.stepSimulation()

        observation, observationAsDict = self._getObservation()

        reward = -self.calculateCost(PD_torque)

        self.t += 1.0 / SIMULATIONFREQUENCY
        if (self.t > self.actionTime) and np.linalg.norm(observationAsDict['baseAngularVelocity']) < 5e-4:
            done = True
        else:
            done = False
        return observation, reward, done, observationAsDict

    def render(self, mode="rgb"):
        print("Hello!!!!!")
        pass

    def close(self):
        p.disconnect()

    def _getObservation(self):
        allJoints = [j.value for j in Joints]
        jointStates = p.getJointStates(self.anymal, allJoints)
        position = np.array([js[0] for js in jointStates])
        velocity = np.array([js[1] for js in jointStates])
        global lastVelocity
        if self.t == 0.0:
            lastVelocity = velocity
        acceleration = (velocity - lastVelocity) * SIMULATIONFREQUENCY
        lastVelocity = velocity
        basePosition, baseOrientation = p.getBasePositionAndOrientation(self.anymal)
        baseVelocity, baseAngularVelocity = p.getBaseVelocity(self.anymal)
        observationAsArray = np.concatenate([
            position,
            velocity,
            basePosition,
            baseOrientation,
            baseVelocity,
            baseAngularVelocity,
            acceleration
        ])
        observationAsDict = {
            "position": position,
            "velocity": velocity,
            "basePosition": basePosition,
            "baseOrientation": baseOrientation,
            "baseVelocity": baseVelocity,
            "baseAngularVelocity": baseAngularVelocity,
            "acceleration": acceleration
        }
        return observationAsArray, observationAsDict

    def calculateCost(self, PD_torque):
        observation, observationAsDict = self._getObservation()

        # Cost function below

        kc = 0.3 ** (0.997 ** self.epoch)
        # global kc  # curriculum factor
        # global lastEpoch
        # # prevEpoch = 800  # The number of epoches before this training, I use this to continue a half-completed training
        # if epoch == 0:
        #     kc = 0.3
        #     lastEpoch = 0
        # if epoch != lastEpoch:
        #     kc = kc ** 0.997
        #     lastIteration = epoch
        # Base orientation cost
        c_o = 6 / SIMULATIONFREQUENCY
        baseOrientation = np.array(p.getEulerFromQuaternion(observationAsDict['baseOrientation']))
        baseOrientation /= np.linalg.norm(baseOrientation)
        baseOrientationCost = c_o * (np.linalg.norm([0, 0, -1] - baseOrientation)) ** 2
        # Joint position cost
        c_HAA = 6 / SIMULATIONFREQUENCY
        c_HFE = 7 / SIMULATIONFREQUENCY
        c_KFE = 7 / SIMULATIONFREQUENCY
        if baseOrientation.dot([0, 0, -1]) < np.cos(0.25 * np.pi):  # which means it hasn't recovered
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
        # Smoothness cost
        c_s = 0.0025 / SIMULATIONFREQUENCY
        global last_PD_torque
        if self.t == 0.0:
            last_PD_torque = 0.0
        smoothnessCost = kc * c_s * (np.linalg.norm(last_PD_torque - PD_torque)) ** 2
        last_PD_torque = PD_torque

        return baseOrientationCost + jointPositionCost + jointVelocityCost + jointAccelerationCost + torqueCost + smoothnessCost

    def setEpoch(self, epoch):
        self.epoch = epoch


if __name__ == "__main__":
    env = Anymal("GUI")
    env.reset()
    input("Press any key to quit.\n")
