# The CMAES method to implement fall recovery trajectory
# 0.2 second per frame in 3-second trajectory. 15*12=180 parameters in total
import sys
import gym
from gym import spaces
import pybullet as p
import numpy as np
from os import path
from enum import IntEnum, unique
import cma  # library for CMAES


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


SIMULATIONFREQUENCY = 500
PGAIN_ORIGIN = 65
DGAIN_ORIGIN = 0.3
PGAIN_NOISE = 0
DGAIN_NOISE = 0
DAMPING = 2.0
MAXTORQUE = 25.0

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
FALL_BASE_POSITION = [0.0, 0.0, 0.22]

REST_JOINT_POSITIONS = [-0.05, 1.45, -2.65,
                        0.05, 1.45, -2.65,
                        -0.05, 1.45, 2.65,
                        0.05, 1.45, 2.65]

EAGLE_JOINT_POSITIONS = [0.0, -1.6, 0.0,
                         0.0, -1.6, 0.0,
                         0.0, -1.6, 0.0,
                         0.0, -1.6, 0.0]


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
            "/home/shawn/Documents/DRL/JueyingProURDF/urdf/jueying.urdf",
            [0.0, 0.0, 1.0]
        )

        self.action_space = spaces.Box(
            low=-2 * np.pi * np.ones(12),
            high=2 * np.pi * np.ones(12)
        )

        maxPosition = np.pi * np.ones(12)
        maxVelocity = 2 * np.ones(12)
        maxBasePositionAndOrientation = np.ones(7)
        maxBaseVelocity = 1 * np.ones(3)
        maxBaseAngularVelocity = 2 * np.pi * np.ones(3)
        observationUpperBound = np.concatenate([
            maxPosition,
            maxVelocity,
            maxBasePositionAndOrientation,
            maxBaseVelocity,
            maxBaseAngularVelocity
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
        self.t = 0.0

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
        if (self.t > actionTime) and np.linalg.norm(observationAsDict['baseAngularVelocity']) < 5e-4:
            done = True
        else:
            done = False
        return observation, reward, done, observationAsDict

    def render(self, mode="rgb"):
        pass

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
        global kc  # curriculum factor
        global lastIteration
        prevIteration = 100  # The iteration before this training
        if iteration == 0:
            kc = 0.3 ** (0.997 ** prevIteration)
            lastIteration = 0
        if iteration != lastIteration:
            kc = kc ** 0.997
            lastIteration = iteration
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

    def close(self):
        p.disconnect()


def HandTuningTrajectory(t):
    #Time
    timePoint = np.array([0, 2, 2.1, 2.2, 2.3])

    # LF_HAA
    timePoint_LF_HAA = np.array([0, 1, 1.1, 1.2, 2, 2.1, 3])
    LF_HAA_Point = np.array([FALL_JOINT_POSITIONS[0], 1.5, -1, -1, 0, -1, 0])
    LF_HAA_Target = np.interp(t, timePoint_LF_HAA, LF_HAA_Point)

    # LF_HFE
    LF_HFE_Point = np.array([FALL_JOINT_POSITIONS[1], 0, -1.57, -1.57, -1.6])
    LF_HFE_Target = np.interp(t, timePoint, LF_HFE_Point)

    # LF_KFE
    LF_KFE_Point = np.array([FALL_JOINT_POSITIONS[2], -2, -1.57, 0, 0])
    LF_KFE_Target = np.interp(t, timePoint, LF_KFE_Point)

    # RF_HAA
    timePoint_RF_HAA = np.append([0, 1], timePoint[1:])
    RF_HAA_Point = np.array([FALL_JOINT_POSITIONS[3], 0, -1.5, 0, 0, 0])
    RF_HAA_Target = np.interp(t, timePoint_RF_HAA, RF_HAA_Point)

    # RF_HFE
    RF_HFE_Point = np.array([FALL_JOINT_POSITIONS[4], -2.0, -2.0, -2.0, -1.6])
    RF_HFE_Target = np.interp(t, timePoint, RF_HFE_Point)

    # RF_KFE
    timePoint_RF_KFE = np.append(timePoint, [3])
    RF_KFE_Point = np.array([FALL_JOINT_POSITIONS[5], -0.2, -1, -1.3, -1.3, 0])
    RF_KFE_Target = np.interp(t, timePoint_RF_KFE, RF_KFE_Point)

    # LH_HAA
    LH_HAA_Target = LF_HAA_Target
    # LH_HFE
    LH_HFE_Target = LF_HFE_Target
    # LH_KFE
    LH_KFE_Target = -LF_KFE_Target

    # RH_HAA
    timePoint_RH_HAA = np.array([0, 0.5, 1, 2, 2.1, 2.2, 2.3])
    RH_HAA_Point = np.array([FALL_JOINT_POSITIONS[9], 0, -1.5, -1.5, 0, 0, 0])
    RH_HAA_Target = np.interp(t, timePoint_RH_HAA, RH_HAA_Point)

    # RH_HFE
    timePoint_RH_HFE = np.array([0, 0.5, 3])
    RH_HFE_Point = np.array([FALL_JOINT_POSITIONS[10], -1.8, -1.6])
    RH_HFE_Target = np.interp(t, timePoint_RH_HFE, RH_HFE_Point)

    # RH_KFE
    timePoint_RH_KFE = np.array([0, 0.5, 1, 2, 2.1, 3])
    RH_KFE_Point = np.array([FALL_JOINT_POSITIONS[11], 0.2, 1, 1, 1.3, 0])
    RH_KFE_Target = np.interp(t, timePoint_RH_KFE, RH_KFE_Point)

    jointTarget = np.array([LF_HAA_Target, LF_HFE_Target, LF_KFE_Target, RF_HAA_Target, RF_HFE_Target, RF_KFE_Target, LH_HAA_Target, LH_HFE_Target, LH_KFE_Target, RH_HAA_Target, RH_HFE_Target, RH_KFE_Target])

    return jointTarget


def jointTargetsAsFunctionOfTime(t):

    timePoint = np.append(0, timeLine)
    tmpTrajectory = trajectory.reshape(len(timeLine), 12).T
    jointTarget = [np.interp(t, timePoint, np.append(FALL_JOINT_POSITIONS[i], tmpTrajectory[i])) for i in range(12)]

    return jointTarget


# Use this kernel function to converts a tracking error to a bounded reward, reference from ETH paper.
def logisticKernal(x):
    return -1/(np.exp(x)+2+np.exp(-x))


if __name__ == "__main__":
    import time
    actionTime = 3
    timeLine = [i*0.2 for i in range(1, int(actionTime/0.2+1))]
    xrecentbest = open('/home/shawn/Documents/DRL/outcmaes_backup/2020.4.21/xrecentbest.dat', 'r')
    xrecentbestList = xrecentbest.readlines()
    tmpList = xrecentbestList.copy()
    for i in range(1, len(tmpList)):
        xrecentbestList[i - 1] = [eval(x) for x in tmpList[i].split()]
    xrecentbestList.pop(-1)
    costList = [x[4] for x in xrecentbestList]
    xbest_index = costList.index(min(costList))
    xbest = xrecentbestList[xbest_index]
    xbest = xbest[5:]
    trajectory = np.array(xbest)
    # initSigma = xrecentbestList[xbest_index][2]
    # trajectory = np.concatenate([HandTuningTrajectory(i) for i in timeLine])
    initSigma = 0.1
    es = cma.CMAEvolutionStrategy(trajectory, initSigma, {'bounds': [-np.pi*np.ones(12), np.pi*np.ones(12)]})
    logger = cma.CMADataLogger().register(es)
    iteration = 0
    totalReward = 0
    env = Anymal("GUI")
    observation = env.reset()

    stupid = input("ATTENTION!!\nHave you backed up last trajectory?\n"
                   "If not, press N, otherwise press any other key to start\n")
    if stupid == "N":
        sys.exit()
    # print("Iterat #Fevals   function value    axis ratio  sigma  minstd maxstd min:sec\n")
    for iteration in range(100):
        solutions = es.ask()
        costs = []
        for s in solutions:
            PGAIN_NOISE = np.random.normal(0,5)
            DGAIN_NOISE = np.random.normal(0,0.05)
            trajectory = s
            for n in range(2 * 3 * SIMULATIONFREQUENCY):
                action = jointTargetsAsFunctionOfTime(env.t)
                observation, reward, done, measurement = env.step(action, addNoise=True)
                if done:
                    break
                totalReward += reward
                # time.sleep(1.0 / SIMULATIONFREQUENCY)  # Normal speed is 1.0, here's to convenient observe
            costs += [-totalReward]
            observation = env.reset()
            totalReward = 0
        es.tell(solutions, costs)
        es.logger.add()
        if iteration%5 == 0:
            es.disp()
    es.result_pretty()
    logger.save()
    # logger.plot()
    input("Press any key to quit, remember to backup trajectory.\n")
    env.close()
