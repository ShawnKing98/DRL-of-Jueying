import gym
from gym import spaces
import pybullet as p
import numpy as np
from os import path
from enum import IntEnum, unique
import matplotlib.pyplot as plt


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
    LF_FOOT = 6
    RF_FOOT = 12
    LH_FOOT =18
    RH_FOOT = 24

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
# FALL_BASE_POSITION = [0.0, 0.0, 1]

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
            # "/home/shawn/Documents/ANYmal/anymal_bedi_urdf/anymal.urdf",
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
        # FALL_BASE_ORIENTATION_tmp = FALL_BASE_ORIENTATION + np.random.uniform(-0.5, 0.5, 4)
        FALL_BASE_ORIENTATION_tmp = FALL_BASE_ORIENTATION + np.random.uniform(0, 0, 4)
        # FALL_BASE_ORIENTATION_tmp = np.random.uniform(-1, 1, 4)
        FALL_BASE_ORIENTATION_tmp /= np.linalg.norm(FALL_BASE_ORIENTATION_tmp)
        p.resetBasePositionAndOrientation(
            self.anymal,
            FALL_BASE_POSITION,
            FALL_BASE_ORIENTATION_tmp
        )

        jointNum = 0
        for joint in Joints:
            # initPositionNoise = np.random.uniform(-0.7, 0.7)
            initPositionNoise = np.random.uniform(-0, 0)
            positionTarget = FALL_JOINT_POSITIONS[jointNum]
            # positionTarget = EAGLE_JOINT_POSITIONS[jointNum]
            p.resetJointState(self.anymal, joint.value, positionTarget+initPositionNoise, 0.0)
            jointNum += 1
        # for _ in range(100):
        #     self.step(FALL_JOINT_POSITIONS)
            # time.sleep(10.0 / SIMULATIONFREQUENCY)  # To observe
        # for _ in range(600):
        #     p.stepSimulation()
        observation, _ = self._getObservation()
        self.t = 0.0
        # input("hold")
        return observation

    def step(self, action, addNoise=False):
        _, measurement = self._getObservation()
        if addNoise:
            global lastTime, torqueNoise
            if self.t == 0.0:
                lastTime = 0
                torqueNoise = np.random.uniform(-6, 6)
            if self.t - lastTime >= 0.1:
                lastTime += 0.1
                torqueNoise = np.random.uniform(-6, 6)
            PGAIN = PGAIN_ORIGIN + PGAIN_NOISE
            DGAIN = DGAIN_ORIGIN + DGAIN_NOISE
        else:
            torqueNoise = 0.0
            PGAIN = PGAIN_ORIGIN
            DGAIN = DGAIN_ORIGIN

        PD_torque = PGAIN * (action - measurement["position"])
        PD_torque -= DGAIN * measurement["velocity"]
        PD_torque = np.clip(PD_torque, -MAXTORQUE, MAXTORQUE)
        maxTorqueList.append(max(abs(PD_torque)))
        joint_torque = PD_torque - DAMPING * measurement["velocity"]

        p.setJointMotorControlArray(
            self.anymal,
            [j.value for j in Joints],
            p.TORQUE_CONTROL,
            forces=joint_torque + torqueNoise
        )
        p.stepSimulation()

        observation, observationAsDict = self._getObservation()
        observationAsDict['torque'] = joint_torque
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

        # Cost function below
        global kc  # curriculum factor
        global lastIteration
        prevIteration = 1500  # The iteration before this training
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


def jointTargetsAsFunctionOfTime(t, timeLine, trajectory):

    timePoint = np.append(0, timeLine)
    tmpTrajectory = trajectory.reshape(len(timeLine), 12).T
    jointTarget = [np.interp(t, timePoint, np.append(FALL_JOINT_POSITIONS[i], tmpTrajectory[i])) for i in range(12)]

    return jointTarget


def logisticKernal(x):
    return -1/(np.exp(x)+2+np.exp(-x))

def DrawFollowingPlot(timeList, trajectoryTargetList, trajectoryObserveList, legend1, legend2):
    for i in range(12):
        tmpTargetList = [target[i] for target in trajectoryTargetList]
        tmpObserveList = [observe[i] for observe in trajectoryObserveList]
        plt.subplot(4,3,i+1)
        plt.plot(timeList, np.array([tmpTargetList,tmpObserveList]).T)
        plt.legend([legend1, legend2])
    return


if __name__ == "__main__":
    import time

    xrecentbest = open('/home/shawn/Documents/DRL/outcmaes_backup/2020.4.21/xrecentbest.dat', 'r')
    xrecentbestList = xrecentbest.readlines()
    tmpList = xrecentbestList.copy()
    for i in range(1, len(tmpList)):
        xrecentbestList[i-1] = [eval(x) for x in tmpList[i].split()]
    xrecentbestList.pop(-1)
    costList = [x[4] for x in xrecentbestList]
    xbest_index = costList.index(min(costList))
    #xbest_index = -1
    xbest = xrecentbestList[xbest_index]
    xbest = xbest[5:]

    maxTorqueList = []

    actionTime = 3
    timeLine = [i * 0.2 for i in range(1, int(actionTime / 0.2 + 1))]
    trajectory = np.concatenate([HandTuningTrajectory(i) for i in timeLine])
    # trajectory = np.array(xbest)
    iteration = 0
    totalReward = 0
    env = Anymal("GUI")
    env.reset()
    # input("Press any key to start\n")
    np.random.seed(66)
    successCount = 0
    for i in range(100):
        observation = env.reset()
        # input("Press any key to start\n")
        # time.sleep(3)
        trajectoryTargetList = []
        trajectoryObserveList = []
        maxTorqueList1 = []
        maxContactImpulseList = []
        maxVelocityList1 = []
        rewardList = []
        timeList = []
        for n in range(2 * 3 * SIMULATIONFREQUENCY):
            # action = HandTuningTrajectory(env.t)
            action = jointTargetsAsFunctionOfTime(env.t, timeLine, trajectory)
            observation, reward, done, measurement = env.step(action, addNoise=False)
            # if reward > 0:
            #     print(measurement["baseOrientationVector"].dot([0, 0, -1]))
            # Observe
            # Trajectory following
            tttorque1 = np.array(measurement['torque'])
            tttorque1 = np.abs(tttorque1).max()
            maxTorqueList1.append(tttorque1)
            trajectoryTargetList.append(action)
            trajectoryObserveList.append(measurement['position'])
            # Contact impulse
            contactPoints = p.getContactPoints(env.anymal)
            contactImpulse = []
            for point in contactPoints:
                if point[2] == 0 and point[3] in [link.value for link in FootLinks]:  # Contact between feet and ground
                    continue
                contactImpulse.append(point[9])
                if point[9]>1500:
                    print("Body A:", point[1], "Body B:", point[2], "Link A:", point[3], "Link B:", point[4], "Time:", env.t)
            if contactImpulse != []:
                maxContactImpulseList.append(max(contactImpulse))
            else:
                maxContactImpulseList.append(0)
            # Max velocity
            maxVelocityList1.append(max(np.abs(measurement["velocity"])))
            # Reward
            rewardList.append(reward)
            timeList.append(env.t)
            time.sleep(1.0 / SIMULATIONFREQUENCY)  # Normal speed is 1.0, here's to convenient observe
            # if done:
            #     break
            totalReward += reward
        print("Total reward is %f\n" % totalReward)
        baseOrientation = np.array(p.getEulerFromQuaternion(measurement['baseOrientation']))
        baseOrientation /= np.linalg.norm(baseOrientation)
        if baseOrientation.dot([0, 0, -1]) > np.cos(0.25 * np.pi) and np.alltrue([measurement['position'][i] < -1 for i in [1, 4, 7, 10]]):
            successCount += 1
            print("I succeed!")
    # trajectory = np.array(xbest)
    # env.reset()
    # for i in range(1):
    #     observation = env.reset()
    #     # input("Press any key to start\n")
    #     # time.sleep(3)
    #     trajectoryTargetList = []
    #     trajectoryObserveList = []
    #     maxTorqueList2 = []
    #     maxContactImpulseList = []
    #     maxVelocityList2 = []
    #     rewardList = []
    #     timeList = []
    #     for n in range(2 * 3 * SIMULATIONFREQUENCY):
    #         # action = HandTuningTrajectory(env.t)
    #         action = jointTargetsAsFunctionOfTime(env.t, timeLine, trajectory)
    #         observation, reward, done, measurement = env.step(action, addNoise=False)
    #         # if reward > 0:
    #         #     print(measurement["baseOrientationVector"].dot([0, 0, -1]))
    #         # Observe
    #         # Trajectory following
    #         tttorque2 = np.array(measurement['torque'])
    #         tttorque2 = np.abs(tttorque2).max()
    #         maxTorqueList2.append(tttorque2)
    #         trajectoryTargetList.append(action)
    #         trajectoryObserveList.append(measurement['position'])
    #         # Contact impulse
    #         contactPoints = p.getContactPoints(env.anymal)
    #         contactImpulse = []
    #         for point in contactPoints:
    #             if point[2] == 0 and point[3] in [link.value for link in FootLinks]:  # Contact between feet and ground
    #                 continue
    #             contactImpulse.append(point[9])
    #             if point[9]>1500:
    #                 print("Body A:", point[1], "Body B:", point[2], "Link A:", point[3], "Link B:", point[4], "Time:", env.t)
    #         if contactImpulse != []:
    #             maxContactImpulseList.append(max(contactImpulse))
    #         else:
    #             maxContactImpulseList.append(0)
    #         # Max velocity
    #         maxVelocityList2.append(max(np.abs(measurement["velocity"])))
    #         # Reward
    #         rewardList.append(reward)
    #         timeList.append(env.t)
    #         time.sleep(1.0 / SIMULATIONFREQUENCY)  # Normal speed is 1.0, here's to convenient observe
    #         if done:
    #             break
    #         totalReward += reward
    #     print("Total reward is %f\n" % totalReward)
    #     baseOrientation = np.array(p.getEulerFromQuaternion(measurement['baseOrientation']))
    #     baseOrientation /= np.linalg.norm(baseOrientation)
    #     if baseOrientation.dot([0, 0, -1]) > np.cos(0.25 * np.pi):
    #         successCount += 1
    #         print("I succeed!")
    # DrawFollowingPlot(timeList, maxTorqueList1, maxTorqueList2, 'Reference', 'CMA-ES')
    # print(successCount/20)
    input("Press any key to quit\n")
    env.close()
