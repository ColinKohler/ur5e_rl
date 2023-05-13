import rospy
import copy
import numpy as np
import scipy.ndimage

from src.ur5e import UR5e
from src.rgbd_sensor import RGBDSensor
from src.force_sensor import ForceSensor

class Env(object):
  def __init__(self, workspace, vision_size, force_obs_len):
    rospy.init_node('ur5e_rl_env')

    self.workspace = workspace
    self.vision_size = vision_size
    self.force_obs_len = force_obs_len
    self.obs_type = ['vision', 'force', 'proprio']

    self.ur5e = UR5e()
    self.rgbd_sensor = RGBDSensor(self.vision_size)
    self.force_sensor = ForceSensor(self.force_obs_len)

    self.num_steps = 0

  def reset(self):
    self.ur5e.reset()
    self.force_sensor.reset()
    self.num_steps = 0

    #return self.getObservation()
    return None

  def step(self, action):
    p, x, y, z, rot = action

    target_pose = self.getActionPose(action)
    self.ur5e.moveToPose(target_pose)
    self.ur5e.sendGripperCmd(p)
    self.num_steps += 1

    #obs = self.getObservation()
    obs = None
    done = self.checkTermination()
    reward = self.getReward()

    return obs, done, reward

  def getActionPose(self, action):
    p, x, y, z, rot = action
    current_pose = self.ur5e.getEEPose()
    target_pose = copy.copy(target_pose)

    target_pose.pos = np.array(current_pose.pos) + np.array([x, y, z])
    target_pose.rot[2] = current_pose.rot[2] + rot
    target_pose.pos[0] = np.clip(
      target_pose.pos[0], self.workspace[0, 0], self.workspace[0, 1]
    )
    target_pose.pos[1] = np.clip(
      target_pose.pos[1], self.workspace[1, 0], self.workspace[1, 1]
    )
    target_pose.pos[2] = np.clip(
      target_pose.pos[2], self.workspace[2, 0], self.workspace[2, 1]
    )

    return target_pose

  def getProprioObservation(self):
    ee_pose = self.ur5e.getEEPose()
    ee_pos = ee_pose.getPosition()
    ee_rz = ee_pose.getEulerOrientation()[-1]
    proprio = np.array([self.robot.gripper.getOpenRatio()] + ee_pos + [ee_rz])
    return proprio

  def getObservation(self):
    ''''''
    obs = list()
    for obs_type in self.obs_type:
      if obs_type == 'vision':
        obs.append(self.rgbd_sensor.getObservation())
      if obs_type == 'force':
        obs.append(self.force_sensor.getObservation())
      if obs_type == 'proprio':
        obs.append(self.getProprioObservation())
    return obs

  def checkTermination(self):
    return None

  def getReward(self):
    return None
