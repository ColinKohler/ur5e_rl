import rospy
import numpy as np
import scipy.ndimage

from src.ur5 import UR5

class Env(object):
  def __init__(self, workspace, heightmap_size):
    rospy.init_node('ur5e_rl_env')

    self.workspace = workspace
    self.heightmap_size = heightmap_size
    self.obs_type = ['vision', 'force', 'proprio']

    self.ur5e = UR5e()

  def reset(self):
    self.ur5e.moveToHome()
    self.ur5e.openGripper()
    self.ur5e.resetLoadCell()

    return self.getObservation()

  def step(self, action):
    pass

  def getVisionObservation(self):
    return None

  def getForceObservation(self):
    return None

  def getProprioObservation(self):
    ee_pose = self.ur5e.getEEPose()
    ee_pos = list(ee_pose.getPosition())
    ee_rz = ee_pose.getEulerOrientation()[-1]
    proprio = np.array([self.robot.gripper.getOpenRatio()] + ee_pos + [ee_rz])
    return proprio

  def getObservation(self):
    ''''''
    obs = list()
    for obs_type in self.obs_type:
      if obs_type == 'vision':
        obs.append(self.getVisionObservation())
      if obs_type == 'force':
        obs.append(self.getForceObservation())
      if obs_type == 'proprio':
        obs.append(self.getProprioObservation())
    return obs
