import tf
import rospy
import time
import copy
import numpy as np
import scipy.ndimage

from src.ur5e import UR5e
from src.rgbd_sensor import RGBDSensor
from src.force_sensor import ForceSensor
from src.tf_proxy import TFProxy
from src.utils import Pose

class Env(object):
  def __init__(self, config):
    rospy.init_node('ur5e_rl_env')
    self.config = config

    self.workspace = self.config.workspace
    self.vision_size = self.config.vision_size
    self.force_obs_len = self.config.force_history
    self.obs_type = self.config.obs_type

    self.tf_proxy = TFProxy()
    self.ur5e = UR5e()
    self.rgbd_sensor = RGBDSensor(self.vision_size)
    self.force_sensor = ForceSensor(self.force_obs_len, self.tf_proxy)

    self.num_steps = 0

  def reset(self):
    self.ur5e.reset()
    time.sleep(2) # TODO: Best to not hardcode this
    self.force_sensor.reset()
    self.num_steps = 0

    return self.getObservation()

  def step(self, action):
    target_pose = self.getActionPose(action)
    self.ur5e.moveToPose(target_pose)
    #self.ur5e.sendGripperCmd(p)
    self.num_steps += 1

    obs = self.getObservation()
    done = self.checkTermination(obs)
    reward = self.getReward(obs)

    return obs, reward, done

  def getActionPose(self, action):
    p, x, y, z, rz = action
    current_pose = self.ur5e.getEEPose()
    current_pos = current_pose.getPosition()
    current_rot = current_pose.getEulerOrientation()

    pos = np.array(current_pos) + np.array([x,y,z])
    rot = np.array(current_rot) + np.array([0, 0, rz])

    pos[0] = np.clip(pos[0], self.workspace[0, 0], self.workspace[0, 1])
    pos[1] = np.clip(pos[1], self.workspace[1, 0], self.workspace[1, 1])
    pos[2] = np.clip(pos[2], self.workspace[2, 0], self.workspace[2, 1])

    #print('Current: {} | {}'.format(current_pos, current_rot))
    #print('Target:  {} | {}'.format(pos, rot))

    target_pose = Pose(*pos, *tf.transformations.quaternion_from_euler(*rot))

    return target_pose

  def getProprioObservation(self):
    ee_pose = self.ur5e.getEEPose()
    ee_pos = ee_pose.getPosition()
    ee_rz = ee_pose.getEulerOrientation()[-1]
    proprio = np.array([self.ur5e.getGripperState()] + ee_pos + [ee_rz])
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
