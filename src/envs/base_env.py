import tf
import rospy
import time
import copy
import numpy as np
import scipy.ndimage
import quaternion
from collections import deque

from src.ur5e import UR5e
from src.rgbd_sensor import RGBDSensor
from src.force_sensor import ForceSensor
from src.tf_proxy import TFProxy
from src.utils import Pose

class BaseEnv(object):
  def __init__(self, config):
    rospy.init_node('ur5e_rl_env')
    self.config = config

    self.workspace = self.config.workspace
    self.vision_size = self.config.obs_size
    self.force_obs_len = self.config.force_history
    self.obs_type = self.config.obs_type

    self.tf_proxy = TFProxy()
    self.ur5e = UR5e()
    self.rgbd_sensor = RGBDSensor(self.vision_size)
    self.force_sensor = ForceSensor(self.force_obs_len, self.tf_proxy)

    self.num_steps = 0
    self.is_holding = False
    self.prev_poses = deque(maxlen=5)

  def reset(self):
    self.resetWorkspace()
    self.ur5e.reset()
    time.sleep(1)
    self.force_sensor.reset()

    self.num_steps = 0
    self.is_holding = False
    self.prev_poses = deque(maxlen=5)
    self.current_pose = self.ur5e.getEEPose()
    self.prev_poses.append(self.current_pose.getPosition())

    self.obs = self.getObservation()
    return self.obs

  def step(self, action):
    p, x, y, z, rz = action

    target_pose = self.getActionPose(action)
    did_move = self.ur5e.moveToPose(target_pose)
    prev_state = self.ur5e.getGripperState()
    self.ur5e.sendGripperCmd(p)
    new_state = self.ur5e.getGripperState()
    self.is_holding =  p < prev_state and p < new_state
    self.num_steps += 1

    self.ur5e.waitUntilNotMoving()
    self.current_pose = self.ur5e.getEEPose()
    self.prev_poses.append(self.current_pose.getPosition())

    self.obs = self.getObservation()
    done = self.checkTermination()
    reward = self.getReward()

    if not did_move:
      done = True

    return self.obs, reward, done

  def getActionPose(self, action):
    p, x, y, z, rz = action
    current_pos = self.current_pose.getPosition()
    current_rot = np.array(self.current_pose.getOrientationQuaternion())
    current_rot = np.quaternion(*current_rot[[3,0,1,2]])

    pos = np.array(current_pos) + np.array([x,y,z])
    delta_rot = np.array(tf.transformations.quaternion_from_euler(0, 0, rz))
    delta_rot = np.quaternion(*delta_rot[[3,0,1,2]])
    rot = quaternion.as_float_array(delta_rot * current_rot)
    rot = rot[[1,2,3,0]]
    rot = [0.5, 0.5, -0.5, 0.5]

    pos[0] = np.clip(pos[0], self.workspace[0, 0], self.workspace[0, 1])
    pos[1] = np.clip(pos[1], self.workspace[1, 0], self.workspace[1, 1])
    pos[2] = np.clip(pos[2], self.workspace[2, 0], self.workspace[2, 1])
    # TODO: This is not the correct way to clip a quaternion
    # TODO: These limits seem wrong but IK fails around here
    #if rot[0] < 0:
    #  rot[0] = 0.708
    #else:
    #  rot[0] = np.clip(rot[0], 0.1548, 0.708)
    #rot[1] = np.clip(rot[1], 0.05, 0.6786)
    #if rot[2] > 0:
    #  rot[2] = -0.705
    #else:
    #  rot[2] = np.clip(rot[2], -0.702, -0.217)
    #rot[3] = np.clip(rot[3], 0.05, 0.684)

    target_pose = Pose(*pos, *rot)

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

  def resetWorkspace(self):
    return None

  def checkTermination(self):
    prev_pose_diff = np.linalg.norm(np.array(self.prev_poses) - self.prev_poses[0], axis=1)
    is_stuck = np.all(prev_pose_diff <= 5e-3) and len(self.prev_poses) == 5
    is_max_step = self.num_steps >= self.max_steps
    return is_max_step or is_stuck

  def getReward(self):
    return None
