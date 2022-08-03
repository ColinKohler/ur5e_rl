import rospy
import numpy as np
import scipy.ndimage

from src.ur5 import UR5
from src.cloud_proxy import CloudProxy

class Env(object):
  def __init__(self, workspace, heightmap_size):
    rospy.init_node('rl_env')

    self.workspace = workspace
    self.heightmap_size = heightmap_size

    self.ur5 = UR5()
    self.cloud_proxy = CloudProxy()

  def reset(self):
    self.ur5.moveToHome()
    self.ur5.openGripper()
    self.ur5.resetLoadCell()

    return self.getObservation()

  # TODO: Take agent action and convert to pose to move the ur5 ee to along with the gripper command
  def step(self, action):
    pass

  def getObservation(self):
    heightmap = self.getHeightmapResonstruct()
    gripper_img = self.getGripperImg()

    heightmap_w_gripper = np.copy(heightmap)
    heightmap_w_gripper[gripper_img.astype(bool)] = 0
    heightmap = heightmap.reshape((1, self.heightmap_size, self.heightmap_size))

    is_holding = self.ur5.holding_state
    return is_holding, heightmap

  def getHeightmapReconstruct(self):
    gripper_pos, _ = self.ur5.getEEPose()
    heightmap = self.cloud_proxy.getProjectImg(0.4, self.heightmap_size, gripper_pos)
    return self.preProcessHeightmap(heightmap)

  def preProcessHeightmap(self, heightmap):
    return scipy.ndimage.median_filter(heightmap, 2)

  def getGripperImg(self):
    img = np.zeros((self.heightmap_size, self.heightmap_size))
    rz = self.ur5.getEndEffectorOrientation()[2]

    gripper_state = self.ur5.getGripperState()
    d = int(32 * gripper_state)
    img[] = 1
    img[] = 1
    img = scipy.ndimage.rotate(img, np.rad2deg(rz), reshape=False, order=0)

    return img
