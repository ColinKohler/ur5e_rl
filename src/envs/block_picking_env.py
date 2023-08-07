import tf
import rospy
import numpy as np
import numpy.random as npr

from src.envs.base_env import BaseEnv
import src.utils as utils
from src.utils import Pose

class BlockPickingEnv(BaseEnv):
  def __init__(self, config):
    super().__init__(config)
    self.pick_height = 0.20
    self.hold_force_th = 100.0

  def getBlockPose(self):
    try:
      pose = utils.convertTfToPose(self.tf_proxy.lookupTransform('base_link', 'block'))
    except:
      input('Block not detected. Please place block back within workspace.')
      pose = utils.convertTfToPose(self.tf_proxy.lookupTransform('base_link', 'block'))

    return pose

  def resetWorkspace(self):
    # Generate new random pose for the block
    new_block_pos = [
      npr.uniform(self.workspace[0,0]+0.05, self.workspace[0,1]-0.05),
      npr.uniform(self.workspace[1,0]+0.05, self.workspace[1,1]-0.05),
      0.04
    ]
    # TODO: Generate random orientation for block
    self.block_pose = Pose(*new_block_pos, -0.5, -0.5, 0.5, -0.5)

    # Pick block if not holding
    if not self.is_holding:
      # Move arm out side of workspace
      self.ur5e.moveToHome()
      self.ur5e.moveToOffsetHome()
      current_block_pose = self.getBlockPose()
      # TODO: Fix ur5e.pick() s.t. it can use the orientation of the block
      current_block_pose.rot = [-0.5, -0.5, 0.5, -0.5]

      # Pick
      self.ur5e.moveToHome()
      block_picked = False
      while not block_picked:
        self.ur5e.pick(current_block_pose)
        block_picked = not self.ur5e.gripper.isClosed()
        if not block_picked:
          self.ur5e.moveToOffsetHome()
          current_block_pose = self.getBlockPose()
          current_block_pose.rot = [-0.5, -0.5, 0.5, -0.5]

    # Place
    self.ur5e.place(self.block_pose)
    self.ur5e.moveToHome()

  def checkTermination(self):
    return super().checkTermination() or (self.getReward() > 0)

  def getReward(self):
    gripper_z = self.current_pose.getPosition()[-1]
    return float(self.is_holding and gripper_z > self.pick_height)
