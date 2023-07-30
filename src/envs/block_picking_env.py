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
    self.max_steps = 50
    self.pick_height = 0.2

  def getBlockPose(self):
    return utils.convertTfToPose(self.tf_proxy.lookupTransform('base_link', 'block'))

  def resetWorkspace(self):
    # Move arm out side of workspace
    self.ur5e.moveToHome()
    self.ur5e.moveToOffsetHome()
    current_block_pose = self.getBlockPose()
    # TODO: Fix ur5e.pick() s.t. it can use the orientation of the block
    current_block_pose.rot = [-0.5, -0.5, 0.5, -0.5]

    # Generate new random pose for the block
    new_block_pos = [
      npr.uniform(self.workspace[0,0]+0.05, self.workspace[0,1]-0.05),
      npr.uniform(self.workspace[1,0]+0.05, self.workspace[1,1]-0.05),
      0.06
    ]
    # TODO: Generate random orientation for block
    self.block_pose = Pose(*new_block_pos, -0.5, -0.5, 0.5, -0.5)

    # Pick and place the block at the new pose
    self.ur5e.moveToHome()
    print(self.ur5e.gripper.getForce())
    # print(self.ur5e.getGripperState())
    block_picked = False
    while not block_picked:
      self.ur5e.pick(current_block_pose)
      # print(self.isHoldingBlock())
      print(self.ur5e.gripper.getForce())
      block_picked = not self.ur5e.gripper.isClosed()
      if not block_picked:
        self.ur5e.moveToOffsetHome()
        current_block_pose = self.getBlockPose()
        current_block_pose.rot = [-0.5, -0.5, 0.5, -0.5]

    # print(self.isHoldingBlock())
    # print(self.ur5e.getGripperState())
    self.ur5e.place(self.block_pose)
    self.ur5e.moveToHome()

  def checkTermination(self, obs):
    return super().checkTermination() or self.isHoldingBlock()

  def getReward(self, obs):
    gripper_z = self.current_pose.getPosition()[-1]
    return float(self.isHoldingBlock() and gripper_z > self.pick_height)

  def isHoldingBlock(self):
    print("pose")
    # print(self.ur5e.gripper.getForce)
    if self.ur5e.gripper.isClosed() :
      print("no")
    else:
      if self.ur5e.getGripperState() <= 0.61:
        print ("holding")
    return True
