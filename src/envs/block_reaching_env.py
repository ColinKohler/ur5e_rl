import tf
import rospy
import numpy as np
import numpy.random as npr

from src.envs.base_env import BaseEnv
import src.utils as utils

class BlockReachingEnv(BaseEnv):
  def __init__(self, config):
    super().__init__(config)
    self.max_steps = 50

  def getBlockPose(self):
    return utils.convertTfToPose(self.tf_proxy.lookupTransform('base_link', 'block'))

  def resetWorkspace(self):
    # Move arm out side of workspace
    self.ur5e.moveToHome()
    self.ur5e.moveToOffsetHome()
    current_block_pose = self.getBlockPose()
    current_block_pose.rot = [-0.5, -0.5, 0.5, -0.5]

    # Generate new random pose for the block
    self.block_pose = [
      npr.uniform(self.workspace[0,0]+0.05, self.workspace[0,1]-0.05),
      npr.uniform(self.workspace[1,0]+0.05, self.workspace[1,1]-0.05),
      0.03
    ]

    # Pick and place the block at the new pose
    self.ur5e.moveToHome()
    self.ur5e.pick(current_block_pose)
    import sys
    sys.exit()
    #self.ur5e.place(self.block_pose)
    #self.ur5e.moveToHome()

  def checkTermination(self, obs):
    is_touching_block = self.touchingBlock(obs)
    return is_touching_block or self.num_steps >= self.max_steps

  def getReward(self, obs):
    return float(self.touchingBlock(obs))

  def touchingBlock(self, obs):
    avg_force = list()
    for i in range(6):
      avg_force.append(np.convolve(obs[1][:,i], np.ones(10), 'valid') / 10)
    avg_force = np.array(avg_force).transpose(1, 0)
    axis_max = np.max(avg_force, axis=0)
    # NOTE: Currently this only works for top down touching.
    return obs[2][3] >= 0.0 and axis_max[2] >= 2.0
