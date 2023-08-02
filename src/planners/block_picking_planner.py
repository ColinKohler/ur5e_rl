from src.planners.base_planner import BasePlanner

import copy
import numpy as np
import numpy.random as npr

class BlockPickingPlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def setNewTarget(self):
    block_pos = np.array(self.env.block_pose.getPosition())
    block_rot = np.array(self.env.block_pose.getEulerOrientation())

    pre_grasp_pos = copy.copy(block_pos)
    pre_grasp_pos[0] += npr.uniform(-0.02, 0.02)
    pre_grasp_pos[1] += npr.uniform(-0.02, 0.02)
    pre_grasp_pos[2] += npr.uniform(0.08, 0.16) + self.env.ur5e.gripper_offset
    pre_grasp_rot = block_rot

    grasp_pos = copy.copy(block_pos)
    grasp_pos[0] += npr.uniform(-0.01, 0.01)
    grasp_pos[1] += npr.uniform(-0.01, 0.01)
    grasp_pos[2] += npr.uniform(-0.02, 0.01) + self.env.ur5e.gripper_offset
    grasp_rot = block_rot

    post_grasp_pos = copy.copy(block_pos)
    post_grasp_pos[0] += npr.uniform(-0.02, 0.02)
    post_grasp_pos[1] += npr.uniform(-0.02, 0.02)
    post_grasp_pos[2] += 0.15 + self.env.ur5e.gripper_offset

    post_grasp_rot = block_rot

    if self.stage == 0:
      self.stage = 1
      self.current_target = (pre_grasp_pos, pre_grasp_rot, 1)
    elif self.stage == 1:
      self.stage = 2
      self.current_target = (grasp_pos, grasp_rot, 0)
    elif self.stage == 2:
      self.stage = 0
      self.current_target = (post_grasp_pos, post_grasp_rot, 0)

  def getNextAction(self):
    if self.env.num_steps == 0:
      self.stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()
