import numpy as np
import numpy.random as npr

class BasePlanner(object):
  def __init__(self, env, config):
    self.env = env
    self.dpos = config.dpos
    self.drot = config.drot

    self.stage = 0

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      primitive = 0 if self.current_target[2] == 0 else 1
      self.current_target = None
    else:
      primitive = 1 #if not self.env.ur5e.gripper.isClosed() else 1
    return np.array([primitive, x, y, z, r])

  def getActionByGoalPose(self, goal_pos, goal_rot):
    current_pos = self.env.current_pose.getPosition()
    current_rot = self.env.current_pose.getEulerOrientation()
    pos_diff = goal_pos - current_pos
    rot_diff = np.array(goal_rot) - current_rot

    pos_diff[pos_diff // self.dpos > 0] = self.dpos
    pos_diff[pos_diff // -self.dpos > 0] = -self.dpos

    rot_diff[rot_diff // self.drot > 0] = self.drot
    rot_diff[rot_diff // -self.drot > 0] = -self.drot

    x, y, z, r = pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]
    return x, y, z, r
