from src.env import Env
import numpy as np
class BlockReachingEnv(Env):
  def __init__(self, config):
    super().__init__(config)
    self.max_steps = 50

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
