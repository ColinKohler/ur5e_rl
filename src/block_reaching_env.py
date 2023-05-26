from src.env import Env
import numpy as np
class BlockReachingEnv(Env):
  def __init__(self, config):
    super().__init__(config)
    self.max_steps = 50

  # TODO:
  #  1.) If gripper is in contact (high force) and >table height?
  def checkTermination(self, num_steps):
    return num_steps >= self.max_steps

  def getReward(self, observation):
    vision, force, proprio = observation

    if (proprio[3] <= 0.0073 and np.average(force)<= -2.0):
      return 1
    return 0
