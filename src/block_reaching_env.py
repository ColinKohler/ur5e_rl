from src.env import Env

class BlockReachingEnv(Env):
  def __init__(self, workspace, vision_size, force_obs_len):
    super().__init__(workspace, vision_size, force_obs_len)
    self.max_steps = 50

  # TODO: Can we come up w/a termination criteria w/o calibrated sensors?
  #  1.) If gripper is in contact (high force) and >table height?
  def checkTermination(self):
    return self.num_steps >= self.max_steps

  def getReward(self):
    return 0
