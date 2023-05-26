import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import matplotlib.pyplot as plt
import time
import rospy
import numpy as np
from scipy.ndimage import rotate

from src.env import Env
from src.block_reaching_env import BlockReachingEnv

from configs import *

if __name__ == '__main__':
  checkpoint = {
    'weights' : None,
    'optimizer_state' : None,
    'training_step' : 0,
    'num_eps' : 0,
    'num_steps' : 0,
  }
  config = BlockReachingConfig(False, 64, results_path='block_centering')

  env = BlockReachingEnv(config)
  time.sleep(1)

  plot = True
  while not rospy.is_shutdown():
    cmd_action = input('Action: ')
    if not cmd_action:
      obs, reward, done = env.step(action)
    elif cmd_action == 'r':
      obs = env.reset()
      reward = 0
      done = False
    else:
      cmd_action = cmd_action.split(' ')
      if len(cmd_action) != 5:
        print('Invalid action given. Required format: p x y z r')
        continue

      dx = float(cmd_action[1]) * config.dpos
      dy = float(cmd_action[2]) * config.dpos
      dz = float(cmd_action[3]) * config.dpos
      dr = float(cmd_action[4]) * config.drot
      action = [0, dx, dy, dz, dr]
      obs, reward, done = env.step(action)

    print(reward, done)
    vision, force, proprio = obs

    if plot:
      fig, ax = plt.subplots(nrows=1, ncols=2)
      ax[0].imshow(vision.squeeze())
      ax[1].plot(force[:,0], label='Fx')
      ax[1].plot(force[:,1], label='Fy')
      ax[1].plot(force[:,2], label='Fz')
      ax[1].plot(force[:,3], label='Mx')
      ax[1].plot(force[:,4], label='My')
      ax[1].plot(force[:,5], label='Mz')
      plt.legend()
      plt.show()
