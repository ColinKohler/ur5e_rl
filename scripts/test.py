import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import matplotlib.pyplot as plt
import time
import rospy
import numpy as np
from scipy.ndimage import rotate

from src.envs.block_reaching_env import BlockReachingEnv

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

      p = float(cmd_action[0])
      dx = float(cmd_action[1]) * config.dpos
      dy = float(cmd_action[2]) * config.dpos
      dz = float(cmd_action[3]) * config.dpos
      dr = float(cmd_action[4]) * config.drot
      action = [p, dx, dy, dz, dr]
      obs, reward, done = env.step(action)

    print(reward, done)
    vision, force, proprio = obs

    if plot:
      fig, ax = plt.subplots(nrows=1, ncols=3)
      ax[0].imshow(vision[:3].transpose(1,2,0))
      ax[1].imshow(vision[-1])
      ax[2].plot(force[:,0], label='Fx')
      ax[2].plot(force[:,1], label='Fy')
      ax[2].plot(force[:,2], label='Fz')
      ax[2].plot(force[:,3], label='Mx')
      ax[2].plot(force[:,4], label='My')
      ax[2].plot(force[:,5], label='Mz')
      plt.legend()
      plt.show()
