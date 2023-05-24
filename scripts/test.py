import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import matplotlib.pyplot as plt
import time
import rospy
import numpy as np
from scipy.ndimage import rotate

from src.env import Env
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

  env = Env(config)
  time.sleep(1)

  while not rospy.is_shutdown():
    cmd_action = input('Action: ')
    if not cmd_action:
      obs, done, reward = env.step(action)
    elif cmd_action == 'r':
      obs = env.reset()
    else:
      cmd_action = cmd_action.split(' ')
      if len(cmd_action) != 5:
        print('Invalid action given. Required format: p x y z r')
        continue

      action = [0.025 * float(a) for a in cmd_action]
      obs, done, reward = env.step(action)

    vision, force, proprio = obs

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(force[:,0], label='Fx')
    ax[0].plot(force[:,1], label='Fy')
    ax[0].plot(force[:,2], label='Fz')
    ax[0].plot(force[:,3], label='Mx')
    ax[0].plot(force[:,4], label='My')
    ax[0].plot(force[:,5], label='Mz')
    ax[1].imshow(vision.squeeze())
    plt.legend()
    plt.show()
