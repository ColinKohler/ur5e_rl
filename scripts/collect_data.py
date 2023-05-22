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
  config = BlockReachingConfig(False, vision_size, results_path=args.results_path)

  replay_buffer_worker = ReplayBuffer.options(num_cpus=0, num_gpus=0).remote(
    checkpoint,
    dict(),
    config
  )

  env = Env(config)
  time.sleep(1)

  while not rospy.is_shutdown():
    action = input('Action: ')
    if action == 'r':
      obs = env.reset()
    else:
      action = action.split(' ')
      if len(action) != 5:
        print('Invalid action given. Required format: p x y z r')
        continue

      obs, done, reward = env.step([0.025 * float(a) for a in action])

    vision, force, proprio = obs

    plt.imshow(rotate(vision.transpose(1, 2, 0), 180, mode='nearest', order=1)[:,:,0]); plt.show()
    plt.plot(force[:,0], label='Fx')
    plt.plot(force[:,1], label='Fy')
    plt.plot(force[:,2], label='Fz')
    plt.plot(force[:,3], label='Mx')
    plt.plot(force[:,4], label='My')
    plt.plot(force[:,5], label='Mz')
    plt.legend()
    plt.show()
