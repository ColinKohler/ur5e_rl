import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import ray
import matplotlib.pyplot as plt
import time
import rospy
import copy
import numpy as np
from scipy.ndimage import rotate

from src.env import Env
from midichlorians.replay_buffer import ReplayBuffer
from midichlorians.shared_storage import SharedStorage
from midichlorians.data_generator import EpisodeHistory
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
  ray.init(num_gpus=config.num_gpus, ignore_reinit_error=True)

  shared_storage = SharedStorage.remote(checkpoint, config)
  replay_buffer = ReplayBuffer.options(num_cpus=0, num_gpus=0).remote(
    checkpoint,
    dict(),
    config
  )

  env = Env(config)
  time.sleep(1)

  eps_history = None
  while not rospy.is_shutdown():
    cmd_action = input('Action: ')
    if not cmd_action:
      obs, done, reward = env.step(action)
      eps_history.logStep(obs[0], obs[1], obs[2], action, 0, 0, 0, config.max_force)
    elif cmd_action == 'q':
      break
    elif cmd_action == 'r':
      if eps_history is not None:
        replay_buffer.add.remote(eps_history, shared_storage)

      obs = env.reset()
      eps_history = EpisodeHistory(is_expert=True)
      eps_history.logStep(obs[0], obs[1], obs[2], np.array([0] * config.action_dim), 0, 0, 0, config.max_force)
    else:
      cmd_action = cmd_action.split(' ')
      if len(cmd_action) != 5:
        print('Invalid action given. Required format: p x y z r')
        continue

      action = np.array([0.025 * float(a) for a in cmd_action])
      obs, done, reward = env.step(action)
      eps_history.logStep(obs[0], obs[1], obs[2], action, 0, 0, 0, config.max_force)

    vision, force, proprio = obs

  buffer = ray.get(replay_buffer.getBuffer.remote())
  shared_storage.saveReplayBuffer.remote(copy.copy(buffer))
  shared_storage.saveCheckpoint.remote()

  ray.shutdown()
