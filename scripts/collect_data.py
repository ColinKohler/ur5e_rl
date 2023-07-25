import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import ray
import argparse
import shutil
import time
import rospy
import copy
import numpy as np
from tqdm import tqdm

from src.envs.block_reaching_env import BlockReachingEnv
from src.planners.block_reaching_planner import BlockReachingPlanner

from svfl.replay_buffer import ReplayBuffer
from svfl.shared_storage import SharedStorage
from svfl.data_generator import EpisodeHistory
from svfl.trainer import Trainer

from configs import *

def collectData(config, num_expert_eps):
  checkpoint = {
    'weights' : None,
    'optimizer_state' : None,
    'training_step' : 0,
    'num_eps' : 0,
    'num_steps' : 0,
  }

  if os.path.exists(config.results_path):
    shutil.rmtree(config.results_path)
  os.makedirs(config.results_path)

  shared_storage = SharedStorage.remote(checkpoint, config)
  replay_buffer = ReplayBuffer.options(num_cpus=0, num_gpus=0).remote(
    checkpoint,
    dict(),
    config
  )
  trainer = Trainer.options(num_cpus=0, num_gpus=0.0).remote(checkpoint, config)

  env = BlockReachingEnv(config)
  planner = BlockReachingPlanner(env, config)
  time.sleep(1)

  num_eps = 0
  print('Generating {} episodes of expert data...'.format(num_expert_eps))

  pbar = tqdm(total=num_expert_eps)
  for expert_eps in range(num_expert_eps):
    obs = env.reset()
    eps_history = EpisodeHistory(is_expert=True)
    eps_history.logStep(obs[0], obs[1], obs[2], np.array([0] * config.action_dim), 0, 0, 0, config.max_force)
    done = False

    while not done:
      expert_action = planner.getNextAction()
      action_idx, action = ray.get(trainer.convertPlanAction.remote(expert_action))
      obs, reward, done = env.step(action.squeeze().tolist())
      value = 0

      # Log step
      eps_history.logStep(
        obs[0],
        obs[1],
        obs[2],
        action_idx.squeeze().numpy(),
        value,
        reward,
        done,
        config.max_force
      )

    # Save expert episode to buffer
    replay_buffer.add.remote(eps_history, shared_storage)
    pbar.update(1)

  buffer = ray.get(replay_buffer.getBuffer.remote())
  ray.get(shared_storage.saveReplayBuffer.remote(copy.copy(buffer)))
  ray.get(shared_storage.saveCheckpoint.remote())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to train on.')
  parser.add_argument('num_eps', type=int,
    help='Number of expert episodes to generate.')
  parser.add_argument('results_path', type=str,
    help='Path to save results & logs to while training.')
  args = parser.parse_args()

  config = BlockReachingConfig(equivariant=True, vision_size=64, results_path=args.results_path)

  ray.init(num_gpus=config.num_gpus, ignore_reinit_error=True)
  collectData(config, args.num_eps)
  ray.shutdown()
