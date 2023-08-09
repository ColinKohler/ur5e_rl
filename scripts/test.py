import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import ray
import signal
import time
import shutil
import argparse
import numpy as np
import torch
from tqdm import tqdm
import pickle

from src.envs.block_reaching_env import BlockReachingEnv
from src.envs.block_picking_env import BlockPickingEnv
from configs import *

from svfl.agent import Agent

from bulletarm_baselines.logger.logger import RayLogger

TASK_CONFIGS = {
  'block_reaching' : BlockReachingConfig,
  'block_picking' : BlockPickingConfig,
}

TASKS = {
  'block_reaching' : BlockReachingEnv,
  'block_picking' : BlockPickingEnv,
}


def test(task, config, checkpoint):
  # Load model
  checkpoint_path = os.path.join(config.results_path,
                                 'model.checkpoint')
  if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print('Loading checkpoint from {}'.format(checkpoint_path))
  else:
    print('Checkpoint not found at {}'.format(checkpoint_path))
    sys.exit()

  env = TASKS[task](config)
  time.sleep(1)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  agent = Agent(config, device, initialize_models=False)
  agent.setWeights(checkpoint['weights'])

  # ---------------------
  # Testing
  # ---------------------
  num_success = 0
  pbar = tqdm(total=args.num_eps)
  pbar.set_description('SR: 0%')
  for i in range(args.num_eps):
    done = False
    obs = env.reset()

    while not done:
      action_idx, action, value = agent.getAction(obs, evaluate=False)
      obs, reward, done = env.step(action[0].tolist())

    num_success += int(reward >= 1)
    pbar.set_description('SR: {}%'.format(int((num_success / (i+1)) * 100)))
    pbar.update(1)
  pbar.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to test on.')
  parser.add_argument('checkpoint', type=str, default=None,
    help='Path to load the model from.')
  parser.add_argument('--num_eps', type=int, default=100,
    help='Number of episodes to test on.')
  parser.add_argument('--num_gpus', type=int, default=1,
    help='Number of GPUs to use for training.')
  parser.add_argument('--encoder', type=str, default='vision+force+proprio',
    help='Type of latent encoder to use')

  args = parser.parse_args()

  config = TASK_CONFIGS[args.task](equivariant=True, vision_size=128, encoder=args.encoder, results_path=args.checkpoint)

  ray.init(num_gpus=config.num_gpus, ignore_reinit_error=True)
  test(args.task, config, args.checkpoint)
  ray.shutdown()
