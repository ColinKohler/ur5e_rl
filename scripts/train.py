import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import ray
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import pickle

from src.envs.block_reaching_env import BlockReachingEnv
from configs import *

from midichlorians.replay_buffer import ReplayBuffer
from midichlorians.shared_storage import SharedStorage
from midichlorians.data_generator import EpisodeHistory
from midichlorians.trainer import Trainer

from bulletarm_baselines.logger.logger import RayLogger

def load(checkpoint):
  pass

def train(config, checkpoint):
  # Initial checkpoint
  checkpoint = {
    'weights' : None,
    'optimizer_state' : None,
    'training_step' : 0,
    'num_eps' : 0,
    'num_steps' : 0,
    'terminate' : False,
  }

  # Create log dir
  if os.path.exists(config.results_path):
    shutil.rmtree(config.results_path)
  os.makedirs(config.results_path)

  # Start logger and trainer
  logger = RayLogger.options(num_cpus=0, num_gpus=0).remote(
    config.results_path,
    config.__dict__,
    checkpoint_interval=config.checkpoint_interval,
    num_eval_eps=0
  )
  trainer = Trainer.options(num_cpus=0, num_gpus=1.0).remote(checkpoint, config)

  # Load replay buffer with expert data
  # TODO: This should not be hard coded
  expert_data_path = '/home/helpinghands/workspace/data/block_reaching/10_expert/replay_buffer.pkl'
  with open(expert_data_path, 'rb') as f:
    data = pickle.load(f)
  checkpoint['num_eps'] = data['num_eps']
  checkpoint['num_steps'] = data['num_steps']
  replay_buffer = ReplayBuffer.options(num_cpus=0, num_gpus=0).remote(
    checkpoint,
    data['buffer'],
    config
  )

  shared_storage = SharedStorage.remote(checkpoint, config)
  shared_storage.setInfo.remote('terminate', False)

  env = BlockReachingEnv(config)
  time.sleep(1)

  eps_history = EpisodeHistory(is_expert=False)
  obs = env.reset()
  done = False

  pbar = tqdm(total=config.training_steps)
  next_batch = replay_buffer.sample.remote(shared_storage)
  for training_step in range(config.training_steps):
    if done:
      replay_buffer.add.remote(eps_history, shared_storage)
      logger.logTrainingEpisode.remote(eps_history.reward_history)

      obs = env.reset()
      eps_history = EpisodeHistory(is_expert=False)
      eps_history.logStep(obs[0], obs[1], obs[2], np.array([0] * config.action_dim), 0, 0, 0, config.max_force)

    action_idxs, action, value = ray.get(trainer.getAction.remote(obs))
    idx_batch, batch = ray.get(next_batch)
    next_batch = replay_buffer.sample.remote(shared_storage)
    obs, reward, done = env.step(action[0].tolist())

    td_error, loss = ray.get(trainer.updateWeights.remote(batch, replay_buffer, shared_storage, logger))
    replay_buffer.updatePriorities.remote(td_error.cpu(), idx_batch)
    eps_history.logStep(
        obs[0],
        obs[1],
        obs[2],
        action_idxs.squeeze().numpy(),
        value.item(),
        reward,
        done,
        config.max_force
    )

    vision, force, proprio = obs
    logger.writeLog.remote()
    pbar.update(1)

  # Saving
  buffer = ray.get(replay_buffer.getBuffer.remote())
  ray.get(trainer.saveWeights.remote(shared_storage))
  ray.get(shared_storage.saveReplayBuffer.remote(buffer))
  ray.get(shared_storage.saveCheckpoint.remote())
  ray.get(logger.exportData.remote())

if __name__ == '__main__':
  parser=  argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to train on.')
  parser.add_argument('--num_gpus', type=int, default=1,
    help='Number of GPUs to use for training.')
  parser.add_argument('--results_path', type=str, default=None,
    help='Path to save results & logs to while training. Defaults to current timestamp.')
  parser.add_argument('--encoder', type=str, default='depth+force+proprio',
    help='Type of latent encoder to use')
  parser.add_argument('--checkpoint', type=str, default=None,
    help='Path to the checkpoint to load.')
  args = parser.parse_args()


  config = BlockReachingConfig(equivariant=True, vision_size=64, results_path=args.results_path)

  ray.init(num_gpus=config.num_gpus, ignore_reinit_error=True)
  train(config, args.checkpoint)
  ray.shutdown()
