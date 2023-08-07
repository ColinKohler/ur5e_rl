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

from svfl.replay_buffer import ReplayBuffer
from svfl.shared_storage import SharedStorage
from svfl.data_generator import EpisodeHistory
from svfl.trainer import Trainer

from bulletarm_baselines.logger.logger import RayLogger

TASK_CONFIGS = {
  'block_reaching' : BlockReachingConfig,
  'block_picking' : BlockPickingConfig,
}

TASKS = {
  'block_reaching' : BlockReachingEnv,
  'block_picking' : BlockPickingEnv,
}

def waitForRayTasks(sleep_time_init=2, sleep_time_loop=0.4):
  time.sleep(sleep_time_init)
  while (ray.cluster_resources() != ray.available_resources()):
    time.sleep(sleep_time_loop)
  return

def load(checkpoint, checkpoint_path=None, replay_buffer_path=None):
  ''' Load the model checkpoint and replay buffer. '''
  if checkpoint_path:
    if os.path.exists(checkpoint_path):
      checkpoint = torch.load(checkpoint_path)
      print('Loading checkpoint from {}'.format(checkpoint_path))
    else:
      print('Checkpoint not found at {}'.format(checkpoint_path))

  replay_buffer = dict()
  if replay_buffer_path:
    if os.path.exists(replay_buffer_path):
      with open(replay_buffer_path, 'rb') as f:
        data = pickle.load(f)

      replay_buffer = data['buffer']
      checkpoint['num_eps'] = data['num_eps']
      checkpoint['num_steps'] = data['num_steps']

      print('Loaded replay buffer at {}'.format(replay_buffer_path))
    else:
      print('Replay buffer not found at {}'.format(replay_buffer_path))

  return checkpoint, replay_buffer

def train(task, config, checkpoint_path, buffer_path):
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

  # Load checkpoint/replay buffer
  log_path = None
  if checkpoint_path:
    log_path =  os.path.join(config.root_path,
                             task,
                             checkpoint_path,
                             'log_data.pkl')
    checkpoint_path = os.path.join(config.root_path,
                                   task,
                                   checkpoint_path,
                                   'model.checkpoint')
  if buffer_path:
    buffer_path = os.path.join(config.root_path,
                               task,
                               buffer_path,
                               'replay_buffer.pkl')

  checkpoint, data_buffer = load(
    checkpoint,
    checkpoint_path=checkpoint_path,
    replay_buffer_path=buffer_path
  )

  # Start logger and trainer
  logger = RayLogger.options(num_cpus=0, num_gpus=0).remote(
    config.results_path,
    config.__dict__,
    checkpoint_interval=config.checkpoint_interval,
    num_eval_eps=0,
    log_file=log_path
  )
  trainer = Trainer.options(num_cpus=0, num_gpus=1.0).remote(checkpoint, config)

  replay_buffer = ReplayBuffer.options(num_cpus=0, num_gpus=0).remote(
    checkpoint,
    data_buffer,
    config
  )

  shared_storage = SharedStorage.remote(checkpoint, config)
  shared_storage.setInfo.remote('terminate', False)

  # Interupt signal handler to save on exit
  def saveOnInt(signum, frame):
    buffer = ray.get(replay_buffer.getBuffer.remote())
    ray.get(trainer.saveWeights.remote(shared_storage))
    ray.get(shared_storage.saveReplayBuffer.remote(buffer))
    ray.get(shared_storage.saveCheckpoint.remote())
    ray.get(logger.exportData.remote())
    #waitForRayTasks()
    ray.shutdown()
  signal.signal(signal.SIGINT, saveOnInt)

  env = TASKS[task](config)
  time.sleep(1)

  # ---------------------
  # Pretraining
  # ---------------------
  if config.pre_training_steps > 0:
    print('Pretrianing for {} steps...'.format(config.pre_training_steps))
    pbar = tqdm(total=config.pre_training_steps)
    for pre_training_step in range(config.pre_training_steps):
      next_batch = replay_buffer.sample.remote(shared_storage)
      idx_batch, batch = ray.get(next_batch)
      td_error, loss = ray.get(trainer.updateWeights.remote(batch, replay_buffer, shared_storage, logger))
      replay_buffer.updatePriorities.remote(td_error.cpu(), idx_batch)
      pbar.update(1)

  # ---------------------
  # Training
  # ---------------------
  print('Training for {} steps...'.format(config.training_steps))
  obs = env.reset()
  eps_history = EpisodeHistory(is_expert=False)
  eps_history.logStep(obs[0], obs[1], obs[2], np.array([0] * config.action_dim), 0, 0, 0, config.max_force)
  done = False

  pbar = tqdm(total=config.training_steps)
  next_batch = replay_buffer.sample.remote(shared_storage)
  for training_step in range(config.training_steps):
    if done:
      replay_buffer.add.remote(eps_history, shared_storage)
      logger.logTrainingEpisode.remote(eps_history.reward_history, eps_history.value_history)

      obs = env.reset()
      eps_history = EpisodeHistory(is_expert=False)
      eps_history.logStep(obs[0], obs[1], obs[2], np.array([0] * config.action_dim), 0, 0, 0, config.max_force)

    action_idx, action, value = ray.get(trainer.getAction.remote(obs))
    obs, reward, done = env.step(action[0].tolist())

    for _ in range(config.training_steps_per_action):
      idx_batch, batch = ray.get(next_batch)
      td_error, loss = ray.get(trainer.updateWeights.remote(batch, replay_buffer, shared_storage, logger))
      replay_buffer.updatePriorities.remote(td_error.cpu(), idx_batch)
      next_batch = replay_buffer.sample.remote(shared_storage)

    # Logging
    eps_history.logStep(
        obs[0],
        obs[1],
        obs[2],
        action_idx.squeeze().numpy(),
        value.item(),
        reward,
        done,
        config.max_force
    )
    logger.writeLog.remote()
    pbar.update(1)

  # Saving
  buffer = ray.get(replay_buffer.getBuffer.remote())
  ray.get(trainer.saveWeights.remote(shared_storage))
  ray.get(shared_storage.saveReplayBuffer.remote(buffer))
  ray.get(shared_storage.saveCheckpoint.remote())
  ray.get(logger.exportData.remote())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to train on.')
  parser.add_argument('--num_gpus', type=int, default=1,
    help='Number of GPUs to use for training.')
  parser.add_argument('--results_path', type=str, default=None,
    help='Path to save results & logs to while training. Defaults to current timestamp.')
  parser.add_argument('--encoder', type=str, default='vision+force+proprio',
    help='Type of latent encoder to use')
  parser.add_argument('--checkpoint', type=str, default=None,
    help='Path to the checkpoint to load.')
  parser.add_argument('--buffer', type=str, default=None,
    help='Path to the replay buffer to load')
  args = parser.parse_args()

  config = TASK_CONFIGS[args.task](equivariant=True, vision_size=128, encoder=args.encoder, results_path=args.results_path)

  ray.init(num_gpus=config.num_gpus, ignore_reinit_error=True)
  train(args.task, config, args.checkpoint, args.buffer)
  ray.shutdown()
