import os
import numpy as np

class Config(object):
  '''
  Base task config.
  '''

  def __init__(self, equivariant=True, vision_size=64, encoder='vision+force+proprio', num_gpus=1):
    # Env
    self.obs_type = ['vision', 'force', 'proprio']
    self.vision_size = vision_size
    self.obs_size = vision_size + 12
    self.vision_channels = 4
    self.force_dim = 6
    self.force_history = 64
    self.max_force = 100
    self.proprio_dim = 5

    self.action_sequence = 'pxyzr'
    self.action_dim =  len(self.action_sequence)

    self.workspace = np.array([[-0.15, 0.15], [0.35, 0.65], [-0.01, 0.25]])
    self.dpos = 0.025
    self.drot = np.pi / 16

    # Model
    self.z_dim = 64
    self.encoder = encoder.split('+')

    # Training
    self.root_path = '/home/helpinghands/workspace/data/'
    self.num_gpus = num_gpus
    self.pre_training_steps = 0
    self.gen_data_on_gpu = False
    self.per_beta_anneal_steps = None
