import rospy
import numpy as np

from src.env import Env

if __name__ == '__main__':
  rospy.init_node('ur5e_rl')

  workspace = np.array(
    [
      [0.2,   0.6],
      [-0.15, 0.15],
      [-0.01, 0.25]
    ]
  )
  vision_size = 64
  force_obs_len = 64
  env = Env(workspace, vision_size, force_obs_len)

  while not rospy.is_shutdown():
    action = input('Action: ')
    action = action.split(' ')
    if len(action) != 5:
      print('Invalid action given. Required format: p x y z r')
      continue

    env.step(action)
