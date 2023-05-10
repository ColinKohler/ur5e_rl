import rospy

from src.env import Env

if __name__ == '__main__':
  rospy.init_node('ur5e_rl')

  workspace = list()
  heightmap_size = 64

  env = Env()

