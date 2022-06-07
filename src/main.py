import rospy

from srv.env import Env

if __name__ == '__main__':
  rospy.init_node('robalto_rl')

  workspace = list()
  heightmap_size = 128

  env = Env()
