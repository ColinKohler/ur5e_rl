import rospy

from src.ur5 import UR5
from src.utils import Pose

if __name__ == '__main__':
  rospy.init_node('robalto_rl')

  ur5 = UR5()
  # NOTE: Makes sure that the home position in src/ur5.py is correct before running this
  # ur5.moveToHome()

  # TODO: Set the test pose to something reasonable
  test_pose = Pose(0., 0., 0., 0., 0., 0.)
  #ur5.moveToPose(test_pose)
