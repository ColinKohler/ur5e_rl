

class UR5(object):
  def __init__(self):
    self.control_pub = rospy.Publisher()
    self.home_joint_pos = list()

  def getEEPose(self):
    pass
