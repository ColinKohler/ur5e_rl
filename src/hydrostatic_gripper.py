import rospy

class HydrostaticGripper(object):
  def __init__(self):
    # TODO: Set these to the correct ros topics
    self.control_sub = rospy.Subscriber('', None, self.updateGripperState)
    self.control_pub = rospy.Publisher('', None, queue_size=1)

    self.state = None

  def updateGripperState(self, msg):
    self.state = msg

  # TODO: Reset gripper to default open pose
  def reset(self):
    pass

  # TODO: Send command to gripper control to set gripper to a specific position
  def setPosition(self, pos):
    pass

  # TODO: Return current gripper pose
  def getPose(self):
    pass
