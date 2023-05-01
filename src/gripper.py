import rospy

class Gripper(object):
  ''' Interface for gripper.

  Args:
    p_min (float): The min joint angle for the gripper.
    p_max (float): The max joint angle for the gripper.
  '''
  def __init__(self, p_min=0, p_max=1):
    # TODO: Set these to the correct ros topics
    self.control_sub = rospy.Subscriber('', None, self.updateGripperState)
    self.control_pub = rospy.Publisher('', None, queue_size=1)

    self.p_min = p_min
    self.p_max = p_max
    self.state = None

    print('Connecting to gripper controller...')
    while self.control_pub.get_num_connections() == 0 or self.state == None:
      rospy.sleep(0.01)

  def updateGripperState(self, msg):
    ''' Callback for gripper control subscriber.

    Sets the gripper state to the most recent message from the gripper controller.

    Args:
      msg (): The message returned by the gripper controller
    '''
    self.state = msg

  # TODO: We need these two functions w/the robotiq gripper so we might not need them.
  def reset(self):
    ''' Reset the gripper. '''
    pass

  def activate(self):
    ''' Activate the gripper. '''
    pass

  def setPosition(self, pos):
    ''' Set the position of the gripper.

    Args:
      pos (float): The postion to set the gripper to (range [0,1])
    '''
    cmd = GripperCmd()
    cmd.position = (1 - pos) * self.p_max
    self.control_pub.publish(cmd)

    # TODO: We want to return a bool when the motion is done?
    return None

  def openGripper(self):
    ''' Open the gripper to the max amount. '''
    self.setPosition(1)

  def getCurrentPosition():
    ''' Get the current positon of the gripper.

    Returns:
      float: The amoun the gripper is open (range [0,1])
    '''
    return 1 - (self.state.position / self.p_max)
