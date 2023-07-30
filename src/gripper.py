import rospy

from robotiq_2f_gripper_control.msg._Robotiq2FGripper_robot_output import Robotiq2FGripper_robot_output as gripperCmd
from robotiq_2f_gripper_control.msg._Robotiq2FGripper_robot_input import Robotiq2FGripper_robot_input as gripperStatus

class Gripper(object):
  ''' Interface for gripper.

  Args:
    p_min (float): The min joint angle for the gripper.
    p_max (float): The max joint angle for the gripper.
  '''
  def __init__(self, p_min=0, p_max=255):
    self.sub = rospy.Subscriber(
      'Robotiq2FGripperRobotInput',
      gripperStatus,
      self.updateGripperState
    )
    self.pub = rospy.Publisher(
      "Robotiq2FGripperRobotOutput",
      gripperCmd,
      queue_size=1
    )
    self.p_min = p_min
    self.p_max = p_max
    self.status = None

    print('Connecting to gripper controller...')
    while self.pub.get_num_connections() == 0 or self.status == None:
      rospy.sleep(0.01)

    if self.status.gACT == 0:
      self.reset()
      self.activate()

  def updateGripperState(self, msg):
    ''' Callback for gripper control subscriber. '''
    self.status = msg

  def reset(self):
    ''' Reset the gripper. '''
    cmd = gripperCmd()
    cmd.rACT = 0

    self.pub.publish(cmd)
    rospy.sleep(0.5)

  def activate(self):
    ''' Activate the gripper. '''
    cmd = gripperCmd()
    cmd.rACT = 1
    cmd.rGTO = 1
    cmd.rSP = 255
    cmd.rFR = 150

    self.pub.publish(cmd)
    rospy.sleep(0.5)

  def setPosition(self, pos, force=255, speed=255):
    ''' Set the position of the gripper.

    Args:
      pos (float): The postion to set the gripper to (range [0,1])
    '''
    cmd = gripperCmd()
    cmd.rACT = 1
    cmd.rGTO = 1
    cmd.rPR = int((1 - pos) * self.p_max)
    cmd.rFR = force
    cmd.rSP = speed
    self.pub.publish(cmd)

  def open(self, force=255, speed=255):
    ''' Open the gripper to the max amount. '''
    self.setPosition(1, force=force, speed=speed)

  def close(self, force=255, speed=255):
    ''' Close the gripper to the min amount. '''
    self.setPosition(0, force=force, speed=speed)

  def getCurrentPosition(self):
    ''' Get the current positon of the gripper.

    Returns:
      float: The amoun the gripper is open (range [0,1])
    '''
    return 1 - (self.status.gPO / self.p_max)

  def isClosed(self):
    return self.status.gPO > 220

  def waitUntilNotMoving(self, max_it=5):
    prev_pos = self.status.gPO
    for i in range(max_it):
      rospy.sleep(0.2)
      curr_pos = self.status.gPO
      if prev_pos == curr_pos:
        return
      prev_pos = curr_pos
  
  # status not giving gripper force rFR or rFO
  def getForce(self):
    return self.status
