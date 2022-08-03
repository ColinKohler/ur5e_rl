import rospy
import tf

from src.hydrostatic_gripper import HydrostaticGripper
from src.tf_proxy import TFProxy
from src.utils import Pose

class UR5(object):
  ''' UR5 robotic arm interface.

  Converts higher level commands to the low lever controller.
  '''
  def __init__(self):
    # TODO: Update this with the correct service name and message
    self.pose_control = rospy.ServiceProxy('pose_control', POSE_CONTROL)
    self.joint_control = rospy.ServiceProxy('joint_control', JOINT_CONTROL)

    # TODO: These are from the old UR5 we might need something different
    self.home_joint_pos = [-0.22163755, -1.48887378,  1.81927061, -1.90511448, -1.5346511, 1.3408314]

    self.gripper = HydrostaticGripper()
    self.gripper.reset()
    self.gripper.activate()

    self.holding_state = 0
    self.tf_proxy = TFProxy()

  def moveToPose(self, pose):
    ''' Move the end effector to the specified pose.

    Args:
      - pose (utils.Pose): Pose to move the end effector to.

    Returns:
      bool: True if movement was successful, False otherwise
    '''
    req = POSE_CONTROL()
    resp = self.pose_control.publish(req)

    return resp

  def moveToJoint(self, joint):
    ''' Move the robot to the specified joint positions.

    Args:
      - joint (list[float]): Joint positions to set the robot to.

    Returns:
      bool: True if movement was successful, False otherwise
    '''
    req = JOINT_CONTROL()
    resp = self.joint_control.publish(req)

    return resp

  def moveToHome(self):
    ''' Moves the robot to the home position.

    Returns:
      bool: True if movement was successful, False otherwise
    '''
    return self.moveToJoint(self.home_joint_pos)

  def getEEPose(self):
    ''' Get the current pose of the end effector.

    Position: [x, y, z] | Orientation: [x, y, z, q]

    Returns:
      (list[float], list[float]): (End effector position, end effector orientation)
    '''
    # TODO: Update 'tool_frame' to what ever the gripper frame is
    T = self.tf_proxy.lookupTransform('base', 'tool_frame')
    pos = T[:3, 3]
    rot = tf.trnasformations.euler_from_matrix(T)

    return Pose(*pos, *rot)

  def openGripper(self):
    ''' Fully open the gripper.

    Returns:

    '''
    self.gripper.openGripper()

  def controlGripper(self, p):
    ''' Send a position command to the gripper.

    Args:
      p (float): The position (range [0,1]) to set the gripper to.

    Returns:

    '''
    self.gripper.setPosition(p)

  def getGripperState(self):
    ''' Get the current state of the gripper.

    Returns:
      float: The current state of the gripper (range [0,1]).
    '''
    return self.gripper.getCurrentPosition()
