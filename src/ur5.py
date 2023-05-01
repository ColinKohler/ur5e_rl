import rospy
import tf

from src.gripper import Gripper
from src.tf_proxy import TFProxy
from src.utils import Pose

class UR5(object):
  ''' UR5 robotic arm interface.

  Converts higher level commands to the low lever controller.
  '''
  def __init__(self):
    self.pose_cmd_pub = rospy.Publisher('compliant_controller/pose_command', PoseStamped, queue_size=1)
    self.ee_pose_sub = rospy.Subscriber('compliant_controller/ee_pose', PoseStamped, eePoseCallback)
    self.ee_pose = None

    # TODO: These are from the old UR5 we might need something different
    self.home_joint_pos = [-0.22163755, -1.48887378,  1.81927061, -1.90511448, -1.5346511, 1.3408314]

    self.gripper = Gripper()
    self.gripper.reset()
    self.gripper.activate()

    self.tf_proxy = TFProxy()

  def eePoseCallback(self, data):
    self.ee_pose = data

  def moveToPose(self, pose):
    ''' Move the end effector to the specified pose.

    Args:
      - pose (utils.Pose): Pose to move the end effector to.

    Returns:
      bool: True if movement was successful, False otherwise
    '''
    ee_pose_pub.publish(pose.getPoseStamped())

  def moveToHome(self):
    ''' Moves the robot to the home position.

    Returns:
      bool: True if movement was successful, False otherwise
    '''
    return self.moveToJoint(self.home_joint_pos)

  def getEEPose(self):
    ''' Get the current pose of the end effector. '''
    pos = self.ee_pose.pose.position
    rot = self.ee_pose.pose.orientation
    return Pose(*pos, *rot)

  def openGripper(self):
    ''' Fully open the gripper. '''
    self.gripper.openGripper()

  def getOpenRatio(self):
    return None

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
