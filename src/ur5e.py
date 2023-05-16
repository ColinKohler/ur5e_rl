import rospy
import tf
import numpy as np

from geometry_msgs.msg import PoseStamped

from src.gripper import Gripper
from src.tf_proxy import TFProxy
from src.utils import Pose

class UR5e(object):
  ''' UR5e robotic arm interface.

  Converts higher level commands to the low lever controller.
  '''
  def __init__(self):
    self.pose_cmd_pub = rospy.Publisher('pose_command', PoseStamped, queue_size=1)
    self.ee_pose_sub = rospy.Subscriber('ee_pose', PoseStamped, self.eePoseCallback)
    self.ee_pose = None

    # TODO: Update these
    self.home_joint_pos = (np.pi/180)*np.array([90.0, -120.0, 90.0, -77.0, -90.0, 180.0])

    #self.gripper = Gripper()
    #self.gripper.reset()
    #self.gripper.activate()

    self.tf_proxy = TFProxy()

  def reset(self):
    pass
    #self.moveToHome()
    #self.openGripper()

  def eePoseCallback(self, data):
    self.ee_pose = data

  def moveToPose(self, pose):
    ''' Move the end effector to the specified pose.

    Args:
      - pose (utils.Pose): Pose to move the end effector to.

    Returns:
      bool: True if movement was successful, False otherwise
    '''
    self.pose_cmd_pub.publish(pose.getPoseStamped())

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
    return Pose(pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w)

  def openGripper(self):
    ''' Fully open the gripper. '''
    self.gripper.openGripper()

  def getOpenRatio(self):
    return None

  def sendGripperCmd(self, p):
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
