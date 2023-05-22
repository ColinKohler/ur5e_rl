import rospy
import tf
import numpy as np
import copy
import time
from scipy.interpolate import InterpolatedUnivariateSpline

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from src.gripper import Gripper
from src.tf_proxy import TFProxy
from src.utils import Pose

class UR5e(object):
  ''' UR5e robotic arm interface.

  Converts higher level commands to the low lever controller.
  '''
  def __init__(self):
    self.joint_cmd_pub = rospy.Publisher('joint_command', JointState, queue_size=1)
    self.pose_cmd_pub = rospy.Publisher('pose_command', PoseStamped, queue_size=1)
    self.ee_pose_sub = rospy.Subscriber('ee_pose', PoseStamped, self.eePoseCallback)
    self.joint_sub = rospy.Subscriber("joint_states", JointState, self.jointStateCallback)

    self.ee_pose = None
    self.joint_state = None
    self.joint_names = ['shoulder_lift_joint', 'shoulder_pan_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    self.joint_positions = np.zeros(6)
    self.joint_reorder = [2,1,0,3,4,5]

    self.home_joint_pos = (np.pi/180)*np.array([75.0, -85.0, 90.0, -95.0, -90.0, 160.0])
    self.home_joint_state = JointState(
      position=self.home_joint_pos,
      velocity=[0] * 6
    )

    self.gripper = Gripper()
    #self.gripper.reset()
    #self.gripper.activate()

    self.tf_proxy = TFProxy()

  def reset(self):
    self.moveToHome()
    #self.openGripper()

  def eePoseCallback(self, data):
    self.ee_pose = data

  def jointStateCallback(self, data):
    self.joint_positions[self.joint_reorder] = data.position
    self.joint_state = data

  def moveToPose(self, pose):
    ''' Move the end effector to the specified pose.

    Args:
      - pose (utils.Pose): Pose to move the end effector to.
    '''
    self.pose_cmd_pub.publish(pose.getPoseStamped())

  def moveToHome(self):
    ''' Moves the robot to the home position. '''
    current_joint_pos = copy.copy(self.joint_positions)
    target_joint_pos = self.home_joint_pos

    speed = 0.1
    max_disp = np.max(np.abs(target_joint_pos-current_joint_pos))
    end_time = max_disp / speed

    traj = [InterpolatedUnivariateSpline([0.,end_time],[current_joint_pos[i], target_joint_pos[i]],k=1) for i in range(6)]
    traj_vel = InterpolatedUnivariateSpline([0.,end_time/2, end_time], [0, 0.1, 0],k=1)
    start_time, loop_time = time.time(), 0
    while loop_time < end_time:
      loop_time = time.time() - start_time
      joint_state = JointState(
        position=[traj[j](loop_time) for j in range(6)],
        velocity=[traj_vel(loop_time)] * 6,
      )
      self.joint_cmd_pub.publish(joint_state)

    joint_state = JointState(
      position=[traj[j](loop_time) for j in range(6)],
      velocity=[0] * 6
    )
    self.joint_cmd_pub.publish(joint_state)

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
