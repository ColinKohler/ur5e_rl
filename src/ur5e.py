import sys
import rospy
import tf
import numpy as np
import copy
import time
from scipy.interpolate import InterpolatedUnivariateSpline
import moveit_commander

from moveit_msgs.msg import DisplayTrajectory
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

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
    self.reset_wrench_pub = rospy.Publisher('reset_wrench', Bool, queue_size=1)
    self.ee_pose_sub = rospy.Subscriber('ee_pose', PoseStamped, self.eePoseCallback)
    self.joint_sub = rospy.Subscriber("joint_states", JointState, self.jointStateCallback)

    self.ee_pose = None
    self.joint_state = None
    self.joint_names = ['shoulder_lift_joint', 'shoulder_pan_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    self.joint_positions = np.zeros(6)
    self.joint_reorder = [2,1,0,3,4,5]

    self.home_joint_pos = (np.pi/180)*np.array([76., -84., 90., -96., -90., 165.])
    self.home_pose = Pose(0, 0.55, 0.25, 0.5, 0.5, -0.5, 0.5)
    self.offset_home_joint_pos = (np.pi/180)*np.array([124., -84., 90., -96., -90., 165.])

    self.gripper = Gripper()
    self.gripper.reset()
    self.gripper.activate()
    self.action_sleep = 1.5

    self.pick_offset = 0.1
    self.place_offset = 0.1
    self.gripper_offset = 0.15

    self.tf_proxy = TFProxy()

    # MoveIt
    self.group_name = 'ur5e_arm'
    self.ee_link = 'ee_link'
    moveit_commander.roscpp_initialize(sys.argv)

    # MoveIt Group
    self.moveit_group = moveit_commander.MoveGroupCommander(self.group_name)
    self.moveit_group.set_planning_time = 0.1
    self.moveit_group.set_goal_position_tolerance(0.01)
    self.moveit_group.set_goal_orientation_tolerance(0.01)

    # MoveIt Planning Scene
    self.moveit_scene = moveit_commander.PlanningSceneInterface()

    table_pose = PoseStamped()
    table_pose.header.frame_id = "base_link"
    table_pose.pose.orientation.w = 1.0
    table_pose.pose.position.y = 0.6
    table_pose.pose.position.z = -0.03
    self.moveit_scene.add_box('table', table_pose, size=(0.75, 1.2, 0.06))

    roof_pose = PoseStamped()
    roof_pose.header.frame_id = "base_link"
    roof_pose.pose.orientation.w = 1.0
    roof_pose.pose.position.y = 0.6
    roof_pose.pose.position.z = 0.8
    self.moveit_scene.add_box('roof', roof_pose, size=(0.75, 1.2, 0.06))

  def reset(self):
    self.moveToHome()
    self.reset_wrench_pub.publish(Bool(True))
    self.openGripper()

  def eePoseCallback(self, data):
    self.ee_pose = data

  def jointStateCallback(self, data):
    self.joint_positions[self.joint_reorder] = data.position
    self.joint_state = data

  def waitUntilNotMoving(self):
    while True:
      prev_joint_position = self.joint_positions.copy()
      rospy.sleep(0.2)
      if np.allclose(prev_joint_position, self.joint_positions, atol=1e-3):
        break

  def moveToPose(self, pose):
    ''' Move the end effector to the specified pose.

    Args:
      - pose (utils.Pose): Pose to move the end effector to.
    '''
    self.pose_cmd_pub.publish(pose.getPoseStamped())
    self.waitUntilNotMoving()

  def pick(self, pose):
    pose.pos[2] = pose.pos[2] + self.gripper_offset
    pre_pose = copy.copy(pose)
    pre_pose.pos[2] = pre_pose.pos[2] + self.pick_offset

    self.moveit_group.clear_pose_targets()
    self.moveit_group.set_pose_target(pre_pose.getPoseStamped())
    success, traj, planning_time, err = self.moveit_group.plan()

    input('move')

    self.moveToJointTraj(traj.joint_trajectory)

    #self.moveit_group.clear_pose_targets()
    #self.moveit_group.set_pose_target(pose.getPoseStamped())
    #success, traj, planning_time, err = self.moveit_group.plan()

    #self.moveToPose(pre_pose)
    #self.moveToPose(pose)
    #self.gripper.close(force=1)
    #self.moveToPose(pre_pose)
    #self.moveToHome()

  def place(self, pose):
    pre_pose = copy.copy(pose)
    pre_pose.z += self.pick_offset

    self.moveToPose(pre_pose)
    self.moveToPose(pose)
    self.gripper.open(speed=100)
    self.moveToPose(pre_pose)
    self.moveToHome()

  def moveToJointTraj(self, traj):
    for pos in traj.points:
      joint_state = JointState(
        position=pos.positions,
        velocity=(np.array(pos.velocities) * 0.1).tolist(),
      )
      self.joint_cmd_pub.publish(joint_state)
    self.waitUntilNotMoving()

  def moveToHome(self):
    ''' Moves the robot to the home position. '''
    self.moveit_group.clear_pose_targets()
    self.moveit_group.set_joint_value_target(self.home_joint_pos)
    success, traj, planning_time, err = self.moveit_group.plan()

    print(len(traj.joint_trajectory.points))

    input('move')

    self.moveToJointTraj(traj.joint_trajectory)

  def moveToOffsetHome(self):
    ''' Moves the robot to the offset home position. '''
    self.moveit_group.clear_pose_targets()
    self.moveit_group.set_joint_value_target(self.offset_home_joint_pos)
    success, traj, planning_time, err = self.moveit_group.plan()

    print(len(traj.joint_trajectory.points))

    input('move')

    self.moveToJointTraj(traj.joint_trajectory)

  def getEEPose(self):
    ''' Get the current pose of the end effector. '''

    pos = self.ee_pose.pose.position
    rot = self.ee_pose.pose.orientation
    return Pose(pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w)

  def openGripper(self):
    self.gripper.open()

  def closeGripper(self):
    self.gripper.close()

  def sendGripperCmd(self, p):
    ''' Send a position command to the gripper. '''
    self.gripper.setPosition(p)

  def getGripperState(self):
    ''' Get the current state of the gripper. '''
    return self.gripper.getCurrentPosition()
