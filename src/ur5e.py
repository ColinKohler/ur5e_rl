import sys
import rospy
import tf
import numpy as np
import copy
import time
from scipy import interpolate
import moveit_commander

from moveit_msgs.msg import DisplayTrajectory
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from src.gripper import Gripper
from src.tf_proxy import TFProxy
from src.utils import Pose
from src import utils

class UR5e(object):
  ''' UR5e robotic arm interface.

  Converts higher level commands to the low lever controller.
  '''
  def __init__(self):
    self.joint_cmd_pub = rospy.Publisher('joint_command', JointState, queue_size=1)
    self.reset_wrench_pub = rospy.Publisher('reset_wrench', Bool, queue_size=1)
    self.joint_sub = rospy.Subscriber("joint_states", JointState, self.jointStateCallback)

    self.ee_pose = None
    self.joint_state = None
    self.joint_names = ['shoulder_lift_joint', 'shoulder_pan_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    self.joint_positions = np.zeros(6)
    self.joint_velocities = np.zeros(6)
    self.joint_reorder = [2,1,0,3,4,5]

    self.home_joint_pos = (np.pi/180)*np.array([76., -84., 90., -96., -90., 165.])
    self.home_pose = Pose(0, 0.55, 0.2+0.12, 0.5, 0.5, -0.5, 0.5)
    self.offset_home_joint_pos = (np.pi/180)*np.array([120., -84., 90., -96., -90., 165.])
    self.max_joint_disp = np.array([0.2, 0.2, 0.2, 0.4, 0.4, 0.6])

    self.gripper = Gripper()
    self.gripper.reset()
    self.gripper.activate()

    self.pick_offset = 0.1
    self.place_offset = 0.1
    self.gripper_offset = 0.10

    self.tf_proxy = TFProxy()

    # MoveIt
    self.group_name = 'ur5e_arm'
    self.ee_link = 'ee_link'
    moveit_commander.roscpp_initialize(sys.argv)

    # MoveIt Group
    self.moveit_group = moveit_commander.MoveGroupCommander(self.group_name)
    self.moveit_group.set_planning_time = 1.0
    #self.moveit_group.set_goal_position_tolerance(0.01)
    #self.moveit_group.set_goal_orientation_tolerance(0.01)

    # MoveIt Planning Scene
    self.moveit_scene = moveit_commander.PlanningSceneInterface()

    table_pose = PoseStamped()
    table_pose.header.frame_id = "base_link"
    table_pose.pose.orientation.w = 1.0
    table_pose.pose.position.y = 0.6
    table_pose.pose.position.z = -0.10
    self.moveit_scene.add_box('table', table_pose, size=(0.75, 1.2, 0.05))

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

  def moveToHome(self):
    ''' Moves the robot to the home position. '''
    #self.moveToJointPose(self.home_joint_pos)
    self.moveToPose(self.home_pose)

  def moveToOffsetHome(self):
    ''' Moves the robot to the offset home position. '''
    self.moveToJointPose(self.offset_home_joint_pos)

  def pick(self, pose):
    pick_pose = copy.deepcopy(pose)
    pick_pose.pos[2] = pose.pos[2] + self.gripper_offset

    pre_pick_pose = copy.deepcopy(pose)
    pre_pick_pose.pos[2] = pre_pick_pose.pos[2] + self.gripper_offset + self.pick_offset

    self.gripper.open(speed=100)
    self.gripper.waitUntilNotMoving()

    self.moveToPose(pre_pick_pose)
    self.moveToPose(pick_pose)

    self.gripper.close(force=1)
    self.gripper.waitUntilNotMoving()

    self.moveToPose(pre_pick_pose)
    self.moveToHome()

  def place(self, pose):
    place_pose = copy.deepcopy(pose)
    place_pose.pos[2] = pose.pos[2] + self.gripper_offset

    pre_place_pose = copy.deepcopy(pose)
    pre_place_pose.pos[2] = pre_place_pose.pos[2] + self.gripper_offset + self.place_offset

    self.moveToPose(pre_place_pose)
    self.moveToPose(place_pose)

    self.gripper.open(speed=100)
    self.gripper.waitUntilNotMoving()

    self.moveToPose(pre_place_pose)
    self.moveToHome()

  def reach(self, pose):
    place_pose = copy.deepcopy(pose)
    place_pose.pos[2] = pose.pos[2] + self.gripper_offset

    pre_place_pose = copy.deepcopy(pose)
    pre_place_pose.pos[2] = pre_place_pose.pos[2] + self.gripper_offset + self.place_offset

    self.moveToPose(pre_place_pose)
    self.moveToHome()

  def moveToPose(self, pose):
    ''' Move the end effector to the specified pose.

    Args:
      - pose (utils.Pose): Pose to move the end effector to.
    '''
    self.moveit_group.clear_pose_targets()
    self.moveit_group.set_pose_target(pose.getPoseStamped())
    success, traj, planning_time, err = self.moveit_group.plan()
    if not success:
      return False

    self.moveToJointTraj(traj.joint_trajectory)
    return True

  def moveToJointPose(self, joint_pos):
    ''' Move the end effector to the specified pose.

    Args:
      - pose (utils.Pose): Pose to move the end effector to.
    '''
    self.moveit_group.clear_pose_targets()
    self.moveit_group.set_joint_value_target(joint_pos)
    success, traj, planning_time, err = self.moveit_group.plan()
    self.moveToJointTraj(traj.joint_trajectory)

  def moveToJointTraj(self, traj):
    for pos in traj.points:
      self.moveTo(pos.positions, pos.velocities)
    self.waitUntilNotMoving()

  def moveTo(self, joint_pos, joint_vel):
    current_joint_pos = copy.deepcopy(self.joint_positions)
    current_joint_vel = copy.deepcopy(self.joint_velocities)

    joint_disp = np.abs(joint_pos - current_joint_pos)
    max_disp = np.max(joint_disp)
    num_steps = max(int(max_disp // 0.05), 2)
    #print('{:.3f} -> {}'.format(max_disp, num_steps))

    #if np.any(np.array(joint_disp) > self.max_joint_disp):
    #  rospy.logerr('Requested movement is too large: {}.'.format(joint_disp))
    #  return

    traj_pos = np.linspace(current_joint_pos, joint_pos, num_steps)[1:]
    traj_vel = np.linspace(current_joint_vel, joint_vel, num_steps)[1:]
    for point, vel in zip(traj_pos, traj_vel):
      joint_state = JointState(
        position=point,
        velocity=vel
      )
      self.joint_cmd_pub.publish(joint_state)
      time.sleep(0.008)

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

  def getEEPose(self):
    ''' Get the current pose of the end effector. '''
    return utils.convertTfToPose(self.tf_proxy.lookupTransform('base_link', 'ee_link'))

  def jointStateCallback(self, data):
    self.joint_positions[self.joint_reorder] = data.position
    self.joint_velocities[self.joint_reorder] = data.velocity
    self.joint_state = data

  def waitUntilNotMoving(self):
    while True:
      prev_joint_position = self.joint_positions.copy()
      rospy.sleep(0.2)
      if np.allclose(prev_joint_position, self.joint_positions, atol=1e-3):
        break
