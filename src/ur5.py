import rospy

from src.hydrostatic_gripper import HydrostaticGripper
from src.tf_proxy import TFProxy

class UR5(object):
  def __init__(self):
    self.control_pub = rospy.Publisher()
    self.home_joint_pos = list()

    self.gripper = HydrostaticGripper()
    self.gripper.reset()
    self.gripper.activate()

    self.holding_state = 0
    self.tf_proxy = TFProxy()

  # TODO: Move the end effector to the specified pose by sending command to controller
  def moveToPose(self, pose):
    pass

  # TODO: Move the ur5 to its home position, probably want to use joint positions here
  def moveToHome(self):
    pass

  def getEEPose(self):
    return self.gripper.getPose()

  def getEEPos(self):
    return self.getEEPose()[0]

  def getEEOrientation(self):
    rot_q =  self.getEEPose()[1]
    return tf.transformation.euler_from_quaternion(rot_q)

  # TODO: Open gripper to the maximum amount
  def openGripper(self):
    pass

  # TODO: Agent will give [0,1] for how open the gripper will be, need to transform this to control somewhere
  def controlGripper(self, p):
    self.gripper.setPosition(p)

  # TODO: Ensure gripper state is [0,1] for agent
  def getGripperState(self):
    return self.gripper.state
