import rospy
import numpy as np

from geometry_msgs.msg import WrenchStamped

class ForceSensor(object):
  def __init__(self, force_obs_len):
    self.force_obs_len = force_obs_len

    self.initial_force = None
    self.force_history = [[0, 0, 0, 0, 0, 0]] * self.force_obs_len
    self.wrench_sub = rospy.Subscriber('wrench', WrenchStamped, self.wrenchCallback)

  # TODO: Check frame for this message, need to be in global (world) frame
  def wrenchCallback(self, data):
    current_wrench = np.array([
      data.wrench.force.x,
      data.wrench.force.y,
      data.wrench.force.z,
      data.wrench.torque.x,
      data.wrench.torque.y,
      data.wrench.torque.z
    ])

    # TODO: Might need to take a average to get good zero'ing
    if self.initial_force is None:
      self.initial_force = current_wrench

    self.force_history.append(current_wrench - self.initial_force)

  def reset(self):
    self.initial_force = None
    self.force_history = [[0, 0, 0, 0, 0, 0]] * self.force_obs_len

  def getObservation(self):
    return np.array(self.force_history[-self.force_obs_len:])
