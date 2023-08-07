import rospy
import numpy as np
import tf2_geometry_msgs

from src.utils import PythonBPF
from geometry_msgs.msg import WrenchStamped

class ForceSensor(object):
  def __init__(self, force_obs_len, tf_proxy):
    self.force_obs_len = force_obs_len
    self.tf_proxy = tf_proxy

    # define bandpass filter parameters
    self.fl = [0.0] * 6
    self.fh = [30.0] * 6
    self.fs = 500
    self.filter = PythonBPF(self.fs, self.fl, self.fh)

    self.initial_force = None
    self.force_history = [[0, 0, 0, 0, 0, 0]] * self.force_obs_len
    self.wrench_sub = rospy.Subscriber('wrench', WrenchStamped, self.wrenchCallback)

  def wrenchCallback(self, data):
    transform = self.tf_proxy.lookupTransform('base_link', data.header.frame_id)
    data = tf2_geometry_msgs.do_transform_wrench(data, transform)

    current_wrench = np.array([
      data.wrench.force.x,
      data.wrench.force.y,
      data.wrench.force.z,
      data.wrench.torque.x,
      data.wrench.torque.y,
      data.wrench.torque.z
    ])

    if self.initial_force is None:
      self.initial_force = current_wrench
      self.filter.calculate_initial_values(current_wrench)

    self.force_history.append(np.array(self.filter.filter(current_wrench)))

  def reset(self):
    self.filter = PythonBPF(self.fs, self.fl, self.fh)
    self.initial_force = None
    self.force_history = [[0, 0, 0, 0, 0, 0]] * self.force_obs_len

  def getObservation(self):
    obs = np.array(self.force_history[-self.force_obs_len:])
    #obs[:] -= self.initial_force
    obs[:,2] -= self.initial_force[2]
    obs[:,5] -= self.initial_force[5]

    return obs
