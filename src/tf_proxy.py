import rospy
import tf
import tf2_ros

class TFProxy(object):
  def __init__(self):
    self.tf_buffer = tf2_ros.Buffer()
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

  def lookupTransform(self, from_frame, to_frame, lookup_time=rospy.Time(0)):
    transform_msg = self.tf_buffer.lookup_transform(from_frame, to_frame, lookup_time, rospy.Duration(1.0))
    return transform_msg
