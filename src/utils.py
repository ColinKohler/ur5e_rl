import tf
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header

class Pose(object):
  ''' Utility class to convert various types of pose formats. '''
  def __init__(self, x, y, z, rx, ry, rz, rw=None):
    self.frame = None #TODO: Update this
    self.x, self.y, self.z = x, y, z

    if rw:
      self.rx, self.ry, self.rz = tf.transformations.euler_from_quaternion(rx, ry, rz, rw)
    else:
      self.rx, self.ry, self.rz = rx, ry, rz

  def getPoseMatrix(self):
    return tf.transformations.matrix_from_euler(self.x, self.y, self.z, self.rz, self.ry, self.rz)

  def getPosition(self):
    return self.x, self.y, self.z

  def getEulerOrientation(self):
    return self.rx, self.ry, self.rz

  def getOrientationQuaternion(self):
    return tf.transformations.quaternion_from_euler(self.rx, self.ry, self.rz)

  def getPoseStamped(self):
    return PoseStamped(
      header=Header(frame_id=self.frame),
      pose=Pose(
        position=Point(*self.getPosition()),
        orientation=Quaternion(*self.getOrientationQuaternion())
      )
    )
