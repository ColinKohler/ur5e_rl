import tf
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header

class Pose(object):
  ''' Utility class to convert various types of pose formats. '''
  def __init__(self, x, y, z, rx, ry, rz, rw=None):
    self.frame = None #TODO: Update this
    self.pos = [x, y, z]

    if rw:
      self.rot = tf.transformations.euler_from_quaternion(rx, ry, rz, rw)
    else:
      self.rot = rx, ry, rz

  def getPoseMatrix(self):
    return tf.transformations.matrix_from_euler(
      self.pos[0], self.pos[1], self.pos[2], self.rot[0], self.rot[1], self.rot[2]
    )

  def getPosition(self):
    return self.pos

  def getEulerOrientation(self):
    return self.rot

  def getOrientationQuaternion(self):
    return tf.transformations.quaternion_from_euler(*self.rot)

  def getPoseStamped(self):
    return PoseStamped(
      header=Header(frame_id=self.frame),
      pose=Pose(
        position=Point(*self.getPosition()),
        orientation=Quaternion(*self.getOrientationQuaternion())
      )
    )
