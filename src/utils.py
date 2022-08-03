import tf

class Pose(object):
  ''' Utility class to convert various types of pose formats. '''
  def __init__(self, x, y, z, rx, ry, rz):
    self.x, self.y, self.z = x, y, z
    self.rx, self.ry, self.rz = rx, ry, rz

  def getPoseMatrix(self):
    return tf.transformations.matrix_from_euler(self.x, self.y, self.z, self.rz, self.ry, self.rz)

  def getPosition(self):
    return self.x, self.y, self.z

  def getEulerOrientation(self):
    return self.rx, self.ry, self.rz

  def getOrientationQuaternion(self):
    return tf.transformations.quaternion_from_euler(self.rx, self.ry, self.rz)
