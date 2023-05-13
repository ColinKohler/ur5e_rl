import rospy
from sensor_msgs.msg import Image

class RGBDSensor(object):
  def __init__(self, vision_size):
    self.vision_size = vision_size

    self.rgb_sub = rospy.Subscriber('', Image, rgbCallback)
    self.depth_sub = rospy.Subscriber('', Image, depthCallback)

  def rgbCallback(self, data):
    self.rgb_data = data.data.reshape((3, self.vision_size, self.vision_size))

  def depthCallback(self, data):
    self.depth_data = data.data.reshape((1, self.vision_size, self.vision_size))

  def getObservation(self):
    return np.concatenate((self.rgb_data, self.depth_data), axis=0)
