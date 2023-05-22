import rospy
import numpy as np

from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge

class RGBDSensor(object):
  def __init__(self, vision_size):
    self.vision_size = vision_size

    #self.rgb_sub = rospy.Subscriber('', Image, self.rgbCallback)
    self.depth_sub = rospy.Subscriber('/camera/depth/image', numpy_msg(Image), self.depthCallback, queue_size=1)
    self.bridge = CvBridge()

  def rgbCallback(self, data):
    self.rgb_data = data.data.reshape((3, self.vision_size, self.vision_size))

  def depthCallback(self, data):
    self.depth_data = data

  def getObservation(self):
    #return np.concatenate((self.rgb_data, self.depth_data), axis=0)
    depth = self.bridge.imgmsg_to_cv2(self.depth_data, desired_encoding='passthrough').reshape(-1, self.depth_data.height , self.depth_data.width)
    return depth
