import rospy
import numpy as np

import skimage
from scipy.ndimage import rotate
import copy
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge

class RGBDSensor(object):
  def __init__(self, vision_size):
    self.vision_size = vision_size

    #self.rgb_sub = rospy.Subscriber('', Image, self.rgbCallback)
    self.depth_sub = rospy.Subscriber('/camera/depth/image', numpy_msg(Image), self.depthCallback, queue_size=1)
    self.bridge = CvBridge()

    # Wait for subscriber to get data
    self.depth_data = None
    print('Waiting for vision data...')
    while self.depth_data is None:
      rospy.sleep(0.1)

  def rgbCallback(self, data):
    self.rgb_data = data.data.reshape((3, self.vision_size, self.vision_size))

  def depthCallback(self, data):
    self.depth_data = data

  def getObservation(self):
    #return np.concatenate((self.rgb_data, self.depth_data), axis=0)
    depth = copy.copy(self.bridge.imgmsg_to_cv2(self.depth_data, desired_encoding='passthrough').reshape(-1, self.depth_data.height , self.depth_data.width).squeeze())

    # Process nans
    mask = np.isnan(depth)
    depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])

    # Resize image and rotate
    depth = skimage.transform.resize(depth, (self.vision_size, self.vision_size))
    depth = rotate(depth, 180)

    # Remove depth past the table
    table_mask = depth > 1.0
    depth[table_mask] = 1

    return depth.reshape(1, self.vision_size, self.vision_size)
