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

    self.obs_pub = rospy.Publisher('/ur5e_rl/rgb_obs', Image, queue_size=1)
    self.rgb_sub = rospy.Subscriber('/camera2/color/image_raw', numpy_msg(Image), self.rgbCallback, queue_size=1)
    self.depth_sub = rospy.Subscriber('/camera2/depth/image_rect_raw', numpy_msg(Image), self.depthCallback, queue_size=1)
    self.bridge = CvBridge()

    self.depth_min = 0.0
    self.depth_max = 0.02

    # Wait for subscriber to get data
    self.rgb_data = None
    self.depth_data = None
    print('Waiting for vision data...')
    while self.depth_data is None or self.rgb_data is None:
      rospy.sleep(0.1)

  def rgbCallback(self, data):
    self.rgb_data = data

  def depthCallback(self, data):
    self.depth_data = data

  def getObservation(self):
    depth = copy.copy(self.bridge.imgmsg_to_cv2(self.depth_data, desired_encoding='passthrough').reshape(-1, self.depth_data.height , self.depth_data.width).squeeze())
    rgb = copy.copy(self.bridge.imgmsg_to_cv2(self.rgb_data, desired_encoding='passthrough'))

    # Process nans
    mask = np.isnan(depth)
    depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])

    # Resize images
    depth = skimage.transform.resize(depth[65:-65,125:-125], (self.vision_size, self.vision_size))
    rgb = skimage.transform.resize(rgb[65:-65,125:-125,:], (self.vision_size, self.vision_size))

    # Process depth image
    depth[depth > self.depth_max] = self.depth_max
    depth = (depth - self.depth_min) / (self.depth_max - self.depth_min)

    # Publish RGB obs for debugging
    self.obs_pub.publish(self.bridge.cv2_to_imgmsg(rgb))

    # Reshape and concatenate
    depth = depth.reshape(1, self.vision_size, self.vision_size)
    rgb = rgb.transpose(2,0,1)
    rgbd = np.concatenate((rgb, depth), axis=0)

    return rgbd
