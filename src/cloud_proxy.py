import tf
import tf2_ros
import rospy
import ros_numpy
import numpy as np
import open3d
import scipy.interpolate

from sensor_msgs.msg import Image
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


class CloudProxy(object):
  def __init__(self):
    self.sensor_1 = '/camera/depth/points'
    self.sensor_1_sub = rospy.Subscriber(self.sensor_1, PointCloud2, self.sensorCallback1, queue_size=1)
    self.cloud_1 = None

    self.sensor_2 = '/camera/depth/points'
    self.sensor_2_sub = rospy.Subscriber(self.sensor_2, PointCloud2, self.sensorCallback2, queue_size=1)
    self.cloud_2 = None

    self.tf_buffer = tf2_ros.Buffer()
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

  def getHeightmap(self, target_size, img_size, gripper_pos=[-0.5, 0, 0.1]):
    while self.cloud_1 is None or self.cloud_2 is None:
      rospy.sleep(0.1)

    cloud = np.concatenate((self.cloud_1, self.cloud_2))
    # Filter workspace and arm/gripper
    # TODO: Update these to be accurate to our setup
    cloud = cloud[(cloud[:,2] > -0.2) * (cloud[:,0] < -0.23) * (cloud[:,0] > -0.8) * (np.abs(cloud[:,1]) < 0.3)]
    cloud = cloud[(cloud[:, 2] < max(gripper_pos[2] - 0.04, 0))]

    # Remove outliers from point cloud
    pcd = open2d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cloud)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    cloud = np.asarray(cl.points)

    # Create view matrix at gripper position
    view_matrix = transformation.euler_matrix(0, np.pi, 0).dot(np.eye(4))
    view_matrix[:3, 3] = [gripper_pos[0], -gripper_pos[1], gripper_pos[2]]
    view_matrix = transformation.euler_matrix(0, 0, -np.pi/2).dot(view_matrix)

    # Transform points onto view matrix
    augment = np.ones((1, cloud.shape[0]))
    pts = np.concatentat((cloud.T, augment), axis=0)
    projection_matrtix = np.array([
      [1 / (target_size / 2), 0,                     0, 0],
      [0,                     1 / (target_size / 2), 0, 0],
      [0,                     0,                     1, 0],
      [0,                     0,                     0, 1]
    ])
    tran_world_pix = np.matmul(projection_matrix, view_matrix)
    pts = np.matmul(tran_world_pix, pts)
    pts[0] = np.round_((pts[0] + 1) * img_size / 2, 2)
    pts[1] = np.round_((pts[1] + 1) * img_size / 2, 2)

    # NOTE: This is doing some sorting junk that I don't quite understand atm
    mask = (pts[0] >= 0) * (pts[0] < img_size) * (pts[1] > 0) * (pts[1] < img_size)
    pts = pts[:, mask]
    min_xy = (pts[1].astype(int) * img_size + pts[0].astype(int))
    ind = np.lexsort(np.stack((pts[2], min_xy)))
    bincount = np.bincount(min_xy)
    cumsum = np.cumsum(bincount)
    cumsum = np.roll(cumsum, 1)
    cumsum[cumsum == np.roll(cumsum, -1)] = 0
    cumsum = np.concatenate((cumsum, -1 * np.ones(img_size * img_size - cumsum.shape[0]))).astype(int)

    # Reconstruct heightmap
    depth = pts[2][ind][cumsum]
    depth[cumsum == 0] = np.nan
    depth = depth.reshape(img_size, img_size)
    return self.interpolate()

  def interpolate(self, depth):
    mask = np.logical_not(np.isnan(depth))
    xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

    data = np.ravel(depth[:, :][mask])
    interp = scipy.interpolate.NearestNDInterpolator(xym, data)
    return interp(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

  def sensorCallback1(self, msg):
    self.cloud_1 = self.processPointCloud(msg)

  def sensorCallback2(self, msg):
    self.cloud_2 = self.processPointCloud(msg)

  def processPointCloud(self, msg):
    cloud_time = msg.header.stamp
    cloud_frame = msg.header.frame_id

    cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
    mask = np.logical_not(np.isnan(cloud).any(axis=1))
    cloud = cloud[mask]

    cTb = self.lookupTransform(cloud_frame, 'base', rospy.Time(0))
    return self.transform(cloud, cTb)

  def transform(self, cloud, T, is_position=True):
    n = cloud.shape[0]

    cloud = cloud.T
    augment = np.ones((1, n)) if is_position else np.zeros((1, n))
    cloud = np.concatenate((cloud, augment), axis=0)
    cloud = np.dot(T, cloud)

    return cloud[:3, :].T

  def lookupTransform(self, from_frame, to_frame, lookup_time=rospy.Time(0)):
    transform_msg = self.tf_buffer.lookup_transform(to_frame, from_frame, lookup_time, rospy.Duration(1.0))
    translation = transform_msg.transform.translation
    rotation = transform_msg.transform.rotation

    pos = [translation.x, translation.y, translation.z]
    quat = [rotation.x, rotation.y, rotation.z, rotation.w]
    T = tf.transformations.quaternion_matrix(quat)
    T[0:3, 3] = pos

    return T
