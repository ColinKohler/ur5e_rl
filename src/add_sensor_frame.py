import rospy
import tf

if __name__ == '__main__':
  rospy.init_node('add_sensor_frame')
  br = tf.TransformBroadcaster()
  rate = rospy.Rate(50.0)

  while not rospy.is_shutdown():
    # TODO: Update these to the sensors we use
    br.sendTransform((0.503, 0.547, 0.762), (-0.2611, 0.649, -0.2638, 0.6641), rospy.Time.now(), "camera_depth_frame", "base_link")
    br.sendTransform((0.503, 0.547, 0.762), (-0.2611, 0.649, -0.2638, 0.6641), rospy.Time.now(), "depth_camera_link", "camera_depth_frame")

    rate.sleep()
