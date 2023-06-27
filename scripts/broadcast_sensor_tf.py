#!/usr/bin/env python3

import rospy
import tf

if __name__ == '__main__':
  rospy.init_node('sensor_tf_broadcaster')
  rate = rospy.Rate(50.0)
  br = tf.TransformBroadcaster()

  while not rospy.is_shutdown():
    br.sendTransform((0, 0.545, 1.140), (0.5, 0.5, -0.5, 0.5), rospy.Time.now(), "camera1_link", "base_link")

    rate.sleep()
