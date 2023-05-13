import rospy
import moveit_commander

class EEPosePublisher(object):
  def __init__(self):
    rospy.init_node('ee_pose_pub', anonymous=True)

    moveit_commander.roscpp_initialize()
    self.move_group = moveit_commander.MoveGroupCommander('manipulator')
    self.ee_pose_pub = rospy.Publisher("/ee_pose", PoseStamped, queue_size=1)

  def run(self):
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
      ee_pose = self.move_group.get_current_pose(end_effector_link="rg2_eef_link")
      self.ee_pose_pub.publish(ee_pose)

      rate.sleep()

if __name__ == '__main__':
  ee_pose_pub = EEPosePublisher()
  ee_pose_pub.run()
