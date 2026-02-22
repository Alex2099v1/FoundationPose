#!/usr/bin/env python
import rospy
from foundation_pose_sam3.srv import GetObjectPose, GetObjectPoseRequest
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

def main():
    rospy.init_node('pose_estimator_test_node')
    pub=rospy.Publisher('/estimated_object_pose', PoseStamped, queue_size=10)
    rospy.wait_for_service('/get_object_pose', timeout=20.0)
    try:
        get_object_pose = rospy.ServiceProxy('/get_object_pose', GetObjectPose)
        request = GetObjectPoseRequest()
        request.object_name = "gray_block"
        response = get_object_pose(request)
        if response.success:
            rospy.loginfo(f"Received {len(response.poses)} poses for object '{request.object_name}'.")
            for pose in response.poses:
                rospy.loginfo(f"Publishing pose: {pose}")
                rate = rospy.Rate(10)
                while not rospy.is_shutdown():
                    pose.header.stamp = rospy.Time.now()
                    pub.publish(pose)
                    rate.sleep()
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    main()