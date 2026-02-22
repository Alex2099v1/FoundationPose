#!/usr/bin/env python
import rospy
from foundation_pose_sam3.srv import GetObjectPose, GetObjectPoseRequest
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
def main():
    rospy.init_node('pose_estimator_test_node')
    rospy.wait_for_service('/get_object_pose', timeout=20.0)
    requested_objects=["gray_block","blue_block"]
    objects_poses={}
    objects_pub = {}
    for obj in requested_objects:
        objects_pub[obj] = rospy.Publisher(f"/{obj}_pose", PoseStamped, queue_size=10)
    try:
        get_object_pose = rospy.ServiceProxy('/get_object_pose', GetObjectPose)
        request = GetObjectPoseRequest()
        for obj in requested_objects:
            request.object_name = obj
            response = get_object_pose(request)
            if not response.success:
                rospy.logwarn(f"Failed to get pose for object '{obj}'")
                continue
            rospy.loginfo(f"Received {len(response.poses)} poses for object '{request.object_name}'.")
            objects_poses[obj] = response.poses
        rete=rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            for obj, poses in objects_poses.items():
                for pose in poses:
                    objects_pub[obj].publish(pose)
            rete.sleep()


    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    main()