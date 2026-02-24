#!/usr/bin/env python
import rospy
from foundation_pose_sam3.srv import GetObjectPose, GetObjectPoseRequest
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import tf
import numpy as np
from visualization_msgs.msg import Marker
def main():
    rospy.init_node('pose_estimator_test_node')
    rospy.wait_for_service('/get_object_pose', timeout=20.0)
    requested_objects=["gray_block","blue_block"]
    colors={"gray_block":(0.5,0.5,0.5,0.5),"blue_block":(0.0,0.0,1.0,0.5)}
    marker_pub = rospy.Publisher('/object_markers', Marker, queue_size=10)
    objects_poses={}
    objects_pub = {}
    tf_listener = tf.TransformListener()

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
            poses=[]
            for pose in response.poses:
                try:
                    tf_listener.waitForTransform(pose.header.frame_id, "table_top", pose.header.stamp, rospy.Duration(1.0))
                    transformed_pose = tf_listener.transformPose("table_top", pose)
                    transformed_pose=fix_transform_axis(transformed_pose)
                    poses.append(transformed_pose)    
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    rospy.logerr(f"TF transformation failed for object '{obj}': {e}")
                    continue
            objects_poses[obj] = poses
        rete=rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            for obj, poses in objects_poses.items():
                for pose in poses:
                    # transform in table_top frame
                    objects_pub[obj].publish(pose)
                    # publish marker for each pose
                    marker = Marker()
                    marker.header = pose.header
                    marker.ns = obj
                    marker.id = poses.index(pose)
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.pose = pose.pose
                    marker.scale.x = 0.06
                    marker.scale.y = 0.02
                    marker.scale.z = 0.02
                    r, g, b, a = colors[obj]
                    marker.color.r = r
                    marker.color.g = g
                    marker.color.b = b
                    marker.color.a = a
                    marker.lifetime = rospy.Duration(0.0)
                    marker_pub.publish(marker)
            rete.sleep()
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


WORLD_UP = np.array([0.0, 0.0, 1.0])

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def fix_transform_axis(transform_msg, parallel_thresh=0.9):
    quat = transform_msg.pose.orientation
    r = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    # If X is already (almost) parallel to world up, don't touch it
    if abs(np.dot(x, WORLD_UP)) > parallel_thresh:
        return transform_msg

    # If Y is (almost) parallel to world up, we want "up" to be Z, so move that axis to Z
    if abs(np.dot(y, WORLD_UP)) > parallel_thresh:
        # Put old Y into Z, and rebuild Y to keep a proper right-handed basis.
        z_new = _normalize(y)
        x_new = _normalize(x - np.dot(x, z_new) * z_new)   # make x orthogonal to z_new
        y_new = np.cross(z_new, x_new)                     # ensures right-handed

        r = np.column_stack((x_new, y_new, z_new))

    # Enforce Z-up (relative to world_up)
    z = r[:, 2]
    if np.dot(z, WORLD_UP) < 0:
        # Flip two axes to keep det=+1 (proper rotation)
        r[:, 0] *= -1  # flip X
        r[:, 2] *= -1  # flip Z

    fixed_quat = R.from_matrix(r).as_quat()
    transform_msg.pose.orientation.x = fixed_quat[0]
    transform_msg.pose.orientation.y = fixed_quat[1]
    transform_msg.pose.orientation.z = fixed_quat[2]
    transform_msg.pose.orientation.w = fixed_quat[3]
    return transform_msg

if __name__ == "__main__":
    main()