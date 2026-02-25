#!/usr/bin/env python
import rospy
from foundation_pose_sam3.srv import GetObjectPose, GetObjectPoseRequest
from scipy.spatial.transform import Rotation as R
import tf
import numpy as np
import os
import yaml
from foundation_pose_sam3.msg import NamedPoseStamped, NamedPoseStampedArray


WORLD_UP = np.array([0.0, 0.0, 1.0])

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def _project_to_xy(v):
    return v - np.dot(v, WORLD_UP) * WORLD_UP

def fix_transform_axis(transform_msg, parallel_thresh=0.9):
    quat = transform_msg.pose.orientation
    r = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    
    x_dot = np.dot(x, WORLD_UP)

    if x_dot > parallel_thresh:
        x_new = WORLD_UP.copy()

        y_proj = _project_to_xy(y)
        if np.linalg.norm(y_proj) < 1e-8:
            y_proj = _project_to_xy(z)

        y_new = _normalize(y_proj)
        z_new = _normalize(np.cross(x_new, y_new))
        y_new = _normalize(np.cross(z_new, x_new))

        r = np.column_stack((x_new, y_new, z_new))

        fixed_quat = R.from_matrix(r).as_quat()
        transform_msg.pose.orientation.x = fixed_quat[0]
        transform_msg.pose.orientation.y = fixed_quat[1]
        transform_msg.pose.orientation.z = fixed_quat[2]
        transform_msg.pose.orientation.w = fixed_quat[3]
        return transform_msg

    if abs(np.dot(y, WORLD_UP)) > parallel_thresh:
        z_new = _normalize(y)
        x_new = _normalize(x - np.dot(x, z_new) * z_new)
        y_new = np.cross(z_new, x_new)
        r = np.column_stack((x_new, y_new, z_new))

    z = r[:, 2]
    if np.dot(z, WORLD_UP) < 0:
        r[:, 0] *= -1
        r[:, 2] *= -1

    z_new = WORLD_UP.copy()

    x_proj = _project_to_xy(r[:, 0])
    if np.linalg.norm(x_proj) < 1e-8:
        x_proj = _project_to_xy(r[:, 1])

    x_new = _normalize(x_proj)
    y_new = _normalize(np.cross(z_new, x_new))
    x_new = _normalize(np.cross(y_new, z_new))

    r = np.column_stack((x_new, y_new, z_new))

    fixed_quat = R.from_matrix(r).as_quat()
    transform_msg.pose.orientation.x = fixed_quat[0]
    transform_msg.pose.orientation.y = fixed_quat[1]
    transform_msg.pose.orientation.z = fixed_quat[2]
    transform_msg.pose.orientation.w = fixed_quat[3]
    return transform_msg


class PosePublisher:
    def __init__(self):
        rospy.init_node('block_pose_publisher_node')
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(_THIS_DIR, '..', 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.objects=self.config['objects']
        self.pose_pub = rospy.Publisher('/detected_object_poses', NamedPoseStampedArray, queue_size=10,latch=True)
        rospy.wait_for_service('/get_object_pose', timeout=20.0)
        self.get_object_pose_client=rospy.ServiceProxy('/get_object_pose', GetObjectPose)
        self.tf_listener = tf.TransformListener()
    
    def get_and_publish_poses(self):
        all_poses=[]
        for obj in self.objects:
            request = GetObjectPoseRequest()
            request.object_name = obj
            try:
                response = self.get_object_pose_client(request)
                if not response.success:
                    for i in range(3):
                        rospy.logwarn(f"Attempt {i+1}: Failed to get pose for object '{obj}', retrying...")
                        rospy.sleep(1.0)  # Wait before retrying
                        response = self.get_object_pose_client(request)
                        if response.success:
                            break
                if response.success:
                    rospy.loginfo(f"Successfully got pose for object '{obj}' with {len(response.poses)} poses.")
                    for pose in response.poses:
                        self.tf_listener.waitForTransform(pose.header.frame_id, "table_top", pose.header.stamp, rospy.Duration(1.0))
                        transformed_pose = self.tf_listener.transformPose("table_top", pose)
                        transformed_pose = fix_transform_axis(transformed_pose)
                        named_pose=NamedPoseStamped()
                        named_pose.name=obj
                        named_pose.pose=transformed_pose
                        all_poses.append(named_pose)
                else:
                    rospy.logwarn(f"Failed to get pose for object '{obj}'")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed for object '{obj}': {e}")
        pose_array_msg=NamedPoseStampedArray()
        pose_array_msg.poses=all_poses
        self.pose_pub.publish(pose_array_msg)
        rospy.loginfo(f"Published poses for {len(all_poses)} objects.")

if __name__ == "__main__":
    pose_publisher = PosePublisher()
    rospy.sleep(1)  
    pose_publisher.get_and_publish_poses()
    rospy.spin()
# Give some time for the publisher to set up