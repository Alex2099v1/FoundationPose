#!/usr/bin/env python
import rospy
import ros_numpy as rnp
import numpy as np
import yaml as pyyaml
import torch
import trimesh
import gc
from PIL import Image 
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from foundation_pose_sam3.srv import GetObjectPose, GetObjectPoseResponse
import os
import sys
from  geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import json 
from sensor_msgs.msg import Image as  RosImage
from sensor_msgs.msg import CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation as R
from threading import Lock
_THIS_FILE = os.path.realpath(__file__)
_THIS_DIR  = os.path.dirname(_THIS_FILE)

# scripts/ -> foundation_pose_sam3/ -> FoundationPose/
FP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Ensure FoundationPose root is importable
if FP_ROOT not in sys.path:
    sys.path.insert(0, FP_ROOT)
from estimater import *
from datareader import * 

class PoseDetectionNode:
    def __init__(self):
        rospy.init_node('pose_detection_node')
        self.lock = Lock()

        # Load configuration
        config_path = os.path.join(_THIS_DIR, '..', 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = pyyaml.safe_load(f)
        camera_info=rospy.wait_for_message(self.config['camera_info_topic'], CameraInfo, timeout=10.0)
        self.K=np.array(camera_info.K).reshape(3,3)
        self.service = rospy.Service('get_object_pose', GetObjectPose, self.handle_get_object_pose)
        self.latest_rgb = None
        self.latest_depth = None
        rgb_subscriber = Subscriber(self.config['rgb_topic'], RosImage)
        deph_subscriber = Subscriber(self.config['depth_topic'], RosImage)
        self.ts = ApproximateTimeSynchronizer([rgb_subscriber, deph_subscriber], queue_size=10, slop=0.3)
        self.ts.registerCallback(self.sync_callback)

        rospy.loginfo("Pose Detection Node initialized and ready to receive requests.")


    def sync_callback(self, rgb_msg, depth_msg):
        with self.lock:
            self.latest_rgb = rgb_msg
            self.latest_depth = depth_msg

    def _ensure_cpu_default_tensors(self):
        # SAM3 prompt encoding calls pin_memory(), which only supports CPU tensors.
        try:
            if torch.tensor(0.0).is_cuda:
                torch.set_default_tensor_type(torch.FloatTensor)
        except Exception as e:
            rospy.logwarn(f"Unable to reset default tensor type to CPU: {e}")

    def handle_get_object_pose(self, req):
        # Convert ROS Image to numpy array
        with self.lock:
            rgb_msg = self.latest_rgb
            depth_msg = self.latest_depth
        try:
                image = rnp.numpify(rgb_msg)
                depth = rnp.numpify(depth_msg)
                depth_encoding = depth_msg.encoding
                rgb_encoding = rgb_msg.encoding
        except Exception as e:
            rospy.logerr(f"Error converting ROS Image to numpy array: {e}")
            return GetObjectPoseResponse(success=False, poses=[])

        depth = self.transform_depth_to_meters(depth, depth_encoding)
        rospy.loginfo(
            f"Service input: rgb_shape={getattr(image, 'shape', None)} rgb_enc={rgb_encoding} "
            f"depth_shape={getattr(depth, 'shape', None)} depth_enc={depth_encoding} "
            f"depth_minmax=({float(np.nanmin(depth)):.4f},{float(np.nanmax(depth)):.4f})"
        )
        poses = self.get_pose_of_object(req.object_name, image, depth)
        if poses is None:
            return GetObjectPoseResponse(success=False, poses=[])
        
        pose_list = []
        for pose in poses:
            msg=PoseStamped()
            msg.pose.position.x = pose[0,3]
            msg.pose.position.y = pose[1,3]
            msg.pose.position.z = pose[2,3]
            # Convert rotation matrix to quaternion
            r = R.from_matrix(pose[:3,:3])
            q = r.as_quat()
            msg.pose.orientation.x = q[0]
            msg.pose.orientation.y = q[1]
            msg.pose.orientation.z = q[2]
            msg.pose.orientation.w = q[3]
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.config['camera_frame']
            pose_list.append(msg)
        return GetObjectPoseResponse(success=True, poses=pose_list)

    def transform_depth_to_meters(self, depth, encoding):
        if depth is None:
            return None

        depth = depth.astype(np.float32, copy=False)
        enc = (encoding or "").lower()

        # FoundationPose expects metric depth (meters).
        if enc in ("16uc1", "mono16"):
            depth *= 1e-3
        elif enc == "32fc1":
            pass
        elif depth.size and np.nanmax(depth) > 100.0:
            # Fallback: depth is likely in millimeters.
            depth *= 1e-3

        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        return depth

    def flush_gpu_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def get_sam3_mask_from_image(self,object_name,image):
        if image is None:
            return None
        self._ensure_cpu_default_tensors()
        model = None
        processor = None
        inference_state = None
        output = None
        masks_cpu = None
        try:
            model = build_sam3_image_model()
            model.eval()
            processor = Sam3Processor(model)
            pil_image = Image.fromarray(image)
            with torch.inference_mode():
                inference_state = processor.set_image(pil_image)
                output = processor.set_text_prompt(state=inference_state, prompt=object_name)
            masks = output["masks"]
            if masks.shape[0] == 0:
                rospy.logwarn(f"No masks detected for object '{object_name}'.")
                return None
            if torch.is_tensor(masks):
                masks_cpu = masks.detach().cpu().numpy().astype(bool)
            else:
                masks_cpu = np.asarray(masks).astype(bool)
            flat_areas = masks_cpu.reshape(masks_cpu.shape[0], -1).sum(axis=1)
            rospy.loginfo(
                f"SAM produced {masks_cpu.shape[0]} mask(s). "
                f"Areas(px) sample={flat_areas[:min(5, len(flat_areas))].tolist()}"
            )
            return masks_cpu
        finally:
            del output
            del inference_state
            del processor
            del model
            self.flush_gpu_memory()
        


    def get_pose_of_object(self,object_name,image,depth):
        if object_name not in self.config['objects']:
            rospy.logwarn(f"Object '{object_name}' not found in configuration.")
            return None 
        rospy.loginfo(f"Using intrinsics K={self.K.tolist()} for image shape={image.shape[:2]}")
        model_path = self.config['objects'][object_name]
        model_folder = os.path.dirname(model_path)
        #look for model_info.json file and load it if exists
        model_info_path = os.path.join(model_folder, "model_info.json")
        symmetry_tfs = None
        if os.path.exists(model_info_path):
            rospy.loginfo(f"Found model_info.json for '{object_name}', loading symmetries.")
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            if "symmetries_discrete" in model_info:
                symmetry_tfs = np.asarray(model_info["symmetries_discrete"], dtype=np.float32)
                angles=[]
                for transform in symmetry_tfs:
                    # Get angles from rotation matrices
                    r = R.from_matrix(transform[:3, :3])
                    angles.append(r.as_rotvec(degrees=True))
                rospy.loginfo(f"Using following symetries:  {angles}")
                    
        mesh = trimesh.load(model_path)
        mask=self.get_sam3_mask_from_image(object_name,image)
        if mask is None:
            return None
        # in case multiple instances of same object, then multiple foundation pose instances
        poses = []
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        for i in range(mask.shape[0]):
            rospy.loginfo(f"Processing mask {i+1}/{mask.shape[0]} for object '{object_name}'")
            single_mask = mask[i, 0].astype(bool)
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, symmetry_tfs=symmetry_tfs, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir="/home/mrrobot/fpose_debug/", debug=0, glctx=glctx)
            pose = est.register(K=self.K, rgb=image, depth=depth, ob_mask=single_mask, iteration=self.config['est_refine_iter'])
            poses.append(pose)
            del est
            self.flush_gpu_memory()

        return poses

if __name__ == '__main__':
    node = PoseDetectionNode()
    rospy.spin()
