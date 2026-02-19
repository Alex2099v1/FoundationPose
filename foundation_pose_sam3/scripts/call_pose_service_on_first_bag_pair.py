#!/usr/bin/env python3
"""
Find the first synchronized RGB/depth pair in a ROS bag, call pose service, and print result.

Default topics:
- RGB:   /realsense/color/image_raw
- Depth: /realsense/aligned_depth_to_color/image_raw
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import rosbag
import rospy
from foundation_pose_sam3.srv import GetObjectPose, GetObjectPoseRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use first synchronized RGB/depth pair from bag to call pose service."
    )
    parser.add_argument("bag_path", help="Path to .bag file")
    parser.add_argument(
        "--object-name",
        required=True,
        help="Object name prompt sent to pose_detection service",
    )
    parser.add_argument(
        "--rgb-topic",
        default="/realsense/color/image_raw",
        help="RGB image topic",
    )
    parser.add_argument(
        "--depth-topic",
        default="/realsense/aligned_depth_to_color/image_raw",
        help="Depth image topic",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=30.0,
        help="Max RGB/depth timestamp delta to accept as a pair (ms)",
    )
    parser.add_argument(
        "--service-name",
        default="/get_object_pose",
        help="Pose service name",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for service availability",
    )
    return parser.parse_args()


def _extract_first_pair(
    bag_path: Path,
    rgb_topic: str,
    depth_topic: str,
    tolerance_sec: float,
):
    rgb_queue = deque()
    depth_queue = deque()

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=[rgb_topic, depth_topic]):
            stamp = msg.header.stamp.to_sec() if hasattr(msg, "header") else None
            if stamp is None:
                continue

            if topic == rgb_topic:
                rgb_queue.append((stamp, msg))
            elif topic == depth_topic:
                depth_queue.append((stamp, msg))

            while rgb_queue and depth_queue:
                rgb_t, rgb_msg = rgb_queue[0]
                depth_t, depth_msg = depth_queue[0]
                dt = rgb_t - depth_t

                if abs(dt) <= tolerance_sec:
                    return rgb_msg, depth_msg, rgb_t, depth_t
                if dt < -tolerance_sec:
                    rgb_queue.popleft()
                else:
                    depth_queue.popleft()

    return None


def _fill_request_images(req: GetObjectPoseRequest, rgb_msg, depth_msg) -> None:
    # Support both field name variants in case local node code and .srv drifted.
    if hasattr(req, "rgb_image") and hasattr(req, "depth_image"):
        req.rgb_image = rgb_msg
        req.depth_image = depth_msg
        return

    if hasattr(req, "image") and hasattr(req, "depth"):
        req.image = rgb_msg
        req.depth = depth_msg
        return

    raise AttributeError(
        "GetObjectPose request has unknown image fields (expected rgb_image/depth_image or image/depth)."
    )


def main() -> None:
    args = parse_args()
    bag_path = Path(args.bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    tol_sec = args.tolerance_ms / 1000.0

    print(f"Reading bag: {bag_path}")
    print(f"RGB topic:   {args.rgb_topic}")
    print(f"Depth topic: {args.depth_topic}")
    print(f"Pair tolerance: {args.tolerance_ms:.1f} ms")

    pair = _extract_first_pair(
        bag_path=bag_path,
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        tolerance_sec=tol_sec,
    )
    if pair is None:
        print("No synchronized RGB/depth pair found in bag with the configured tolerance.")
        return

    rgb_msg, depth_msg, rgb_t, depth_t = pair
    print(f"Found first pair: rgb_t={rgb_t:.6f}, depth_t={depth_t:.6f}, dt={(rgb_t - depth_t)*1000:.2f} ms")

    rospy.init_node("call_pose_service_on_first_bag_pair", anonymous=True)
    print(f"Waiting for service: {args.service_name} (timeout={args.wait_timeout:.1f}s)")
    rospy.wait_for_service(args.service_name, timeout=args.wait_timeout)
    client = rospy.ServiceProxy(args.service_name, GetObjectPose)

    req = GetObjectPoseRequest()
    req.object_name = args.object_name
    _fill_request_images(req, rgb_msg, depth_msg)

    print(f"Calling service for object_name='{args.object_name}'...")
    resp = client(req)

    print("\nService response:")
    print(f"success: {resp.success}")
    print(f"poses: {len(resp.poses)}")

    for idx, pose_stamped in enumerate(resp.poses):
        p = pose_stamped.pose.position
        q = pose_stamped.pose.orientation
        print(
            f"[{idx}] frame='{pose_stamped.header.frame_id}' "
            f"position=({p.x:.6f}, {p.y:.6f}, {p.z:.6f}) "
            f"quaternion=({q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f})"
        )


if __name__ == "__main__":
    main()
