#!/usr/bin/env python3
"""
Semantic Depth OctoMap ROS2 Node

Simplified node that uses VoxelMappingHelper for all heavy lifting.
Subscribes to depth, pose, and semantic info to create semantic voxel maps.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2

import numpy as np
from cv_bridge import CvBridge
import time
import json
import math
from typing import Optional

# Import helpers
try:
	from resilience.voxel_mapping_helper import VoxelMappingHelper
	from resilience.semantic_info_bridge import SemanticHotspotSubscriber
	HELPERS_AVAILABLE = True
except ImportError:
	HELPERS_AVAILABLE = False


class SemanticDepthOctoMapNode(Node):
	"""Simplified semantic depth octomap node using helpers."""

	def __init__(self):
		super().__init__('semantic_depth_octomap_node')

		# Parameters
		self.declare_parameters('', [
			('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered'),
			('camera_info_topic', '/robot_1/sensors/front_stereo/left/camera_info'),
			('pose_topic', '/robot_1/sensors/front_stereo/pose'),
			('embedding_topic', '/voxel_embeddings'),
			('confidence_topic', '/voxel_confidences'),
			('map_frame', 'map'),
			('voxel_resolution', 0.1),
			('max_range', 5.0),
			('min_range', 0.1),
			('probability_hit', 0.7),
			('probability_miss', 0.4),
			('occupancy_threshold', 0.5),
			('publish_markers', True),
			('publish_stats', True),
			('publish_colored_cloud', True),
			('use_cube_list_markers', True),
			('max_markers', 30000),
			('marker_publish_rate', 1.0),
			('stats_publish_rate', 1.0),
			('pose_is_base_link', True),
			('apply_optical_frame_rotation', True),
			('cam_to_base_rpy_deg', [0.0, 0.0, 0.0]),
			('cam_to_base_xyz', [0.0, 0.0, 0.0]),
			('embedding_dim', 1152),
			('enable_semantic_mapping', True),
			('semantic_similarity_threshold', 0.6),
			('buffers_directory', '/home/navin/ros2_ws/src/buffers'),
			('segmentation_config_path', ''),
		])

		params = self.get_parameters([
			'depth_topic', 'camera_info_topic', 'pose_topic', 'embedding_topic', 'confidence_topic',
			'map_frame', 'voxel_resolution', 'max_range', 'min_range', 'probability_hit',
			'probability_miss', 'occupancy_threshold', 'publish_markers', 'publish_stats',
			'publish_colored_cloud', 'use_cube_list_markers', 'max_markers', 'marker_publish_rate',
			'stats_publish_rate', 'pose_is_base_link', 'apply_optical_frame_rotation',
			'cam_to_base_rpy_deg', 'cam_to_base_xyz', 'embedding_dim', 'enable_semantic_mapping',
			'semantic_similarity_threshold', 'buffers_directory', 'segmentation_config_path'
		])

		# Extract parameter values
		(self.depth_topic, self.camera_info_topic, self.pose_topic, self.embedding_topic, self.confidence_topic,
		 self.map_frame, self.voxel_resolution, self.max_range, self.min_range, self.prob_hit,
		 self.prob_miss, self.occ_thresh, self.publish_markers, self.publish_stats, self.publish_colored_cloud,
		 self.use_cube_list_markers, self.max_markers, self.marker_publish_rate, self.stats_publish_rate,
		 self.pose_is_base_link, self.apply_optical_frame_rotation, self.cam_to_base_rpy_deg, self.cam_to_base_xyz,
		 self.embedding_dim, self.enable_semantic_mapping, self.semantic_similarity_threshold,
		 self.buffers_directory, self.segmentation_config_path) = [p.value for p in params]

		# State
		self.bridge = CvBridge()
		self.camera_intrinsics = None
		self.latest_pose = None
		self.latest_embeddings = None
		self.latest_confidence = None
		self.last_marker_pub = 0.0
		self.last_stats_pub = 0.0

		# Initialize voxel mapping helper
		if not HELPERS_AVAILABLE:
			self.get_logger().error("VoxelMappingHelper not available!")
			return

		self.voxel_helper = VoxelMappingHelper(
			voxel_resolution=float(self.voxel_resolution),
			probability_hit=float(self.prob_hit),
			probability_miss=float(self.prob_miss),
			occupancy_threshold=float(self.occ_thresh),
			embedding_dim=int(self.embedding_dim),
			enable_semantic_mapping=bool(self.enable_semantic_mapping),
			semantic_similarity_threshold=float(self.semantic_similarity_threshold),
			map_frame=str(self.map_frame),
			verbose=False
		)

		# Load existing embeddings
		if isinstance(self.buffers_directory, str) and len(self.buffers_directory) > 0:
			loaded = self.voxel_helper.load_vlm_embeddings_from_buffers(self.buffers_directory)
			if loaded > 0:
				self.get_logger().info(f"Loaded {loaded} VLM embeddings from buffers")

		# Initialize clean semantic hotspot subscriber
		# Load config from naradio_processor if available
		config = {}
		if hasattr(self, 'segmentation_config_path') and self.segmentation_config_path:
			try:
				import yaml
				with open(self.segmentation_config_path, 'r') as f:
					config = yaml.safe_load(f)
			except Exception as e:
				self.get_logger().warn(f"Could not load semantic config: {e}")
		
		self.semantic_bridge = SemanticHotspotSubscriber(self, self.voxel_helper, config)

		# Precompute transforms
		self.R_opt_to_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
		self.R_cam_to_base_extra = self._rpy_deg_to_rot(self.cam_to_base_rpy_deg)
		self.t_cam_to_base_extra = np.array(self.cam_to_base_xyz, dtype=np.float32)

		# QoS
		sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

		# Subscribers
		self.create_subscription(Image, self.depth_topic, self.depth_callback, sensor_qos)
		self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
		self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 10)
		# NOTE: No longer subscribing to pixel embeddings/confidence - using smart semantic regions instead

		# Publishers
		self.marker_pub = self.create_publisher(MarkerArray, '/semantic_octomap_markers', 10) if self.publish_markers else None
		self.stats_pub = self.create_publisher(String, '/semantic_octomap_stats', 10) if self.publish_stats else None
		self.cloud_pub = self.create_publisher(PointCloud2, '/semantic_octomap_colored_cloud', 10) if self.publish_colored_cloud else None

		self.get_logger().info(
			f"SemanticDepthOctoMapNode initialized:\n"
			f"  - Voxel resolution: {self.voxel_resolution}m\n"
			f"  - Semantic mapping: {'ENABLED' if self.enable_semantic_mapping else 'DISABLED'}\n"
			f"  - Topics: depth={self.depth_topic}, pose={self.pose_topic}"
		)

	def camera_info_callback(self, msg: CameraInfo):
		if self.camera_intrinsics is None:
			self.camera_intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
			self.get_logger().info(f"Camera intrinsics set: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}")

	def pose_callback(self, msg: PoseStamped):
		self.latest_pose = msg

	# NOTE: Embedding and confidence callbacks removed - using smart semantic regions instead

	def depth_callback(self, msg: Image):
		if self.camera_intrinsics is None:
			self.get_logger().warn("No camera intrinsics received yet")
			return
		if self.latest_pose is None:
			self.get_logger().warn("No pose received yet") 
			return

		# Log occasionally to avoid spam
		if hasattr(self, '_debug_counter'):
			self._debug_counter += 1
		else:
			self._debug_counter = 1
		
		if self._debug_counter % 30 == 1:  # Log every 30th frame
			self.get_logger().info(f"Processing depth frame #{self._debug_counter}: {msg.width}x{msg.height}")
		
		try:
			# Convert depth
			depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
			depth_m = self._depth_to_meters(depth, msg.encoding)
			if depth_m is None:
				return

			# Convert to world points
			points_world, u_indices, v_indices = self._depth_to_world_points(depth_m, self.camera_intrinsics, self.latest_pose)
			if points_world is None or len(points_world) == 0:
				return

			# Range filter
			origin = self._pose_position(self.latest_pose)
			dist = np.linalg.norm(points_world - origin, axis=1)
			mask = (dist >= float(self.min_range)) & (dist <= float(self.max_range))
			points_world = points_world[mask]
			u_sel, v_sel = u_indices[mask], v_indices[mask]

			if points_world.size == 0:
				return

			# SMART SEMANTIC: Regular voxel mapping without continuous embeddings
			# Semantic labeling happens when semantic regions are received
			if self._debug_counter % 30 == 1:  # Log every 30th frame
				self.get_logger().info(f"Updating voxel map with {len(points_world)} points")
			self.voxel_helper.update_map(points_world, origin, hit_confidences=None, voxel_embeddings=None)

			# Periodic publishing
			self._periodic_publishing()

		except Exception as e:
			self.get_logger().error(f"Error in depth callback: {e}")

	def _depth_to_meters(self, depth, encoding: str):
		try:
			enc = (encoding or '').lower()
			if '16uc1' in enc or 'mono16' in enc:
				return depth.astype(np.float32) / 1000.0
			elif '32fc1' in enc or 'float32' in enc:
				return depth.astype(np.float32)
			else:
				return depth.astype(np.float32) / 1000.0
		except Exception:
			return None

	def _depth_to_world_points(self, depth_m: np.ndarray, intrinsics, pose: PoseStamped):
		try:
			fx, fy, cx, cy = intrinsics
			h, w = depth_m.shape
			u, v = np.meshgrid(np.arange(w), np.arange(h))
			z = depth_m
			valid = np.isfinite(z) & (z > 0.0)
			if not np.any(valid):
				return None, None, None

			u, v, z = u[valid], v[valid], z[valid]
			x = (u - cx) * z / fx
			y = (v - cy) * z / fy
			pts_cam = np.stack([x, y, z], axis=1)

			# Transform to base if needed
			if bool(self.pose_is_base_link):
				pts_cam = pts_cam @ (self.R_opt_to_base.T if bool(self.apply_optical_frame_rotation) else np.eye(3, dtype=np.float32))
				pts_cam = pts_cam @ self.R_cam_to_base_extra.T + self.t_cam_to_base_extra

			# World transform
			R_world = self._quat_to_rot(self._pose_quat(pose))
			p_world = self._pose_position(pose)
			pts_world = pts_cam @ R_world.T + p_world
			return pts_world, u, v
		except Exception:
			return None, None, None

	# NOTE: Pixel confidence/embedding extraction methods removed - using smart semantic regions instead

	def _periodic_publishing(self):
		now = time.time()
		
		if self.marker_pub and (now - self.last_marker_pub) >= float(self.marker_publish_rate):
			markers = self.voxel_helper.create_markers(int(self.max_markers), bool(self.use_cube_list_markers))
			self.marker_pub.publish(markers)
			marker_count = len(markers.markers) if hasattr(markers, 'markers') else 0
			self.get_logger().info(f"Published {marker_count} voxel markers")
			self.last_marker_pub = now

		if self.cloud_pub:
			cloud = self.voxel_helper.create_colored_cloud(int(self.max_markers))
			if cloud:
				self.cloud_pub.publish(cloud)

		if self.stats_pub and (now - self.last_stats_pub) >= float(self.stats_publish_rate):
			stats = self.voxel_helper.get_statistics()
			self.stats_pub.publish(String(data=json.dumps(stats)))
			self.last_stats_pub = now

	def _pose_position(self, pose: PoseStamped):
		return np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)

	def _pose_quat(self, pose: PoseStamped):
		q = pose.pose.orientation
		return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

	def _quat_to_rot(self, q: np.ndarray):
		x, y, z, w = q
		n = x*x + y*y + z*z + w*w
		if n < 1e-8:
			return np.eye(3, dtype=np.float32)
		s = 2.0 / n
		xx, yy, zz = x*x*s, y*y*s, z*z*s
		xy, xz, yz = x*y*s, x*z*s, y*z*s
		wx, wy, wz = w*x*s, w*y*s, w*z*s
		return np.array([
			[1.0 - (yy + zz), xy - wz, xz + wy],
			[xy + wz, 1.0 - (xx + zz), yz - wx],
			[xz - wy, yz + wx, 1.0 - (xx + yy)]
		], dtype=np.float32)

	def _rpy_deg_to_rot(self, rpy_deg):
		try:
			roll, pitch, yaw = [math.radians(float(x)) for x in rpy_deg]
			cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
			Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
			Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
			Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
			return Rz @ Ry @ Rx
		except Exception:
			return np.eye(3, dtype=np.float32)


def main():
	rclpy.init()
	node = SemanticDepthOctoMapNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main() 