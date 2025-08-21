#!/usr/bin/env python3
"""
Depth OctoMap ROS2 Node

Subscribes to:
- sensor_msgs/Image (depth)
- sensor_msgs/CameraInfo (intrinsics)
- geometry_msgs/PoseStamped (camera/robot pose)

Builds an OctoMap in real time and publishes visualization topics.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String, ColorRGBA
from std_msgs.msg import Header

import numpy as np
from cv_bridge import CvBridge
import time
import json
import math

# Try to use OctoMap backend; fall back to NumPy voxel grid if unavailable
try:
	import octomap
	OCTOMAP_AVAILABLE = True
except Exception:
	OCTOMAP_AVAILABLE = False

# For colored cloud
try:
	import sensor_msgs_py.point_cloud2 as pc2
	PC2_AVAILABLE = True
except Exception:
	PC2_AVAILABLE = False


class DepthOctoMapNode(Node):
	"""ROS2 node for real-time OctoMap generation from depth images and poses."""

	def __init__(self):
		super().__init__('depth_octomap_node')

		# Parameters
		self.declare_parameters('', [
			('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered'),
			('camera_info_topic', '/robot_1/sensors/front_stereo/left/camera_info'),
			('pose_topic', '/robot_1/sensors/front_stereo/pose'),
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
			# Alignment/extrinsics
			('pose_is_base_link', True),
			('apply_optical_frame_rotation', True),
			('cam_to_base_rpy_deg', [0.0, 0.0, 0.0]),  # roll, pitch, yaw (deg)
			('cam_to_base_xyz', [0.0, 0.0, 0.0]),      # meters
		])

		(self.depth_topic,
		 self.camera_info_topic,
		 self.pose_topic,
		 self.map_frame,
		 self.voxel_resolution,
		 self.max_range,
		 self.min_range,
		 self.prob_hit,
		 self.prob_miss,
		 self.occ_thresh,
		 self.publish_markers,
		 self.publish_stats,
		 self.publish_colored_cloud,
		 self.use_cube_list_markers,
		 self.max_markers,
		 self.marker_publish_rate,
		 self.stats_publish_rate,
		 self.pose_is_base_link,
		 self.apply_optical_frame_rotation,
		 self.cam_to_base_rpy_deg,
		 self.cam_to_base_xyz) = [p.value for p in self.get_parameters([
			'depth_topic', 'camera_info_topic', 'pose_topic', 'map_frame',
			'voxel_resolution', 'max_range', 'min_range', 'probability_hit',
			'probability_miss', 'occupancy_threshold', 'publish_markers',
			'publish_stats', 'publish_colored_cloud', 'use_cube_list_markers', 'max_markers',
			'marker_publish_rate', 'stats_publish_rate',
			'pose_is_base_link', 'apply_optical_frame_rotation',
			'cam_to_base_rpy_deg', 'cam_to_base_xyz'])]

		# Initialize backend
		self._init_backend()

		# State
		self.bridge = CvBridge()
		self.camera_intrinsics = None  # [fx, fy, cx, cy]
		self.latest_pose: PoseStamped = None
		self.total_points = 0
		self.total_frames = 0
		self.last_marker_pub = 0.0
		self.last_stats_pub = 0.0

		# Precompute extrinsics
		self.R_opt_to_base = np.array([[0.0, 0.0, 1.0],
										   [-1.0, 0.0, 0.0],
										   [0.0, -1.0, 0.0]], dtype=np.float32)  # REP103 optical->base
		self.R_cam_to_base_extra = self._rpy_deg_to_rot(self.cam_to_base_rpy_deg)
		self.t_cam_to_base_extra = np.array(self.cam_to_base_xyz, dtype=np.float32)

		# QoS suitable for sensor data
		sensor_qos = QoSProfile(
			reliability=ReliabilityPolicy.BEST_EFFORT,
			history=HistoryPolicy.KEEP_LAST,
			depth=5,
		)

		# Subscribers
		self.create_subscription(Image, self.depth_topic, self.depth_callback, sensor_qos)
		self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
		self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 10)

		# Publishers
		if self.publish_markers:
			self.marker_pub = self.create_publisher(MarkerArray, '/octomap_markers', 10)
		else:
			self.marker_pub = None
		self.stats_pub = self.create_publisher(String, '/octomap_stats', 10) if self.publish_stats else None
		self.cloud_pub = self.create_publisher(PointCloud2, '/octomap_colored_cloud', 10) if self.publish_colored_cloud and PC2_AVAILABLE else None

		self.get_logger().info(
			f"DepthOctoMapNode initialized. Backend={'OctoMap' if OCTOMAP_AVAILABLE else 'NumPy'} | "
			f"depth: {self.depth_topic}, camera_info: {self.camera_info_topic}, pose: {self.pose_topic}\n"
			f"pose_is_base_link={self.pose_is_base_link}, apply_optical_frame_rotation={self.apply_optical_frame_rotation}\n"
			f"cam_to_base_rpy_deg={self.cam_to_base_rpy_deg}, cam_to_base_xyz={self.cam_to_base_xyz}\n"
			f"use_cube_list_markers={self.use_cube_list_markers}, publish_colored_cloud={self.publish_colored_cloud}")

	def _init_backend(self):
		if OCTOMAP_AVAILABLE:
			self.octree = octomap.OcTree(float(self.voxel_resolution))
			self.octree.setProbHit(float(self.prob_hit))
			self.octree.setProbMiss(float(self.prob_miss))
			self.octree.setClampingThresMin(0.12)
			self.octree.setClampingThresMax(0.97)
			self.octree.setOccupancyThres(float(self.occ_thresh))
			self.backend = 'octomap'
		else:
			self.voxel_dict = {}
			self.backend = 'numpy'

	def camera_info_callback(self, msg: CameraInfo):
		if self.camera_intrinsics is None:
			self.camera_intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
			self.get_logger().info(f"Camera intrinsics set: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}")

	def pose_callback(self, msg: PoseStamped):
		self.latest_pose = msg

	def depth_callback(self, msg: Image):
		if self.camera_intrinsics is None or self.latest_pose is None:
			return

		# Convert depth to cv image
		try:
			depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
		except Exception:
			return

		# Normalize depth to meters based on encoding
		depth_m = self._depth_to_meters(depth, msg.encoding)
		if depth_m is None:
			return

		# Convert to points in world
		points_world = self._depth_to_world_points(depth_m, self.camera_intrinsics, self.latest_pose)
		if points_world is None or len(points_world) == 0:
			return

		# Range filter
		dist = np.linalg.norm(points_world - self._pose_position(self.latest_pose), axis=1)
		mask = (dist >= float(self.min_range)) & (dist <= float(self.max_range))
		points_world = points_world[mask]
		if points_world.size == 0:
			return

		# Update backend
		if self.backend == 'octomap':
			self._update_octomap(points_world, self._pose_position(self.latest_pose))
		else:
			self._update_numpy(points_world)

		self.total_points += points_world.shape[0]
		self.total_frames += 1

		# Periodic publications
		now = time.time()
		if self.publish_markers and self.marker_pub and (now - self.last_marker_pub) >= float(self.marker_publish_rate):
			markers = self._create_markers()
			if markers:
				self.marker_pub.publish(markers)
			self.last_marker_pub = now

		if self.publish_colored_cloud and self.cloud_pub and PC2_AVAILABLE:
			cloud_msg = self._create_colored_cloud()
			if cloud_msg:
				self.cloud_pub.publish(cloud_msg)

		if self.publish_stats and self.stats_pub and (now - self.last_stats_pub) >= float(self.stats_publish_rate):
			stats = self._stats()
			self.stats_pub.publish(String(data=json.dumps(stats)))
			self.last_stats_pub = now

	def _depth_to_meters(self, depth, encoding: str):
		try:
			enc = (encoding or '').lower()
			if '16uc1' in enc or 'mono16' in enc:
				# millimeters
				return depth.astype(np.float32) / 1000.0
			elif '32fc1' in enc or 'float32' in enc:
				return depth.astype(np.float32)
			else:
				# Fallback: assume mm
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
				return None

			u = u[valid]
			v = v[valid]
			z = z[valid]
			x = (u - cx) * z / fx
			y = (v - cy) * z / fy
			pts_cam = np.stack([x, y, z], axis=1)  # camera optical frame by default

			# If the pose is base_link, convert camera optical -> base via fixed transforms
			if bool(self.pose_is_base_link):
				pts_cam = pts_cam @ (self.R_opt_to_base.T if bool(self.apply_optical_frame_rotation) else np.eye(3, dtype=np.float32))
				pts_cam = pts_cam @ self.R_cam_to_base_extra.T + self.t_cam_to_base_extra

			# World transform from pose
			R_world = self._quat_to_rot(self._pose_quat(pose))
			p_world = self._pose_position(pose)
			pts_world = pts_cam @ R_world.T + p_world
			return pts_world
		except Exception:
			return None

	def _pose_position(self, pose: PoseStamped):
		return np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)

	def _pose_quat(self, pose: PoseStamped):
		q = pose.pose.orientation
		return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

	def _quat_to_rot(self, q: np.ndarray):
		x, y, z, w = q
		# Normalized quaternion to rotation matrix
		n = x*x + y*y + z*z + w*w
		if n < 1e-8:
			return np.eye(3, dtype=np.float32)
		s = 2.0 / n
		xx, yy, zz = x*x*s, y*y*s, z*z*s
		xy, xz, yz = x*y*s, x*z*s, y*z*s
		wx, wy, wz = w*x*s, w*y*s, w*z*s
		R = np.array([
			[1.0 - (yy + zz), xy - wz, xz + wy],
			[xy + wz, 1.0 - (xx + zz), yz - wx],
			[xz - wy, yz + wx, 1.0 - (xx + yy)]
		], dtype=np.float32)
		return R

	def _rpy_deg_to_rot(self, rpy_deg):
		try:
			roll = math.radians(float(rpy_deg[0]))
			pitch = math.radians(float(rpy_deg[1]))
			yaw = math.radians(float(rpy_deg[2]))
			cr, sr = math.cos(roll), math.sin(roll)
			cp, sp = math.cos(pitch), math.sin(pitch)
			cy, sy = math.cos(yaw), math.sin(yaw)
			Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
			Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
			Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
			return Rz @ Ry @ Rx
		except Exception:
			return np.eye(3, dtype=np.float32)

	def _update_octomap(self, points_world: np.ndarray, origin_world: np.ndarray):
		try:
			origin = octomap.point3d(float(origin_world[0]), float(origin_world[1]), float(origin_world[2]))
			# Insert rays for each hit point
			for pt in points_world:
				end = octomap.point3d(float(pt[0]), float(pt[1]), float(pt[2]))
				self.octree.insertRay(origin, end)
			self.octree.updateInnerOccupancy()
		except Exception as e:
			self.get_logger().warn(f"OctoMap update error: {e}")

	def _update_numpy(self, points_world: np.ndarray):
		vox = np.floor(points_world / float(self.voxel_resolution)).astype(np.int32)
		for v in vox:
			key = (int(v[0]), int(v[1]), int(v[2]))
			self.voxel_dict[key] = min(1.0, self.voxel_dict.get(key, 0.6) + 0.1)

	def _create_markers(self):
		try:
			ma = MarkerArray()
			# Clear previous markers to avoid RViz persistence issues
			del_all = Marker()
			del_all.header.frame_id = self.map_frame
			del_all.header.stamp = self.get_clock().now().to_msg()
			del_all.action = Marker.DELETEALL
			ma.markers.append(del_all)

			if self.backend == 'octomap':
				if bool(self.use_cube_list_markers):
					m = Marker()
					m.header.frame_id = self.map_frame
					m.header.stamp = self.get_clock().now().to_msg()
					m.id = 0
					m.type = Marker.CUBE_LIST
					m.action = Marker.ADD
					m.scale.x = float(self.voxel_resolution)
					m.scale.y = float(self.voxel_resolution)
					m.scale.z = float(self.voxel_resolution)
					m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
					count = 0
					m.points = []
					m.colors = []
					for it in self.octree.begin_leafs():
						if not self.octree.isNodeOccupied(it):
							continue
						prob = self.octree.getNodeProbability(it)
						coord = it.getCoordinate()
						pt = Point()
						pt.x = coord.x(); pt.y = coord.y(); pt.z = coord.z()
						m.points.append(pt)
						m.colors.append(self._prob_to_color(prob))
						count += 1
						if count >= int(self.max_markers):
							break
					ma.markers.append(m)
				else:
					marker_id = 0
					for it in self.octree.begin_leafs():
						if self.octree.isNodeOccupied(it):
							coord = it.getCoordinate()
							size = it.getSize()
							m = Marker()
							m.header.frame_id = self.map_frame
							m.header.stamp = self.get_clock().now().to_msg()
							m.id = marker_id
							m.type = Marker.CUBE
							m.action = Marker.ADD
							m.pose.position.x = coord.x()
							m.pose.position.y = coord.y()
							m.pose.position.z = coord.z()
							m.pose.orientation.w = 1.0
							m.scale.x = size
							m.scale.y = size
							m.scale.z = size
							prob = self.octree.getNodeProbability(it)
							m.color = self._prob_to_color(prob)
							ma.markers.append(m)
							marker_id += 1
							if marker_id >= int(self.max_markers):
								break
			else:
				# NumPy backend
				if bool(self.use_cube_list_markers):
					m = Marker()
					m.header.frame_id = self.map_frame
					m.header.stamp = self.get_clock().now().to_msg()
					m.id = 0
					m.type = Marker.CUBE_LIST
					m.action = Marker.ADD
					m.scale.x = float(self.voxel_resolution)
					m.scale.y = float(self.voxel_resolution)
					m.scale.z = float(self.voxel_resolution)
					m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
					count = 0
					m.points = []
					m.colors = []
					for (vx, vy, vz), p in self.voxel_dict.items():
						if p < float(self.occ_thresh):
							continue
						pt = Point()
						pt.x = vx * float(self.voxel_resolution)
						pt.y = vy * float(self.voxel_resolution)
						pt.z = vz * float(self.voxel_resolution)
						m.points.append(pt)
						m.colors.append(self._prob_to_color(p))
						count += 1
						if count >= int(self.max_markers):
							break
					ma.markers.append(m)
				else:
					marker_id = 0
					for (vx, vy, vz), p in list(self.voxel_dict.items()):
						if p < float(self.occ_thresh):
							continue
						m = Marker()
						m.header.frame_id = self.map_frame
						m.header.stamp = self.get_clock().now().to_msg()
						m.id = marker_id
						m.type = Marker.CUBE
						m.action = Marker.ADD
						m.pose.position.x = vx * float(self.voxel_resolution)
						m.pose.position.y = vy * float(self.voxel_resolution)
						m.pose.position.z = vz * float(self.voxel_resolution)
						m.pose.orientation.w = 1.0
						m.scale.x = float(self.voxel_resolution)
						m.scale.y = float(self.voxel_resolution)
						m.scale.z = float(self.voxel_resolution)
						m.color = self._prob_to_color(p)
						ma.markers.append(m)
						marker_id += 1
						if marker_id >= int(self.max_markers):
							break
			return ma
		except Exception:
			return None

	def _create_colored_cloud(self):
		try:
			if not PC2_AVAILABLE:
				return None
			points = []
			colors = []
			if self.backend == 'octomap':
				count = 0
				for it in self.octree.begin_leafs():
					if not self.octree.isNodeOccupied(it):
						continue
					coord = it.getCoordinate()
					prob = self.octree.getNodeProbability(it)
					points.append([coord.x(), coord.y(), coord.z()])
					c = self._prob_to_color(prob)
					colors.append([int(c.r*255), int(c.g*255), int(c.b*255)])
					count += 1
					if count >= int(self.max_markers):
						break
			else:
				for (vx, vy, vz), p in self.voxel_dict.items():
					if p < float(self.occ_thresh):
						continue
					points.append([vx*float(self.voxel_resolution), vy*float(self.voxel_resolution), vz*float(self.voxel_resolution)])
					c = self._prob_to_color(p)
					colors.append([int(c.r*255), int(c.g*255), int(c.b*255)])
			if not points:
				return None
			header = Header()
			header.frame_id = self.map_frame
			header.stamp = self.get_clock().now().to_msg()
			dtype = [
				('x', np.float32), ('y', np.float32), ('z', np.float32),
				('r', np.uint8), ('g', np.uint8), ('b', np.uint8)
			]
			arr = np.zeros(len(points), dtype=dtype)
			arr['x'] = np.asarray(points, dtype=np.float32)[:,0]
			arr['y'] = np.asarray(points, dtype=np.float32)[:,1]
			arr['z'] = np.asarray(points, dtype=np.float32)[:,2]
			arr['r'] = np.asarray(colors, dtype=np.uint8)[:,0]
			arr['g'] = np.asarray(colors, dtype=np.uint8)[:,1]
			arr['b'] = np.asarray(colors, dtype=np.uint8)[:,2]
			msg = pc2.create_cloud(header, [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='r', offset=12, datatype=pc2.PointField.UINT8, count=1),
				pc2.PointField(name='g', offset=13, datatype=pc2.PointField.UINT8, count=1),
				pc2.PointField(name='b', offset=14, datatype=pc2.PointField.UINT8, count=1),
			], arr.tolist())
			return msg
		except Exception:
			return None

	def _prob_to_color(self, p: float) -> ColorRGBA:
		p = float(p)
		red = min(1.0, max(0.0, (p - 0.5) * 2.0))
		green = min(1.0, max(0.0, (0.5 - p) * 2.0))
		return ColorRGBA(r=red, g=green, b=0.1, a=0.8)

	def _stats(self):
		stats = {
			'backend': self.backend,
			'voxel_resolution': float(self.voxel_resolution),
			'total_points': int(self.total_points),
			'frames': int(self.total_frames),
			'has_intrinsics': self.camera_intrinsics is not None,
			'has_pose': self.latest_pose is not None,
		}
		if self.backend == 'octomap':
			try:
				stats.update({
					'leaf_nodes': int(self.octree.getNumLeafNodes()),
					'tree_depth': int(self.octree.getTreeDepth()),
				})
			except Exception:
				pass
		else:
			stats.update({'voxels': int(len(self.voxel_dict))})
		return stats


def main():
	rclpy.init()
	node = DepthOctoMapNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main() 