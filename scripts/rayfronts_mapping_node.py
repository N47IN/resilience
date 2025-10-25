#!/usr/bin/env python3
"""
RayFronts Occupancy Mapping ROS2 Node

Simplified node that uses RayFronts OccupancyVdbMap for efficient 3D mapping.
Subscribes to depth, pose, and RGB to create occupancy voxel maps using OpenVDB.
Provides automatic depth noise reduction and memory-efficient sparse storage.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2

import numpy as np
import torch
from cv_bridge import CvBridge
import time
import json
import math
from typing import Optional, List
import sensor_msgs_py.point_cloud2 as pc2
import threading
import cv2
import os

# Import RayFronts mapping
try:
    import sys
    sys.path.append('/home/navin/ros2_ws/src/resilience/RayFronts')
    from rayfronts.mapping import OccupancyVDBMap  # Note: VDB not Vdb
    from rayfronts import geometry3d as g3d
    RAYFRONTS_AVAILABLE = True
except ImportError as e:
    print(f"RayFronts not available: {e}")
    RAYFRONTS_AVAILABLE = False


class RayFrontsMappingNode(Node):
    """RayFronts occupancy mapping node using OpenVDB for efficient 3D mapping."""

    def __init__(self):
        super().__init__('rayfronts_mapping_node')

        # Professional startup message
        self.get_logger().info("=" * 60)
        self.get_logger().info("RAYFRONTS OCCUPANCY MAPPING SYSTEM INITIALIZING")
        self.get_logger().info("=" * 60)

        if not RAYFRONTS_AVAILABLE:
            self.get_logger().error("RayFronts not available! Please check installation.")
            return

        # Parameters
        self.declare_parameters('', [
            ('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered'),
            ('camera_info_topic', '/robot_1/sensors/front_stereo/left/camera_info'),
            ('pose_topic', '/robot_1/sensors/front_stereo/pose'),
            ('rgb_topic', '/robot_1/sensors/front_stereo/left/image_rect_color'),
            ('map_frame', 'map'),
            ('voxel_resolution', 0.1),
            ('max_range', 3.0),
            ('min_range', 0.1),
            ('max_empty_cnt', 3),
            ('max_occ_cnt', 5),
            ('occ_observ_weight', 5),
            ('occ_thickness', 2),
            ('occ_pruning_tolerance', 2),
            ('occ_pruning_period', 1),
            ('vox_accum_period', 1),
            ('max_pts_per_frame', 1000),
            ('max_empty_pts_per_frame', 1000),
            ('max_depth_sensing', 1.0),
            ('pose_is_base_link', True),
            ('apply_optical_frame_rotation', True),
            ('cam_to_base_rpy_deg', [0.0, 0.0, 0.0]),
            ('cam_to_base_xyz', [0.0, 0.0, 0.0]),
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
            ('cam_to_base_xyz', [0.0, 0.0, 0.0])
        ])

        params = self.get_parameters([
            'depth_topic', 'camera_info_topic', 'pose_topic', 'rgb_topic',
            'map_frame', 'voxel_resolution', 'max_range', 'min_range',
            'max_empty_cnt', 'max_occ_cnt', 'occ_observ_weight', 'occ_thickness',
            'occ_pruning_tolerance', 'occ_pruning_period', 'vox_accum_period',
            'max_pts_per_frame', 'max_empty_pts_per_frame', 'max_depth_sensing',
            'publish_markers', 'publish_stats', 'publish_colored_cloud',
            'use_cube_list_markers', 'max_markers', 'marker_publish_rate', 'stats_publish_rate',
            'pose_is_base_link', 'apply_optical_frame_rotation', 'cam_to_base_rpy_deg', 'cam_to_base_xyz'
        ])

        # Extract parameter values
        (self.depth_topic, self.camera_info_topic, self.pose_topic, self.rgb_topic,
         self.map_frame, self.voxel_resolution, self.max_range, self.min_range,
         self.max_empty_cnt, self.max_occ_cnt, self.occ_observ_weight, self.occ_thickness,
         self.occ_pruning_tolerance, self.occ_pruning_period, self.vox_accum_period,
         self.max_pts_per_frame, self.max_empty_pts_per_frame, self.max_depth_sensing,
         self.publish_markers, self.publish_stats, self.publish_colored_cloud,
         self.use_cube_list_markers, self.max_markers, self.marker_publish_rate, self.stats_publish_rate,
         self.pose_is_base_link, self.apply_optical_frame_rotation, self.cam_to_base_rpy_deg, self.cam_to_base_xyz) = [p.value for p in params]

        # Load topic configuration
        self.load_topic_configuration()

        # State
        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.latest_pose = None
        self.last_marker_pub = 0.0
        self.last_stats_pub = 0.0

        # Precompute transforms
        self.R_opt_to_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
        self.R_cam_to_base_extra = self._rpy_deg_to_rot(self.cam_to_base_rpy_deg)
        self.t_cam_to_base_extra = np.array(self.cam_to_base_xyz, dtype=np.float32)

        # Initialize RayFronts mapper
        self._initialize_rayfronts_mapper()

        # QoS
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        # Subscribers
        self.create_subscription(Image, self.depth_topic, self.depth_callback, sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 10)
        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, sensor_qos)

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/rayfronts_occupancy_markers', 10) if self.publish_markers else None
        self.stats_pub = self.create_publisher(String, '/rayfronts_mapping_stats', 10) if self.publish_stats else None
        self.cloud_pub = self.create_publisher(PointCloud2, '/rayfronts_occupancy_cloud', 10) if self.publish_colored_cloud else None
        self.frontiers_pub = self.create_publisher(PointCloud2, '/rayfronts_frontiers', 10)

        self.get_logger().info("=" * 60)
        self.get_logger().info("RAYFRONTS OCCUPANCY MAPPING SYSTEM READY")
        self.get_logger().info("=" * 60)
        
        self.get_logger().info(f"Mapping Configuration:")
        self.get_logger().info(f"   Voxel resolution: {self.voxel_resolution}m")
        self.get_logger().info(f"   Max range: {self.max_range}m")
        self.get_logger().info(f"   Min range: {self.min_range}m")
        self.get_logger().info(f"   Max empty count: {self.max_empty_cnt}")
        self.get_logger().info(f"   Max occupied count: {self.max_occ_cnt}")
        self.get_logger().info(f"   Occupancy weight: {self.occ_observ_weight}")
        
        self.get_logger().info(f"Topics:")
        self.get_logger().info(f"   Depth: {self.depth_topic}")
        self.get_logger().info(f"   RGB: {self.rgb_topic}")
        self.get_logger().info(f"   Pose: {self.pose_topic}")
        
        self.get_logger().info("=" * 60)

    def load_topic_configuration(self):
        """Load topic configuration from mapping config file."""
        try:
            import yaml
            from ament_index_python.packages import get_package_share_directory
            package_dir = get_package_share_directory('resilience')
            config_path = os.path.join(package_dir, 'config', 'mapping_config.yaml')
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract topic configuration
            topics = config.get('topics', {})
            
            # Input topics - use mapping config topics
            self.depth_topic = topics.get('depth_topic', self.depth_topic)
            self.camera_info_topic = topics.get('camera_info_topic', self.camera_info_topic)
            self.pose_topic = topics.get('pose_topic', self.pose_topic)
            
            # RGB topic - use main config since mapping config doesn't have it
            main_config_path = os.path.join(package_dir, 'config', 'main_config.yaml')
            try:
                with open(main_config_path, 'r') as f:
                    main_config = yaml.safe_load(f)
                main_topics = main_config.get('topics', {})
                self.rgb_topic = main_topics.get('rgb_topic', self.rgb_topic)
            except:
                pass  # Keep default RGB topic
            
            self.get_logger().info(f"Topic configuration loaded from: {config_path}")
            
        except Exception as e:
            self.get_logger().warn(f"Using default topic configuration: {e}")

    def _initialize_rayfronts_mapper(self):
        """Initialize the RayFronts occupancy mapper."""
        try:
            # Create dummy intrinsics (will be updated when camera info is received)
            dummy_intrinsics = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
            
            self.mapper = OccupancyVDBMap(
                intrinsics_3x3=dummy_intrinsics,
                device="cuda" if torch.cuda.is_available() else "cpu",
                visualizer=None,  # No visualizer for now
                clip_bbox=None,
                max_pts_per_frame=int(self.max_pts_per_frame),
                vox_size=float(self.voxel_resolution),
                vox_accum_period=int(self.vox_accum_period),
                max_empty_pts_per_frame=int(self.max_empty_pts_per_frame),
                max_depth_sensing=float(self.max_depth_sensing) if self.max_depth_sensing > 0 else -1,
                max_empty_cnt=int(self.max_empty_cnt),
                max_occ_cnt=int(self.max_occ_cnt),
                occ_observ_weight=int(self.occ_observ_weight),
                occ_thickness=int(self.occ_thickness),
                occ_pruning_tolerance=int(self.occ_pruning_tolerance),
                occ_pruning_period=int(self.occ_pruning_period)
            )
            
            self.get_logger().info("RayFronts OccupancyVdbMap initialized successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize RayFronts mapper: {e}")
            raise

    def camera_info_callback(self, msg: CameraInfo):
        """Update camera intrinsics when camera info is received."""
        if self.camera_intrinsics is None:
            # Update mapper intrinsics
            intrinsics = torch.tensor([
                [msg.k[0], msg.k[1], msg.k[2]],
                [msg.k[3], msg.k[4], msg.k[5]],
                [msg.k[6], msg.k[7], msg.k[8]]
            ], dtype=torch.float32)
            
            self.mapper.intrinsics_3x3 = intrinsics.to(self.mapper.device)
            self.camera_intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
            self.get_logger().info(f"Camera intrinsics set: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}")

    def pose_callback(self, msg: PoseStamped):
        """Store latest pose for processing."""
        self.latest_pose = msg
        # Debug: Log pose information
        self.get_logger().debug(f"Pose received: pos=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f}), "
                               f"quat=({msg.pose.orientation.x:.3f}, {msg.pose.orientation.y:.3f}, {msg.pose.orientation.z:.3f}, {msg.pose.orientation.w:.3f})")

    def rgb_callback(self, msg: Image):
        """Store latest RGB image for processing."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f"Failed to process RGB image: {e}")

    def depth_callback(self, msg: Image):
        """Process depth frame with RayFronts mapping."""
        if self.camera_intrinsics is None or self.latest_pose is None:
            self.get_logger().warn("No camera intrinsics or pose available yet")
            return

        try:
            # Convert depth image
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_m = self._depth_to_meters(depth, msg.encoding)
            if depth_m is None:
                return

            # Convert to torch tensors and move to correct device
            device = self.mapper.device
            depth_tensor = torch.from_numpy(depth_m).float().unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHxW
            
            # Convert RGB if available
            if hasattr(self, 'latest_rgb'):
                rgb_tensor = torch.from_numpy(self.latest_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0  # 1x3xHxW
            else:
                # Create dummy RGB tensor
                h, w = depth_m.shape
                rgb_tensor = torch.zeros(1, 3, h, w, dtype=torch.float32).to(device)

            # Convert pose to 4x4 matrix
            pose_4x4 = self._pose_to_4x4_matrix(self.latest_pose)

            # Debug: Log transformation matrix
            pose_np = pose_4x4.cpu().numpy()[0]  # Convert back to numpy for logging
            self.get_logger().debug(f"Pose matrix: pos=({pose_np[0,3]:.2f}, {pose_np[1,3]:.2f}, {pose_np[2,3]:.2f})")
            
            # Process with RayFronts mapper
            update_info = self.mapper.process_posed_rgbd(
                rgb_img=rgb_tensor,
                depth_img=depth_tensor,
                pose_4x4=pose_4x4
            )

            # Periodic publishing
            self._periodic_publishing()

        except Exception as e:
            self.get_logger().error(f"Error processing depth frame: {e}")
            import traceback
            traceback.print_exc()

    def _depth_to_meters(self, depth, encoding: str):
        """Convert depth image to meters."""
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

    def _rpy_deg_to_rot(self, rpy_deg):
        """Convert roll-pitch-yaw in degrees to rotation matrix."""
        try:
            import math
            roll, pitch, yaw = [math.radians(float(x)) for x in rpy_deg]
            cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
            return Rz @ Ry @ Rx
        except Exception:
            return np.eye(3, dtype=np.float32)

    def _pose_to_4x4_matrix(self, pose: PoseStamped) -> torch.Tensor:
        """Convert PoseStamped to 4x4 transformation matrix with proper coordinate frame handling."""
        # Position
        p = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)
        
        # Orientation (quaternion to rotation matrix)
        q = np.array([pose.pose.orientation.x, pose.pose.orientation.y, 
                     pose.pose.orientation.z, pose.pose.orientation.w], dtype=np.float32)
        R = self._quat_to_rot(q)
        
        # Apply coordinate frame transformation if pose is in base_link frame
        if bool(self.pose_is_base_link):
            # Transform from base_link to camera frame
            # This is the inverse of the transformation used in depth_octomap_node
            
            # Step 1: Transform pose from base_link to camera frame
            # Apply camera-to-base transformation (inverse)
            p = p - self.t_cam_to_base_extra
            R = R @ self.R_cam_to_base_extra
            
            # Step 2: Apply optical frame rotation (inverse)
            if bool(self.apply_optical_frame_rotation):
                R = R @ self.R_opt_to_base
        
        # Create 4x4 matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = p
        
        # Convert to tensor and move to same device as mapper
        device = self.mapper.device
        return torch.from_numpy(T).unsqueeze(0).to(device)  # 1x4x4

    def _quat_to_rot(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
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
        """Convert RPY angles in degrees to rotation matrix."""
        try:
            roll, pitch, yaw = [math.radians(float(x)) for x in rpy_deg]
            cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
            return Rz @ Ry @ Rx
        except Exception:
            return np.eye(3, dtype=np.float32)

    def _periodic_publishing(self):
        """Periodic publishing of map data."""
        now = time.time()
        
        # Publish occupancy point cloud
        if self.cloud_pub:
            cloud = self._create_occupancy_pointcloud()
            if cloud:
                self.cloud_pub.publish(cloud)
                self.get_logger().debug(f"Published occupancy point cloud")

    def _create_occupancy_pointcloud(self) -> Optional[PointCloud2]:
        """Create occupancy point cloud from RayFronts map."""
        try:
            if self.mapper.is_empty():
                return None
            
            # Get occupancy data from OpenVDB
            import rayfronts_cpp
            pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.mapper.occ_map_vdb)
            
            if pc_xyz_occ_size.shape[0] == 0:
                return None
            
            # Filter occupied voxels
            occupied_mask = pc_xyz_occ_size[:, -2] > 0
            occupied_points = pc_xyz_occ_size[occupied_mask]
            
            if len(occupied_points) == 0:
                return None
            
            # Create PointCloud2 message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.map_frame
            
            # Create structured array with XYZ + occupancy
            cloud_data = np.empty(len(occupied_points), dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32), 
                ('occupancy', np.float32)
            ])
            
            cloud_data['x'] = occupied_points[:, 0]
            cloud_data['y'] = occupied_points[:, 1]
            cloud_data['z'] = occupied_points[:, 2]
            cloud_data['occupancy'] = occupied_points[:, -2]  # Occupancy value
            
            # Create PointCloud2 message
            cloud_msg = PointCloud2()
            cloud_msg.header = header
            
            # Define the fields
            cloud_msg.fields = [
                pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='occupancy', offset=12, datatype=pc2.PointField.FLOAT32, count=1)
            ]
            
            # Set the message properties
            cloud_msg.point_step = 16  # 4 bytes per float * 4 fields
            cloud_msg.width = len(occupied_points)
            cloud_msg.height = 1
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
            cloud_msg.is_dense = True
            
            # Set the data
            cloud_msg.data = cloud_data.tobytes()
            
            return cloud_msg
            
        except Exception as e:
            self.get_logger().error(f"Error creating occupancy point cloud: {e}")
            return None


def main():
    rclpy.init()
    node = RayFrontsMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()