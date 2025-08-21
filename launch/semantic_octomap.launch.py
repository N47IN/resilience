#!/usr/bin/env python3
"""
Semantic OctoMap Launch File

Launches the enhanced depth octomap node with semantic voxel mapping capabilities.
This node creates occupancy maps and overlays semantic labels based on VLM similarity.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Declare launch arguments
    declare_depth_topic = DeclareLaunchArgument(
        'depth_topic',
        default_value='/robot_1/sensors/front_stereo/depth/depth_registered',
        description='Depth image topic'
    )
    
    declare_camera_info_topic = DeclareLaunchArgument(
        'camera_info_topic', 
        default_value='/robot_1/sensors/front_stereo/left/camera_info',
        description='Camera info topic'
    )
    
    declare_pose_topic = DeclareLaunchArgument(
        'pose_topic',
        default_value='/robot_1/sensors/front_stereo/pose',
        description='Camera/robot pose topic'
    )
    
    declare_embedding_topic = DeclareLaunchArgument(
        'embedding_topic',
        default_value='',
        description='Optional per-pixel embedding topic'
    )
    
    declare_confidence_topic = DeclareLaunchArgument(
        'confidence_topic',
        default_value='',
        description='Optional per-pixel confidence topic'
    )
    
    declare_vlm_answer_topic = DeclareLaunchArgument(
        'vlm_answer_topic',
        default_value='/vlm_answer',
        description='VLM answer topic for semantic labeling'
    )
    
    declare_voxel_resolution = DeclareLaunchArgument(
        'voxel_resolution',
        default_value='0.1',
        description='Voxel resolution in meters'
    )
    
    declare_max_range = DeclareLaunchArgument(
        'max_range',
        default_value='5.0',
        description='Maximum depth range'
    )
    
    declare_semantic_threshold = DeclareLaunchArgument(
        'semantic_similarity_threshold',
        default_value='0.6',
        description='Cosine similarity threshold for semantic labeling'
    )
    
    declare_buffers_directory = DeclareLaunchArgument(
        'buffers_directory',
        default_value='/home/navin/ros2_ws/src/buffers',
        description='Directory containing saved buffer data with VLM embeddings'
    )
    
    declare_enable_semantic = DeclareLaunchArgument(
        'enable_semantic_mapping',
        default_value='true',
        description='Enable semantic voxel mapping with VLM similarity'
    )
    
    # Semantic OctoMap Node
    semantic_octomap_node = Node(
        package='resilience',
        executable='depth_octomap_node.py',
        name='semantic_octomap_node',
        output='screen',
        parameters=[{
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'pose_topic': LaunchConfiguration('pose_topic'),
            'embedding_topic': LaunchConfiguration('embedding_topic'),
            'confidence_topic': LaunchConfiguration('confidence_topic'),
            'vlm_answer_topic': LaunchConfiguration('vlm_answer_topic'),
            'map_frame': 'map',
            'voxel_resolution': LaunchConfiguration('voxel_resolution'),
            'max_range': LaunchConfiguration('max_range'),
            'min_range': 0.1,
            'probability_hit': 0.7,
            'probability_miss': 0.4,
            'occupancy_threshold': 0.5,
            'publish_markers': True,
            'publish_stats': True,
            'publish_colored_cloud': True,
            'use_cube_list_markers': True,
            'max_markers': 30000,
            'marker_publish_rate': 1.0,
            'stats_publish_rate': 1.0,
            'pose_is_base_link': True,
            'apply_optical_frame_rotation': True,
            'cam_to_base_rpy_deg': [0.0, 0.0, 0.0],
            'cam_to_base_xyz': [0.0, 0.0, 0.0],
            'use_embeddings': True,
            'embedding_dim': 1152,
            'default_hit_confidence': 1.0,
            'enable_semantic_mapping': LaunchConfiguration('enable_semantic_mapping'),
            'semantic_similarity_threshold': LaunchConfiguration('semantic_similarity_threshold'),
            'buffers_directory': LaunchConfiguration('buffers_directory')
        }],
        remappings=[
            ('/octomap_markers', '/semantic_octomap_markers'),
            ('/octomap_stats', '/semantic_octomap_stats'),
            ('/octomap_colored_cloud', '/semantic_octomap_colored_cloud')
        ]
    )
    
    return LaunchDescription([
        declare_depth_topic,
        declare_camera_info_topic,
        declare_pose_topic,
        declare_embedding_topic,
        declare_confidence_topic,
        declare_vlm_answer_topic,
        declare_voxel_resolution,
        declare_max_range,
        declare_semantic_threshold,
        declare_buffers_directory,
        declare_enable_semantic,
        semantic_octomap_node
    ]) 