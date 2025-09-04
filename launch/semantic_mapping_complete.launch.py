#!/usr/bin/env python3
"""
Complete Semantic Mapping Launch File

Launches the main resilience node, semantic octomap node, and narration display node together.
The main node handles VLM processing and publishes semantic info,
while the octomap node handles voxel mapping with semantic coloring,
and the narration display node renders narration + queries VLM and publishes /vlm_answer.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    declare_voxel_resolution = DeclareLaunchArgument(
        'voxel_resolution',
        default_value='0.1',
        description='Voxel resolution in meters'
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
    
    # Main Resilience Node (handles VLM processing and semantic info publishing)
    main_resilience_node = Node(
        package='resilience',
        executable='main.py',
        name='resilience_main_node',
        output='screen',
        parameters=[{
            # Add any specific parameters for main node here
        }]
    )
    
    # Semantic OctoMap Node (handles voxel mapping with semantic coloring)
    semantic_octomap_node = Node(
        package='resilience',
        executable='depth_octomap_node.py',
        name='semantic_octomap_node',
        output='screen',
        parameters=[{
            # Sensor topics (from rosbag or live robot)
            'depth_topic': '/robot_1/sensors/front_stereo/depth/depth_registered',
            'camera_info_topic': '/robot_1/sensors/front_stereo/left/camera_info',
            'pose_topic': '/robot_1/sensors/front_stereo/pose',
            
            # Semantic bridge topics (from main node)
            'embedding_topic': '/voxel_embeddings',
            'confidence_topic': '/voxel_confidences',
            
            # Mapping parameters
            'map_frame': 'map',
            'voxel_resolution': LaunchConfiguration('voxel_resolution'),
            'max_range': 1.5,
            'min_range': 0.3,
            'probability_hit': 0.7,
            'probability_miss': 0.4,
            'occupancy_threshold': 0.5,
            
            # Visualization
            'publish_markers': True,
            'publish_stats': True,
            'publish_colored_cloud': True,
            'use_cube_list_markers': True,
            'max_markers': 30000,
            'marker_publish_rate': 1.0,
            'stats_publish_rate': 1.0,
            
            # Transform parameters
            'pose_is_base_link': True,
            'apply_optical_frame_rotation': True,
            'cam_to_base_rpy_deg': [0.0, 0.0, 0.0],
            'cam_to_base_xyz': [0.0, 0.0, 0.0],
            
            # Semantic parameters
            'embedding_dim': 1152,
                            'enable_semantic_mapping': True,
                'semantic_similarity_threshold': LaunchConfiguration('semantic_similarity_threshold'),
                'buffers_directory': LaunchConfiguration('buffers_directory'),
                'main_config_path': '/home/navin/ros2_ws/src/resilience/config/main_config.yaml'
        }],
        remappings=[
            # Remap output topics to avoid conflicts
            ('/semantic_octomap_markers', '/semantic_voxel_markers'),
            ('/semantic_octomap_stats', '/semantic_voxel_stats'),
            ('/semantic_octomap_colored_cloud', '/semantic_voxel_cloud')
        ]
    )

    # Narration Display Node (renders narration, queries VLM, publishes /vlm_answer)
    narration_display_node = Node(
        package='resilience',
        executable='narration_display_node.py',
        name='narration_display_node',
        output='screen',
        parameters=[{}]
    )
    
    return LaunchDescription([
        declare_voxel_resolution,
        declare_semantic_threshold,
        declare_buffers_directory,
        main_resilience_node,
        semantic_octomap_node,
        narration_display_node
    ]) 