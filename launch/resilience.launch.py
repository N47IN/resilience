#!/usr/bin/env python3
"""
Complete Resilience System Launch File

Launches the full resilience system with three main components:
1. Main resilience node (main.py) - handles drift detection, NARadio processing, and semantic mapping
2. Depth octomap node (depth_octomap_node.py) - creates semantic voxel maps with reduced max_range
3. Narration display node (narration_display_node.py) - displays narration and queries VLM

This launch file provides a complete system for resilience monitoring with semantic mapping.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Declare launch arguments for main resilience node
    declare_flip_y_axis = DeclareLaunchArgument(
        'flip_y_axis',
        default_value='false',
        description='Whether to flip Y axis for coordinate system'
    )
    
    declare_use_tf = DeclareLaunchArgument(
        'use_tf',
        default_value='false',
        description='Whether to use TF for coordinate transforms'
    )
    
    declare_radio_model_version = DeclareLaunchArgument(
        'radio_model_version',
        default_value='radio_v2.5-b',
        description='NARadio model version'
    )
    
    declare_radio_lang_model = DeclareLaunchArgument(
        'radio_lang_model',
        default_value='siglip',
        description='NARadio language model'
    )
    
    declare_radio_input_resolution = DeclareLaunchArgument(
        'radio_input_resolution',
        default_value='512',
        description='NARadio input resolution'
    )
    
    declare_enable_naradio_visualization = DeclareLaunchArgument(
        'enable_naradio_visualization',
        default_value='true',
        description='Whether to enable NARadio visualization'
    )
    
    declare_enable_combined_segmentation = DeclareLaunchArgument(
        'enable_combined_segmentation',
        default_value='true',
        description='Whether to enable combined segmentation'
    )
    
    declare_enable_voxel_mapping = DeclareLaunchArgument(
        'enable_voxel_mapping',
        default_value='true',
        description='Whether to enable voxel mapping'
    )
    
    declare_pose_is_base_link = DeclareLaunchArgument(
        'pose_is_base_link',
        default_value='true',
        description='Whether pose is base link'
    )
    
    # Declare launch arguments for depth octomap node
    declare_max_range = DeclareLaunchArgument(
        'max_range',
        default_value='1.5',
        description='Maximum depth range for octomap'
    )
    
    declare_voxel_resolution = DeclareLaunchArgument(
        'voxel_resolution',
        default_value='0.1',
        description='Voxel resolution in meters'
    )
    
    declare_min_range = DeclareLaunchArgument(
        'min_range',
        default_value='0.1',
        description='Minimum depth range'
    )
    
    declare_enable_semantic_mapping = DeclareLaunchArgument(
        'enable_semantic_mapping',
        default_value='true',
        description='Enable semantic voxel mapping with VLM similarity'
    )
    
    declare_semantic_similarity_threshold = DeclareLaunchArgument(
        'semantic_similarity_threshold',
        default_value='0.6',
        description='Cosine similarity threshold for semantic labeling'
    )
    
    declare_buffers_directory = DeclareLaunchArgument(
        'buffers_directory',
        default_value='/home/navin/ros2_ws/src/buffers',
        description='Directory containing saved buffer data with VLM embeddings'
    )
    
    # Declare launch arguments for narration display node
    declare_vlm_api_key = DeclareLaunchArgument(
        'vlm_api_key',
        default_value='',
        description='API key for VLM service (optional)'
    )
    
    declare_vlm_base_url = DeclareLaunchArgument(
        'vlm_base_url',
        default_value='http://localhost:8000/v1',
        description='Base URL for VLM API'
    )
    
    declare_vlm_model = DeclareLaunchArgument(
        'vlm_model',
        default_value='llava-v1.6-mistral-7b',
        description='VLM model to use'
    )
    
    declare_save_images = DeclareLaunchArgument(
        'save_images',
        default_value='true',
        description='Whether to save images with narration text'
    )
    
    # Main Resilience Node
    main_resilience_node = Node(
        package='resilience',
        executable='main.py',
        name='resilience_node',
        output='screen',
        parameters=[{
            'flip_y_axis': LaunchConfiguration('flip_y_axis'),
            'use_tf': LaunchConfiguration('use_tf'),
            'radio_model_version': LaunchConfiguration('radio_model_version'),
            'radio_lang_model': LaunchConfiguration('radio_lang_model'),
            'radio_input_resolution': LaunchConfiguration('radio_input_resolution'),
            'enable_naradio_visualization': LaunchConfiguration('enable_naradio_visualization'),
            'enable_combined_segmentation': LaunchConfiguration('enable_combined_segmentation'),
            'enable_voxel_mapping': LaunchConfiguration('enable_voxel_mapping'),
            'pose_is_base_link': LaunchConfiguration('pose_is_base_link'),
            'main_config_path': '',
            'mapping_config_path': ''
        }],
        remappings=[
            ('/robot_1/sensors/front_stereo/right/image', '/robot_1/sensors/front_stereo/right/image'),
            ('/robot_1/sensors/front_stereo/depth/depth_registered', '/robot_1/sensors/front_stereo/depth/depth_registered'),
            ('/robot_1/sensors/front_stereo/pose', '/robot_1/sensors/front_stereo/pose'),
            ('/robot_1/sensors/front_stereo/right/camera_info', '/robot_1/sensors/front_stereo/right/camera_info'),
            ('/vlm_answer', '/vlm_answer'),
        ]
    )
    
    # Depth Octomap Node (with reduced max_range for semantic mapping)
    depth_octomap_node = Node(
        package='resilience',
        executable='depth_octomap_node.py',
        name='semantic_depth_octomap_node',
        output='screen',
        parameters=[{
            'depth_topic': '/robot_1/sensors/front_stereo/depth/depth_registered',
            'camera_info_topic': '/robot_1/sensors/front_stereo/left/camera_info',
            'pose_topic': '/robot_1/sensors/front_stereo/pose',
            'map_frame': 'map',
            'voxel_resolution': LaunchConfiguration('voxel_resolution'),
            'max_range': LaunchConfiguration('max_range'),
            'min_range': LaunchConfiguration('min_range'),
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
            'pose_is_base_link': LaunchConfiguration('pose_is_base_link'),
            'apply_optical_frame_rotation': True,
            'cam_to_base_rpy_deg': [0.0, 0.0, 0.0],
            'cam_to_base_xyz': [0.0, 0.0, 0.0],
            'embedding_dim': 1152,
            'enable_semantic_mapping': LaunchConfiguration('enable_semantic_mapping'),
            'semantic_similarity_threshold': LaunchConfiguration('semantic_similarity_threshold'),
            'buffers_directory': LaunchConfiguration('buffers_directory'),
            'bridge_queue_max_size': 100,
            'bridge_queue_process_interval': 0.1,
            'enable_voxel_mapping': LaunchConfiguration('enable_voxel_mapping'),
            'sync_buffer_seconds': 2.0,
            'inactivity_threshold_seconds': 2.5,
            'semantic_export_directory': LaunchConfiguration('buffers_directory'),
            'mapping_config_path': '',
            'nominal_path': '/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json',
            'main_config_path': ''
        }],
        remappings=[
            ('/semantic_hotspots', '/semantic_hotspots'),
            ('/semantic_hotspot_mask', '/semantic_hotspot_mask'),
            ('/semantic_octomap_markers', '/semantic_octomap_markers'),
            ('/semantic_octomap_stats', '/semantic_octomap_stats'),
            ('/semantic_octomap_colored_cloud', '/semantic_octomap_colored_cloud'),
            ('/semantic_voxels_only', '/semantic_voxels_only'),
            ('/gp_field_visualization', '/gp_field_visualization'),
            ('/semantic_costmap', '/semantic_costmap')
        ]
    )
    
    # Narration Display Node
    narration_display_node = Node(
        package='resilience',
        executable='narration_display_node.py',
        name='narration_display_node',
        output='screen',
        parameters=[{
            'vlm_api_key': LaunchConfiguration('vlm_api_key'),
            'vlm_base_url': LaunchConfiguration('vlm_base_url'),
            'vlm_model': LaunchConfiguration('vlm_model'),
            'save_images': LaunchConfiguration('save_images')
        }],
        remappings=[
            ('/narration_image', '/narration_image'),
            ('/narration_text', '/narration_text'),
            ('/vlm_answer', '/vlm_answer')
        ]
    )
    
    return LaunchDescription([
        # Main resilience node arguments
        declare_flip_y_axis,
        declare_use_tf,
        declare_radio_model_version,
        declare_radio_lang_model,
        declare_radio_input_resolution,
        declare_enable_naradio_visualization,
        declare_enable_combined_segmentation,
        declare_enable_voxel_mapping,
        declare_pose_is_base_link,
        
        # Depth octomap node arguments
        declare_max_range,
        declare_voxel_resolution,
        declare_min_range,
        declare_enable_semantic_mapping,
        declare_semantic_similarity_threshold,
        declare_buffers_directory,
        
        # Narration display node arguments
        declare_vlm_api_key,
        declare_vlm_base_url,
        declare_vlm_model,
        declare_save_images,
        
        # Launch all nodes
        main_resilience_node,
        depth_octomap_node,
        narration_display_node
    ])
