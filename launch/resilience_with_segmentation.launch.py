#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directory
    package_dir = get_package_share_directory('resilience')
    
    # Default config path
    default_config_path = os.path.join(package_dir, 'config', 'combined_segmentation_config.yaml')
    
    # Launch arguments
    enable_combined_segmentation_arg = DeclareLaunchArgument(
        'enable_combined_segmentation',
        default_value='true',
        description='Enable combined segmentation processing'
    )
    
    segmentation_config_path_arg = DeclareLaunchArgument(
        'segmentation_config_path',
        default_value=default_config_path,
        description='Path to combined segmentation configuration file'
    )
    
    publish_original_mask_arg = DeclareLaunchArgument(
        'publish_original_mask',
        default_value='true',
        description='Publish original segmentation mask'
    )
    
    publish_refined_mask_arg = DeclareLaunchArgument(
        'publish_refined_mask',
        default_value='true',
        description='Publish refined segmentation mask with DBSCAN'
    )
    
    # NARadio parameters
    radio_model_version_arg = DeclareLaunchArgument(
        'radio_model_version',
        default_value='radio_v2.5-b',
        description='NARadio model version'
    )
    
    radio_lang_model_arg = DeclareLaunchArgument(
        'radio_lang_model',
        default_value='siglip',
        description='NARadio language model'
    )
    
    radio_input_resolution_arg = DeclareLaunchArgument(
        'radio_input_resolution',
        default_value='512',
        description='NARadio input resolution'
    )
    
    enable_naradio_visualization_arg = DeclareLaunchArgument(
        'enable_naradio_visualization',
        default_value='true',
        description='Enable NARadio visualization'
    )
    
    # YOLO-SAM parameters
    min_confidence_arg = DeclareLaunchArgument(
        'min_confidence',
        default_value='0.05',
        description='Minimum confidence for YOLO detections'
    )
    
    min_detection_distance_arg = DeclareLaunchArgument(
        'min_detection_distance',
        default_value='0.5',
        description='Minimum detection distance'
    )
    
    max_detection_distance_arg = DeclareLaunchArgument(
        'max_detection_distance',
        default_value='2.0',
        description='Maximum detection distance'
    )
    
    # Topics
    rgb_topic_arg = DeclareLaunchArgument(
        'rgb_topic',
        default_value='/robot_1/sensors/front_stereo/right/image',
        description='RGB image topic'
    )
    
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/robot_1/sensors/front_stereo/depth/depth_registered',
        description='Depth image topic'
    )
    
    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic',
        default_value='/robot_1/sensors/front_stereo/pose',
        description='Pose topic'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/robot_1/sensors/front_stereo/right/camera_info',
        description='Camera info topic'
    )
    
    # Resilience node
    resilience_node = Node(
        package='resilience',
        executable='main.py',
        name='resilience_node',
        output='screen',
        parameters=[{
            # Combined Segmentation parameters
            'enable_combined_segmentation': LaunchConfiguration('enable_combined_segmentation'),
            'segmentation_config_path': LaunchConfiguration('segmentation_config_path'),
            'publish_original_mask': LaunchConfiguration('publish_original_mask'),
            'publish_refined_mask': LaunchConfiguration('publish_refined_mask'),
            
            # NARadio parameters
            'radio_model_version': LaunchConfiguration('radio_model_version'),
            'radio_lang_model': LaunchConfiguration('radio_lang_model'),
            'radio_input_resolution': LaunchConfiguration('radio_input_resolution'),
            'enable_naradio_visualization': LaunchConfiguration('enable_naradio_visualization'),
            
            # YOLO-SAM parameters
            'min_confidence': LaunchConfiguration('min_confidence'),
            'min_detection_distance': LaunchConfiguration('min_detection_distance'),
            'max_detection_distance': LaunchConfiguration('max_detection_distance'),
            
            # Topics
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'pose_topic': LaunchConfiguration('pose_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
        }],
        remappings=[
            ('/rgb_image', LaunchConfiguration('rgb_topic')),
            ('/depth_image', LaunchConfiguration('depth_topic')),
            ('/pose', LaunchConfiguration('pose_topic')),
            ('/camera_info', LaunchConfiguration('camera_info_topic')),
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        enable_combined_segmentation_arg,
        segmentation_config_path_arg,
        publish_original_mask_arg,
        publish_refined_mask_arg,
        radio_model_version_arg,
        radio_lang_model_arg,
        radio_input_resolution_arg,
        enable_naradio_visualization_arg,
        min_confidence_arg,
        min_detection_distance_arg,
        max_detection_distance_arg,
        rgb_topic_arg,
        depth_topic_arg,
        pose_topic_arg,
        camera_info_topic_arg,
        
        # Nodes
        resilience_node,
    ]) 