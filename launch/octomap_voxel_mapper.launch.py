#!/usr/bin/env python3
"""
Launch file for OctoMap Voxel Mapper

This launch file starts the OctoMap voxel mapping node with configurable parameters
for different scenarios and robot configurations.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    package_dir = get_package_share_directory('resilience')
    
    # Declare launch arguments
    launch_args = [
        # Basic parameters
        DeclareLaunchArgument(
            'voxel_resolution',
            default_value='0.1',
            description='Voxel size in meters (smaller = higher resolution)'
        ),
        DeclareLaunchArgument(
            'max_range',
            default_value='10.0',
            description='Maximum sensor range in meters'
        ),
        DeclareLaunchArgument(
            'min_range',
            default_value='0.1',
            description='Minimum sensor range in meters'
        ),
        
        # Probability parameters
        DeclareLaunchArgument(
            'probability_hit',
            default_value='0.7',
            description='Probability for occupied voxels (0.5-1.0)'
        ),
        DeclareLaunchArgument(
            'probability_miss',
            default_value='0.4',
            description='Probability for free voxels (0.0-0.5)'
        ),
        DeclareLaunchArgument(
            'occupancy_threshold',
            default_value='0.5',
            description='Threshold for occupied voxels (0.0-1.0)'
        ),
        
        # Topic configuration
        DeclareLaunchArgument(
            'point_cloud_topic',
            default_value='/robot_1/sensors/front_stereo/depth/points',
            description='Point cloud topic name'
        ),
        DeclareLaunchArgument(
            'pose_topic',
            default_value='/robot_1/sensors/front_stereo/pose',
            description='Pose topic name'
        ),
        DeclareLaunchArgument(
            'rgb_topic',
            default_value='/robot_1/sensors/front_stereo/right/image',
            description='RGB image topic name (optional)'
        ),
        
        # Frame configuration
        DeclareLaunchArgument(
            'map_frame',
            default_value='map',
            description='Map frame ID'
        ),
        DeclareLaunchArgument(
            'sensor_frame',
            default_value='camera_link',
            description='Sensor frame ID'
        ),
        
        # Visualization parameters
        DeclareLaunchArgument(
            'publish_markers',
            default_value='true',
            description='Publish visualization markers'
        ),
        DeclareLaunchArgument(
            'publish_grid',
            default_value='true',
            description='Publish 2D occupancy grid'
        ),
        DeclareLaunchArgument(
            'publish_colored_cloud',
            default_value='true',
            description='Publish colored point cloud'
        ),
        
        # Processing parameters
        DeclareLaunchArgument(
            'filter_outliers',
            default_value='true',
            description='Enable statistical outlier removal'
        ),
        DeclareLaunchArgument(
            'downsample_voxels',
            default_value='true',
            description='Enable voxel downsampling'
        ),
        
        # Performance parameters
        DeclareLaunchArgument(
            'update_frequency',
            default_value='2.0',
            description='Map update frequency in Hz'
        ),
        DeclareLaunchArgument(
            'visualization_frequency',
            default_value='1.0',
            description='Visualization publishing frequency in Hz'
        ),
        
        # Storage parameters
        DeclareLaunchArgument(
            'save_maps_enabled',
            default_value='true',
            description='Enable automatic map saving'
        ),
        DeclareLaunchArgument(
            'map_save_directory',
            default_value='/home/navin/ros2_ws/src/buffers/octomaps',
            description='Directory to save maps'
        ),
        
        # Advanced parameters
        DeclareLaunchArgument(
            'clamping_thres_min',
            default_value='0.12',
            description='Minimum clamping threshold'
        ),
        DeclareLaunchArgument(
            'clamping_thres_max',
            default_value='0.97',
            description='Maximum clamping threshold'
        ),
        
        # Node configuration
        DeclareLaunchArgument(
            'node_name',
            default_value='octomap_voxel_mapper',
            description='Name of the mapper node'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Log level (debug, info, warn, error)'
        ),
    ]
    
    # Create the node
    octomap_node = Node(
        package='resilience',
        executable='octomap_voxel_mapper.py',
        name=LaunchConfiguration('node_name'),
        output='screen',
        parameters=[{
            # Basic parameters
            'voxel_resolution': LaunchConfiguration('voxel_resolution'),
            'max_range': LaunchConfiguration('max_range'),
            'min_range': LaunchConfiguration('min_range'),
            
            # Probability parameters
            'probability_hit': LaunchConfiguration('probability_hit'),
            'probability_miss': LaunchConfiguration('probability_miss'),
            'occupancy_threshold': LaunchConfiguration('occupancy_threshold'),
            'clamping_thres_min': LaunchConfiguration('clamping_thres_min'),
            'clamping_thres_max': LaunchConfiguration('clamping_thres_max'),
            
            # Topic configuration
            'point_cloud_topic': LaunchConfiguration('point_cloud_topic'),
            'pose_topic': LaunchConfiguration('pose_topic'),
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            
            # Frame configuration
            'map_frame': LaunchConfiguration('map_frame'),
            'sensor_frame': LaunchConfiguration('sensor_frame'),
            
            # Visualization parameters
            'publish_markers': LaunchConfiguration('publish_markers'),
            'publish_grid': LaunchConfiguration('publish_grid'),
            'publish_colored_cloud': LaunchConfiguration('publish_colored_cloud'),
            
            # Processing parameters
            'filter_outliers': LaunchConfiguration('filter_outliers'),
            'downsample_voxels': LaunchConfiguration('downsample_voxels'),
            
            # Performance parameters
            'update_frequency': LaunchConfiguration('update_frequency'),
            'visualization_frequency': LaunchConfiguration('visualization_frequency'),
            
            # Storage parameters
            'save_maps_enabled': LaunchConfiguration('save_maps_enabled'),
            'map_save_directory': LaunchConfiguration('map_save_directory'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    # Info messages
    info_messages = [
        LogInfo(msg=[
            'Starting OctoMap Voxel Mapper with parameters:',
            '\n  Voxel resolution: ', LaunchConfiguration('voxel_resolution'), 'm',
            '\n  Sensor range: ', LaunchConfiguration('min_range'), 'm - ', LaunchConfiguration('max_range'), 'm',
            '\n  Update frequency: ', LaunchConfiguration('update_frequency'), 'Hz',
            '\n  Point cloud topic: ', LaunchConfiguration('point_cloud_topic'),
            '\n  Pose topic: ', LaunchConfiguration('pose_topic'),
            '\n  Map frame: ', LaunchConfiguration('map_frame'),
        ]),
    ]
    
    return LaunchDescription(launch_args + info_messages + [octomap_node])


# Preset configurations for common use cases
def generate_high_resolution_config():
    """High resolution mapping configuration."""
    return {
        'voxel_resolution': '0.05',
        'update_frequency': '1.0',
        'visualization_frequency': '0.5',
        'filter_outliers': 'true',
        'downsample_voxels': 'true',
    }


def generate_fast_mapping_config():
    """Fast mapping configuration for real-time use."""
    return {
        'voxel_resolution': '0.2',
        'update_frequency': '5.0',
        'visualization_frequency': '2.0',
        'filter_outliers': 'false',
        'downsample_voxels': 'true',
    }


def generate_memory_efficient_config():
    """Memory efficient configuration for limited resources."""
    return {
        'voxel_resolution': '0.15',
        'max_range': '5.0',
        'update_frequency': '1.0',
        'visualization_frequency': '0.5',
        'publish_markers': 'false',
        'publish_colored_cloud': 'false',
        'downsample_voxels': 'true',
    } 