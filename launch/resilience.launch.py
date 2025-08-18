#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'min_confidence',
            default_value='0.05',
            description='Minimum confidence for YOLO detection'
        ),
        DeclareLaunchArgument(
            'min_detection_distance',
            default_value='0.5',
            description='Minimum detection distance in meters'
        ),
        DeclareLaunchArgument(
            'max_detection_distance',
            default_value='2.0',
            description='Maximum detection distance in meters'
        ),
        DeclareLaunchArgument(
            'yolo_model',
            default_value='yolov8l-world.pt',
            description='YOLO model to use'
        ),
        DeclareLaunchArgument(
            'yolo_imgsz',
            default_value='480',
            description='YOLO input image size'
        ),
        DeclareLaunchArgument(
            'flip_y_axis',
            default_value='false',
            description='Whether to flip Y axis for coordinate system'
        ),
        DeclareLaunchArgument(
            'use_tf',
            default_value='false',
            description='Whether to use TF for coordinate transforms'
        ),
        DeclareLaunchArgument(
            'disable_yolo_printing',
            default_value='true',
            description='Whether to disable YOLO default printing'
        ),
        DeclareLaunchArgument(
            'radio_model_version',
            default_value='radio_v2.5-b',
            description='NARadio model version'
        ),
        DeclareLaunchArgument(
            'radio_lang_model',
            default_value='siglip',
            description='NARadio language model'
        ),
        DeclareLaunchArgument(
            'radio_input_resolution',
            default_value='512',
            description='NARadio input resolution'
        ),
        DeclareLaunchArgument(
            'enable_naradio_visualization',
            default_value='true',
            description='Whether to enable NARadio visualization'
        ),
        
        # Launch the resilience node
        Node(
            package='resilience',
            executable='main.py',
            name='resilience_node',
            output='screen',
            parameters=[{
                'min_confidence': LaunchConfiguration('min_confidence'),
                'min_detection_distance': LaunchConfiguration('min_detection_distance'),
                'max_detection_distance': LaunchConfiguration('max_detection_distance'),
                'yolo_model': LaunchConfiguration('yolo_model'),
                'yolo_imgsz': LaunchConfiguration('yolo_imgsz'),
                'flip_y_axis': LaunchConfiguration('flip_y_axis'),
                'use_tf': LaunchConfiguration('use_tf'),
                'disable_yolo_printing': LaunchConfiguration('disable_yolo_printing'),
                'radio_model_version': LaunchConfiguration('radio_model_version'),
                'radio_lang_model': LaunchConfiguration('radio_lang_model'),
                'radio_input_resolution': LaunchConfiguration('radio_input_resolution'),
                'enable_naradio_visualization': LaunchConfiguration('enable_naradio_visualization'),
            }],
            remappings=[
                ('/robot_1/sensors/front_stereo/right/image', '/robot_1/sensors/front_stereo/right/image'),
                ('/robot_1/sensors/front_stereo/depth/depth_registered', '/robot_1/sensors/front_stereo/depth/depth_registered'),
                ('/robot_1/sensors/front_stereo/pose', '/robot_1/sensors/front_stereo/pose'),
                ('/robot_1/sensors/front_stereo/right/camera_info', '/robot_1/sensors/front_stereo/right/camera_info'),
                ('/vlm_answer', '/vlm_answer'),
            ]
        )
    ]) 