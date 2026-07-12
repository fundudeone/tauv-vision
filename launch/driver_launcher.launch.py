from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.actions import TimerAction
import os
from launch_ros.actions import Node

def generate_launch_description():
    mapping_launch_path = os.path.join(
        get_package_share_directory('tauv_vision'), 
        'launch', 
        'driver.launch.py'
    )

    config_yaml_path = os.path.join(
        get_package_share_directory('tauv_vision'),
        'config',
        'rtab_launch.yaml'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mapping_launch_path),
            launch_arguments={
                'name': 'oak',
                'params_file': config_yaml_path,
            }.items()
        ),

        Node(
            name='camera_transform_publisher',
            package='tf2_ros',
            executable='static_transform_publisher',
            # Arguments: x y z yaw pitch roll parent_frame child_frame
            arguments = ['0.19', '0', '-0.04', '3.141592', '3.141592', '0', 'os/base_link', 'oak_parent_frame']
        )
    ])