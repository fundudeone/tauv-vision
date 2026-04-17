from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.actions import TimerAction
import os

def generate_launch_description():
    mapping_launch_path = os.path.join(
        get_package_share_directory('depthai_ros_driver_v3'), 
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
                'params_file': config_yaml_path
            }.items()
        ),
    ])