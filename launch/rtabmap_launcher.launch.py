import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration("name").perform(context)
    depthai_prefix = get_package_share_directory("depthai_ros_driver_v3")

    params_file = LaunchConfiguration("params_file")

    parameters = [
        {
            "frame_id": "os/base_link",
            "subscribe_rgb": True,
            "subscribe_depth": True,
            "subscribe_odom_info": False,
            "approx_sync": True,
            # RTAB-Map's parameters should be strings:
            'Mem/NotLinkedNodesKept':'false',
            "Mem/RehearsalSimilarity": "0.85",
            "Rtabmap/DetectionRate": "3.0",
            'visual_odometry' : 'false',
            "RGBD/LinearUpdate": "0.0",
            "RGBD/AngularUpdate": "0.0",
            "cloud_voxel_size": "0.01",       # lower = denser (default is 0.05)
            "cloud_decimation": "4",          # sample every Nth pixel
            "Proj/MaxDepth": "3.5",           # cap depth to where stereo is actually reliable
            "Proj/MinDepth": "0.3",           # ignore points too close for stereo to resolve
        }
    ]

    remappings = [
        ("rgb/image", name + "/rgb/image_raw"),
        ("rgb/camera_info", name + "/rgb/camera_info"),
        ("depth/image", name + "/stereo/image_raw"),
        ("odom", "/odometry/filtered")
    ]

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(depthai_prefix, "launch", "driver.launch.py")
            ),
            launch_arguments={"name": name, "params_file": params_file}.items(),
        ),
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            parameters=parameters,
            remappings=remappings,
            arguments=["--delete_db_on_start"] # delete cached map while testing since its garbage until this works
        ),
        # extremely annoying hack to make sure that the camera parent transform is just os/base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            # Arguments: x y z yaw pitch roll parent_frame child_frame
            arguments = ['0', '0', '0', '0', '3.141592', '0', 'os/base_link', 'oak_parent_frame']
        )
    ]


def generate_launch_description():
    tauv_vision_prefix = get_package_share_directory("tauv_vision")
    declared_arguments = [
        DeclareLaunchArgument("name", default_value="oak"),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(tauv_vision_prefix, "config", "rtab_launch.yaml"),
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )