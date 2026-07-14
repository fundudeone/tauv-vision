import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile


def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration("name").perform(context)
    params_file = ParameterFile(LaunchConfiguration("params_file"), allow_substs=True)

    remappings = [
        ("rgb/image", name + "/rgb/image_raw"),
        ("rgb/camera_info", name + "/rgb/camera_info"),
        ("depth/image", name + "/stereo/image_raw"),
        ("odom", "/odometry/filtered"),
    ]

    return [
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[
                params_file,
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
            remappings=remappings,
            # Belt-and-suspenders alongside database_path:"" in the params
            # file -- wipe any stale on-disk db instead of loading it.
            arguments=["--delete_db_on_start"],
        ),
    ]


def generate_launch_description():
    tauv_vision_prefix = get_package_share_directory("tauv_vision")

    declared_arguments = [
        DeclareLaunchArgument(
            "name",
            default_value="oak",
            description="Topic namespace the OAK-D publishes under (live on the vehicle, or as recorded in a rosbag).",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Set true when replaying a rosbag with `ros2 bag play --clock`.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(
                tauv_vision_prefix, "config", "rtabmap_pointcloud.yaml"
            ),
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
