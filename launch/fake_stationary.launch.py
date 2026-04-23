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
from launch.actions import ExecuteProcess

publishPoseScript = """
import rclpy
from nav_msgs.msg import Odometry;
rclpy.init()
n=rclpy.create_node("fake_odom_node")
p=n.create_publisher(Odometry, "/odometry/filtered", 10)

def c(): 
    m=Odometry()
    m.header.stamp=n.get_clock().now().to_msg()
    m.header.frame_id="odom"
    m.child_frame_id="base_link"
    m.pose.pose.orientation.w=1.0
    p.publish(m)

n.create_timer(0.1, c)
rclpy.spin(n)
"""

def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration("name").perform(context)

    return [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            # Arguments: x y z yaw pitch roll parent_frame child_frame
            arguments = ['0', '0', '0', '0', '0', '0', 'odom', 'os/base_link']
        ),
        ExecuteProcess(
            cmd=['python3', '-c', publishPoseScript],
            output='screen'
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