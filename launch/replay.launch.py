from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription 
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """Replay a recorded bag through bannerTask and expose the result to Foxglove.

    The bag carries rgb as compressed only, so it is republished to raw before
    bannerTask sees it. Depth, camera_info and tf come straight off the bag, so
    no rtabmap/odometry stack is needed here.
    """
    # The bag lives in the source tree (too large to install into share/), and
    # the repo is mounted at /tauv-mono inside the container.
    default_bag = (
        '/tauv-mono/ros_ws/src/tauv_vision/rosbag_osprey_2026.07.14_14.23.09'
    )

    bag_arg = DeclareLaunchArgument('bag', default_value=default_bag)
    rate_arg = DeclareLaunchArgument('rate', default_value='1.0')
    start_arg = DeclareLaunchArgument('start_offset', default_value='0.0')

    play_bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', LaunchConfiguration('bag'),
            '--clock',
            '--rate', LaunchConfiguration('rate'),
            '--start-offset', LaunchConfiguration('start_offset'),
            '--delay', '3.0',
        ],
        output='screen',
    )

    # `--start-offset` drops every message before the offset, and /tf_static is
    # a single burst at the head of the bag -- so playing from an offset would
    # otherwise lose the whole static transform tree and break every TF lookup.
    # This node latches those transforms regardless of where playback starts.
    tf_static = Node(
        package='tauv_vision',
        executable='tfStaticFromBag',
        name='tf_static_from_bag',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'bag': LaunchConfiguration('bag'),
        }],
    )

    decompressor = Node(
        package='image_transport',
        executable='republish',
        name='rgb_decompressor',
        arguments=[
            'compressed', 'in/compressed:=oak/rgb/image_raw/compressed',
            'raw', 'out:=oak/rgb/image_raw',
        ],
        parameters=[{'use_sim_time': True}],
    )

    # banner_task = Node(
    #     package='tauv_vision',
    #     executable='bannerTask',
    #     name='banner_task',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': True,
    #     }],
    # )

    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'use_sim_time': True,
            'port': 9876,
            'send_buffer_limit': 10000000,  # increase for pointclouds
            'capabilities': ['client_publish', 'connection_graph', 'parameter_updates'],
        }],
    )

    rtablaunch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('tauv_vision'), 'launch', 'rtabmap_pointcloud.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    return LaunchDescription([
        bag_arg,
        rate_arg,
        start_arg,
        tf_static,
        play_bag,
        decompressor,
        rtablaunch,
        foxglove,
    ])
