import os
from glob import glob

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _collect_available_worlds(pkg_bme_ros2_navigation, worlds_root):
    available_worlds = set()

    package_worlds_dir = os.path.join(pkg_bme_ros2_navigation, 'worlds')
    if os.path.isdir(package_worlds_dir):
        for entry in os.listdir(package_worlds_dir):
            name, ext = os.path.splitext(entry)
            if ext in ('.sdf', '.world'):
                available_worlds.add(name)

    if os.path.isdir(worlds_root):
        for world_dir in os.listdir(worlds_root):
            world_root = os.path.join(worlds_root, world_dir)
            if not os.path.isdir(world_root):
                continue
            if (
                os.path.isfile(os.path.join(world_root, f'{world_dir}.world')) or
                os.path.isfile(os.path.join(world_root, f'{world_dir}.sdf'))
            ):
                available_worlds.add(world_dir)

    return sorted(available_worlds)


def _resolve_world_path(pkg_bme_ros2_navigation, worlds_root, world_name):
    candidate = world_name.strip()
    if not candidate:
        raise RuntimeError('World argument cannot be empty.')

    if os.path.isabs(candidate):
        if os.path.isfile(candidate):
            return candidate
        raise RuntimeError(f"World file does not exist: '{candidate}'")

    for ext in ('.sdf', '.world'):
        package_path = os.path.join(pkg_bme_ros2_navigation, 'worlds', f'{candidate}{ext}')
        if os.path.isfile(package_path):
            return package_path

    extra_dir = os.path.join(worlds_root, candidate)
    for ext in ('.world', '.sdf'):
        extra_path = os.path.join(extra_dir, f'{candidate}{ext}')
        if os.path.isfile(extra_path):
            return extra_path

    available_worlds = _collect_available_worlds(pkg_bme_ros2_navigation, worlds_root)
    raise RuntimeError(
        f"Unknown world '{world_name}'. Available worlds: {', '.join(available_worlds)}"
    )


def _build_world_launch(context, pkg_ros_gz_sim, pkg_bme_ros2_navigation, worlds_root):
    selected_world = LaunchConfiguration('world').perform(context)
    world_path = _resolve_world_path(pkg_bme_ros2_navigation, worlds_root, selected_world)
    gz_args = f'{world_path} -r -v -v1'

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
            ),
            launch_arguments={
                'gz_args': gz_args,
                'on_exit_shutdown': 'true',
            }.items(),
        )
    ]


def generate_launch_description():
    pkg_bme_ros2_navigation = get_package_share_directory('bme_ros2_navigation')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    workspace_root = os.path.abspath(
        os.path.join(pkg_bme_ros2_navigation, '..', '..', '..', '..')
    )
    worlds_root = os.path.join(workspace_root, 'simulation', 'worlds')
    extra_world_model_paths = [
        model_path for model_path in glob(os.path.join(worlds_root, '*', 'models'))
        if os.path.isdir(model_path)
    ]
    resource_paths = [
        os.path.dirname(pkg_bme_ros2_navigation),
        os.path.join(
            workspace_root,
            'simulation',
            'Week-7-8-ROS2-Navigation',
            'gazebo_models',
        ),
    ] + extra_world_model_paths
    current_resource_paths = [
        p for p in os.environ.get('GZ_SIM_RESOURCE_PATH', '').split(os.pathsep) if p
    ]
    for path in resource_paths:
        if os.path.isdir(path) and path not in current_resource_paths:
            current_resource_paths.append(path)
    os.environ['GZ_SIM_RESOURCE_PATH'] = os.pathsep.join(current_resource_paths)

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='bookstore',
        description="World name (e.g. 'bookstore', 'corridor') or absolute world file path",
    )

    world_launch = OpaqueFunction(
        function=_build_world_launch,
        args=[pkg_ros_gz_sim, pkg_bme_ros2_navigation, worlds_root],
    )

    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(world_launch)
    return launchDescriptionObject
