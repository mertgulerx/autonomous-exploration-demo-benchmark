# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare reproducible per-worker files without owning simulation processes."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

from .known_pose import (
    BASE_FRAME,
    GROUND_TRUTH_RATE_HZ,
    GROUND_TRUTH_RECORD_PERIOD_NS,
    MAPPING_CONDITION,
    add_ground_truth_publisher,
    ground_truth_bridge,
    model_frame,
)
from .mapping_scan import FILL_OFFSET_M, MAPPING_SCAN_TOPIC, THRESHOLD_MARGIN_M


# The E0 moving-object system (world_collection/plugins/moving_objects) and the ROS topic its pose
# message is bridged to for the worker's startup check (docs/plans/worlds/I5, open point 1 (a)).
MOVING_OBJECTS_SYSTEM = "benchmark::MovingObjects"
MOVING_OBJECTS_TOPIC = "/benchmark/moving_objects/pose"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remove_unused_sensors(root):
    """Preserve links/inertias/collisions while suppressing unused rendering."""
    removed = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "sensor" and child.get("type") in (
                "camera",
                "depth_camera",
                "rgbd_camera",
                "thermal_camera",
                "navsat",
            ):
                removed.append(child.get("name"))
                parent.remove(child)
    return removed


def visualization_description(description, lookup):
    """Resolve vendor meshes in the worker's SDK for the observer's RobotModel.

    This is a presentation copy. The spawned native description stays intact.
    SDKs can contain different revisions of identically named mesh packages.
    """
    root = ET.fromstring(description)
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("package://"):
            package, separator, relative = filename[len("package://") :].partition("/")
            if not separator:
                raise ValueError(f"Invalid robot mesh URI: {filename}")
            path = Path(lookup(package)) / relative
            if not path.is_file():
                raise ValueError(f"Robot mesh unavailable: {path}")
            mesh.set("filename", path.resolve().as_uri())
    return ET.tostring(root, encoding="unicode")


def navigation_parameters(baseline, profile):
    parameters = deepcopy(baseline)
    if profile.get("adapter") == "mogi":
        settings = profile.get("benchmark_overrides", {})
        server = parameters["controller_server"]["ros__parameters"]
        follow = server["FollowPath"]
        smoother = parameters["velocity_smoother"]["ros__parameters"]
        for name, axis, mppi, dwb, threshold in (
            ("movement_speed", 0, "vx_max", "max_vel_x", "min_x_velocity_threshold"),
            (
                "turning_speed",
                2,
                "wz_max",
                "max_vel_theta",
                "min_theta_velocity_threshold",
            ),
        ):
            if name not in settings:
                continue
            speed = settings[name]
            if "MPPIController" in follow["plugin"]:
                follow[mppi] = speed
                if axis == 0:
                    follow["vx_min"] = max(-speed, min(follow["vx_min"], speed))
            elif "DWBLocalPlanner" in follow["plugin"]:
                follow[dwb] = speed
                if axis == 0:
                    follow["max_speed_xy"] = speed
            else:
                raise ValueError(
                    "MOGI speed settings require the shared MPPI or DWB controller"
                )
            smoother["max_velocity"][axis] = speed
            smoother["min_velocity"][axis] = -speed
            smoother["deadband_velocity"][axis] = min(
                smoother["deadband_velocity"][axis], speed / 10
            )
            server[threshold] = min(server[threshold], speed / 10)
        if "turning_speed" in settings:
            behavior = parameters["behavior_server"]["ros__parameters"]
            behavior["max_rotational_vel"] = settings["turning_speed"]
            behavior["min_rotational_vel"] = min(
                behavior["min_rotational_vel"], settings["turning_speed"]
            )
        if "lidar_range" in settings:
            for costmap in ("local_costmap", "global_costmap"):
                values = parameters[costmap][costmap]["ros__parameters"]
                for layer in values.values():
                    if not isinstance(layer, dict):
                        continue
                    for source in layer.get("observation_sources", "").split():
                        for key in ("raytrace_max_range", "obstacle_max_range"):
                            if key in layer[source]:
                                layer[source][key] = min(
                                    layer[source][key], settings["lidar_range"]
                                )
        return parameters
    vx, vy, wz = (
        profile[k]
        for k in ("max_linear_speed", "max_lateral_speed", "max_angular_speed")
    )
    ax, aw = profile["linear_acceleration"], profile["angular_acceleration"]
    ay = profile.get("lateral_acceleration", ax) if vy else 0.0
    dx = profile.get("linear_deceleration", ax)
    dy = profile.get("lateral_deceleration", ay) if vy else 0.0
    dw = profile.get("angular_deceleration", aw)
    server = parameters["controller_server"]["ros__parameters"]
    follow = server["FollowPath"]
    if "MPPIController" in follow["plugin"]:
        follow.update(
            vx_max=vx,
            vy_max=vy,
            wz_max=wz,
            ax_max=ax,
            ax_min=-dx,
            ay_max=ay,
            az_max=min(aw, dw),
            motion_model="Omni" if vy else "DiffDrive",
        )
    elif "DWBLocalPlanner" in follow["plugin"]:
        follow.update(
            max_vel_x=vx,
            max_vel_y=vy,
            max_vel_theta=wz,
            max_speed_xy=vx,
            acc_lim_x=ax,
            acc_lim_y=ay,
            acc_lim_theta=aw,
            decel_lim_x=-dx,
            decel_lim_y=-dy,
            decel_lim_theta=-dw,
        )
        if vy:
            follow["vy_samples"] = max(5, follow.get("vy_samples", 1))
    else:
        raise ValueError(
            "Robot limit adaptation requires the shared MPPI or DWB controller"
        )
    if vy:
        server["min_y_velocity_threshold"] = min(
            server["min_y_velocity_threshold"], 0.03
        )
    for costmap in ("local_costmap", "global_costmap"):
        values = parameters[costmap][costmap]["ros__parameters"]
        values.pop("robot_radius", None)
        values["footprint"] = json.dumps(profile["footprint"])
        padding = values.get("footprint_padding", 0.01)
        radius = max(
            math.hypot(abs(x) + padding, abs(y) + padding)
            for x, y in profile["footprint"]
        )
        values["inflation_layer"]["inflation_radius"] = max(
            values["inflation_layer"]["inflation_radius"], radius + 0.01
        )
    smoother = parameters["velocity_smoother"]["ros__parameters"]
    smoother.update(
        max_velocity=[vx, vy, wz],
        min_velocity=[-vx, -vy, -wz],
        max_accel=[ax, ay, aw],
        max_decel=[-dx, -dy, -dw],
    )
    behavior = parameters["behavior_server"]["ros__parameters"]
    behavior["max_rotational_vel"] = min(behavior["max_rotational_vel"], wz)
    behavior["rotational_acc_lim"] = min(behavior["rotational_acc_lim"], aw)
    return parameters


def enforce_drive_limits(root, profile):
    """Constrain the native Gazebo diff-drive actuator using platform settings."""
    if profile.get("adapter") == "mogi":
        settings = profile.get("benchmark_overrides", {})
        values = {
            tag: settings[name]
            for name, tag in (
                ("movement_speed", "max_linear_velocity"),
                ("turning_speed", "max_angular_velocity"),
            )
            if name in settings
        }
    else:
        values = dict(
            max_linear_velocity=profile["max_linear_speed"],
            max_angular_velocity=profile["max_angular_speed"],
            max_linear_acceleration=profile["linear_acceleration"],
            max_angular_acceleration=profile["angular_acceleration"],
        )
    for plugin in root.iter("plugin"):
        if not plugin.get("name", "").endswith("::DiffDrive"):
            continue
        if profile.get("adapter") == "mogi" and values:
            generic = plugin.find("max_velocity")
            if generic is not None:
                generic.text = str(max(float(generic.text), *values.values()))
        for key, value in values.items():
            for tag, limit in ((key, value), (key.replace("max_", "min_", 1), -value)):
                element = plugin.find(tag)
                if element is None:
                    element = ET.SubElement(plugin, tag)
                element.text = str(limit)


def turtlebot3_lidar(root, profile):
    """Apply the catalog's LDS-02 range to the pinned legacy LDS model.

    The upstream scan sampling, update rate and Gaussian noise remain unchanged.
    Retain the native sensor pose, frames and physical robot geometry.
    """
    lidar = profile["lidar"]
    if lidar.get("model") != "LDS-02":
        raise ValueError("TurtleBot3 requires the benchmark LDS-02 lidar profile")
    sensors = [
        sensor for sensor in root.iter("sensor") if sensor.get("type") == "gpu_lidar"
    ]
    if len(sensors) != 1:
        raise ValueError("TurtleBot3 model must contain exactly one GPU lidar")
    sensor = sensors[0]
    # The pinned Waffle Pi uses SDF's legacy <ray> spelling, Burger <lidar>.
    limits = sensor.find("lidar/range")
    if limits is None:
        limits = sensor.find("ray/range")
    if limits is None or limits.find("min") is None or limits.find("max") is None:
        raise ValueError("TurtleBot3 lidar model is missing its range limits")
    sensor.set("name", "lds_02")
    limits.find("min").text = str(lidar["range_min"])
    limits.find("max").text = str(lidar["range_max"])


def turtlebot3_sensor_frames(root):
    """Use the native link frame instead of Gazebo's scoped sensor name.

    The bridge preserves this header; robot_state_publisher publishes the
    corresponding link. Sensor poses, noise and update rates stay native.
    """
    for link in root.iter("link"):
        for sensor in link.findall("sensor"):
            if sensor.get("type") != "imu":
                continue
            frame = sensor.find("gz_frame_id")
            if frame is None:
                frame = ET.SubElement(sensor, "gz_frame_id")
            frame.text = link.get("name")


def system_library(filename, search_path=None):
    """The library Gazebo loads for a system plugin's SDF ``filename``.

    Workers launch a copy of the world, so a system library is found only through
    GZ_SIM_SYSTEM_PLUGIN_PATH, searched in order like Gazebo does.
    """
    if search_path is None:
        search_path = os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    names = [filename, "lib" + filename + ".so", filename + ".so"]
    for directory in filter(None, search_path.split(os.pathsep)):
        for name in names:
            path = Path(directory) / name
            if path.is_file():
                return path.resolve()
    raise ValueError(
        f"Moving-object system library {filename!r} is not on GZ_SIM_SYSTEM_PLUGIN_PATH"
    )


def moving_objects(scenario, world_root, search_path=None):
    """Provenance of a world's moving objects: (record, bridge entry), or (None, None).

    The registry entry (``scenario["moving_objects"]``) marks a moving-object world. Its
    ``world.yaml`` (``scenario["source"]["world_yaml"]``) holds the E0 block, which must agree
    with the registry entry and with the system plugin element of the world SDF. The record goes
    into the attempt's manifest, the bridge entry serves the worker's startup check.
    """
    entry = scenario.get("moving_objects")
    if not entry:
        return None, None
    if not isinstance(entry, dict):
        raise ValueError("Registry moving_objects must be a mapping")
    source = scenario.get("source") or {}
    path = Path(source.get("world_yaml") or "")
    if not path.is_absolute() or not path.is_file():
        raise ValueError(
            "Moving-object world requires an absolute source.world_yaml: " + str(path)
        )
    digest = sha256(path)
    if source.get("world_yaml_sha256") != digest:
        raise ValueError(f"world.yaml changed since registration: {path}")
    block = (yaml.safe_load(path.read_text()) or {}).get("moving_objects")
    if not isinstance(block, dict) or not isinstance(block.get("objects"), list):
        raise ValueError(f"world.yaml has no moving_objects block with objects: {path}")
    ids = [item.get("id") for item in block["objects"] if isinstance(item, dict)]
    if not ids or len(ids) != len(block["objects"]) or len(set(ids)) != len(ids):
        raise ValueError(f"world.yaml moving objects need unique ids: {path}")
    differing = sorted(
        key for key in entry if key in block and entry[key] != block[key]
    )
    if "count" in entry and entry["count"] != len(ids):
        differing.append("count (objects)")
    if differing:
        raise ValueError(
            "Registry moving_objects disagrees with world.yaml: " + ", ".join(differing)
        )
    system = MOVING_OBJECTS_SYSTEM
    if block.get("system", system) != system:
        raise ValueError(f"world.yaml moving objects need the {system} system: {path}")
    plugins = [
        plugin
        for plugin in world_root.find("world").findall("plugin")
        if plugin.get("name") == system
    ]
    if len(plugins) != 1:
        raise ValueError(f"World SDF must declare the {system} system exactly once")
    plugin = plugins[0]
    declared = dict(
        topic=(plugin.findtext("topic") or "").strip(),
        velocity_mode=(plugin.findtext("velocity_mode") or "").strip(),
        objects=[
            (item.findtext("model") or "").strip() for item in plugin.findall("object")
        ],
    )
    expected = dict(
        topic=block.get("topic"), velocity_mode=block.get("velocity_mode"), objects=ids
    )
    differing = sorted(key for key in declared if declared[key] != expected[key])
    if differing:
        raise ValueError(
            "World SDF moving objects disagree with world.yaml: " + ", ".join(differing)
        )
    library = system_library(plugin.get("filename", ""), search_path)
    record = dict(
        registry=entry,
        world_yaml=str(path),
        world_yaml_sha256=digest,
        block=block,
        objects=ids,
        topic=MOVING_OBJECTS_TOPIC,
        plugin=dict(
            name=system,
            filename=plugin.get("filename"),
            library=str(library),
            library_sha256=sha256(library),
        ),
    )
    bridge = dict(
        ros_topic_name=MOVING_OBJECTS_TOPIC,
        gz_topic_name=block["topic"],
        ros_type_name="tf2_msgs/msg/TFMessage",
        gz_type_name="gz.msgs.Pose_V",
        direction="GZ_TO_ROS",
    )
    return record, bridge


def spawn_pose(profile, scenario):
    """Resolve only height; benchmark XY/yaw and MOGI's pose stay unchanged."""
    pose = list(scenario["spawn"])
    if profile.get("spawn_z_policy") != "scenario":
        pose[2] = profile.get("spawn_z_by_world", {}).get(
            scenario["world"], profile["spawn_z"]
        )
    return pose


def mapping_parameters():
    """The condition-defining values of MAPPING_CONDITION (comparison_profile)."""
    return dict(
        fill_offset_m=FILL_OFFSET_M,
        threshold_margin_m=THRESHOLD_MARGIN_M,
        ground_truth_rate_hz=GROUND_TRUTH_RATE_HZ,
        ground_truth_record_period_ns=GROUND_TRUTH_RECORD_PERIOD_NS,
    )


def prepare_known_pose(profile, directory, generated, bridge):
    """Ground-truth pose for every run (mapping-known-pose-v1, I10).

    The PosePublisher goes into the spawned model and its bridge into the bridge
    list; known_pose_tf turns the model pose into odom->base_footprint. Writes
    known_pose.yaml and ``generated["known_pose"]``.
    """
    spawn = ET.parse(generated["spawn_file"])
    frame = model_frame(spawn.getroot(), ET.fromstring(generated["robot_description"]))
    settings = add_ground_truth_publisher(spawn.getroot())
    spawn.write(generated["spawn_file"], encoding="unicode", xml_declaration=True)
    bridge.append(ground_truth_bridge())
    parameters = dict(
        use_sim_time=True,
        model=profile["model"],
        model_frame=frame,
        base_frame=BASE_FRAME,
        odom_frame="odom",
        spawn_pose=[float(value) for value in generated["spawn_pose"]],
    )
    path = Path(directory) / "known_pose.yaml"
    path.write_text(
        yaml.safe_dump({"known_pose_tf": {"ros__parameters": parameters}})
    )
    generated["known_pose"] = dict(
        parameters=str(path),
        model_frame=frame,
        ground_truth_settings=settings,
        tf="odom->base_footprint, planar, stamped with the ground-truth sim time",
        odom_origin="base_footprint at the commanded spawn pose",
    )


def prepare_mapping(profile, directory, generated, shared):
    """Per-robot SLAM Toolbox file and the mapping scan filter configuration.

    Scans are placed at the known pose (no scan matching, no loop closing) and the
    mapping topic carries no-return beams as V = range_max - FILL_OFFSET_M below
    the sensor maximum; max_laser_range = range_max - THRESHOLD_MARGIN_M keeps V
    beyond Karto's range threshold, so such beams clear free space and mark no
    obstacle. Writes slam.yaml, mapping_scan.yaml and ``generated["slam"]``,
    ``generated["mapping_scan"]`` and ``generated["mapping_condition"]``.
    """
    range_max = float(profile["lidar"]["range_max"])
    threshold = round(range_max - THRESHOLD_MARGIN_M, 6)
    slam = yaml.safe_load(Path(shared).read_text())
    values = slam["slam_toolbox"]["ros__parameters"]
    values.update(
        scan_topic=MAPPING_SCAN_TOPIC,
        use_scan_matching=False,
        do_loop_closing=False,
        max_laser_range=threshold,
        scan_buffer_maximum_scan_distance=threshold,
    )
    slam_path = Path(directory) / "slam.yaml"
    slam_path.write_text(yaml.safe_dump(slam, sort_keys=False))
    generated["slam"] = str(slam_path)
    filter_path = Path(directory) / "mapping_scan.yaml"
    parameters = dict(
        use_sim_time=True,
        input_topic="/scan",
        output_topic=MAPPING_SCAN_TOPIC,
        max_laser_range=threshold,
        fill_offset=FILL_OFFSET_M,
    )
    filter_path.write_text(
        yaml.safe_dump({"mapping_scan_filter": {"ros__parameters": parameters}})
    )
    generated["mapping_scan"] = dict(
        parameters=str(filter_path),
        input="/scan",
        output=MAPPING_SCAN_TOPIC,
        replaced="readings >= message range_max (no-return +inf included)",
        replacement=f"message range_max - {FILL_OFFSET_M}",
        unchanged="NaN, -inf and every reading below range_max",
        catalog_range_max=range_max,
        max_laser_range=threshold,
    )
    generated["mapping_condition"] = dict(
        id=MAPPING_CONDITION,
        pose_source="Gazebo PosePublisher model pose via known_pose_tf",
        ground_truth_rate_hz=GROUND_TRUTH_RATE_HZ,
        scan_matching=False,
        loop_closing=False,
        slam_shared_source=str(shared),
    )


def prepare(profile, scenario, directory, lookup=None):
    from ament_index_python.packages import get_package_share_directory
    import xacro

    lookup = lookup or get_package_share_directory
    directory = Path(directory) / "robot_configuration"
    directory.mkdir(parents=True, exist_ok=True)
    baseline = Path(lookup("bme_ros2_navigation"))
    nav = navigation_parameters(
        yaml.safe_load((baseline / "config/navigation.yaml").read_text()), profile
    )
    nav_path = directory / "navigation.yaml"
    nav_path.write_text(yaml.safe_dump(nav, sort_keys=False))
    world = ET.parse(scenario["world_file"])
    physics = world.getroot().find("world/physics")
    if physics is None:
        physics = ET.SubElement(
            world.getroot().find("world"), "physics", name="benchmark", type="ode"
        )
    step = physics.find("max_step_size")
    if step is None:
        step = ET.SubElement(physics, "max_step_size")
    step.text = str(profile["physics_step"])
    world_path = directory / "world.sdf"
    world.write(world_path, encoding="unicode", xml_declaration=True)
    bridge = [
        dict(
            ros_topic_name="/clock",
            gz_topic_name="/world/" + scenario["world_name"] + "/clock",
            ros_type_name="rosgraph_msgs/msg/Clock",
            gz_type_name="gz.msgs.Clock",
            direction="GZ_TO_ROS",
        )
    ]
    generated = dict(
        profile=profile,
        world=str(world_path),
        scenario=scenario,
        spawn_pose=spawn_pose(profile, scenario),
        directory=str(directory),
    )
    movers, mover_bridge = moving_objects(scenario, world.getroot())
    if movers:
        # Recorded with or without diagnostics; worlds without movers record nothing new.
        generated["moving_objects"] = movers
        bridge.append(mover_bridge)
    family = profile["adapter"]
    if family == "mogi":
        description = xacro.process_file(str(baseline / "urdf/mogi_bot.urdf")).toxml()
        plant = ET.fromstring(description)
        enforce_drive_limits(plant, profile)
        if "lidar_range" in profile.get("benchmark_overrides", {}):
            sensors = [s for s in plant.iter("sensor") if s.get("type") == "gpu_lidar"]
            if len(sensors) != 1 or sensors[0].find("lidar/range/max") is None:
                raise ValueError(
                    "MOGI description must contain one GPU lidar with range limits"
                )
            sensors[0].find("lidar/range/max").text = str(profile["lidar"]["range_max"])
        generated["disabled_sensors"] = remove_unused_sensors(plant)
        description = ET.tostring(plant, encoding="unicode")
        generated["robot_description"] = description
        generated["spawn_file"] = str(directory / "robot.urdf")
        Path(generated["spawn_file"]).write_text(description)
        for entry in yaml.safe_load((baseline / "config/gz_bridge.yaml").read_text()):
            if entry["ros_type_name"] not in (
                "rosgraph_msgs/msg/Clock",
                "sensor_msgs/msg/CameraInfo",
            ):
                bridge.append(entry)
        # Known pose (I10): the EKF keeps odometry/filtered but no longer owns the TF.
        ekf = yaml.safe_load((baseline / "config/ekf.yaml").read_text())
        ekf["ekf_filter_node"]["ros__parameters"]["publish_tf"] = False
        generated["ekf"] = str(directory / "ekf.yaml")
        Path(generated["ekf"]).write_text(yaml.safe_dump(ekf, sort_keys=False))
    elif family == "turtlebot3":
        share = Path(lookup("turtlebot3_gazebo"))
        generated["resource_paths"] = [str(share / "models")]
        generated["robot_description"] = (
            share / "urdf" / (profile["model"] + ".urdf")
        ).read_text()
        sdf = ET.parse(share / "models" / profile["model"] / "model.sdf")
        turtlebot3_lidar(sdf.getroot(), profile)
        enforce_drive_limits(sdf.getroot(), profile)
        turtlebot3_sensor_frames(sdf.getroot())
        generated["disabled_sensors"] = remove_unused_sensors(sdf.getroot())
        # The model publishes measured differential-drive odometry; its TF is not
        # bridged (known pose, I10: the ground-truth node owns odom->base_footprint).
        generated["spawn_file"] = str(directory / "robot.sdf")
        sdf.write(generated["spawn_file"], encoding="unicode", xml_declaration=True)
        for topic, ros, gz, direction in (
            ("/cmd_vel", "geometry_msgs/msg/Twist", "gz.msgs.Twist", "ROS_TO_GZ"),
            ("/odom", "nav_msgs/msg/Odometry", "gz.msgs.Odometry", "GZ_TO_ROS"),
            ("/imu", "sensor_msgs/msg/Imu", "gz.msgs.IMU", "GZ_TO_ROS"),
            ("/scan", "sensor_msgs/msg/LaserScan", "gz.msgs.LaserScan", "GZ_TO_ROS"),
            (
                "/joint_states",
                "sensor_msgs/msg/JointState",
                "gz.msgs.Model",
                "GZ_TO_ROS",
            ),
        ):
            bridge.append(
                dict(
                    ros_topic_name=topic,
                    gz_topic_name=topic,
                    ros_type_name=ros,
                    gz_type_name=gz,
                    direction=direction,
                )
            )
    else:
        # Native integrations append spawn/control actions and resolved files.
        from .wheeled import prepare_wheeled

        prepare_wheeled(profile, scenario, directory, lookup, generated, bridge)
    if profile.get("self_filter"):
        raw = "/robot/scan/raw"
        for entry in bridge:
            if entry["ros_topic_name"] == "/scan":
                entry["ros_topic_name"] = raw
        settings = dict(
            profile["self_filter"], box_frame="base_footprint", invert=False
        )
        filter_config = directory / "scan_filter.yaml"
        filter_config.write_text(
            yaml.safe_dump(
                {
                    "scan_to_scan_filter_chain": {
                        "ros__parameters": {
                            "use_sim_time": True,
                            "filter1": {
                                "name": "robot_body",
                                "type": "laser_filters/LaserScanBoxFilter",
                                "params": settings,
                            },
                        }
                    }
                },
                sort_keys=False,
            )
        )
        generated["scan_filter"] = str(filter_config)
        generated["scan_preprocessing"] = dict(
            type="laser_filters/LaserScanBoxFilter",
            input=raw,
            output="/scan",
            rejected_value="NaN",
            parameters=settings,
        )
    prepare_known_pose(profile, directory, generated, bridge)
    bridge_path = directory / "bridge.yaml"
    bridge_path.write_text(yaml.safe_dump(bridge, sort_keys=False))
    generated.update(bridge=str(bridge_path), navigation=str(nav_path))
    prepare_mapping(
        profile, directory, generated, baseline / "config/slam_toolbox_mapping.yaml"
    )
    slam_path = Path(generated["slam"])
    generated["visualization_description"] = visualization_description(
        generated["robot_description"], lookup
    )
    generated.setdefault("resource_paths", []).extend(
        sorted({str(Path(lookup(package)).parent) for package in profile["packages"]})
    )
    # Sensor identity is independent of worker-specific controller file paths.
    model = ET.parse(generated["spawn_file"]).getroot()
    sensors = [
        ET.tostring(element, encoding="unicode") for element in model.iter("sensor")
    ]
    generated["sensor_configuration"] = sensors
    generated["comparison_profile"] = dict(
        robot_id=profile["id"],
        profile_version=profile["version"],
        catalog_profile_sha256=profile.get("catalog_profile_sha256"),
        dependencies=profile.get("dependencies", {}),
        physics_step=profile["physics_step"],
        physics_sha256=hashlib.sha256(
            ET.tostring(world.getroot().find("world/physics"))
        ).hexdigest(),
        sensor_sha256=hashlib.sha256(
            json.dumps(sensors, sort_keys=True).encode()
        ).hexdigest(),
        scan_preprocessing=generated.get("scan_preprocessing"),
        odometry_provenance=generated.get("odometry_provenance"),
        description_sha256=hashlib.sha256(
            Path(generated["spawn_file"])
            .read_text()
            .replace(str(directory), "${WORKER_DIRECTORY}")
            .encode()
        ).hexdigest(),
        controllers_sha256=(
            sha256(directory / "controllers.yaml")
            if (directory / "controllers.yaml").is_file()
            else None
        ),
        navigation_sha256=sha256(nav_path),
        slam_sha256=sha256(slam_path),
        mapping_condition=MAPPING_CONDITION,
        mapping_parameters=mapping_parameters(),
    )
    for path in directory.rglob("*"):
        if path.is_file():
            generated.setdefault("sha256", {})[str(path.relative_to(directory))] = (
                sha256(path)
            )
    config = directory / "launch.json"
    config.write_text(json.dumps(generated, indent=2))
    return generated, str(config)
