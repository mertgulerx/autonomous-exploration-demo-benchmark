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

"""Vendor wheeled descriptions/controllers without vendor worlds or navigation."""

from pathlib import Path
import math
import xml.etree.ElementTree as ET
import yaml
from .adapters import remove_unused_sensors


def add_bridge(bridge, ros, gz, ros_type, gz_type, direction="GZ_TO_ROS"):
    bridge.append(
        dict(
            ros_topic_name=ros,
            gz_topic_name=gz,
            ros_type_name=ros_type,
            gz_type_name=gz_type,
            direction=direction,
        )
    )


def sensor_interfaces(plant, bridge, selected_lidar=None):
    """Only transport names change; sensor range/noise/rate remain native."""
    for group in plant.findall("gazebo"):
        for sensor in list(group.findall("sensor")):
            kind = sensor.get("type")
            if kind == "gpu_lidar":
                if selected_lidar and sensor.get("name") != selected_lidar:
                    group.remove(sensor)
                    continue
                ros, message, gz = (
                    "/scan",
                    "sensor_msgs/msg/LaserScan",
                    "gz.msgs.LaserScan",
                )
            elif kind == "imu":
                ros, message, gz = "/imu", "sensor_msgs/msg/Imu", "gz.msgs.IMU"
            else:
                continue
            topic = sensor.find("topic")
            if topic is None:
                topic = ET.SubElement(sensor, "topic")
            topic.text = "/benchmark" + ros
            frame = sensor.find("gz_frame_id")
            if frame is None:
                frame = ET.SubElement(sensor, "gz_frame_id")
            frame.text = group.get("reference")
            add_bridge(bridge, ros, topic.text, message, gz)
    # World owns its rendering/contact/IMU systems.
    for parent in plant.iter():
        for child in list(parent):
            if child.tag == "plugin" and child.get("name", "").endswith("::Sensors"):
                parent.remove(child)


def save_description(plant, directory, generated):
    generated["disabled_sensors"] = remove_unused_sensors(plant)
    generated["robot_description"] = ET.tostring(plant, encoding="unicode")
    generated["spawn_file"] = str(directory / "robot.urdf")
    Path(generated["spawn_file"]).write_text(generated["robot_description"])


def ensure_footprint(plant):
    if plant.find("link[@name='base_link']") is None:
        raise ValueError(
            "Native description has no base_link; check xacro expansion before spawning"
        )
    # Re-root the fixed ancestor chain, including vendors that put the footprint
    # below a body_link rather than directly below base_link. Physical relative
    # transforms remain identical and the controller is the sole odom TF owner.
    incoming = {j.find("child").get("link"): j for j in plant.findall("joint")}
    child = "base_footprint"
    ancestors = []
    while child in incoming:
        joint = incoming[child]
        if joint.get("type") != "fixed":
            raise ValueError("Native footprint ancestor must be fixed")
        ancestors.append(joint)
        child = joint.find("parent").get("link")
    for joint in ancestors:
        parent, child = joint.find("parent"), joint.find("child")
        parent_name = parent.get("link")
        parent.set("link", child.get("link"))
        child.set("link", parent_name)
        origin = joint.find("origin")
        if origin is not None:
            roll, pitch, yaw = map(float, origin.get("rpy", "0 0 0").split())
            if abs(roll) > 1e-9 or abs(pitch) > 1e-9:
                raise ValueError(
                    "Non-planar native footprint transform requires a dedicated adapter"
                )
            x, y, z = map(float, origin.get("xyz", "0 0 0").split())
            c, s = math.cos(yaw), math.sin(yaw)
            origin.set("xyz", f"{-c*x-s*y} {s*x-c*y} {-z}")
            origin.set("rpy", f"0 0 {-yaw}")
    if plant.find("link[@name='base_footprint']") is None:
        ET.SubElement(plant, "link", name="base_footprint")
        joint = ET.SubElement(
            plant, "joint", name="benchmark_footprint_joint", type="fixed"
        )
        ET.SubElement(joint, "parent", link="base_footprint")
        ET.SubElement(joint, "child", link="base_link")
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")


def replace_controller_file(plant, filename):
    for plugin in plant.iter("plugin"):
        if "ros2_control" in plugin.get("filename", ""):
            parameters = plugin.find("parameters")
            if parameters is None:
                parameters = ET.SubElement(plugin, "parameters")
            parameters.text = str(filename)


def remap_controller(plant, name):
    for plugin in plant.iter("plugin"):
        if "ros2_control" not in plugin.get("filename", ""):
            continue
        ros = plugin.find("ros")
        if ros is None:
            ros = ET.SubElement(plugin, "ros")
        targets = {
            name + "/odom": "/odom",
            name + "/odometry": "/odom",
            name + "/cmd_vel": "/" + name + "/cmd_vel",
            "~/odom": "/odom",
            "~/odometry": "/odom",
            "~/cmd_vel": "/" + name + "/cmd_vel",
            "~/reference": "/" + name + "/reference",
            "joint_states": "/joint_states",
        }
        for rule in list(ros.findall("remapping")):
            if (rule.text or "").split(":=", 1)[0].lstrip("/") in targets:
                ros.remove(rule)
        for source, target in targets.items():
            ET.SubElement(ros, "remapping").text = source + ":=" + target


def prepare_wheeled(profile, scenario, directory, lookup, generated, bridge):
    import xacro

    adapter = profile["adapter"]
    if adapter == "turtlebot4":
        source = Path(lookup("irobot_create_control")) / "config/control.yaml"
        controls = yaml.safe_load(source.read_text())["/**"]
        controls["controller_manager"]["ros__parameters"]["use_sim_time"] = True
        drive = controls["diffdrive_controller"]["ros__parameters"]
        drive.update(
            use_sim_time=True,
            base_frame_id="base_footprint",
            enable_odom_tf=True,
            use_stamped_vel=True,
        )
        controller_file = directory / "controllers.yaml"
        controller_file.write_text(yaml.safe_dump(controls, sort_keys=False))
        description = (
            Path(lookup("turtlebot4_description"))
            / "urdf"
            / profile["model"]
            / "turtlebot4.urdf.xacro"
        )
        plant = ET.fromstring(
            xacro.process_file(
                str(description), mappings={"gazebo": "ignition", "namespace": ""}
            ).toxml()
        )
        # The zero-length upstream footprint joint is reversed so odometry has
        # the benchmark's canonical tree; this does not move any physical link.
        ensure_footprint(plant)
        replace_controller_file(plant, controller_file)
        remap_controller(plant, "diffdrive_controller")
        sensor_interfaces(plant, bridge, "rplidar")
        generated["controllers"] = ["joint_state_broadcaster", "diffdrive_controller"]
        generated["command_topic"] = "/diffdrive_controller/cmd_vel"
        generated["odometry_provenance"] = (
            "Native closed-loop ros2_control wheel feedback"
        )
        save_description(plant, directory, generated)
    elif adapter == "rosbot":
        prepare_rosbot(profile, directory, lookup, generated, bridge)
    elif adapter == "husarion_ugv":
        prepare_ugv(profile, directory, lookup, generated, bridge)
    else:
        raise ValueError(f"{profile['label']}: unsupported native adapter")


def prepare_rosbot(profile, directory, lookup, generated, bridge):
    import xacro

    model = profile["model"]
    drive_name = "differential_drive_controller"
    source = Path(lookup("rosbot_controller")) / "config" / model / "controllers.yaml"
    content = (
        source.read_text()
        .replace("<namespace>/", "")
        .replace("<manipulator_state>", "inactive")
    )
    content = content.replace("<mecanum>", "False").replace(
        "<drive_controller>", drive_name
    )
    controls = yaml.safe_load(content)
    drive = controls["/**"][drive_name]["ros__parameters"]
    drive.update(use_sim_time=True, enable_odom_tf=True, base_frame_id="base_footprint")
    if profile.get("measurement_odom_topic"):
        drive["enable_odom_tf"] = False
    controls["/**"]["controller_manager"]["ros__parameters"]["use_sim_time"] = True
    controller_file = directory / "controllers.yaml"
    controller_file.write_text(yaml.safe_dump(controls, sort_keys=False))
    share = Path(lookup("rosbot_description"))
    mappings = dict(
        use_sim="True",
        controller_config=str(controller_file),
        namespace="",
        components_config=str(
            share / "config" / model / (profile["configuration"] + ".yaml")
        ),
        mecanum="False",
    )
    plant = ET.fromstring(
        xacro.process_file(
            str(share / "urdf" / (model + ".urdf.xacro")), mappings=mappings
        ).toxml()
    )
    ensure_footprint(plant)
    sensor_interfaces(plant, bridge, "lidar")
    generated["controllers"] = ["joint_state_broadcaster", drive_name]
    generated["command_topic"] = "/" + drive_name + "/cmd_vel"
    generated["odometry_provenance"] = (
        "Native closed-loop Husarion ros2_control wheel feedback"
    )
    generated["drive_controller"] = drive_name
    # Namespace-free native controller output is the sole odometry authority.
    remap_controller(plant, drive_name)
    if profile.get("measurement_odom_topic"):
        for rule in plant.iter("remapping"):
            if (rule.text or "").endswith(":=/odom"):
                rule.text = (
                    rule.text.split(":=", 1)[0]
                    + ":="
                    + profile["measurement_odom_topic"]
                )
        params = yaml.safe_load(
            (Path(lookup("rosbot_localization")) / "config/config.yaml").read_text()
        )["/**"]["ekf_node"]["ros__parameters"]
        params.update(
            use_sim_time=True,
            base_link_frame="base_footprint",
            odom0=profile["measurement_odom_topic"],
            imu0="/imu",
        )
        filename = directory / "ekf.yaml"
        filename.write_text(
            yaml.safe_dump(
                {"ekf_filter_node": {"ros__parameters": params}}, sort_keys=False
            )
        )
        generated["ekf"] = str(filename)
        generated["odometry_provenance"] = (
            "Native ROSbot EKF: measured wheel velocities and IMU yaw/rate; raw wheel metrics retained"
        )
    save_description(plant, directory, generated)


def prepare_ugv(profile, directory, lookup, generated, bridge):
    import xacro

    model = profile["model"]
    wheels = "WH01" if model == "panther" else "WH05"
    source = (
        Path(lookup("husarion_ugv_controller"))
        / "config"
        / (wheels + "_controller.yaml")
    )
    content = (
        source.read_text()
        .replace("<namespace>/", "")
        .replace(
            "<static_covariance_orientation>",
            "[0.001849, 0.0, 0.0, 0.0, 0.001849, 0.0, 0.0, 0.0, 0.001849]",
        )
    )
    controls = yaml.safe_load(content)
    drive = controls["/**"]["drive_controller"]["ros__parameters"]
    drive.update(use_sim_time=True, enable_odom_tf=True, base_frame_id="base_footprint")
    controls["/**"]["controller_manager"]["ros__parameters"]["use_sim_time"] = True
    controller_file = directory / "controllers.yaml"
    controller_file.write_text(yaml.safe_dump(controls, sort_keys=False))
    components = directory / "components.yaml"
    components.write_text(
        yaml.safe_dump(
            dict(
                components=[
                    dict(
                        type="LDR06",
                        parent_link="mount_link",
                        xyz="0.0 0.0 0.0",
                        rpy="0.0 0.0 0.0",
                    )
                ]
            ),
            sort_keys=False,
        )
    )
    share = Path(lookup("husarion_ugv_description"))
    plant = ET.fromstring(
        xacro.process_file(
            str(share / "urdf" / (model + ".urdf.xacro")),
            mappings=dict(
                use_sim="true",
                namespace="",
                controller_config_file=str(controller_file),
                battery_config_file="",
                components_config_path=str(components),
            ),
        ).toxml()
    )
    ensure_footprint(plant)
    sensor_interfaces(plant, bridge)
    remap_controller(plant, "drive_controller")
    generated["controllers"] = ["joint_state_broadcaster", "drive_controller"]
    generated["command_topic"] = "/drive_controller/cmd_vel"
    generated["odometry_provenance"] = "Native Husarion UGV closed-loop wheel feedback"
    save_description(plant, directory, generated)


def wheeled_actions(config):
    from launch_ros.actions import Node

    controllers = config["controllers"]
    arguments = controllers + [
        "--activate-as-group",
        "--controller-manager-timeout",
        "90",
    ]
    if config.get("controller_ros_arguments"):
        arguments += ["--controller-ros-args", config["controller_ros_arguments"]]
    actions = [
        Node(
            package="controller_manager",
            executable="spawner",
            output="screen",
            arguments=arguments,
        )
    ]
    if config["command_topic"] != "/cmd_vel":
        profile = config["profile"]
        actions.append(
            Node(
                package="exploration_benchmark_robots",
                executable="command_bridge",
                output="screen",
                parameters=[
                    dict(
                        use_sim_time=True,
                        output=config["command_topic"],
                        max_linear_speed=profile["max_linear_speed"],
                        max_lateral_speed=profile["max_lateral_speed"],
                        max_angular_speed=profile["max_angular_speed"],
                    )
                ],
            )
        )
    return actions
