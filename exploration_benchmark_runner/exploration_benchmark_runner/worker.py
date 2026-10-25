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

"""One isolated simulation, native control, and a lightweight retained observer.

ROS callbacks run in the worker context on a separate executor. The supervision
loop and observer services run on the main thread, serializing lifecycle changes.
After a comparison ends, its worker context and heavy processes are disposed;
only the observer's cached presentation and frozen measurements remain.
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path as FilePath
import queue
import signal
import subprocess
import threading
import time
import traceback

import psutil
import rclpy
from rclpy.action import ActionClient
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
import yaml
from tf2_ros import Buffer, TransformListener, TransformException
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist, PoseStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String, Empty, Bool
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import ListControllers
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray, Marker
from exploration_benchmark_interfaces.msg import (
    RunState,
    RunMetrics,
    CoverageMilestone,
)
from exploration_benchmark_interfaces.srv import WorkerCommand
from .configuration import Specification
from .processes import ProcessScope
from .artifacts import Recorder
from .measurements import Measurements
from .coverage import Reference, Calculator, Coverage
from .relay import prefix_frames
from . import adapters
from exploration_benchmark_robots.adapters import (
    MOVING_OBJECTS_TOPIC,
    prepare as prepare_robot,
)
from exploration_benchmark_robots.known_pose import (
    GROUND_TRUTH_TOPIC,
    MAPPING_STATUS_TOPIC,
    SLAM_TOOLBOX_PATCH,
    SLAM_TOOLBOX_PATCH_MARKER,
    SLAM_TOOLBOX_PATCH_SHA256,
    odom_origin,
)

# RunMetrics fields that are None when unmeasured (measurements.py, coverage.py).
NULLABLE_METRICS = frozenset(
    {
        "cpu_percent",
        "rss_mib",
        "cpu_average_percent",
        "rss_average_mib",
        "cpu_seconds",
        "coverage_covered_m2",
    }
)
LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
# Two latched publishers (known_pose_tf, mapping_scan_filter) share the status topic.
MAPPING_STATUS_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
# Integer status fields per mapping node; a status without them is ignored.
MAPPING_STATUS_COUNTS = {
    "known_pose_tf": ("published", "foreign"),
    "mapping_scan_filter": ("scans", "rejected_scans"),
}
# Executables of the known-pose condition (I10) the runtime image must provide.
MAPPING_EXECUTABLES = ("known_pose_tf", "mapping_scan_filter")
# Manual runner: the 100 Hz /tf (known-pose TF) is forwarded to the observer domain at
# most every 50 ms of simulation time, and at once when a new frame appears (the
# period less 1 µs, so float clock steps keep 20 Hz).
TF_FORWARD_PERIOD_S = 0.05 - 1e-6
NAV_NODES = (
    "controller_server",
    "planner_server",
    "smoother_server",
    "route_server",
    "docking_server",
    "behavior_server",
    "bt_navigator",
    "waypoint_follower",
    "velocity_smoother",
    "collision_monitor",
)
NAV_DISCOVERY_SETTLE_SECONDS = 3.0
NAV_ACTIVATION_TIMEOUT_SECONDS = 60.0
# Frontier insight node (suite runs): its output topic and the attempt log it writes.
INSIGHT_TOPIC = "/benchmark/frontier_insight"
INSIGHT_LOG = "insight.log"


def require_mapping_executables():
    """Refuse a runtime image older than I10 (no known-pose nodes)."""
    from ament_index_python.packages import PackageNotFoundError, get_package_prefix

    try:
        prefix = FilePath(get_package_prefix("exploration_benchmark_robots"))
    except PackageNotFoundError as exc:
        raise RuntimeError("exploration_benchmark_robots is not installed") from exc
    missing = [
        name
        for name in MAPPING_EXECUTABLES
        if not (prefix / "lib/exploration_benchmark_robots" / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "runtime image older than I10 (known-pose mapping): rebuild the runtime "
            f"image, then the suite image (missing {', '.join(missing)} in {prefix})"
        )
    # mapping-known-pose-v4: the SLAM Toolbox that launches must be the patched build.
    try:
        slam_prefix = FilePath(get_package_prefix("slam_toolbox"))
    except PackageNotFoundError as exc:
        raise RuntimeError("slam_toolbox is not installed") from exc
    marker = slam_prefix / SLAM_TOOLBOX_PATCH_MARKER
    found = marker.read_text().strip() if marker.is_file() else None
    if found != SLAM_TOOLBOX_PATCH_SHA256:
        raise RuntimeError(
            f"SLAM Toolbox in {slam_prefix} is not the build with {SLAM_TOOLBOX_PATCH} "
            "(mapping-known-pose-v4): rebuild the runtime image, then the suite image "
            f"(marker {'missing' if found is None else 'differs'})"
        )


class Worker:
    def __init__(self, config):
        self.initialized = False
        from .launch_spec import WorkerLaunchSpec

        config = WorkerLaunchSpec.from_configuration(config).dictionary()
        self.config = config
        self.campaign = bool(config.get("suite_identity"))
        self.runtime = None
        self.single = config["single"]
        self.id = config["run_id"]
        self.slot = config["slot"]
        self.session = config["session_root"]
        self.root = self.session + "/runs/" + self.id
        self.spec = Specification(**config["spec"])
        self.profile = config["profile"]
        self.phase, self.outcome, self.detail = "STARTING", "", ""
        self.native_code = -1
        self.explorer_identity = None
        self.start_sim = self.start_unix = 0.0
        self.measurements = None
        descriptor = config["scenario"]["coverage_reference"]
        self.coverage_reference = Reference(descriptor["metadata_file"])
        self.coverage_reference.validate_scenario(
            config["project_root"], config["scenario"], allow_custom_spawn=self.campaign
        )
        if self.coverage_reference.metadata_hash != descriptor["metadata_sha256"]:
            raise ValueError("Coverage reference changed after the start request")
        self.coverage = None
        self.map_received_wall = self.map_received_unix = None
        self.motion_coverage_input = None
        self.motion_recorded = False
        self.recorder = None
        self.cache, self.tf_cache = {}, {"tf": {}, "tf_static": {}}
        self.tf_forwarded = None
        self.marker_cache = {}
        self.streams, self.active_states, self.state_futures = {}, {}, {}
        self.nav_state_received = {}
        self.nav_discovered_since = None
        self.nav_start_future = None
        self.nav_start_requested_wall = None
        self.clock_value = 0.0
        self.clock_received = False
        self.stationary_since = None
        self.speed = float("inf")
        self.closed = False
        self.initial_pose_sent = False
        self.paused = False
        self.nav_started = False
        self.critical_processes = []
        self.lock = threading.RLock()
        self.commands = queue.Queue()
        self.terminal = queue.Queue()
        self.path = Path()
        self.path.header.frame_id = "map"
        self.subscriptions = []
        self.publishers = {}
        self.retained_visuals = {}
        self.started_wall = time.monotonic()
        self.last_sample = self.last_nav_query = self.last_path = (
            self.last_checkpoint
        ) = 0.0
        self.source_closed = False
        self.logged_state = None
        self.simulation_scope = ProcessScope(config["process_token"] + "_simulation")
        self.explorer_scope = ProcessScope(config["process_token"] + "_explorer")
        self.insight_scope = ProcessScope(config["process_token"] + "_insight")
        self.insight_subscribed = False
        if self.campaign:
            from benchmark_suite.worker_runtime import WorkerRuntime

            self.runtime = WorkerRuntime(self)
        else:
            self.observer_context = Context()
            rclpy.init(context=self.observer_context, domain_id=config["observer_domain"],
                       signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
            self.observer = Node("benchmark_observer_" + self.slot,
                                 context=self.observer_context, use_global_arguments=False)
            self.observer_executor = SingleThreadedExecutor(context=self.observer_context)
            self.observer_executor.add_node(self.observer)
            self.state_pub = self.observer.create_publisher(
                RunState, self.session + "/workers/" + self.slot + "/state", LATCHED)
            self.metrics_pub = self.observer.create_publisher(RunMetrics, self.session + "/metrics", 20)
            self.observer.create_service(WorkerCommand, self.session + "/workers/" + self.slot + "/command", self.command)
        self.worker_context = Context()
        rclpy.init(
            context=self.worker_context,
            domain_id=config["domain_id"],
            signal_handler_options=rclpy.signals.SignalHandlerOptions.NO,
        )
        self.worker = Node(
            "benchmark_worker_" + self.slot,
            context=self.worker_context,
            use_global_arguments=False,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        if self.campaign:
            from exploration_benchmark_telemetry.msg import NavigationEvent

            self.worker.create_subscription(
                NavigationEvent, "/benchmark/navigation_events", self.runtime.navigation, 256
            )
            self.worker.create_subscription(
                NavigateToPose.Impl.FeedbackMessage, "/navigate_to_pose/_action/feedback",
                self.runtime.feedback, qos_profile_sensor_data
            )
            if config.get("suite_diagnostics", True):
                # Every run publishes the pose (known-pose TF); diagnostics record it.
                self.worker.create_subscription(
                    TFMessage,
                    GROUND_TRUTH_TOPIC,
                    self.runtime.receive_ground_truth_pose,
                    qos_profile_sensor_data,
                )
            try:
                from exploration_benchmark_insight.msg import FrontierInsight
            except ImportError:
                self.insight_subscribed = False  # start_insight records why
            else:
                self.worker.create_subscription(
                    FrontierInsight, INSIGHT_TOPIC, self.runtime.receive_insight, 10
                )
                self.insight_subscribed = True
        # Moving-object worlds (registry moving_objects, I5 open point 1 (a)): startup waits for one
        # pose message of every mover; launch_simulation sets the expected objects from world.yaml.
        self.moving_objects = None
        self.moving_objects_started = None
        self.moving_objects_recorded = False
        if config["scenario"].get("moving_objects"):
            self.subscriptions.append(
                self.worker.create_subscription(
                    TFMessage,
                    MOVING_OBJECTS_TOPIC,
                    self.receive_moving_objects,
                    qos_profile_sensor_data,
                )
            )
        # Known-pose mapping condition (I10): known_pose_tf and mapping_scan_filter
        # report on one latched topic; readiness needs both. A second publisher of
        # base_footprint's parent (TF conflict) or a withheld mapping scan fails
        # readiness, or after it the run (mapping_failure).
        self.mapping_status = {}
        self.subscriptions.append(
            self.worker.create_subscription(
                String,
                MAPPING_STATUS_TOPIC,
                self.receive_mapping_status,
                MAPPING_STATUS_QOS,
            )
        )
        self.worker_executor = SingleThreadedExecutor(context=self.worker_context)
        self.worker_executor.add_node(self.worker)
        self.buffer = Buffer(node=self.worker)
        self.listener = TransformListener(self.buffer, self.worker)
        self.nav = ActionClient(self.worker, NavigateToPose, "/navigate_to_pose")
        self.nav_through = ActionClient(
            self.worker, NavigateThroughPoses, "/navigate_through_poses"
        )
        self.cancel_clients = [
            self.worker.create_client(CancelGoal, name + "/_action/cancel_goal")
            for name in ("/navigate_to_pose", "/navigate_through_poses")
        ]
        self.velocity_pub = self.worker.create_publisher(Twist, "/cmd_vel", 10)
        self.lifecycle_clients = {
            name: self.worker.create_client(GetState, "/" + name + "/get_state")
            for name in NAV_NODES
        }
        self.nav_start_client = self.worker.create_client(
            ManageLifecycleNodes, "/lifecycle_manager_navigation/manage_nodes"
        )
        for kind, topic, channel, qos in (
            (Clock, "/clock", "clock", qos_profile_sensor_data),
            (LaserScan, "/scan", "scan", qos_profile_sensor_data),
            (Imu, "/imu", "imu", qos_profile_sensor_data),
            (Odometry, "/odom", "odom", qos_profile_sensor_data),
            (OccupancyGrid, "/map", "map", LATCHED),
            (String, "/robot_description", "robot_description", LATCHED),
            (TFMessage, "/tf", "tf", qos_profile_sensor_data),
            (TFMessage, "/tf_static", "tf_static", LATCHED),
            (Path, "/plan", "plan", qos_profile_sensor_data),
            (MarkerArray, "/explore/frontiers", "frontiers", qos_profile_sensor_data),
            (
                OccupancyGrid,
                "/local_costmap/costmap",
                "local_costmap/costmap",
                qos_profile_sensor_data,
            ),
            (
                OccupancyGrid,
                "/global_costmap/costmap",
                "global_costmap/costmap",
                LATCHED,
            ),
        ):
            if self.campaign and channel not in ("clock", "scan", "imu", "odom", "map", "plan", "tf"):
                continue
            if not self.campaign and channel not in ("imu", "odom"):
                self.publishers[channel] = self.observer.create_publisher(
                    kind, topic if self.single else self.root + "/" + channel, LATCHED
                )
            self.subscriptions.append(
                self.worker.create_subscription(
                    kind,
                    topic,
                    lambda message, c=channel: self.receive(c, message),
                    qos,
                )
            )
        if self.config["robot"].get("measurement_odom_topic"):
            self.subscriptions.append(
                self.worker.create_subscription(
                    Odometry,
                    self.config["robot"]["measurement_odom_topic"],
                    lambda message: self.receive("raw_odom", message),
                    qos_profile_sensor_data,
                )
            )
        if self.campaign:
            pass
        elif not self.single:
            self.publishers["path"] = self.observer.create_publisher(
                Path, self.root + "/path", LATCHED
            )
        else:
            self.publishers["path"] = self.observer.create_publisher(
                Path, "/explore/traversed_path", LATCHED
            )
            self.initial_pub = self.worker.create_publisher(
                PoseStamped, "/explore/path_tracker/initial_pose", LATCHED
            )
            self.subscriptions.append(
                self.worker.create_subscription(
                    Empty, "/explore/reset_traveled_path", self.reset_path, LATCHED
                )
            )
        self.subscriptions.append(
            self.worker.create_subscription(
                Empty,
                "/exploration_complete",
                lambda _: self.native_completed("primary completion event"),
                LATCHED,
            )
        )
        self.thread = threading.Thread(target=self.spin_worker, daemon=True)
        self.robot_ready_boolean = None
        self.robot_start_future = None
        self.robot_start_requested = None
        self.robot_start_client = None
        self.native_controller_names = []
        self.native_controller_states = {}
        self.native_controller_future = None
        self.native_controller_query_wall = 0.0
        self.native_controller_client = self.worker.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        robot = config["robot"]
        if robot.get("startup_trigger_service"):
            self.robot_start_client = self.worker.create_client(
                Trigger, robot["startup_trigger_service"]
            )
            self.subscriptions.append(
                self.worker.create_subscription(
                    Bool,
                    robot["readiness_boolean_topic"],
                    self.receive_robot_ready_boolean,
                    LATCHED,
                )
            )
        self.robot_diagnostics = {}
        for topic in ("/cmd_vel", "/cmd_vel_smoothed", "/cmd_vel_nav"):
            self.subscriptions.append(
                self.worker.create_subscription(
                    Twist,
                    topic,
                    lambda message, source=topic: self.receive_command_diagnostic(
                        source, message
                    ),
                    10,
                )
            )
        self.thread.start()
        self.make_recorder(config["result_directory"])
        self.launch_simulation()
        self.publish_state()
        self.initialized = True

    def environment(self, *, shared=False):
        env = dict(
            os.environ,
            ROS_DOMAIN_ID=str(self.config["domain_id"]),
            GZ_PARTITION=self.config["gazebo_partition"],
            BENCHMARK_WORKER_OWNER=self.config["process_token"],
            BENCHMARK_PROJECT_ROOT=self.config["project_root"],
        )
        env["ROS_LOG_DIR"] = str(FilePath(self.config["result_directory"]) / "ros_logs")
        # Jazzy controller spawners use a file lock beneath ROS_HOME. Isolate it
        # as well as DDS/Gazebo, or unrelated workers serialize their startup.
        env["ROS_HOME"] = str(FilePath(self.config["result_directory"]) / "ros_home")
        # Gazebo EGL headless rendering works without an X server.
        env.pop("DISPLAY", None)
        if shared:
            from exploration_benchmark_robots.runtime import shared_environment

            env = shared_environment(self.config["robot"], env)
        return env

    def cleanup_initialization(self, error):
        for name in ("explorer_scope", "insight_scope", "simulation_scope"):
            scope = getattr(self, name, None)
            if scope:
                scope.stop()
        executor = getattr(self, "worker_executor", None)
        if executor:
            executor.shutdown(timeout_sec=3)
        thread = getattr(self, "thread", None)
        if thread and thread.is_alive():
            thread.join(timeout=3)
        node = getattr(self, "worker", None)
        if node:
            node.destroy_node()
        context = getattr(self, "worker_context", None)
        if context and context.ok():
            context.shutdown()
        recorder = getattr(self, "recorder", None)
        if recorder:
            recorder.finish("startup_failure", str(error), {}, None, None, -1, [])
        runtime = getattr(self, "runtime", None)
        if runtime:
            runtime.close()
        else:
            executor = getattr(self, "observer_executor", None)
            if executor:
                executor.shutdown(timeout_sec=3)
            node = getattr(self, "observer", None)
            if node:
                node.destroy_node()
            context = getattr(self, "observer_context", None)
            if context and context.ok():
                context.shutdown()

    def spin_worker(self):
        try:
            self.worker_executor.spin()
        except (ExternalShutdownException, rclpy.executors.ShutdownException):
            pass
        except Exception as exc:
            self.terminal.put(("runtime_failure", f"Worker executor failed: {exc}", -1))

    def make_recorder(self, directory):
        self.config["result_directory"] = directory
        destination = FilePath(directory)
        destination.mkdir(parents=True, exist_ok=True)
        profile = self.profile
        if self.campaign:
            from benchmark_suite.recording import Recorder as SuiteRecorder

            self.recorder = SuiteRecorder(directory, dict(self.config, spec=self.spec.dictionary()), self.coverage_reference)
            # Every suite row carries the latest /clock (dataset schema 10).
            self.recorder.clock_source = lambda: self.clock_value if self.clock_received else None
        else:
            self.recorder = Recorder(directory, dict(self.config, spec=self.spec.dictionary()),
                                     profile, self.config["project_root"], coverage_reference=self.coverage_reference)
        params, overrides = adapters.resolve_parameters(self.spec, directory)
        self.parameter_file = params
        self.recorder.manifest["instrumentation_overrides"] = overrides
        self.recorder.flush()

    def launch_simulation(self):
        directory = FilePath(self.config["result_directory"])
        directory.mkdir(parents=True, exist_ok=True)
        scenario = self.config["scenario"]
        from ament_index_python.packages import get_package_share_directory

        robot = self.config["robot"]
        require_mapping_executables()
        generated, robot_file = prepare_robot(robot, scenario, directory)
        if self.campaign and self.config.get("suite_diagnostics", True):
            from benchmark_suite.instrumentation import prepare as instrument

            instrument(generated)
        if "moving_objects" in generated:
            with self.lock:
                self.moving_objects = list(generated["moving_objects"]["objects"])
        self.native_controller_names = generated.get("controllers", [])
        self.navigation_parameters = generated["navigation"]
        self.slam_parameters = generated["slam"]
        self.recorder.manifest["robot_configuration"] = generated
        self.recorder.flush()
        args = [
            "ros2",
            "launch",
            "bme_ros2_navigation",
            "world.launch.py",
            "world:=" + generated["world"],
            "headless:=true",
            "seed:=" + str(self.config["seed"]),
        ]
        env = self.environment()
        env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
            generated.get("resource_paths", []) + [env.get("GZ_SIM_RESOURCE_PATH", "")]
        )
        self.spawn_process = self.simulation_scope.start(
            args,
            env,
            directory / "simulation.log",
            self.config["project_root"],
        )
        self.robot_process = self.simulation_scope.start(
            [
                "ros2",
                "launch",
                "exploration_benchmark_robots",
                "robot.launch.py",
                "configuration:=" + robot_file,
            ],
            env,
            directory / "robot.log",
            self.config["project_root"],
        )
        identity = psutil.Process(self.robot_process.pid)
        self.critical_processes.append((identity.pid, identity.create_time()))

    def command(self, request, response):
        if request.command == "start" and self.phase not in ("READY", "FINISHED"):
            response.accepted, response.message = False, "Worker is not ready"
            return response
        if request.command == "start" and self.source_closed:
            response.accepted, response.message = (
                False,
                "Finished comparison cannot be restarted",
            )
            return response
        self.commands.put(request)
        response.accepted, response.message = True, "Queued"
        return response

    def receive(self, channel, message):
        with self.lock:
            if self.source_closed or self.phase == "STOPPING":
                return
            now = time.monotonic()
            self.streams[channel] = now
            if channel == "clock":
                self.clock_value = message.clock.sec + message.clock.nanosec * 1e-9
                self.clock_received = True
            elif channel == "odom":
                velocity = message.twist.twist
                self.speed = max(
                    math.hypot(velocity.linear.x, velocity.linear.y),
                    abs(velocity.angular.z),
                )
            measurement_channel = (
                "raw_odom"
                if self.config["robot"].get("measurement_odom_topic")
                else "odom"
            )
            if (
                channel == measurement_channel
                and self.measurements
                and self.phase == "RUNNING"
            ):
                self.measure_odometry(message, now)
            if self.campaign and channel in ("scan", "imu", "tf", "tf_static", "robot_description"):
                return
            if channel in ("tf", "tf_static"):
                frames = self.tf_cache[channel]
                appeared = False
                for transform in message.transforms:
                    appeared |= transform.child_frame_id not in frames
                    frames[transform.child_frame_id] = deepcopy(transform)
                if channel == "tf":
                    # The cache always holds the newest transforms (republished on pause
                    # and stop); only the forward to the observer domain is throttled.
                    self.cache[channel] = TFMessage(transforms=list(frames.values()))
                    if (
                        not appeared
                        and self.tf_forwarded is not None
                        and 0
                        <= self.clock_value - self.tf_forwarded
                        < TF_FORWARD_PERIOD_S
                    ):
                        return
                    self.tf_forwarded = self.clock_value
                message = TFMessage(transforms=list(frames.values()))
            if channel == "frontiers":
                for marker in message.markers:
                    key = (marker.ns, marker.id)
                    if marker.action == Marker.DELETEALL:
                        self.marker_cache.clear()
                    elif marker.action == Marker.DELETE:
                        self.marker_cache.pop(key, None)
                    else:
                        self.marker_cache[key] = deepcopy(marker)
                clear = Marker(action=Marker.DELETEALL)
                clear.header.frame_id = "map"
                self.cache[channel] = MarkerArray(
                    markers=[clear] + list(self.marker_cache.values())
                )
            else:
                if channel == "robot_description":
                    from ament_index_python.packages import get_package_share_directory
                    from exploration_benchmark_robots.adapters import (
                        visualization_description,
                    )

                    message = String(
                        data=visualization_description(
                            message.data, get_package_share_directory
                        )
                    )
                self.cache[channel] = deepcopy(message)
            if channel == "map":
                self.map_received_wall, self.map_received_unix = now, time.time()
            if not self.single and channel in self.publishers:
                self.publishers[channel].publish(prefix_frames(message, self.id))

    def receive_command_diagnostic(self, topic, message):
        with self.lock:
            if self.phase != "RUNNING":
                return
            if self.runtime:
                self.runtime.command(topic, message)
            values = (message.linear.x, message.linear.y, message.angular.z)
            counters = self.robot_diagnostics.setdefault("commands", {})
            entry = counters.setdefault(
                topic, {"samples": 0, "nonzero_samples": 0, "max_abs": [0.0, 0.0, 0.0]}
            )
            entry["samples"] += 1
            entry["nonzero_samples"] += int(any(abs(v) > 0.01 for v in values))
            entry["max_abs"] = [
                max(old, abs(value)) for old, value in zip(entry["max_abs"], values)
            ]

    def receive_moving_objects(self, message):
        """Keep the first pose message that holds every mover (executor thread, no blocking).

        The E0 system publishes only once every object resolved, one pose per object in its SDF
        order. The bridge forwards a pose's name only when Gazebo sets it as child_frame_id (E0
        sets Pose.name only), so the name set is compared when every pose is named and the object
        count always.
        """
        with self.lock:
            expected = self.moving_objects
            if expected is None or self.moving_objects_started is not None:
                return
            names = [t.child_frame_id.split("::")[-1] for t in message.transforms]
            named = bool(names) and all(names)
            if len(names) != len(expected) or (
                named and sorted(names) != sorted(expected)
            ):
                return
            stamp = message.transforms[0].header.stamp
            self.moving_objects_started = dict(
                clock_ns=stamp.sec * 10**9 + stamp.nanosec,
                objects=len(names),
                names_carried=named,
            )

    def check_moving_objects(self):
        """The movers' startup check: a readiness item until they publish, then recorded once."""
        with self.lock:
            if self.moving_objects is None or self.moving_objects_recorded:
                return None
            started = self.moving_objects_started
            if started is None:
                return "moving objects"
            self.moving_objects_recorded = True
            record = self.recorder.manifest["robot_configuration"]["moving_objects"]
            record["started"] = dict(started)
            self.recorder.event("moving_objects_started", **started)
            self.recorder.flush()
        return None

    def receive_robot_ready_boolean(self, message):
        self.robot_ready_boolean = message.data
        if (
            self.phase == "RUNNING"
            and message.data != self.config["robot"]["readiness_boolean_expected"]
        ):
            self.terminal.put(
                ("runtime_failure", "Native simulated emergency stop engaged", -1)
            )

    def receive_mapping_status(self, message):
        try:
            status = json.loads(message.data)
        except ValueError:
            return
        if not isinstance(status, dict):
            return
        counts = MAPPING_STATUS_COUNTS.get(status.get("node"))
        if (
            counts is None
            or not isinstance(status.get("ok"), bool)
            or not all(
                isinstance(status.get(name), int)
                and not isinstance(status.get(name), bool)
                for name in counts
            )
        ):
            return
        with self.lock:
            self.mapping_status[status["node"]] = status

    def check_mapping_condition(self):
        """Readiness items of the known-pose condition (empty when satisfied)."""
        with self.lock:
            pose = self.mapping_status.get("known_pose_tf")
            scan = self.mapping_status.get("mapping_scan_filter")
        missing = []
        if pose and pose["foreign"] > 0:
            missing.append("TF conflict (second publisher of base_footprint's parent)")
        elif not pose or not pose["ok"]:
            missing.append("known-pose TF")
        if not scan or not scan["ok"]:
            missing.append("mapping scan")
        return missing

    def mapping_failure(self):
        """Why the known-pose condition no longer holds after readiness, or None.

        A second base_footprint parent publisher (TF conflict) or a withheld mapping
        scan ends a READY run as startup_failure and a RUNNING run as runtime_failure.
        """
        with self.lock:
            pose = self.mapping_status.get("known_pose_tf") or {}
            scan = self.mapping_status.get("mapping_scan_filter") or {}
        if pose.get("foreign", 0) > 0:
            return (
                "TF conflict: a second publisher of base_footprint's parent; "
                "see robot.log"
            )
        if scan.get("rejected_scans", 0) > 0:
            return (
                "Mapping scan withheld: scan range_max inconsistent with "
                "max_laser_range; see robot.log"
            )
        return None

    def initialize_robot(self, wall):
        """Release a native simulation latch once; retain its runtime mechanism."""
        if self.robot_start_client is None:
            return
        if self.robot_start_future:
            if self.robot_start_future.done():
                response = self.robot_start_future.result()
                if not response or not response.success:
                    raise RuntimeError(
                        "Native robot initialization rejected; see robot.log"
                    )
                if self.recorder:
                    self.recorder.event(
                        "native_robot_initialized",
                        service=self.config["robot"]["startup_trigger_service"],
                    )
                self.robot_start_future = None
            elif wall - self.robot_start_requested > 5:
                raise RuntimeError(
                    "Native robot initialization timed out; see robot.log"
                )
        elif (
            self.robot_start_requested is None
            and self.clock_value > 0
            and self.robot_start_client.service_is_ready()
        ):
            self.robot_start_requested = wall
            self.robot_start_future = self.robot_start_client.call_async(
                Trigger.Request()
            )

    def check_native_controllers(self, wall):
        """Gate readiness and report deactivation; temporary DDS gaps remain diagnostic."""
        if not self.native_controller_names:
            return
        if self.native_controller_future:
            future = self.native_controller_future
            if future.done():
                try:
                    response = future.result()
                except Exception:
                    response = None
                if response is not None:
                    self.native_controller_states = {
                        c.name: c.state for c in response.controller
                    }
                self.native_controller_future = None
                inactive = [
                    name
                    for name in self.native_controller_names
                    if self.native_controller_states.get(name) != "active"
                ]
                if self.phase == "RUNNING" and response is not None and inactive:
                    self.terminal.put(
                        (
                            "runtime_failure",
                            "Native controllers became inactive: "
                            + ", ".join(inactive),
                            -1,
                        )
                    )
            elif wall - self.native_controller_query_wall > 3:
                self.native_controller_client.remove_pending_request(future)
                self.native_controller_future = None
        if (
            self.native_controller_future is None
            and wall - self.native_controller_query_wall >= 1
            and self.native_controller_client.service_is_ready()
        ):
            self.native_controller_query_wall = wall
            self.native_controller_future = self.native_controller_client.call_async(
                ListControllers.Request()
            )

    def measure_odometry(self, message, wall):
        position, orientation = (
            message.pose.pose.position,
            message.pose.pose.orientation,
        )
        yaw = math.atan2(
            2 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1 - 2 * (orientation.y**2 + orientation.z**2),
        )
        stamp = message.header.stamp
        simulation = stamp.sec + stamp.nanosec * 1e-9
        started = self.measurements.odometry(
            position.x, position.y, yaw=yaw, wall=wall, simulation=simulation
        )
        if started:
            # Cached messages are immutable after insertion. Retain the map that
            # existed at first movement without doing geometry work in callbacks.
            self.motion_coverage_input = (
                self.cache.get("map"),
                self.map_received_wall,
                self.map_received_unix,
            )
        if self.runtime:
            self.runtime.odometry(message, wall)

    def prepare_coverage(self):
        """Place the SLAM map frame in the world (coverage, Monitor, odometry).

        Known pose (I10): map -> odom is the identity and odom is base_footprint at
        the commanded spawn pose, so world_from_map is exactly that pose
        (known_pose.odom_origin, the value known_pose_tf uses). The anchor of earlier
        conditions, spawn composed with (map -> base_footprint)^-1 at readiness, is
        recorded as a check: it differs from the exact value by the robot's settling
        offset between the spawn command and rest.
        """
        transform = self.buffer.lookup_transform(
            "map", "base_footprint", Time()
        ).transform
        q = transform.rotation
        map_yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
        )
        spawn = self.config["scenario"]["spawn"]
        x, y = transform.translation.x, transform.translation.y
        yaw = spawn[3] - map_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        estimate = (spawn[0] - c * x + s * y, spawn[1] - s * x - c * y, yaw)
        odom = self.buffer.lookup_transform("map", "odom", Time()).transform
        q = odom.rotation
        map_odom = [
            odom.translation.x,
            odom.translation.y,
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
        ]
        alignment = odom_origin(spawn)
        self.coverage = Coverage(Calculator(self.coverage_reference, alignment))
        if self.runtime:
            self.runtime.align_odometry(alignment)
        self.recorder.manifest["coverage"]["alignment"] = dict(
            world_from_map=list(alignment),
            anchor="known_pose_spawn",
            simulation=self.clock_value,
            map_base_pose=[x, y, map_yaw],
            map_odom_pose=map_odom,
            spawn=spawn,
            check=dict(
                anchor="stationary_spawn_and_map_to_base_tf",
                world_from_map=list(estimate),
                offset_m=math.hypot(
                    estimate[0] - alignment[0], estimate[1] - alignment[1]
                ),
                offset_yaw_rad=math.atan2(
                    math.sin(estimate[2] - alignment[2]),
                    math.cos(estimate[2] - alignment[2]),
                ),
            ),
        )
        self.recorder.flush()

    def coverage_tick(self, wall, *, force=False):
        if (
            self.coverage is None
            or self.coverage.frozen
            or (self.phase != "RUNNING" and not (force and self.phase == "STOPPING"))
        ):
            return
        with self.lock:
            simulation = self.clock_value
            grid = self.cache.get("map")
            received_wall, received_unix = (
                self.map_received_wall,
                self.map_received_unix,
            )
            motion = self.motion_coverage_input
            origins = (
                self.measurements.movement_simulation_start,
                self.measurements.movement_wall_start,
            )
        if origins[0] is not None and self.coverage.motion_sim is None:
            row = self.coverage.start_motion(*origins, *(motion or (None, None, None)))
            if row and self.recorder:
                self.recorder.coverage_sample(row)
        row = self.coverage.update(
            simulation, wall, grid, received_wall, received_unix, force=force
        )
        if row and self.recorder:
            self.recorder.coverage_sample(row)
        if self.campaign:
            self.runtime.explored_snapshot(self.coverage, grid, simulation, final=force)
        if self.recorder:
            self.recorder.coverage_checkpoint(self.coverage)

    def publish_metrics(self, snapshot):
        if self.runtime:
            self.runtime.publish_metrics(snapshot)
            return
        # The message has no null: unmeasured values travel as NaN (the panel shows "—").
        fields = {
            name: float("nan") if value is None and name in NULLABLE_METRICS else value
            for name, value in snapshot.items()
        }
        fields["coverage_milestones"] = [
            CoverageMilestone(
                percent=m["percent"],
                reached=m["reached"],
                simulation_elapsed=m["simulation_elapsed"] or 0.0,
            )
            for m in fields.get("coverage_milestones", [])
        ]
        self.metrics_pub.publish(RunMetrics(run_id=self.id, **fields))

    def tf_ready(self):
        try:
            if not self.buffer.can_transform(
                "map", "odom", Time()
            ) or not self.buffer.can_transform("odom", "base_footprint", Time()):
                return False
            transform = self.buffer.lookup_transform("map", "base_footprint", Time())
            stamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
            return self.clock_value - stamp <= 1.0
        except TransformException:
            return False

    def readiness(self, wall):
        self.initialize_robot(wall)
        self.check_native_controllers(wall)
        missing = [
            name
            for name in self.config["robot"]["required_streams"]
            if wall - self.streams.get(name, 0) > 2.0
        ]
        if "map" not in self.cache or not self.cache["map"].data:
            missing.append("map")
        if not self.tf_ready():
            missing.append("TF chain")
        movers = self.check_moving_objects()
        if movers:
            missing.append(movers)
        missing.extend(self.check_mapping_condition())
        missing.extend(
            name
            for name in self.native_controller_names
            if self.native_controller_states.get(name) != "active"
        )
        robot = self.config["robot"]
        if self.robot_start_client is not None and (
            self.robot_start_requested is None
            or self.robot_start_future is not None
            or self.robot_ready_boolean != robot["readiness_boolean_expected"]
        ):
            missing.append("native simulated emergency-stop release")
        if not self.nav.server_is_ready() or not self.nav_through.server_is_ready():
            missing.append("navigation actions")
        if wall - self.last_nav_query >= 1.0:
            self.last_nav_query = wall
            for name, client in self.lifecycle_clients.items():
                pending = self.state_futures.get(name)
                if pending:
                    future, requested_wall = pending
                    if future.done():
                        try:
                            self.active_states[name] = (
                                future.result().current_state.id == 3
                            )
                            self.nav_state_received[name] = wall
                        except Exception:
                            self.active_states[name] = False
                        self.state_futures.pop(name, None)
                    elif wall - requested_wall >= 3:
                        # DDS endpoint discovery can lose an initial request.
                        # Never hold readiness on an unanswered future forever.
                        client.remove_pending_request(future)
                        future.cancel()
                        self.state_futures.pop(name, None)
                        self.active_states[name] = False
                        self.nav_state_received.pop(name, None)
                if name not in self.state_futures and client.service_is_ready():
                    self.state_futures[name] = (
                        client.call_async(GetState.Request()),
                        wall,
                    )
        self.start_navigation_when_discovered(wall)
        missing.extend(name for name in NAV_NODES if not self.active_states.get(name))
        if missing or self.speed > 0.03:
            self.stationary_since = None
        elif self.stationary_since is None:
            self.stationary_since = self.clock_value
        warmup_duration = 0.0 if self.single else 5.0
        stationary_ready = (
            self.stationary_since is not None
            and self.clock_value - self.stationary_since >= warmup_duration
        )
        if missing:
            self.detail = "Waiting for: " + ", ".join(missing)
        elif stationary_ready:
            self.detail = "Ready"
        elif self.single:
            self.detail = "Waiting for robot to be stationary"
        else:
            self.detail = "Stationary warm-up (5 simulation seconds)"
        return not missing and stationary_ready

    def start_navigation_when_discovered(self, wall):
        """Let request/reply discovery settle before Nav2's first transition.

        Fast DDS can discover a request endpoint before its reply endpoint.
        Nav2 autostart then waits indefinitely for a lost change_state response.
        Launch inactive, confirm every server replies, and allow discovery to
        settle before asking the native manager to perform its usual startup.
        Never resend an unconfirmed state-changing request.
        """
        if not self.nav_started:
            return
        if self.nav_start_requested_wall is not None:
            if self.nav_start_future is None:
                return
            if self.nav_start_future.done():
                if not self.nav_start_future.result().success:
                    raise RuntimeError(
                        "Nav2 lifecycle manager rejected startup; see navigation.log"
                    )
            elif wall - self.nav_start_requested_wall > NAV_ACTIVATION_TIMEOUT_SECONDS:
                if all(self.active_states.get(name) for name in NAV_NODES):
                    # A lost manager response must not mask verified active nodes.
                    self.nav_start_client.remove_pending_request(self.nav_start_future)
                    self.nav_start_future.cancel()
                    self.nav_start_future = None
                else:
                    raise RuntimeError(
                        "Nav2 activation timed out; lifecycle reply may be missing. "
                        "See navigation.log"
                    )
            return
        discovered = self.nav_start_client.service_is_ready() and all(
            wall - self.nav_state_received.get(name, float("-inf")) < 3
            for name in NAV_NODES
        )
        if not discovered:
            self.nav_discovered_since = None
            return
        if self.nav_discovered_since is None:
            self.nav_discovered_since = wall
        if wall - self.nav_discovered_since < NAV_DISCOVERY_SETTLE_SECONDS:
            return
        self.nav_start_future = self.nav_start_client.call_async(
            ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.STARTUP)
        )
        self.nav_start_requested_wall = wall
        self.recorder.event(
            "navigation_start_requested",
            discovery_settle_wall_seconds=NAV_DISCOVERY_SETTLE_SECONDS,
            activation_timeout_wall_seconds=NAV_ACTIVATION_TIMEOUT_SECONDS,
        )

    def gazebo_pause(self, paused):
        command = [
            "gz",
            "service",
            "-s",
            "/world/" + self.config["scenario"]["world_name"] + "/control",
            "--reqtype",
            "gz.msgs.WorldControl",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            "pause: " + str(paused).lower(),
        ]
        result = subprocess.run(
            command, env=self.environment(), capture_output=True, text=True, timeout=4
        )
        if result.returncode or "true" not in result.stdout.lower():
            raise RuntimeError(
                f"Gazebo pause={paused} failed: {result.stdout} {result.stderr}"
            )
        self.paused = paused
        if self.recorder:
            self.recorder.event(
                "gazebo_pause", paused=paused, simulation=self.clock_value
            )

    def start_explorer(self):
        if self.paused:
            self.gazebo_pause(False)
        self.start_sim, self.start_unix = self.clock_value, time.time()
        with self.lock:
            self.measurements = Measurements(
                time.monotonic(), self.start_sim, wait_for_motion=True
            )
            self.motion_recorded = False
            measurement_channel = (
                "raw_odom"
                if self.config["robot"].get("measurement_odom_topic")
                else "odom"
            )
            if measurement_channel in self.cache:
                self.measure_odometry(
                    self.cache[measurement_channel], self.measurements.wall_start
                )
            self.measurements.sample(time.monotonic(), self.clock_value)
        self.terminal = queue.Queue()
        self.explorer_identity = None
        self.native_code = -1
        self.path = Path()
        self.path.header.frame_id = "map"
        self.phase, self.outcome, self.detail = (
            "RUNNING",
            "",
            "Exploration active; waiting for first movement",
        )
        profile = self.profile
        self.explorer_scope = ProcessScope(self.config["process_token"] + "_explorer")
        self.explorer_process = self.explorer_scope.start(
            adapters.command(profile, self.parameter_file),
            self.environment(shared=True),
            FilePath(self.config["result_directory"]) / "explorer.log",
            self.config["project_root"],
        )
        self.recorder.manifest.update(
            simulation_start=self.start_sim,
            wall_start_unix=self.start_unix,
            phase="RUNNING",
            explorer_command=self.explorer_scope.commands[-1],
            timing_policy={
                "elapsed_origin": "first_raw_odometry_movement",
                "translation_threshold_m": Measurements.MOTION_TRANSLATION_METRES,
                "rotation_threshold_rad": Measurements.MOTION_ROTATION_RADIANS,
                "simulation_origin": "odometry_header_stamp",
                "wall_origin": "odometry_receipt_monotonic_time",
                "time_limit_origin": "first_raw_odometry_movement" if self.campaign else "explorer_activation",
                "resource_and_quality_origin": "explorer_activation",
            },
            first_motion=None,
        )
        self.recorder.event(
            "exploration_started",
            simulation=self.start_sim,
            wall_start_unix=self.start_unix,
        )
        if self.campaign:
            self.start_insight()
        self.recorder.flush()
        self.publish_state()

    def start_insight(self):
        """Start the read-only frontier insight node beside the explorer (suite runs).

        A missing package or a failed start is recorded and leaves the run unchanged:
        the Monitor then shows no explorer analysis for it.
        """
        from ament_index_python.packages import PackageNotFoundError, get_package_prefix

        directory = FilePath(self.config["result_directory"])
        manifest = dict(started=False, topic=INSIGHT_TOPIC, log=INSIGHT_LOG)
        try:
            if not self.insight_subscribed:
                raise PackageNotFoundError("exploration_benchmark_insight messages")
            get_package_prefix("exploration_benchmark_insight")
            resolved = yaml.safe_load(FilePath(self.parameter_file).read_text())
            parameters = dict(
                resolved["frontier_explorer"]["ros__parameters"], use_sim_time=True
            )
            path = directory / "insight_parameters.yaml"
            document = {"frontier_insight": {"ros__parameters": parameters}}
            path.write_text(yaml.safe_dump(document, sort_keys=False))
            command = [
                "ros2",
                "run",
                "exploration_benchmark_insight",
                "frontier_insight",
                "--ros-args",
                "--params-file",
                str(path),
            ]
            self.insight_scope = ProcessScope(self.config["process_token"] + "_insight")
            self.insight_scope.start(
                command,
                self.environment(shared=True),
                directory / INSIGHT_LOG,
                self.config["project_root"],
            )
            manifest.update(started=True, command=command, parameters_file=path.name)
        except PackageNotFoundError:
            manifest["reason"] = "exploration_benchmark_insight is not importable"
        except (
            OSError, KeyError, TypeError, ValueError, psutil.Error, yaml.YAMLError
        ) as error:
            manifest["reason"] = f"Insight node did not start: {error}"
        self.recorder.manifest["insight"] = manifest

    def native_completed(self, detail):
        if self.phase == "RUNNING":
            self.terminal.put(("completed", detail, -1))

    def reset_path(self, _):
        if not self.single or self.phase == "RUNNING":
            return
        self.path.poses.clear()
        self.cache["path"] = self.path
        self.publishers["path"].publish(self.path)

    def publish_initial_pose(self):
        if self.campaign or not self.single or self.initial_pose_sent:
            return
        try:
            transform = self.buffer.lookup_transform("map", "base_footprint", Time())
        except TransformException:
            return
        pose = PoseStamped()
        pose.header = transform.header
        pose.header.frame_id = "map"
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        self.initial_pub.publish(pose)
        self.initial_pose_sent = True

    def record_path(self):
        if self.campaign or self.phase != "RUNNING":
            return
        try:
            transform = self.buffer.lookup_transform("map", "base_footprint", Time())
        except TransformException:
            return
        pose = PoseStamped()
        pose.header = transform.header
        pose.header.frame_id = "map"
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        if self.path.poses:
            last = self.path.poses[-1].pose.position
            if (
                math.hypot(last.x - pose.pose.position.x, last.y - pose.pose.position.y)
                < 0.05
            ):
                return
        elif self.single:
            self.publish_initial_pose()
        self.path.header.stamp = pose.header.stamp
        self.path.poses.append(pose)
        self.cache["path"] = self.path
        self.publishers["path"].publish(
            self.path if self.single else prefix_frames(self.path, self.id)
        )

    def sample(self, wall, *, coverage_force=False):
        if not self.measurements:
            return
        self.coverage_tick(wall, force=coverage_force)
        cpu = rss = None
        try:
            if self.explorer_identity is None:
                process = self.explorer_scope.explorer(
                    self.profile["package"], self.profile["executable"]
                )
            else:
                process = psutil.Process(self.explorer_identity[0])
            if process is not None:
                identity = (process.pid, process.create_time())
                if (
                    self.explorer_identity is not None
                    and self.explorer_identity != identity
                ):
                    self.terminal.put(
                        (
                            "runtime_failure",
                            "Explorer process identity changed during the run",
                            -1,
                        )
                    )
                else:
                    self.explorer_identity = identity
                    times = process.cpu_times()
                    cpu = times.user + times.system
                    rss = process.memory_info().rss / 1024**2
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        gaps = [
            name
            for name in self.config["robot"]["required_streams"] + ["tf"]
            if wall - self.measurements.wall_start > 2
            and wall - self.streams.get(name, self.started_wall) > 2.0
        ]
        with self.lock:
            snapshot = self.measurements.sample(wall, self.clock_value, cpu, rss, gaps)
            if self.coverage:
                coverage = self.coverage.snapshot()
                snapshot.update(coverage)
                self.measurements.latest.update(coverage)
            if self.measurements.motion_started and not self.motion_recorded:
                self.motion_recorded = True
                first_motion = dict(
                    simulation=self.measurements.movement_simulation_start,
                    wall_start_unix=self.start_unix
                    + self.measurements.movement_wall_start
                    - self.measurements.wall_start,
                    # Monotonic wall origin of the run-level RTF (dataset schema 10).
                    monotonic_ns=round(self.measurements.movement_wall_start * 1e9),
                )
                self.detail = "Exploration active"
                if self.recorder:
                    self.recorder.manifest["first_motion"] = first_motion
                    self.recorder.event("first_motion", **first_motion)
                    self.recorder.flush()
        if self.recorder and self.phase == "RUNNING":
            if self.campaign:
                self.recorder.origin = self.measurements.movement_simulation_start
                snapshot["explorer_pid"] = self.explorer_identity[0] if self.explorer_identity else None
                snapshot["explorer_resources_available"] = cpu is not None and rss is not None
                self.runtime.sample_resources(wall)
            self.recorder.sample(snapshot)
            # timing-rule-v2: the recorder's windows decide rejection; the engine no longer computes it.
            if self.campaign and self.recorder.timing.rejected:
                self.terminal.put(
                    ("timing_rejection", "RTF below 0.90 for ten wall seconds", -1)
                )
        self.publish_metrics(snapshot)

    def stop(self, outcome, detail="", native_code=-1):
        if self.phase in ("STOPPING", "FINISHED"):
            return
        recorder = self.recorder
        if self.campaign and recorder and recorder.error:
            outcome, detail = "storage_failure", str(recorder.error)
            self.recorder = None
        if self.campaign and self.phase == "RUNNING":
            self.runtime.check_insight()  # an exit in the last second before the stop counts too
        # Close callback input atomically at the trigger. Geometry calculation
        # and process accounting must not admit another map/odometry afterwards.
        with self.lock:
            self.phase, self.outcome, self.detail, self.native_code = (
                "STOPPING",
                outcome,
                detail,
                native_code,
            )
            terminal_wall = time.monotonic()
            if self.campaign and self.recorder:
                self.recorder.manifest.update(
                    terminal_trigger_utc_ns=time.time_ns(),
                    terminal_trigger_monotonic_ns=round(terminal_wall * 1e9),
                    terminal_trigger_simulation_ns=round(self.clock_value * 1e9),
                )
        # Freeze before cancellation and process teardown add extra distance/time.
        if self.measurements:
            try:
                self.sample(terminal_wall, coverage_force=True)
            except OSError as error:
                if not self.campaign:
                    raise
                self.outcome, self.detail = "storage_failure", str(error)
                self.recorder = None
            with self.lock:
                self.measurements.freeze()
                if self.coverage:
                    self.measurements.latest.update(self.coverage.freeze())
            try:
                if self.recorder:
                    if self.coverage:
                        self.recorder.coverage_checkpoint(self.coverage)
                    self.recorder.sample(self.measurements.latest)
            except OSError as error:
                if not self.campaign:
                    raise
                self.outcome, self.detail = "storage_failure", str(error)
                self.recorder = None
            if self.campaign:
                self.publish_metrics(self.measurements.latest)
        self.publish_state()
        for client in self.cancel_clients:
            if client.service_is_ready():
                client.call_async(CancelGoal.Request())
        for _ in range(3):
            self.velocity_pub.publish(Twist())
            time.sleep(0.05)
        self.explorer_scope.stop()
        # The insight node only reads: stop it after the explorer, never delaying it.
        self.insight_scope.stop()
        self.simulation_scope.stop()
        self.close_source()
        if recorder:
            recorder.manifest["robot_diagnostics"] = self.robot_diagnostics
            with self.lock:
                recorder.manifest["mapping_status"] = deepcopy(self.mapping_status)
            if self.campaign and "insight" in recorder.manifest:
                recorder.manifest["insight"].update(self.runtime.insight_manifest())
            snapshot = self.measurements.latest if self.measurements else {}
            commands = (
                self.simulation_scope.commands
                + self.explorer_scope.commands
                + self.insight_scope.commands
            )
            try:
                extra = {}
                if self.campaign:
                    self.runtime.close_goals(recorder)
                    extra["run_facts"] = self.runtime.run_facts(recorder)
                recorder.finish(
                    self.outcome,
                    self.detail,
                    snapshot,
                    self.cache.get("map"),
                    self.path,
                    native_code,
                    commands,
                    **extra,
                )
            except OSError as error:
                if not self.campaign:
                    raise
                self.outcome, self.detail = "storage_failure", str(error)
                recorder.thread.join(timeout=3)
            self.recorder = None
        self.phase = "FINISHED"
        self.publish_state()
        if self.measurements:
            self.sample(time.monotonic())
        if self.campaign:
            self.closed = True

    def close_source(self):
        if self.source_closed:
            return
        self.source_closed = True
        self.worker_executor.shutdown(timeout_sec=3)
        self.thread.join(timeout=3)
        self.worker.destroy_node()
        if self.worker_context.ok():
            self.worker_context.shutdown()
        self.retained_visuals = {
            channel: message if self.single else prefix_frames(message, self.id)
            for channel, message in self.cache.items()
            if channel in self.publishers
        }

    def publish_state(self):
        if self.runtime:
            self.runtime.publish_state()
            return
        description = (self.phase, self.outcome, self.detail)
        if description != self.logged_state:
            self.logged_state = description
            self.observer.get_logger().info(" ".join(description).strip())
        self.state_pub.publish(
            RunState(
                run_id=self.id,
                robot_id=self.config["robot"]["id"],
                spec=self.spec.message(),
                phase=self.phase,
                outcome=self.outcome,
                detail=self.detail,
                domain_id=self.config["domain_id"],
                gazebo_partition=self.config["gazebo_partition"],
                topic_root=self.root,
                result_directory=self.config["result_directory"],
                simulation_start=self.start_sim,
                wall_start_unix=self.start_unix,
                native_result_code=self.native_code,
            )
        )

    def tick(self):
        wall = time.monotonic()
        if self.runtime:
            self.runtime.tick()
            if self.recorder and self.recorder.error:
                self.stop("storage_failure", str(self.recorder.error))
                return
        while not self.commands.empty():
            request = self.commands.get_nowait()
            if request.command == "start":
                self.start_explorer()
            elif request.command in ("stop", "cancel", "shutdown"):
                reason = "cancelled" if request.command == "cancel" else "manual_stop"
                self.stop(
                    reason,
                    "Session control request",
                )
                if request.command == "shutdown":
                    self.closed = True
            elif request.command == "runtime_failure":
                self.stop("runtime_failure", "Coordinator detected worker failure")
        if self.phase == "STARTING":
            if not self.nav_started and self.clock_value > 0:
                self.simulation_scope.start(
                    [
                        "ros2",
                        "launch",
                        "bme_ros2_navigation",
                        "navigation_with_slam.launch.py",
                        "rviz:=false",
                        "nav2_autostart:=false",
                        "navigation_params_file:=" + self.navigation_parameters,
                        "slam_params_file:=" + self.slam_parameters,
                    ],
                    self.environment(shared=True),
                    FilePath(self.config["result_directory"]) / "navigation.log",
                    self.config["project_root"],
                )
                self.nav_started = True
            if self.readiness(wall):
                try:
                    self.prepare_coverage()
                except (TransformException, ValueError) as exc:
                    self.stop("startup_failure", f"Coverage alignment failed: {exc}")
                    return
                if not self.campaign:
                    self.gazebo_pause(True)
                self.phase = "READY"
                self.detail = "Ready" if self.single else "Ready; warm-up complete"
                self.publish_initial_pose()
                # Readiness has identified a complete stack. Cache identities
                # of its native executables so a child crash is visible even
                # when ROS launch itself remains alive.
                for process in self.simulation_scope.discover():
                    try:
                        arguments = process.cmdline()
                        if any(
                            "/lib/" in arg and not arg.endswith("/ros2")
                            for arg in arguments
                        ):
                            self.critical_processes.append(
                                (process.pid, process.create_time())
                            )
                    except psutil.NoSuchProcess:
                        pass
                if self.recorder:
                    self.recorder.event("ready", simulation=self.clock_value)
                if self.campaign:
                    self.start_explorer()
            elif wall - self.started_wall > self.config.get("startup_timeout", 180):
                self.stop("startup_failure", self.detail)
        if self.phase in ("STARTING", "READY", "RUNNING"):
            if any(root.poll() is not None for root in self.simulation_scope.roots):
                self.stop(
                    "startup_failure" if self.phase != "RUNNING" else "runtime_failure",
                    "Simulation launch exited",
                )
        if self.phase in ("READY", "RUNNING"):
            failure = self.mapping_failure()
            if failure:
                outcome = (
                    "startup_failure" if self.phase == "READY" else "runtime_failure"
                )
                self.terminal.put((outcome, failure, -1))
        if self.phase == "RUNNING":
            self.coverage_tick(wall)
            self.check_native_controllers(wall)
            identities = list(self.critical_processes)
            if self.explorer_identity is not None:
                identities.append(self.explorer_identity)
            for pid, started in identities:
                try:
                    process = psutil.Process(pid)
                    alive = (
                        process.create_time() == started
                        and process.status() != psutil.STATUS_ZOMBIE
                    )
                except psutil.NoSuchProcess:
                    alive = False
                if not alive:
                    self.terminal.put(
                        (
                            "runtime_failure",
                            f"Simulation component exited (owned PID {pid})",
                            -1,
                        )
                    )
                    break
            # Wait briefly for a native completion callback before treating an
            # otherwise unexpected explorer exit as a runtime failure.
            if (
                self.explorer_process.poll() is not None
                and wall - self.measurements.wall_start > 2
            ):
                self.terminal.put(
                    (
                        "runtime_failure",
                        f"Explorer exited with code {self.explorer_process.returncode}",
                        -1,
                    )
                )
            limit = self.config["simulation_time_limit"]
            limit_origin = self.measurements.movement_simulation_start if self.campaign else self.start_sim
            if limit and limit_origin is not None and self.clock_value - limit_origin >= limit:
                self.terminal.put(
                    ("time_limit", f"Simulation-time limit ({limit:g} seconds)", -1)
                )
            if self.campaign:
                if not self.measurements.motion_started and self.clock_value-self.start_sim >= 180:
                    self.terminal.put(("no_movement", "No planar motion within 180 simulation seconds after activation", -1))
                required = self.config["robot"]["required_streams"] + ["clock"]
                stale = [name for name in required if wall-self.streams.get(name,self.started_wall)>10.]
                if stale:
                    self.terminal.put(("runtime_failure", "Required stream stopped: " + ", ".join(stale), -1))
            if wall - self.last_path >= 0.2:
                self.last_path = wall
                self.record_path()
        while not self.terminal.empty():
            outcome, detail, code = self.terminal.get_nowait()
            if self.phase in ("STARTING", "READY", "RUNNING"):
                self.stop(outcome, detail, code)
        if self.recorder and wall - self.last_checkpoint >= 5:
            self.last_checkpoint = wall
            with self.lock:
                self.recorder.save_visuals(self.cache.get("map"), self.path)
                self.recorder.flush()
        if wall - self.last_sample >= 1:
            self.last_sample = wall
            self.publish_state()
            if self.measurements:
                self.sample(wall)
            if self.source_closed:
                for channel, message in self.retained_visuals.items():
                    self.publishers[channel].publish(message)

    def run(self):
        try:
            while not self.closed:
                if self.campaign:
                    # The 50 ms pause between ticks; suite requests are answered during
                    # it instead of waiting for the next tick.
                    self.runtime.serve(0.05)
                else:
                    self.observer_executor.spin_once(timeout_sec=0.05)
                self.tick()
        except Exception as error:
            self.stop(
                (
                    "startup_failure"
                    if self.phase in ("STARTING", "READY")
                    else "runtime_failure"
                ),
                str(error),
            )
            raise
        finally:
            if self.phase not in ("STOPPING", "FINISHED"):
                self.stop("manual_stop", "Worker application exit")
            self.explorer_scope.stop()
            self.insight_scope.stop()
            self.simulation_scope.stop()
            self.close_source()
            if self.runtime:
                self.runtime.close()
            else:
                self.observer_executor.shutdown()
                self.observer.destroy_node()
                if self.observer_context.ok():
                    self.observer_context.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    args = parser.parse_args()
    config = json.loads(FilePath(args.configuration).read_text())
    worker = None

    def exit_signal(_sig, _frame):
        if worker is not None:
            worker.closed = True

    signal.signal(signal.SIGINT, exit_signal)
    signal.signal(signal.SIGTERM, exit_signal)
    try:
        worker = Worker.__new__(Worker)
        Worker.__init__(worker, config)
        worker.run()
    except BaseException as exc:
        traceback.print_exc()
        if worker is not None:
            try:
                if not getattr(worker, "initialized", False):
                    worker.cleanup_initialization(exc)
                elif worker.phase not in ("STOPPING", "FINISHED"):
                    worker.stop(
                        (
                            "startup_failure"
                            if worker.phase in ("STARTING", "READY")
                            else "runtime_failure"
                        ),
                        str(exc),
                    )
                worker.closed = True
            except BaseException:
                traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
