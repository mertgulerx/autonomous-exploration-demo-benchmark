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

from collections import deque
from copy import deepcopy
import json
import itertools
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
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Twist, PoseStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import DriveOnHeading, NavigateToPose, NavigateThroughPoses, Spin
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import Bool, String, Empty
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
from .coverage import (
    COVERAGE_STOP_DELAY_S,
    COVERAGE_STOP_DETAIL,
    COVERAGE_STOP_PERCENT,
    COVERAGE_STOP_RULE,
    STUCK_COMPLETED_DETAIL,
    STUCK_COMPLETION_PERCENT,
    STUCK_COVERAGE_MAX_AGE_S,
    STUCK_DETAIL,
    STUCK_GAIN_PP,
    STUCK_NO_GAIN_DETAIL,
    STUCK_NO_GAIN_RADIUS_M,
    STUCK_NO_GAIN_WINDOW_S,
    STUCK_POSE_MAX_AGE_S,
    STUCK_RADIUS_M,
    STUCK_RULE,
    STUCK_SAMPLE_PERIOD_S,
    STUCK_WINDOW_S,
    WATCHDOG_HOLD_PROFILES,
    WATCHDOG_MAX,
    WATCHDOG_NUDGE_M,
    WATCHDOG_PROFILES,
    WATCHDOG_RELOCATE_M,
    WATCHDOG_RULE,
    WATCHDOG_SECTOR_RAD,
    WATCHDOG_SPEED_MPS,
    WATCHDOG_WINDOW_S,
    Calculator,
    Coverage,
    Reference,
)
from .relay import prefix_frames
from . import adapters
from exploration_benchmark_robots.adapters import (
    MOVING_OBJECTS_TOPIC,
    prepare as prepare_robot,
)
from exploration_benchmark_robots.known_pose import (
    GROUND_TRUTH_TOPIC,
    MAPPING_STATUS_TOPIC,
    SLAM_TOOLBOX_PATCHES,
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
# A start topic's true (profiles.yaml ``start: topic``) waits for this many start_after messages after the
# explorer's subscriber appears: at 1 Hz (the global costmap) at least one full period after discovery.
START_AFTER_MESSAGES = 2
# rrt-explore's true is repeated every wall second until its state reads ACTIVE, for at most this long (A5).
START_REPEAT_S = 30.0
# An explorer process exit becomes a runtime failure this many wall seconds after it is first seen, so that a
# native completion sent just before the exit ends the run as completed.
EXPLORER_EXIT_GRACE_S = 2.0
# Start sweep (maintainer, 26 October 2026): once the stack is ready and before the stationary warm-up, the robot turns
# START_SWEEP_RAD and back to its spawn heading at START_SWEEP_SPEED, so SLAM Toolbox (one scan per 0.05 rad turned)
# adds scans between the first scan's rays; a still robot's first map is otherwise one sparse starburst scan. 0.8 rad/s
# is below every catalog robot's limit (Waffle Pi 1.0). A sweep not finished within START_SWEEP_TIMEOUT_S simulated
# seconds ends where the robot is. Measurement starts at the first movement after activation, as before.
START_SWEEP_RAD = 0.3
START_SWEEP_SPEED = 0.8
START_SWEEP_TIMEOUT_S = 10.0
# Frontier insight node (suite runs): its output topic and the attempt log it writes.
INSIGHT_TOPIC = "/benchmark/frontier_insight"
INSIGHT_LOG = "insight.log"


def process_alive(pid, started):
    """Whether the process ``pid`` with creation time ``started`` still runs (not replaced, not a zombie)."""
    try:
        process = psutil.Process(pid)
        return process.create_time() == started and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


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
    # mapping-known-pose-v4: the SLAM Toolbox that launches must be the patched build (both patches since v8).
    try:
        slam_prefix = FilePath(get_package_prefix("slam_toolbox"))
    except PackageNotFoundError as exc:
        raise RuntimeError("slam_toolbox is not installed") from exc
    marker = slam_prefix / SLAM_TOOLBOX_PATCH_MARKER
    found = marker.read_text().strip() if marker.is_file() else None
    if found != SLAM_TOOLBOX_PATCH_SHA256:
        raise RuntimeError(
            f"SLAM Toolbox in {slam_prefix} is not the build with {', '.join(SLAM_TOOLBOX_PATCHES)} "
            "(mapping-known-pose-v8): rebuild the runtime image, then the suite image "
            f"(marker {'missing' if found is None else 'differs'})"
        )



def yaw_of(rotation):
    """The yaw (radians) of a quaternion message."""
    return math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
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
        # The triggering sample of the coverage stop (coverage-stop-v3), once reached (check_coverage_stop).
        self.coverage_stop = None
        self.stopped_by_coverage = False
        # The stuck-robot stop (stuck-stop-v1): true poses (simulation time, x, y) since movement onset back to
        # the window's anchor, the latest valid coverage, the trigger and whether it ended the run.
        self.stuck_samples = deque()
        # Valid post-motion coverage samples (simulation time, percent) for the no-gain condition.
        self.stuck_coverage = deque()
        self.last_valid_coverage = None
        self.stuck_stop = None
        self.stopped_by_stuck = False
        # The stuck watchdog (WATCHDOG_RULE) of the explorers in WATCHDOG_PROFILES: its interventions for the
        # manifest, its true poses (simulation time, x, y) back to the window's anchor, the intervention in
        # progress and the latest scan, which rates the headings.
        self.watchdog = (
            dict(rule=WATCHDOG_RULE, interventions=[]) if self.spec.profile in WATCHDOG_PROFILES else None
        )
        self.watchdog_samples = deque()
        self.watchdog_step = None
        self.watchdog_held = False
        self.latest_scan = None
        self.native_code = -1
        # Who caused a startup or runtime failure (C1, reported-outcomes-v4): "explorer" for the failures that
        # explorer_failure() queues, "stack" for every other failure; null for other outcomes.
        self.failure_source = None
        self.explorer_identity = None
        self.explorer_identities = {}
        # Wall time an explorer exit was first seen (EXPLORER_EXIT_GRACE_S), else None.
        self.explorer_exit_wall = None
        self.explorer_started_wall = 0.0
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
        # The start sweep (START_SWEEP_RAD): its state while it runs, then its record.
        self.sweep = None
        self.sweep_done = False
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
        self.spin_client = ActionClient(self.worker, Spin, "/spin")
        self.drive_client = ActionClient(self.worker, DriveOnHeading, "/drive_on_heading")
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
        self.setup_start_and_completion()
        self.thread = threading.Thread(target=self.spin_worker, daemon=True)
        self.native_controller_names = []
        self.native_controller_states = {}
        self.native_controller_future = None
        self.native_controller_query_wall = 0.0
        self.native_controller_client = self.worker.create_client(
            ListControllers, "/controller_manager/list_controllers"
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
        params, overrides = adapters.resolve_parameters(
            self.spec,
            directory,
            profile,
            robot=self.config["robot"],
            spawn=self.config["scenario"]["spawn"],
            reference=self.coverage_reference,
        )
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
            elif channel == "scan":
                self.latest_scan = message
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
            self.check_coverage_stop(row)
        row = self.coverage.update(
            simulation, wall, grid, received_wall, received_unix, force=force
        )
        if row and self.recorder:
            self.recorder.coverage_sample(row)
        self.check_coverage_stop(row)
        if self.campaign:
            self.runtime.explored_snapshot(self.coverage, grid, simulation, final=force)
        if self.recorder:
            self.recorder.coverage_checkpoint(self.coverage)

    def check_coverage_stop(self, row):
        """The coverage stop's trigger (COVERAGE_STOP_RULE): the sample that reaches the stop milestone starts its
        delay; ``coverage_stop_due`` ends the run once the delay has passed. Every valid sample also updates
        ``last_valid_coverage``, which the stuck-robot stop (``stuck_tick``) reads."""
        if not row:
            return
        if row["coverage_status"] == "valid":
            self.last_valid_coverage = row["coverage_percent"]
            if row["simulation_elapsed"] is not None:
                self.stuck_coverage.append((row["simulation"], row["coverage_percent"]))
        if self.phase != "RUNNING" or self.coverage_stop is not None:
            return
        for milestone in row["milestones_crossed"]:
            if milestone["percent"] == COVERAGE_STOP_PERCENT:
                self.coverage_stop = dict(
                    rule=COVERAGE_STOP_RULE,
                    threshold_percent=COVERAGE_STOP_PERCENT,
                    delay_seconds=COVERAGE_STOP_DELAY_S,
                    simulation=row["simulation"],
                    simulation_elapsed=row["simulation_elapsed"],
                    coverage_percent=row["coverage_percent"],
                    stop_simulation=None,
                )

    def coverage_stop_due(self):
        """Queue the coverage stop COVERAGE_STOP_DELAY_S simulated seconds after its trigger sample; the
        explorer's completion, the time limit and failures stay possible until then (first queued wins)."""
        stop = self.coverage_stop
        if stop is None or stop["stop_simulation"] is not None or self.phase != "RUNNING":
            return
        if self.clock_value - stop["simulation"] >= COVERAGE_STOP_DELAY_S:
            stop["stop_simulation"] = self.clock_value
            self.terminal.put(("completed", COVERAGE_STOP_DETAIL, -1))

    def stuck_tick(self):
        """The stuck-robot stop (STUCK_RULE): sample the true pose (TF map->base_footprint) after movement onset,
        at most every STUCK_SAMPLE_PERIOD_S simulated seconds. The run is stuck when the robot stayed within
        STUCK_RADIUS_M of its pose STUCK_WINDOW_S earlier ("still"), or within STUCK_NO_GAIN_RADIUS_M of its pose
        STUCK_NO_GAIN_WINDOW_S earlier while valid coverage gained less than STUCK_GAIN_PP over that window
        ("no_gain"); each anchor is the latest sample at or before the window start. Queue "completed" at or
        above STUCK_COMPLETION_PERCENT of the latest valid coverage, else "stuck". Triggers once."""
        if (
            self.stuck_stop is not None
            or self.watchdog_step is not None  # waits while the stuck watchdog intervenes
            or self.phase != "RUNNING"
            or not self.measurements
            or self.measurements.movement_simulation_start is None
        ):
            return
        now = self.clock_value
        samples = self.stuck_samples
        if samples and now < samples[-1][0]:
            samples.clear()  # the simulation clock went back: restart the windows
            self.stuck_coverage.clear()
        if samples and now - samples[-1][0] < STUCK_SAMPLE_PERIOD_S:
            return
        try:
            transform = self.buffer.lookup_transform("map", "base_footprint", Time())
        except TransformException:
            return
        stamp = transform.header.stamp
        if now - (stamp.sec + stamp.nanosec * 1e-9) > STUCK_POSE_MAX_AGE_S:
            return  # a stale pose is no evidence that the robot stands still
        if samples and now - samples[-1][0] > STUCK_POSE_MAX_AGE_S:
            samples.clear()  # an unobserved gap: restart the windows
        translation = transform.transform.translation
        samples.append((now, translation.x, translation.y))
        keep = now - max(STUCK_WINDOW_S, STUCK_NO_GAIN_WINDOW_S)
        while len(samples) >= 2 and samples[1][0] <= keep:
            samples.popleft()
        history = self.stuck_coverage
        while len(history) >= 2 and history[1][0] <= keep:
            history.popleft()
        gain = None
        condition, anchor = "still", self.stuck_anchor(now - STUCK_WINDOW_S, STUCK_RADIUS_M)
        if anchor is None:
            condition = "no_gain"
            anchor = self.stuck_anchor(now - STUCK_NO_GAIN_WINDOW_S, STUCK_NO_GAIN_RADIUS_M)
            gain = None if anchor is None else self.stuck_coverage_gain(anchor[0], now)
            if gain is None or gain >= STUCK_GAIN_PP:
                return
        coverage = self.last_valid_coverage
        completed = coverage is not None and coverage >= STUCK_COMPLETION_PERCENT
        still = condition == "still"
        self.stuck_stop = dict(
            rule=STUCK_RULE,
            condition=condition,
            window_seconds=STUCK_WINDOW_S if still else STUCK_NO_GAIN_WINDOW_S,
            radius_m=STUCK_RADIUS_M if still else STUCK_NO_GAIN_RADIUS_M,
            gain_threshold_pp=None if still else STUCK_GAIN_PP,
            coverage_gain_pp=gain,
            completion_percent=STUCK_COMPLETION_PERCENT,
            simulation=now,
            simulation_elapsed=now - self.measurements.movement_simulation_start,
            anchor_simulation=anchor[0],
            coverage_percent=coverage,
            outcome="completed" if completed else "stuck",
        )
        if completed:
            self.terminal.put(("completed", STUCK_COMPLETED_DETAIL, -1))
        else:
            self.terminal.put(("stuck", STUCK_DETAIL if still else STUCK_NO_GAIN_DETAIL, -1))

    def stuck_anchor(self, cutoff, radius):
        """The latest kept pose at or before ``cutoff`` when every pose since lies within ``radius`` of it."""
        start = None
        for index, sample in enumerate(self.stuck_samples):
            if sample[0] > cutoff:
                break
            start = index
        if start is None:
            return None
        anchor = self.stuck_samples[start]
        for _, x, y in itertools.islice(self.stuck_samples, start, None):
            if math.hypot(x - anchor[1], y - anchor[2]) >= radius:
                return None
        return anchor

    def stuck_coverage_gain(self, since, now):
        """Valid coverage gained since ``since`` (pp): the highest valid sample after it minus the latest one at or
        before it. None without a sample at or before ``since`` or without one in the last
        STUCK_COVERAGE_MAX_AGE_S (no evidence of the current coverage)."""
        history = self.stuck_coverage
        before = [value for time, value in history if time <= since]
        if not before or now - history[-1][0] > STUCK_COVERAGE_MAX_AGE_S:
            return None
        start = before[-1]
        return max([start, *(value for time, value in history if time > since)]) - start

    def true_pose(self):
        """The true pose (TF map->base_footprint) as (x, y, yaw), or None when unavailable or older than
        STUCK_POSE_MAX_AGE_S simulated seconds."""
        try:
            transform = self.buffer.lookup_transform("map", "base_footprint", Time())
        except TransformException:
            return None
        stamp = transform.header.stamp
        if self.clock_value - (stamp.sec + stamp.nanosec * 1e-9) > STUCK_POSE_MAX_AGE_S:
            return None
        translation, rotation = transform.transform.translation, transform.transform.rotation
        return translation.x, translation.y, yaw_of(rotation)

    def watchdog_tick(self):
        """The stuck watchdog (WATCHDOG_RULE): after activation, a true pose within STUCK_RADIUS_M of its pose
        WATCHDOG_WINDOW_S simulated seconds earlier starts an intervention (watchdog_start), one at a time and at most
        WATCHDOG_MAX per run. Poses are sampled as the stuck stop samples them (STUCK_SAMPLE_PERIOD_S,
        STUCK_POSE_MAX_AGE_S)."""
        with self.lock:
            watchdog = self.watchdog
            if (
                watchdog is None
                or self.phase != "RUNNING"
                or self.watchdog_step is not None
                or len(watchdog["interventions"]) >= WATCHDOG_MAX
            ):
                return
            now = self.clock_value
            samples = self.watchdog_samples
            if samples and now < samples[-1][0]:
                samples.clear()  # the simulation clock went back: restart the window
            if samples and now - samples[-1][0] < STUCK_SAMPLE_PERIOD_S:
                return
            pose = self.true_pose()
            if pose is None:
                return
            if samples and now - samples[-1][0] > STUCK_POSE_MAX_AGE_S:
                samples.clear()  # an unobserved gap: restart the window
            samples.append((now, pose[0], pose[1]))
            cutoff = now - WATCHDOG_WINDOW_S
            while len(samples) >= 2 and samples[1][0] <= cutoff:
                samples.popleft()
            anchor = samples[0]
            if anchor[0] > cutoff or any(
                math.hypot(x - anchor[1], y - anchor[2]) >= STUCK_RADIUS_M for _, x, y in samples
            ):
                return
            self.watchdog_start(now, pose)

    def watchdog_start(self, now, pose):
        """Start an intervention: the first a nudge (a full turn for fresh scans, then WATCHDOG_NUDGE_M toward the
        most open heading), the later ones a relocation (WATCHDOG_RELOCATE_M toward it)."""
        interventions = self.watchdog["interventions"]
        kind = "nudge" if not interventions else "relocate"
        step = dict(
            number=len(interventions) + 1,
            kind=kind,
            simulation=now,
            start=[pose[0], pose[1]],
            heading=None,
            distance_m=None,
            result=None,
            end_simulation=None,
        )
        interventions.append(step)
        self.watchdog_step = step
        self.watchdog_restart()
        if self.spec.profile in WATCHDOG_HOLD_PROFILES:
            step["held"] = True
            self.watchdog_hold(True)
        self.worker.get_logger().info(f"Stuck watchdog: {kind} {step['number']} at {now:.1f} s")
        if kind == "nudge":
            self.watchdog_spin(2 * math.pi, self.watchdog_turn)
        else:
            self.watchdog_turn()

    def watchdog_hold(self, held):
        """Pause (SIGSTOP) the explorer's owned processes and cancel its Nav2 goals, or resume them (SIGCONT);
        idempotent, so every stop path may release."""
        with self.lock:
            if held == self.watchdog_held:
                return
            self.watchdog_held = held
            processes = self.explorer_scope.live()
        for process in processes:
            try:
                process.send_signal(signal.SIGSTOP if held else signal.SIGCONT)
            except psutil.NoSuchProcess:
                pass
        if held:
            for client in self.cancel_clients:
                if client.service_is_ready():
                    client.call_async(CancelGoal.Request())

    def watchdog_restart(self):
        """Restart the watchdog's and the stuck stop's windows (when an intervention starts and when it ends)."""
        self.watchdog_samples.clear()
        self.stuck_samples.clear()
        self.stuck_coverage.clear()

    def watchdog_turn(self, _result=None):
        """Turn to the most open heading of the latest scan, then drive."""
        with self.lock:
            step = self.watchdog_step
            if step is None:
                return
            heading = self.open_heading()
            if heading is None:
                self.watchdog_end("no_scan")
                return
            step["heading"] = heading
        self.watchdog_spin(heading, self.watchdog_drive)

    def watchdog_drive(self, _result=None):
        """Drive the intervention's distance straight ahead; the behaviour server stops it before a collision."""
        with self.lock:
            step = self.watchdog_step
            if step is None:
                return
            distance = WATCHDOG_NUDGE_M if step["kind"] == "nudge" else WATCHDOG_RELOCATE_M
        goal = DriveOnHeading.Goal()
        goal.target = Point(x=distance)
        goal.speed = WATCHDOG_SPEED_MPS
        goal.time_allowance = Duration(sec=int(distance / WATCHDOG_SPEED_MPS) + 10)
        self.watchdog_send(self.drive_client, goal, self.watchdog_end)

    def watchdog_spin(self, yaw, then):
        """Turn by ``yaw`` (radians, positive counterclockwise), then call ``then`` whatever the turn's result,
        unless the behaviour server is unavailable or refused the goal."""
        if abs(yaw) < 0.05:
            then()
            return
        goal = Spin.Goal()
        goal.target_yaw = float(yaw)
        goal.time_allowance = Duration(sec=int(abs(yaw) / 0.3) + 10)

        def turned(result):
            if result in ("unavailable", "rejected"):
                self.watchdog_end(result)
            else:
                then(result)

        self.watchdog_send(self.spin_client, goal, turned)

    def watchdog_send(self, client, goal, done):
        """Send a behaviour goal; ``done`` receives "succeeded", "aborted", "canceled", "rejected",
        "unavailable" or "failed"."""
        if not client.server_is_ready():
            done("unavailable")
            return

        def finished(future):
            status = future.result().status
            done(
                "succeeded"
                if status == GoalStatus.STATUS_SUCCEEDED
                else "aborted"
                if status == GoalStatus.STATUS_ABORTED
                else "canceled"
                if status == GoalStatus.STATUS_CANCELED
                else "failed"
            )

        def accepted(future):
            handle = future.result()
            if handle is None or not handle.accepted:
                done("rejected")
                return
            handle.get_result_async().add_done_callback(finished)

        client.send_goal_async(goal).add_done_callback(accepted)

    def watchdog_end(self, result):
        """Record how the intervention ended and how far the robot got, then restart the windows."""
        with self.lock:
            step = self.watchdog_step
            if step is None:
                return
            pose = self.true_pose()
            step["result"] = result
            step["end_simulation"] = self.clock_value
            step["distance_m"] = (
                None
                if pose is None
                else math.hypot(pose[0] - step["start"][0], pose[1] - step["start"][1])
            )
            self.watchdog_step = None
            self.watchdog_restart()
            self.watchdog_hold(False)
        self.worker.get_logger().info(
            f"Stuck watchdog: {step['kind']} {step['number']} {result}, moved {step['distance_m']}"
        )

    def open_heading(self):
        """The heading (radians, robot frame) whose ±WATCHDOG_SECTOR_RAD sector of the latest scan has the largest
        smallest range: no return counts as the scan's maximum range, invalid and too-short readings as 0. None
        without a scan or its transform."""
        scan = self.latest_scan
        if scan is None or not scan.ranges or scan.angle_increment <= 0:
            return None
        try:
            transform = self.buffer.lookup_transform("base_footprint", scan.header.frame_id, Time())
        except TransformException:
            return None
        ranges = [
            scan.range_max if math.isinf(value) and value > 0
            else value if math.isfinite(value) and value >= scan.range_min
            else 0.0
            for value in scan.ranges
        ]
        count = len(ranges)
        width = max(1, int(WATCHDOG_SECTOR_RAD / scan.angle_increment))
        best = None
        for index in range(0, count, max(1, count // 72)):
            sector = min(ranges[(index + offset) % count] for offset in range(-width, width + 1))
            if best is None or sector > best[0]:
                best = (sector, index)
        angle = scan.angle_min + best[1] * scan.angle_increment + yaw_of(transform.transform.rotation)
        return math.atan2(math.sin(angle), math.cos(angle))

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
        if not missing and not self.sweep_done:
            self.sweep_tick()
            if not self.sweep_done:
                self.stationary_since = None
                self.detail = f"Start sweep (±{START_SWEEP_RAD:g} rad)"
                return False
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

    def sweep_tick(self):
        """One step of the start sweep: turn to +START_SWEEP_RAD from the ready heading (ground-truth odom ->
        base_footprint), then back to it, then stop; the record goes into the manifest (``start_sweep``)."""
        try:
            rotation = self.buffer.lookup_transform("odom", "base_footprint", Time()).transform.rotation
        except TransformException:
            return
        yaw = math.atan2(
            2 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1 - 2 * (rotation.y**2 + rotation.z**2),
        )
        if self.sweep is None:
            self.sweep = dict(origin=yaw, outward=True, started=self.clock_value, peak=0.0)
        delta = math.atan2(math.sin(yaw - self.sweep["origin"]), math.cos(yaw - self.sweep["origin"]))
        self.sweep["peak"] = max(self.sweep["peak"], delta)
        timed_out = self.clock_value - self.sweep["started"] > START_SWEEP_TIMEOUT_S
        if self.sweep["outward"] and (delta >= START_SWEEP_RAD or timed_out):
            self.sweep["outward"] = False
        command = Twist()
        if not self.sweep["outward"] and (delta <= 0.0 or timed_out):
            self.velocity_pub.publish(command)
            self.sweep_done = True
            record = dict(
                angle_rad=START_SWEEP_RAD,
                speed_rad_s=START_SWEEP_SPEED,
                peak_rad=round(self.sweep["peak"], 4),
                final_offset_rad=round(delta, 4),
                duration_s=round(self.clock_value - self.sweep["started"], 3),
                timed_out=timed_out,
            )
            if self.recorder:
                self.recorder.manifest["start_sweep"] = record
            return
        command.angular.z = START_SWEEP_SPEED if self.sweep["outward"] else -START_SWEEP_SPEED
        self.velocity_pub.publish(command)

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
        self.explorer_identities = {}
        # Wall time an explorer exit was first seen (EXPLORER_EXIT_GRACE_S), else None.
        self.explorer_exit_wall = None
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
        self.explorer_started_wall = time.monotonic()
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
        if self.campaign and profile.get("insight", True):
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

    def setup_start_and_completion(self):
        """What ends a run as completed besides the coverage stop (``completion`` in profiles.yaml), and the
        start topic of explorers that wait for one (``start: topic``)."""
        completion = self.profile.get(
            "completion", {"kind": "event", "topic": "/exploration_complete"}
        )
        self.explore_client = None
        self.explore_goal = None
        self.explore_handle = None
        self.start_publisher = None
        # start_sent: true was published (a stop then publishes false), last at start_sent_wall; start_settled:
        # nothing more to try (ACTIVE seen or refused).
        self.start_sent = False
        self.start_first_wall = self.start_sent_wall = 0.0
        self.start_settled = False
        self.explorer_active = False
        if self.profile.get("start") == "topic":
            self.start_publisher = self.worker.create_publisher(
                Bool, self.profile["start_topic"], 10
            )
            # rrt-explore indexes its costmap on its first loop tick without checking it arrived (rrt.cpp
            # find_frontiers): true goes out only once start_subscriber is on the graph and the OccupancyGrid
            # topic start_after has delivered START_AFTER_MESSAGES messages since.
            self.start_after_count = None
            self.subscriptions.append(
                self.worker.create_subscription(
                    OccupancyGrid,
                    self.profile["start_after"],
                    self.count_start_after,
                    LATCHED,
                )
            )
        if completion["kind"] == "none":
            # TDLE (decision X1): only the coverage stop or the time limit ends its runs.
            pass
        elif completion["kind"] == "event":
            self.subscriptions.append(
                self.worker.create_subscription(
                    Empty,
                    completion["topic"],
                    lambda _: self.native_completed("primary completion event"),
                    LATCHED,
                )
            )
        elif completion["kind"] == "explore_lite_status":
            from explore_lite_msgs.msg import ExploreStatus

            def status(message):
                if message.status == ExploreStatus.EXPLORATION_COMPLETE:
                    self.native_completed("explore_lite EXPLORATION_COMPLETE")

            # explore/status is transient-local with depth 10.
            self.subscriptions.append(
                self.worker.create_subscription(
                    ExploreStatus,
                    completion["topic"],
                    status,
                    QoSProfile(
                        depth=10,
                        reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    ),
                )
            )
        elif completion["kind"] == "explore_action":
            from roadmap_explorer_msgs.action import Explore

            self.explore_client = ActionClient(self.worker, Explore, completion["action"])
        elif completion["kind"] == "rrt_state":
            from rrt_explore.msg import ExplorationState

            def state(message):
                # DONE is 0, the value of an unset status: it counts only after ACTIVE.
                if message.status == ExplorationState.ACTIVE:
                    self.explorer_active = True
                elif message.status == ExplorationState.DONE and self.explorer_active:
                    self.native_completed("rrt-explore DONE")

            self.subscriptions.append(
                self.worker.create_subscription(
                    ExplorationState, completion["topic"], state, 10
                )
            )
        else:
            raise ValueError(f"Unknown completion kind: {completion['kind']}")

    def start_exploration_action(self, wall):
        """roadmap-explorer: one Explore goal (a new session) once its server is active; its result ends the run
        (NO_MORE_REACHABLE_FRONTIERS: completed, other codes: runtime failure)."""
        if self.explore_client is None or self.explore_goal is not None:
            return
        if not self.explore_client.server_is_ready():
            if wall - self.explorer_started_wall > 120:
                self.explorer_failure(
                    "startup_failure", "Explore action server not available within 120 s"
                )
                self.explore_goal = False
            return
        from roadmap_explorer_msgs.action import Explore

        goal = Explore.Goal()
        goal.exploration_bringup_mode = Explore.Goal.NEW_EXPLORATION_SESSION
        self.explore_goal = self.explore_client.send_goal_async(goal)
        self.explore_goal.add_done_callback(self.exploration_accepted)
        if self.recorder:
            self.recorder.event("exploration_goal_sent", simulation=self.clock_value)

    def count_start_after(self, _):
        if self.start_after_count is not None:
            self.start_after_count += 1

    def start_exploration_topic(self, wall):
        """rrt-explore: ``true`` on its enable topic once the node subscribes to it and its costmap has arrived
        (``setup_start_and_completion``), refused after 120 s (C1: a missing subscriber is the explorer's
        failure, missing start_after messages the stack's); repeated every wall second until its state is ACTIVE,
        for at most START_REPEAT_S (A5: a first message lost on a new connection would leave it idle; a repeated
        true is a no-op while it runs, but would restart it after DONE)."""
        if self.start_publisher is None or self.start_settled:
            return
        if self.start_sent:
            if self.explorer_active or wall - self.start_first_wall > START_REPEAT_S:
                self.start_settled = True
            elif wall - self.start_sent_wall >= 1.0:
                self.start_publisher.publish(Bool(data=True))
                self.start_sent_wall = wall
            return
        if self.start_after_count is None:
            expected = self.profile["start_subscriber"]
            if any(
                info.node_namespace.rstrip("/") + "/" + info.node_name == expected
                for info in self.worker.get_subscriptions_info_by_topic(
                    self.profile["start_topic"]
                )
            ):
                self.start_after_count = 0
        if self.start_after_count is None or self.start_after_count < START_AFTER_MESSAGES:
            if wall - self.explorer_started_wall > 120:
                # The explorer's own subscriber missing is the explorer's failure; the stack's start_after topic
                # (Nav2's global costmap) not publishing is the stack's (C1).
                if self.start_after_count is None:
                    self.explorer_failure(
                        "startup_failure",
                        f"Explorer start: no {self.profile['start_subscriber']} on "
                        f"{self.profile['start_topic']} within 120 s",
                    )
                else:
                    self.terminal.put(
                        (
                            "startup_failure",
                            f"Explorer start: no {self.profile['start_after']} messages within 120 s",
                            -1,
                        )
                    )
                self.start_settled = True
            return
        self.start_publisher.publish(Bool(data=True))
        self.start_sent = True
        self.start_first_wall = self.start_sent_wall = wall
        if self.recorder:
            self.recorder.event("exploration_start_sent", simulation=self.clock_value)

    def exploration_accepted(self, future):
        handle = future.result()
        if handle is None or not handle.accepted:
            self.explorer_failure("startup_failure", "Explore goal rejected")
            return
        self.explore_handle = handle
        handle.get_result_async().add_done_callback(self.exploration_finished)

    def exploration_finished(self, future):
        from roadmap_explorer_msgs.action import Explore

        response = future.result()
        if response is None or response.result is None:
            self.explorer_failure("runtime_failure", "Explore action ended without a result")
            return
        code = response.result.error_code
        if code == Explore.Result.NO_MORE_REACHABLE_FRONTIERS:
            self.native_completed("roadmap-explorer NO_MORE_REACHABLE_FRONTIERS")
        elif self.phase == "RUNNING":
            self.explorer_failure(
                "runtime_failure",
                f"roadmap-explorer ended with error code {code} (goal status {response.status})",
            )

    def native_completed(self, detail):
        if self.phase == "RUNNING":
            self.terminal.put(("completed", detail, -1))

    def explorer_failure(self, outcome, detail):
        """Queue a failure the explorer caused (C1): its process exited or was replaced, roadmap-explorer's action
        ended without success, or a start protocol was refused."""
        self.terminal.put((outcome, detail, -1, "explorer"))

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

    def explorer_resources(self):
        """CPU seconds and RSS (MiB) summed over the explorer's processes (``executables`` in profiles.yaml,
        decision E7), each found once in the explorer's process scope; a changed identity is a runtime failure.
        The first executable is the explorer itself (``explorer_identity``). A sample counts only when every listed
        process was read; otherwise both values are unavailable (None)."""
        executables = self.profile.get(
            "executables", [f"{self.profile['package']}/{self.profile['executable']}"]
        )
        cpu = rss = 0.0
        read = 0
        for index, name in enumerate(executables):
            package, executable = name.split("/")
            known = self.explorer_identities.get(name)
            try:
                if known is None:
                    process = self.explorer_scope.explorer(package, executable)
                else:
                    process = psutil.Process(known[0])
                if process is None:
                    continue
                identity = (process.pid, process.create_time())
                if known is not None and known != identity:
                    self.explorer_failure(
                        "runtime_failure",
                        f"Explorer process identity changed during the run ({executable})",
                    )
                    continue
                self.explorer_identities[name] = identity
                if index == 0:
                    self.explorer_identity = identity
                times = process.cpu_times()
                cpu += times.user + times.system
                rss += process.memory_info().rss / 1024**2
                read += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if read != len(executables):
            return None, None
        return cpu, rss

    def sample(self, wall, *, coverage_force=False):
        if not self.measurements:
            return
        self.coverage_tick(wall, force=coverage_force)
        cpu, rss = self.explorer_resources()
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
                if self.phase == "RUNNING":  # never overwrite a terminal reason's detail
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

    def stop(self, outcome, detail="", native_code=-1, failure_source="stack"):
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
            # Decided at the accepted terminal reason, before any later sample or failure.
            self.failure_source = (
                failure_source if outcome in ("startup_failure", "runtime_failure") else None
            )
            self.stopped_by_coverage = (outcome, detail) == (
                "completed",
                COVERAGE_STOP_DETAIL,
            )
            self.stopped_by_stuck = (outcome, detail) in (
                ("completed", STUCK_COMPLETED_DETAIL),
                ("stuck", STUCK_DETAIL),
                ("stuck", STUCK_NO_GAIN_DETAIL),
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
        # The explorer first, so it sends no new Nav2 goal while the Nav2 goals are cancelled: roadmap-explorer's
        # session goal, rrt-explore's enable topic (false stops its loop and cancels its goals).
        if self.explore_handle is not None:
            self.explore_handle.cancel_goal_async()
        if self.start_sent:
            self.start_publisher.publish(Bool(data=False))
        for client in self.cancel_clients:
            if client.service_is_ready():
                client.call_async(CancelGoal.Request())
        for _ in range(3):
            self.velocity_pub.publish(Twist())
            time.sleep(0.05)
        self.watchdog_hold(False)  # a paused explorer would ignore SIGINT and SIGTERM
        self.explorer_scope.stop()
        # The insight node only reads: stop it after the explorer, never delaying it.
        self.insight_scope.stop()
        self.simulation_scope.stop()
        self.close_source()
        if recorder:
            recorder.manifest["robot_diagnostics"] = self.robot_diagnostics
            # What ended a completed run: the explorer's completion event, the coverage stop (coverage-stop-v3) or
            # the stuck-robot stop at high coverage (stuck-stop-v1).
            stopped_by_rule = self.outcome == "completed" and self.stopped_by_coverage
            stuck_completed = self.outcome == "completed" and self.stopped_by_stuck
            recorder.manifest["completion_source"] = (
                (
                    "coverage_stop"
                    if stopped_by_rule
                    else "stuck_stop"
                    if stuck_completed
                    else "explorer"
                )
                if self.outcome == "completed"
                else None
            )
            # The stuck-robot stop's trigger when it ended the run (stuck-stop-v1), else null.
            recorder.manifest["stuck_stop"] = (
                self.stuck_stop if self.stopped_by_stuck else None
            )
            # The stuck watchdog's interventions (WATCHDOG_RULE); null for explorers without it.
            recorder.manifest["watchdog"] = self.watchdog
            recorder.manifest["coverage_stop"] = (
                self.coverage_stop if stopped_by_rule else None
            )
            # Who caused a startup or runtime failure (C1): "explorer" or "stack", else null (a storage failure
            # that replaced the outcome after the trigger carries none).
            recorder.manifest["failure_source"] = (
                self.failure_source
                if self.outcome in ("startup_failure", "runtime_failure")
                else None
            )
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
            for pid, started in self.critical_processes:
                if not process_alive(pid, started):
                    self.terminal.put(
                        (
                            "runtime_failure",
                            f"Simulation component exited (owned PID {pid})",
                            -1,
                        )
                    )
                    break
            # The explorer's listed processes (``executables``) and its launch process: an exit is the explorer's
            # failure (C1), queued EXPLORER_EXIT_GRACE_S wall s after it is first seen so that a native completion
            # callback sent just before the exit is taken first.
            dead = next(
                (
                    f"Explorer process exited ({name}, owned PID {pid})"
                    for name, (pid, started) in self.explorer_identities.items()
                    if not process_alive(pid, started)
                ),
                None,
            )
            if dead is None and self.explorer_process.poll() is not None:
                dead = f"Explorer exited with code {self.explorer_process.returncode}"
            if dead is None:
                self.explorer_exit_wall = None
            elif self.explorer_exit_wall is None:
                self.explorer_exit_wall = wall
            elif wall - self.explorer_exit_wall >= EXPLORER_EXIT_GRACE_S:
                self.explorer_failure("runtime_failure", dead)
            self.start_exploration_action(wall)
            self.start_exploration_topic(wall)
            self.coverage_stop_due()
            limit = self.config["simulation_time_limit"]
            limit_origin = self.measurements.movement_simulation_start if self.campaign else self.start_sim
            if limit and limit_origin is not None and self.clock_value - limit_origin >= limit:
                self.terminal.put(
                    ("time_limit", f"Simulation-time limit ({limit:g} seconds)", -1)
                )
            else:
                self.watchdog_tick()
                self.stuck_tick()
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
            outcome, detail, code, *source = self.terminal.get_nowait()
            if self.phase in ("STARTING", "READY", "RUNNING"):
                self.stop(outcome, detail, code, failure_source=source[0] if source else "stack")
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
            self.watchdog_hold(False)
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
