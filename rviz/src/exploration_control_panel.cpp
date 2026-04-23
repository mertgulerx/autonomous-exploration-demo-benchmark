/*
Copyright 2026 Mert Güler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/


#include "utils/exploration_control_panel.hpp"

#include <unistd.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
// slam_toolbox service APIs used for runtime map reset.
#include <slam_toolbox/srv/clear.hpp>
#include <slam_toolbox/srv/clear_queue.hpp>
#include <slam_toolbox/srv/reset.hpp>

#include <QFile>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMetaObject>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QTextStream>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <stdexcept>

#include <pluginlib/class_list_macros.hpp>

namespace utils
{

// Panel constructor builds the full UI and wires runtime orchestration primitives.
// Keeping this setup centralized makes behavior predictable when the panel is recreated.
ExplorationControlPanel::ExplorationControlPanel(QWidget * parent)
: rviz_common::Panel(parent),
  process_(std::make_unique<QProcess>(this)),
  stats_timer_(std::make_unique<QTimer>(this))
{
  auto * root_layout = new QVBoxLayout();

  auto * group_box = new QGroupBox("Exploration Packages");
  auto * group_layout = new QGridLayout();

  // Row helper keeps explorer configuration declarative and avoids repetitive widget glue.
  // Each call registers one launch profile plus the process-matching metadata used later.
  auto add_explorer_row =
    [this, group_layout](int row, const QString & display_name, const QString & package_name,
    const QString & launch_file, const QString & launch_arguments, const QString & process_pattern,
    const QString & active_path_package_name,
    bool use_ros2_run = false, const QString & executable_name = QString(),
    const QString & post_start_command = QString()) {
      ExplorerEntry entry;
      entry.display_name = display_name;
      entry.active_path_package_name = active_path_package_name;
      entry.package_name = package_name;
      entry.launch_file = launch_file;
      entry.launch_arguments = launch_arguments;
      entry.executable_name = executable_name;
      entry.use_ros2_run = use_ros2_run;
      entry.post_start_command = post_start_command;
      entry.process_match_pattern = process_pattern;
      entry.start_button = new QPushButton("Start");
      connect(entry.start_button, &QPushButton::clicked, this, &ExplorationControlPanel::onStartClicked);

      auto * name_label = new QLabel(entry.display_name);
      group_layout->addWidget(name_label, row, 0);
      group_layout->addWidget(entry.start_button, row, 1);
      explorers_.push_back(entry);
    };

  add_explorer_row(
    0,
    "frontier_exploration_ros2 (mrtsp)",
    "frontier_exploration_ros2",
    "frontier_explorer.launch.py",
    "params_file:=config/frontier_exploration_ros2/config.yaml",
    "frontier_explorer",
    "frontier_exploration_ros2_mrtsp");

  add_explorer_row(
    1,
    "frontier_exploration_ros2 (nearest)",
    "frontier_exploration_ros2",
    "frontier_explorer.launch.py",
    "params_file:=config/frontier_exploration_ros2/config_nearest.yaml",
    "frontier_explorer",
    "frontier_exploration_ros2_nearest");

  add_explorer_row(
    2,
    "m_explore_ros2",
    "explore_lite",
    "explore.launch.py",
    "",
    "(__node:=explore_node|explore_lite/.*/explore)",
    "m_explore_ros2");

  add_explorer_row(
    3,
    "nav2_wavefront_frontier_exploration",
    "nav2_wfd",
    "",
    "",
    "(nav2_wfd/.*/explore|ros2 run nav2_wfd explore)",
    "",
    true,
    "explore");

  add_explorer_row(
    4,
    "roadmap_explorer",
    "rviz_autonomous_exploration_benchmark",
    "roadmap_explorer_compat.launch.py",
    "",
    "roadmap_exploration_server",
    "roadmap_explorer",
    false,
    "",
    "for i in $(seq 1 30); do "
    "ros2 action info /roadmap_explorer >/dev/null 2>&1 && break; "
    "sleep 1; "
    "done; "
    "ros2 action send_goal /roadmap_explorer roadmap_explorer_msgs/action/Explore "
    "\"{exploration_bringup_mode: 0, load_from_folder: {data: ''}, session_name: {data: ''}}\" "
    ">> /dev/null 2>&1");

  // Control buttons intentionally separate run/stop/reset concerns so operators can recover
  // from partial failures without restarting the entire RViz session.
  group_box->setLayout(group_layout);
  root_layout->addWidget(group_box);

  auto * controls_layout = new QHBoxLayout();
  end_button_ = new QPushButton("End");
  end_button_->setEnabled(false);
  connect(end_button_, &QPushButton::clicked, this, &ExplorationControlPanel::onEndClicked);
  controls_layout->addWidget(end_button_);
  path_reset_button_ = new QPushButton("Path Reset");
  connect(path_reset_button_, &QPushButton::clicked, this, &ExplorationControlPanel::onPathResetClicked);
  controls_layout->addWidget(path_reset_button_);
  map_reset_button_ = new QPushButton("Map Reset");
  connect(map_reset_button_, &QPushButton::clicked, this, &ExplorationControlPanel::onMapResetClicked);
  controls_layout->addWidget(map_reset_button_);
  return_start_button_ = new QPushButton("Return Start");
  connect(return_start_button_, &QPushButton::clicked, this, &ExplorationControlPanel::onReturnStartClicked);
  controls_layout->addWidget(return_start_button_);
  controls_layout->addStretch();
  root_layout->addLayout(controls_layout);

  status_label_ = new QLabel("Status: idle");
  pid_label_ = new QLabel("PID: -");
  cpu_label_ = new QLabel("CPU: -");
  ram_label_ = new QLabel("RAM: -");
  distance_label_ = new QLabel("Distance: -");
  time_label_ = new QLabel("Time: -");
  root_layout->addWidget(status_label_);
  root_layout->addWidget(pid_label_);
  root_layout->addWidget(cpu_label_);
  root_layout->addWidget(ram_label_);
  root_layout->addWidget(distance_label_);
  root_layout->addWidget(time_label_);
  root_layout->addStretch();

  setLayout(root_layout);

  // Process and timer signals are hooked once so run state transitions stay consistent across
  // all explorer backends.
  connect(
    process_.get(), QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
    this, &ExplorationControlPanel::onProcessFinished);
  connect(
    process_.get(), &QProcess::errorOccurred,
    this, &ExplorationControlPanel::onProcessError);
  connect(
    stats_timer_.get(), &QTimer::timeout,
    this, &ExplorationControlPanel::updateProcessStats);
  stats_timer_->setInterval(1000);

  // A dedicated metrics node lets the panel publish resets/markers and consume odometry without
  // depending on the lifecycle of whichever explorer process is currently running.
  const uint64_t node_stamp = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
  const std::string node_name =
    "exploration_control_panel_metrics_" + std::to_string(node_stamp);
  metrics_node_ = std::make_shared<rclcpp::Node>(node_name);
  metrics_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  metrics_executor_->add_node(metrics_node_);
  traveled_path_clear_pub_ = metrics_node_->create_publisher<nav_msgs::msg::Path>(
    "/explore/traversed_path",
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
  active_path_package_pub_ = metrics_node_->create_publisher<std_msgs::msg::String>(
    "/explore/path_tracker/active_package", 10);
  traveled_path_reset_pub_ = metrics_node_->create_publisher<std_msgs::msg::Empty>(
    "/explore/reset_traveled_path",
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
  frontier_clear_pub_ =
    metrics_node_->create_publisher<visualization_msgs::msg::MarkerArray>("/explore/frontiers", 10);
  navigate_to_pose_client_ =
    rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(metrics_node_, "navigate_to_pose");
  initial_pose_sub_ = metrics_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/explore/path_tracker/initial_pose",
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
    [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { onInitialPose(msg); });
  odom_sub_ = metrics_node_->create_subscription<nav_msgs::msg::Odometry>(
    "/odom",
    rclcpp::QoS(rclcpp::KeepLast(20)).best_effort(),
    [this](const nav_msgs::msg::Odometry::SharedPtr msg) { onOdometry(msg); });

  // Spin in a lightweight polling loop so callbacks continue even when UI-triggered operations
  // run in separate worker threads.
  ros_spin_thread_ = std::thread([this]() {
      while (!stop_ros_thread_ && rclcpp::ok()) {
        if (metrics_executor_) {
          metrics_executor_->spin_some();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });
}

// Destructor enforces an ordered shutdown: stop ROS loop, stop external processes, join workers.
// The order prevents callbacks from touching panel state after UI teardown starts.
ExplorationControlPanel::~ExplorationControlPanel()
{
  stop_ros_thread_ = true;
  if (metrics_executor_) {
    metrics_executor_->cancel();
  }
  stopExplorer();
  if (operation_thread_.joinable()) {
    operation_thread_.join();
  }
  if (ros_spin_thread_.joinable()) {
    ros_spin_thread_.join();
  }
  if (metrics_executor_ && metrics_node_) {
    metrics_executor_->remove_node(metrics_node_);
  }
}

// Start action is routed by sender button, so one slot can serve all explorer rows.
void ExplorationControlPanel::onStartClicked()
{
  // Prevent overlapping starts; only one explorer process is supervised at a time.
  if (process_->state() != QProcess::NotRunning) {
    return;
  }

  // Sender-based routing keeps button wiring simple even with dynamic row construction.
  auto * button = qobject_cast<QPushButton *>(sender());
  if (!button) {
    return;
  }

  // Find the row that owns the clicked button and launch that specific explorer profile.
  for (const auto & entry : explorers_) {
    if (entry.start_button == button) {
      startExplorer(entry);
      return;
    }
  }
}

// End action freezes final averages before terminating, so users keep a summary snapshot.
void ExplorationControlPanel::onEndClicked()
{
  // Force one last sampling tick so averages include the latest process counters.
  updateProcessStats();
  if (resource_sample_count_ > 0) {
    const double avg_cpu = cpu_sum_percent_ / static_cast<double>(resource_sample_count_);
    const double avg_ram = ram_sum_mb_ / static_cast<double>(resource_sample_count_);
    cpu_label_->setText(QString("CPU(avg): %1%").arg(avg_cpu, 0, 'f', 1));
    ram_label_->setText(QString("RAM(avg): %1 MB").arg(avg_ram, 0, 'f', 1));
  } else {
    // Keep UI explicit when no valid sample was collected during this run.
    cpu_label_->setText("CPU(avg): -");
    ram_label_->setText("RAM(avg): - MB");
  }
  // Clear stale frontier overlays so the next run starts from a visually clean state.
  clearFrontierMarkers();
  stopExplorer();
}

// Path reset is blocked while running to avoid mixing metrics from two logical runs.
void ExplorationControlPanel::onPathResetClicked()
{
  if (process_->state() != QProcess::NotRunning) {
    setStatusText("Status: path reset blocked (exploration is running)");
    return;
  }

  requestPathTrackerReset();
  clearTraveledPath();
  // Local distance accumulator must be reset alongside tracker state.
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    has_last_odom_ = false;
    traveled_distance_m_ = 0.0;
  }
  distance_label_->setText("Distance: 0.00 m");
  setStatusText("Status: path reset OK");
}

// Map reset tries discovered slam_toolbox services first, then falls back to conservative defaults.
// This keeps behavior resilient across slight naming/type differences between deployments.
void ExplorationControlPanel::onMapResetClicked()
{
  if (process_->state() != QProcess::NotRunning) {
    setStatusText("Status: map reset blocked (exploration is running)");
    return;
  }
  if (operation_in_progress_) {
    setStatusText("Status: map reset ignored (another request is still running)");
    return;
  }

  clearFrontierMarkers();

  auto discovered = discoverSlamResetServices();
  if (discovered.isEmpty()) {
    // Conservative fallback if graph query is late on startup.
    discovered.push_back({"/slam_toolbox/reset", "slam_toolbox/srv/Reset"});
    discovered.push_back({"/slam_toolbox/clear", "slam_toolbox/srv/Clear"});
    discovered.push_back({"/slam_toolbox/clear_queue", "slam_toolbox/srv/ClearQueue"});
    discovered.push_back({"/slam_toolbox/reset", "std_srvs/srv/Empty"});
    discovered.push_back({"/slam_toolbox/clear", "std_srvs/srv/Empty"});
    discovered.push_back({"/slam_toolbox/clear_queue", "std_srvs/srv/Empty"});
  }

  setStatusText(QString("Status: map reset requested (%1 candidate service(s))").arg(discovered.size()));
  startBackgroundOperation([this, discovered]() {
      for (const auto & candidate : discovered) {
        if (stop_ros_thread_) {
          return;
        }
        if (callSlamResetService(candidate.first, candidate.second, 2000)) {
          postStatusText(
            QString("Status: map reset OK (%1, %2)").arg(candidate.first, candidate.second));
          return;
        }
      }

      postStatusText(
        QString("Status: map reset failed (checked %1 candidate service(s))").arg(discovered.size()));
    });
}

// Return-to-start uses the same serialized background lane as map reset to avoid overlapping
// long-running requests from multiple buttons.
void ExplorationControlPanel::onReturnStartClicked()
{
  if (operation_in_progress_) {
    setStatusText("Status: return start ignored (another request is still running)");
    return;
  }
  sendReturnToStartGoal();
}

// Process-finished callback is the canonical place where run-scoped counters are reset.
void ExplorationControlPanel::onProcessFinished(int exit_code, QProcess::ExitStatus)
{
  setRunningUiState(false);
  const QString status = QString("Status: stopped (%1), exit code=%2")
    .arg(active_explorer_name_.isEmpty() ? "unknown" : active_explorer_name_)
    .arg(exit_code);
  setStatusText(status);
  pid_label_->setText("PID: -");
  // Reset all run-scoped telemetry and process matching caches for a clean next start.
  exploration_running_ = false;
  has_run_start_time_ = false;
  monitored_pid_ = -1;
  last_cpu_ticks_ = 0;
  last_cpu_check_time_ = {};
  last_cpu_pid_ = -1;
  active_process_pattern_.clear();
  active_explorer_name_.clear();
  active_log_file_.clear();
}

// Launch errors are surfaced with explorer context so operator logs stay actionable.
void ExplorationControlPanel::onProcessError(QProcess::ProcessError)
{
  setStatusText(
    QString("Status: launch error (%1)")
    .arg(active_explorer_name_.isEmpty() ? "unknown" : active_explorer_name_));
}

// startExplorer builds one shell command that sources overlays, truncates log, and execs launcher.
// Using a single shell keeps environment handling identical for all explorer backends.
void ExplorationControlPanel::startExplorer(const ExplorerEntry & entry)
{
  const QString project_root = projectRootPath();
  const QString install_setup = project_root + "/install/setup.bash";
  const std::filesystem::path log_dir =
    std::filesystem::path(project_root.toStdString()) / "logs" / "exploration";
  std::error_code ec;
  // Best-effort directory creation; launch may still proceed if folder already exists.
  std::filesystem::create_directories(log_dir, ec);
  const QString log_file = QString::fromStdString(
    (log_dir / (safeLogFileStem(entry.display_name).toStdString() + ".log")).string());
  active_log_file_ = log_file;

  QString command = "source /opt/ros/jazzy/setup.bash";
  command += " && ";
  command += "source ";
  command += shellQuote(install_setup);
  // Truncate previous log contents so each run has an isolated log timeline.
  command += " && : > ";
  command += shellQuote(log_file);
  command += " && ";
  command += "(";
  if (entry.use_ros2_run) {
    // run-mode path is used for single-node packages that do not provide launch files.
    command += "exec ros2 run ";
    command += entry.package_name;
    command += " ";
    command += entry.executable_name;
    command += " --ros-args -p use_sim_time:=true";
  } else {
    // launch-mode path supports richer bringup stacks with params/remappings.
    command += "exec ros2 launch ";
    command += entry.package_name;
    command += " ";
    command += entry.launch_file;
    command += " use_sim_time:=true";
    if (!entry.launch_arguments.trimmed().isEmpty()) {
      command += " ";
      command += entry.launch_arguments;
    }
  }
  command += ")";
  command += " >> ";
  command += shellQuote(log_file);
  command += " 2>&1";

  // The project root is used as working directory so relative launch arguments remain valid.
  process_->setWorkingDirectory(project_root);
  process_->start("bash", {"-lc", command});

  if (!process_->waitForStarted(3000)) {
    setStatusText(QString("Status: failed to start %1").arg(entry.display_name));
    pid_label_->setText("PID: -");
    return;
  }

  active_explorer_name_ = entry.display_name;
  if (!entry.active_path_package_name.trimmed().isEmpty()) {
    setActivePathPackage(entry.active_path_package_name);
  } else {
    // Fallback keeps path tracker usable even when row has no explicit package channel.
    setActivePathPackage(active_explorer_name_);
  }
  active_process_pattern_ = entry.process_match_pattern;
  if (!entry.post_start_command.trimmed().isEmpty()) {
    // Optional post-start hook is detached intentionally: it should not block the launch process.
    QString post_start = "source /opt/ros/jazzy/setup.bash";
    post_start += " && source ";
    post_start += shellQuote(install_setup);
    post_start += " && ";
    post_start += entry.post_start_command;
    QProcess::startDetached("bash", {"-lc", post_start}, project_root);
  }
  {
    // Distance always restarts from zero at explorer launch boundary.
    std::lock_guard<std::mutex> lock(odom_mutex_);
    has_last_odom_ = false;
    traveled_distance_m_ = 0.0;
  }
  cpu_sum_percent_ = 0.0;
  ram_sum_mb_ = 0.0;
  resource_sample_count_ = 0;
  // CPU delta math requires clean previous-sample state on each run.
  run_start_time_ = std::chrono::steady_clock::now();
  has_run_start_time_ = true;
  monitored_pid_ = -1;
  last_cpu_ticks_ = 0;
  last_cpu_check_time_ = {};
  last_cpu_pid_ = -1;
  exploration_running_ = true;
  setRunningUiState(true);
  setStatusText(QString("Status: running (%1)").arg(entry.display_name));
  pid_label_->setText("PID: resolving...");
  cpu_label_->setText("CPU: collecting...");
  ram_label_->setText("RAM: collecting...");
  distance_label_->setText("Distance: 0.00 m");
  time_label_->setText("Time: 00:00");
  stats_timer_->start();
}

// stopExplorer handles both running and already-exited cases, then enforces tree-wide cleanup.
// This defensive approach is required because some launch trees leave grandchildren behind.
void ExplorationControlPanel::stopExplorer()
{
  exploration_running_ = false;
  const qint64 launch_pid = process_->processId();
  const qint64 tracked_pid = monitored_pid_;
  QVector<qint64> known_pids = collectProcessTreePids(launch_pid);
  known_pids += collectProcessTreePids(tracked_pid);
  std::sort(known_pids.begin(), known_pids.end());
  known_pids.erase(std::unique(known_pids.begin(), known_pids.end()), known_pids.end());

  if (process_->state() == QProcess::NotRunning) {
    // Process may have exited unexpectedly; still enforce descendant cleanup by PID set.
    signalPids(known_pids, "TERM");
    signalProcessTree(launch_pid, "TERM");
    signalProcessTree(tracked_pid, "TERM");
    signalPids(known_pids, "KILL");
    signalProcessTree(launch_pid, "KILL");
    signalProcessTree(tracked_pid, "KILL");
    stats_timer_->stop();
    return;
  }

  process_->terminate();
  // Escalate to kill if graceful termination does not complete promptly.
  if (!process_->waitForFinished(2500)) {
    process_->kill();
    process_->waitForFinished(1000);
  }

  // Some launch trees (notably explore_lite) may keep descendants alive after
  // the parent exits, so enforce a tree-wide shutdown.
  signalPids(known_pids, "TERM");
  signalProcessTree(launch_pid, "TERM");
  signalProcessTree(tracked_pid, "TERM");
  signalPids(known_pids, "KILL");
  signalProcessTree(launch_pid, "KILL");
  signalProcessTree(tracked_pid, "KILL");
  stats_timer_->stop();
}

// UI enable/disable policy keeps mutually exclusive actions obvious to the operator.
void ExplorationControlPanel::setRunningUiState(bool running)
{
  for (const auto & entry : explorers_) {
    entry.start_button->setEnabled(!running);
  }
  end_button_->setEnabled(running);
  path_reset_button_->setEnabled(!running);
  map_reset_button_->setEnabled(!running);
  return_start_button_->setEnabled(true);
}

// setStatusText updates directly on the UI thread.
void ExplorationControlPanel::setStatusText(const QString & text)
{
  status_label_->setText(text);
}

// postStatusText is safe from worker threads by marshaling updates onto the Qt event loop.
void ExplorationControlPanel::postStatusText(const QString & text)
{
  QMetaObject::invokeMethod(
    this,
    [this, text]() { setStatusText(text); },
    Qt::QueuedConnection);
}

// Background operation lane serializes expensive requests (map reset, return-to-start, etc.).
// Reusing one worker thread avoids concurrent service storms from repeated clicks.
void ExplorationControlPanel::startBackgroundOperation(std::function<void()> task)
{
  // Single-flight guard keeps reset/goal actions serialized and easier to reason about.
  if (operation_in_progress_.exchange(true)) {
    return;
  }

  // Join previous worker before reusing the slot to avoid detached thread accumulation.
  if (operation_thread_.joinable()) {
    operation_thread_.join();
  }

  operation_thread_ = std::thread([this, task = std::move(task)]() mutable {
      try {
        task();
      } catch (const std::exception & ex) {
        postStatusText(QString("Status: background request failed (%1)").arg(ex.what()));
      } catch (...) {
        postStatusText("Status: background request failed");
      }
      operation_in_progress_ = false;
    });
}

// Initial pose is latched and reused as return-to-start target.
void ExplorationControlPanel::onInitialPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  std::lock_guard<std::mutex> lock(odom_mutex_);
  initial_pose_ = *msg;
  has_initial_pose_ = true;
}

// Project root is derived from installed share path so the panel works from build/install layouts.
QString ExplorationControlPanel::projectRootPath() const
{
  const auto benchmark_rviz_share = std::filesystem::path(
    ament_index_cpp::get_package_share_directory("rviz_autonomous_exploration_benchmark"));
  auto project_root = benchmark_rviz_share;
  // install/<pkg>/share/<pkg> -> workspace root (four parent hops).
  for (int i = 0; i < 4; ++i) {
    project_root = project_root.parent_path();
  }
  return QString::fromStdString(project_root.string());
}

// Shell quoting is intentionally minimal and only escapes single quotes for bash -lc commands.
QString ExplorationControlPanel::shellQuote(const QString & value) const
{
  QString escaped = value;
  escaped.replace("'", "'\\''");
  return "'" + escaped + "'";
}

// Log file stems are normalized so explorer names can safely map to filesystem paths.
QString ExplorationControlPanel::safeLogFileStem(const QString & value) const
{
  QString stem = value.trimmed();
  stem.replace(QRegularExpression("[^A-Za-z0-9._-]"), "_");
  if (stem.isEmpty()) {
    return "explorer";
  }
  return stem;
}

// runCommand is a short-lived helper for host-side introspection commands.
// Timeout is intentionally small to avoid stalling UI-triggered flows.
QString ExplorationControlPanel::runCommand(const QString & command) const
{
  QProcess proc;
  proc.start("bash", {"-lc", command});
  // Keep helper responsive; callers use it for control-plane checks, not long jobs.
  if (!proc.waitForFinished(300)) {
    return {};
  }
  return QString::fromUtf8(proc.readAllStandardOutput()).trimmed();
}

// Recursive shell function sends a signal to all descendants before signaling root.
// This catches wrapper-heavy launch stacks where children outlive their parent.
void ExplorationControlPanel::signalProcessTree(qint64 root_pid, const QString & signal_name) const
{
  if (root_pid <= 0) {
    return;
  }

  const QString escaped_signal = shellQuote(signal_name);
  runCommand(
    QString(
      "root=%1; sig=%2; "
      "signal_descendants() { "
      "  local p=\"$1\" child; "
      "  for child in $(ps -o pid= --ppid \"$p\"); do "
      "    signal_descendants \"$child\"; "
      "    kill -s \"$sig\" \"$child\" 2>/dev/null || true; "
      "  done; "
      "}; "
      "signal_descendants \"$root\"; "
      "kill -s \"$sig\" \"$root\" 2>/dev/null || true")
      .arg(root_pid)
      .arg(escaped_signal));
}

// /proc-based traversal avoids external tools and gives deterministic child discovery.
QVector<qint64> ExplorationControlPanel::collectProcessTreePids(qint64 root_pid) const
{
  QVector<qint64> pids;
  if (root_pid <= 0) {
    return pids;
  }

  QVector<qint64> pending{root_pid};
  while (!pending.isEmpty()) {
    const qint64 pid = pending.back();
    pending.pop_back();
    if (pid <= 0 || pids.contains(pid)) {
      continue;
    }

    pids.push_back(pid);

    QFile children_file(QString("/proc/%1/task/%1/children").arg(pid));
    if (!children_file.open(QIODevice::ReadOnly)) {
      continue;
    }

    const QList<QByteArray> child_parts = children_file.readAll().trimmed().split(' ');
    for (const QByteArray & child_part : child_parts) {
      if (child_part.isEmpty()) {
        continue;
      }

      bool ok = false;
      const qint64 child_pid = child_part.toLongLong(&ok);
      if (ok && child_pid > 0) {
        pending.push_back(child_pid);
      }
    }
  }

  std::sort(pids.begin(), pids.end());
  pids.erase(std::unique(pids.begin(), pids.end()), pids.end());
  return pids;
}

// signalPids batches explicit known PIDs when already discovered locally.
void ExplorationControlPanel::signalPids(const QVector<qint64> & pids, const QString & signal_name) const
{
  if (pids.isEmpty()) {
    return;
  }

  const QString escaped_signal = shellQuote(signal_name);
  QStringList pid_args;
  // Filter invalid PIDs defensively before constructing shell command.
  for (qint64 pid : pids) {
    if (pid > 0) {
      pid_args.push_back(QString::number(pid));
    }
  }
  if (pid_args.isEmpty()) {
    return;
  }

  runCommand(
    QString("kill -s %1 %2 2>/dev/null || true")
      .arg(escaped_signal)
      .arg(pid_args.join(' ')));
}

// Liveness check uses /proc presence as a cheap single-process probe.
bool ExplorationControlPanel::isPidAlive(qint64 pid) const
{
  if (pid <= 0) {
    return false;
  }
  return QFile::exists(QString("/proc/%1/stat").arg(pid));
}

// resolveMonitoredPid maps panel-level process patterns onto descendants of the launch PID.
// Restricting scope to the launch tree prevents cross-terminal false positives.
qint64 ExplorationControlPanel::resolveMonitoredPid() const
{
  if (monitored_pid_ > 0 && isPidAlive(monitored_pid_)) {
    return monitored_pid_;
  }

  if (active_process_pattern_.isEmpty()) {
    return -1;
  }

  // Resolve only from descendants of the panel-started launch process.
  // This avoids picking unrelated processes from other terminals/sessions.
  const qint64 launch_pid = process_->processId();
  if (launch_pid <= 0) {
    return -1;
  }

  const QRegularExpression pattern(active_process_pattern_);
  if (!pattern.isValid()) {
    return -1;
  }

  qint64 matched_pid = -1;
  const QVector<qint64> tree_pids = collectProcessTreePids(launch_pid);
  for (qint64 pid : tree_pids) {
    if (pid == launch_pid) {
      continue;
    }

    QFile cmdline_file(QString("/proc/%1/cmdline").arg(pid));
    QString command_line;
    if (cmdline_file.open(QIODevice::ReadOnly)) {
      // /proc cmdline is NUL-separated; replace with spaces for regex matching.
      QByteArray cmdline = cmdline_file.readAll();
      cmdline.replace('\0', ' ');
      command_line = QString::fromLocal8Bit(cmdline).trimmed();
    }

    if (command_line.isEmpty()) {
      // Fallback to comm when cmdline is unavailable (kernel threads/permission quirks).
      QFile comm_file(QString("/proc/%1/comm").arg(pid));
      if (comm_file.open(QIODevice::ReadOnly)) {
        command_line = QString::fromLocal8Bit(comm_file.readAll()).trimmed();
      }
    }

    if (command_line.isEmpty() || command_line.contains(".launch.py")) {
      continue;
    }

    if (pattern.match(command_line).hasMatch()) {
      // Keep last match in traversal; usually the deepest child is the true worker process.
      matched_pid = pid;
    }
  }

  return matched_pid;
}

// Elapsed time is rendered as MM:SS for compact panel display.
QString ExplorationControlPanel::formatElapsedTime(qint64 elapsed_seconds) const
{
  const qint64 minutes = elapsed_seconds / 60;
  const qint64 seconds = elapsed_seconds % 60;
  return QString("%1:%2")
    .arg(minutes, 2, 10, QLatin1Char('0'))
    .arg(seconds, 2, 10, QLatin1Char('0'));
}

// Frontier markers are cleared with DELETEALL so stale visual state is removed in one publish.
void ExplorationControlPanel::clearFrontierMarkers()
{
  if (!frontier_clear_pub_ || !metrics_node_) {
    return;
  }

  visualization_msgs::msg::MarkerArray clear_msg;
  visualization_msgs::msg::Marker marker;
  marker.header.stamp = metrics_node_->now();
  marker.header.frame_id = "map";
  marker.action = visualization_msgs::msg::Marker::DELETEALL;
  clear_msg.markers.push_back(marker);
  frontier_clear_pub_->publish(clear_msg);
}

// Traveled path reset republishes an empty path in map frame for deterministic RViz clearing.
void ExplorationControlPanel::clearTraveledPath()
{
  if (!traveled_path_clear_pub_ || !metrics_node_) {
    return;
  }

  nav_msgs::msg::Path clear_path;
  clear_path.header.stamp = metrics_node_->now();
  clear_path.header.frame_id = "map";
  clear_path.poses.clear();
  traveled_path_clear_pub_->publish(clear_path);
}

// Tracker reset topic clears per-package path history inside the tracker node.
void ExplorationControlPanel::requestPathTrackerReset()
{
  if (!traveled_path_reset_pub_) {
    return;
  }
  std_msgs::msg::Empty msg;
  traveled_path_reset_pub_->publish(msg);
}

// Active package broadcast routes new path samples into the selected package channel.
void ExplorationControlPanel::setActivePathPackage(const QString & package_name)
{
  if (!active_path_package_pub_) {
    return;
  }
  std_msgs::msg::String msg;
  msg.data = package_name.toStdString();
  active_path_package_pub_->publish(msg);
}

// Return-to-start uses the latched initial pose and Nav2 action client with async callbacks.
void ExplorationControlPanel::sendReturnToStartGoal()
{
  if (!metrics_node_ || !navigate_to_pose_client_) {
    setStatusText("Status: return start failed (client unavailable)");
    return;
  }

  geometry_msgs::msg::PoseStamped initial_pose;
  {
    // Copy under lock once so asynchronous action setup uses a stable snapshot.
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (!has_initial_pose_) {
      setStatusText("Status: return start failed (initial pose not available yet)");
      return;
    }
    initial_pose = initial_pose_;
  }

  setStatusText("Status: return start requested");
  startBackgroundOperation([this, initial_pose]() {
      if (!navigate_to_pose_client_->wait_for_action_server(std::chrono::milliseconds(1200))) {
        postStatusText("Status: return start failed (navigate_to_pose unavailable)");
        return;
      }

      nav2_msgs::action::NavigateToPose::Goal goal;
      // Stamp at send time so Nav2 receives a current header even for old latched poses.
      goal.pose.header.stamp = metrics_node_->now();
      goal.pose.header.frame_id = initial_pose.header.frame_id.empty() ? "map" : initial_pose.header.frame_id;
      goal.pose.pose = initial_pose.pose;

      rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions options;
      options.goal_response_callback =
        [this](const rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>::SharedPtr & handle) {
          if (!handle) {
            postStatusText("Status: return start rejected");
          } else {
            postStatusText("Status: return start accepted");
          }
        };
      options.result_callback =
        [this](const rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>::WrappedResult & result) {
          // Map action result codes directly to operator-facing status messages.
          if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            postStatusText("Status: return start completed");
          } else if (result.code == rclcpp_action::ResultCode::ABORTED) {
            postStatusText("Status: return start aborted");
          } else if (result.code == rclcpp_action::ResultCode::CANCELED) {
            postStatusText("Status: return start canceled");
          } else {
            postStatusText("Status: return start failed");
          }
        };
      navigate_to_pose_client_->async_send_goal(goal, options);
    });
}

// Generic Empty service helper is used for fallback reset endpoints.
bool ExplorationControlPanel::callEmptyService(const QString & service_name, int timeout_ms)
{
  if (!metrics_node_ || stop_ros_thread_) {
    return false;
  }

  auto client =
    metrics_node_->create_client<std_srvs::srv::Empty>(service_name.toStdString());
  if (!client->wait_for_service(std::chrono::milliseconds(timeout_ms))) {
    return false;
  }

  auto request = std::make_shared<std_srvs::srv::Empty::Request>();
  auto future = client->async_send_request(request);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  // Poll in short waits so shutdown can interrupt promptly.
  while (!stop_ros_thread_ && std::chrono::steady_clock::now() < deadline) {
    if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
      (void)future.get();
      return true;
    }
  }
  return false;
}

// Service discovery filters for slam/reset-like semantics, then ranks candidates by preference.
QVector<QPair<QString, QString>> ExplorationControlPanel::discoverSlamResetServices() const
{
  QVector<QPair<QString, QString>> services;
  if (!metrics_node_) {
    return services;
  }

  const auto graph = metrics_node_->get_service_names_and_types();
  for (const auto & item : graph) {
    const auto & service_name = item.first;
    const auto & types = item.second;
    const QString qname = QString::fromStdString(service_name);
    const QString lname = qname.toLower();
    for (const auto & t : types) {
      const QString qtype = QString::fromStdString(t);
      const bool supported_type =
        (qtype == "slam_toolbox/srv/Reset") ||
        (qtype == "slam_toolbox/srv/Clear") ||
        (qtype == "slam_toolbox/srv/ClearQueue") ||
        (qtype == "std_srvs/srv/Empty");
      if (!supported_type) {
        continue;
      }

      // Require both "slam-like" and "reset-like" signals to reduce false positives.
      const QString ltype = qtype.toLower();
      const bool reset_like_name = lname.contains("reset") || lname.contains("clear");
      const bool reset_like_type = ltype.contains("reset") || ltype.contains("clear");
      const bool slam_like = lname.contains("slam") || lname.contains("toolbox") || ltype.contains("slam_toolbox");
      if (slam_like && (reset_like_name || reset_like_type)) {
        services.push_back({qname, qtype});
      }
    }
  }

  std::sort(services.begin(), services.end(), [](const QPair<QString, QString> & a,
    const QPair<QString, QString> & b) {
    const QString na = a.first.toLower();
    const QString nb = b.first.toLower();
    const QString ta = a.second.toLower();
    const QString tb = b.second.toLower();
    // Prefer semantic reset endpoints before broader clear/empty fallbacks.
    auto score = [](const QString & name, const QString & type) {
      if (type.contains("slam_toolbox/srv/reset")) {return 0;}
      if (name.contains("reset")) {return 1;}
      if (type.contains("slam_toolbox/srv/clear") && !type.contains("queue")) {return 2;}
      if (name.contains("clear") && !name.contains("queue")) {return 3;}
      if (type.contains("clearqueue")) {return 4;}
      if (name.contains("clear_queue")) {return 5;}
      if (type.contains("std_srvs/srv/empty")) {return 6;}
      return 3;
    };
    const int sa = score(na, ta);
    const int sb = score(nb, tb);
    if (sa != sb) {
      return sa < sb;
    }
    if (na != nb) {
      return na < nb;
    }
    return ta < tb;
  });

  return services;
}

// Type-dispatched reset caller keeps request/response semantics specific to each service API.
bool ExplorationControlPanel::callSlamResetService(
  const QString & service_name, const QString & service_type, int timeout_ms)
{
  if (service_type == "std_srvs/srv/Empty") {
    // Legacy/bridged deployments may only expose Empty services.
    return callEmptyService(service_name, timeout_ms);
  }

  if (!metrics_node_ || stop_ros_thread_) {
    return false;
  }

  if (service_type == "slam_toolbox/srv/Reset") {
    auto client = metrics_node_->create_client<slam_toolbox::srv::Reset>(service_name.toStdString());
    if (!client->wait_for_service(std::chrono::milliseconds(timeout_ms))) {
      return false;
    }
    auto request = std::make_shared<slam_toolbox::srv::Reset::Request>();
    // Keep measurements flowing after reset so exploration can resume immediately.
    request->pause_new_measurements = false;
    auto future = client->async_send_request(request);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!stop_ros_thread_ && std::chrono::steady_clock::now() < deadline) {
      if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
        auto response = future.get();
        return response && response->result == 0;
      }
    }
    return false;
  }

  if (service_type == "slam_toolbox/srv/Clear") {
    auto client = metrics_node_->create_client<slam_toolbox::srv::Clear>(service_name.toStdString());
    if (!client->wait_for_service(std::chrono::milliseconds(timeout_ms))) {
      return false;
    }
    auto request = std::make_shared<slam_toolbox::srv::Clear::Request>();
    auto future = client->async_send_request(request);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!stop_ros_thread_ && std::chrono::steady_clock::now() < deadline) {
      if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
        (void)future.get();
        return true;
      }
    }
    return false;
  }

  if (service_type == "slam_toolbox/srv/ClearQueue") {
    auto client = metrics_node_->create_client<slam_toolbox::srv::ClearQueue>(service_name.toStdString());
    if (!client->wait_for_service(std::chrono::milliseconds(timeout_ms))) {
      return false;
    }
    auto request = std::make_shared<slam_toolbox::srv::ClearQueue::Request>();
    auto future = client->async_send_request(request);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!stop_ros_thread_ && std::chrono::steady_clock::now() < deadline) {
      if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
        auto response = future.get();
        return response && response->status;
      }
    }
    return false;
  }

  return false;
}

// updateProcessStats resolves active PID, samples /proc CPU/RAM, and updates run aggregates.
// Missing or changing process state is handled defensively to avoid stale readings.
void ExplorationControlPanel::updateProcessStats()
{
  if (process_->state() == QProcess::NotRunning) {
    stats_timer_->stop();
    return;
  }

  if (has_run_start_time_) {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - run_start_time_);
    const qint64 elapsed_seconds = std::max<qint64>(0, elapsed.count());
    time_label_->setText(QString("Time: %1").arg(formatElapsedTime(elapsed_seconds)));
  } else {
    time_label_->setText("Time: -");
  }

  double traveled_distance_m = 0.0;
  {
    // Copy shared metric under lock and render outside lock to keep callback contention low.
    std::lock_guard<std::mutex> lock(odom_mutex_);
    traveled_distance_m = traveled_distance_m_;
  }
  distance_label_->setText(QString("Distance: %1 m").arg(traveled_distance_m, 0, 'f', 2));

  if (monitored_pid_ <= 0 || !isPidAlive(monitored_pid_)) {
    const qint64 resolved_pid = resolveMonitoredPid();
    if (resolved_pid > 0) {
      monitored_pid_ = resolved_pid;
    } else {
      monitored_pid_ = -1;
    }
  }

  const qint64 pid = monitored_pid_;
  if (pid <= 0) {
    pid_label_->setText("PID: -");
    cpu_label_->setText("CPU: -");
    ram_label_->setText("RAM: - MB");
    last_cpu_ticks_ = 0;
    last_cpu_check_time_ = {};
    last_cpu_pid_ = -1;
    return;
  }
  // Show the resolved worker PID (not necessarily the top-level launch shell PID).
  pid_label_->setText(QString("PID: %1").arg(pid));

  QFile stat_file(QString("/proc/%1/stat").arg(pid));
  QFile statm_file(QString("/proc/%1/statm").arg(pid));
  if (!stat_file.open(QIODevice::ReadOnly) || !statm_file.open(QIODevice::ReadOnly)) {
    cpu_label_->setText("CPU: -");
    ram_label_->setText("RAM: - MB");
    last_cpu_ticks_ = 0;
    last_cpu_check_time_ = {};
    last_cpu_pid_ = -1;
    return;
  }

  bool ram_ok = false;
  double rss_mb = 0.0;
  // /proc/<pid>/statm second field is resident pages; convert to MB via page size.
  const QList<QByteArray> statm_parts = statm_file.readAll().trimmed().split(' ');
  if (statm_parts.size() >= 2) {
    bool ok = false;
    const uint64_t rss_pages = statm_parts[1].toULongLong(&ok);
    if (ok) {
      rss_mb = (rss_pages * static_cast<double>(sysconf(_SC_PAGESIZE))) / (1024.0 * 1024.0);
      ram_label_->setText(QString("RAM: %1 MB").arg(rss_mb, 0, 'f', 1));
      ram_ok = true;
    }
  }
  if (!ram_ok) {
    ram_label_->setText("RAM: - MB");
  }

  // cpu_ok: this refresh produced a numeric CPU percentage that can be displayed and aggregated.
  // cpu_state_updated: we parsed counters successfully, so baseline state is now synchronized.
  // cpu_percent: instantaneous estimate from delta(process_cpu_time) / delta(wall_time).
  // Keeping these flags explicit makes UI/output behavior deterministic in partial-failure cases.
  bool cpu_ok = false;
  bool cpu_state_updated = false;
  double cpu_percent = 0.0;
  // /proc/<pid>/stat includes process name in parentheses, and that name may contain spaces.
  // Example: "1234 (my process) R ...". Splitting naively by space would break field indexing.
  // We locate the last ')' first, then parse positional numeric fields from the remainder.
  const QByteArray stat_data = stat_file.readAll();
  const int comm_end = stat_data.lastIndexOf(')');
  if (comm_end > 0 && comm_end + 2 < stat_data.size()) {
    // Slice after ") " so index 0 now corresponds to kernel's original "state" field.
    const QList<QByteArray> stat_parts = stat_data.mid(comm_end + 2).trimmed().split(' ');
    // After stripping "pid (comm)", indexes 11/12 correspond to utime/stime in jiffies.
    // We need both user+kernel ticks to represent total process CPU time.
    if (stat_parts.size() > 12) {
      bool utime_ok = false;
      bool stime_ok = false;
      // Parse as unsigned 64-bit to avoid overflow on long-lived processes.
      const uint64_t utime = stat_parts[11].toULongLong(&utime_ok);
      const uint64_t stime = stat_parts[12].toULongLong(&stime_ok);
      if (utime_ok && stime_ok) {
        // total_ticks is cumulative CPU time consumed by the process since start.
        const uint64_t total_ticks = utime + stime;
        // steady_clock avoids wall-clock jumps (NTP/timezone updates) during delta computation.
        const auto now = std::chrono::steady_clock::now();
        cpu_state_updated = true;

        // CPU percent requires two samples for the same PID.
        // On first sample (or when tracked PID changes), we only prime the baseline.
        if (last_cpu_pid_ == pid) {
          // CPU usage is computed from process tick delta over wall-clock elapsed interval.
          const double elapsed_sec = std::chrono::duration<double>(now - last_cpu_check_time_).count();
          if (elapsed_sec > 0.0) {
            // Guard against counter regressions (PID recycle or proc race) by clamping delta.
            const uint64_t ticks_diff = total_ticks >= last_cpu_ticks_ ? total_ticks - last_cpu_ticks_ : 0;
            // _SC_CLK_TCK converts jiffies -> seconds for this platform.
            const double hz = static_cast<double>(sysconf(_SC_CLK_TCK));
            if (hz > 0.0) {
              // Percentage may exceed 100 on multicore if process is multithreaded.
              cpu_percent = (static_cast<double>(ticks_diff) / hz) / elapsed_sec * 100.0;
              cpu_label_->setText(QString("CPU: %1%").arg(cpu_percent, 0, 'f', 1));
              cpu_ok = true;
            }
          }
        }

        if (!cpu_ok) {
          // We intentionally avoid showing "0.0%" for primer samples because that is misleading.
          // "collecting..." tells operators that one more tick is needed for a delta-based metric.
          cpu_label_->setText("CPU: collecting...");
        }

        // Always refresh baseline after a successful parse so next tick has delta reference.
        last_cpu_ticks_ = total_ticks;
        last_cpu_check_time_ = now;
        last_cpu_pid_ = pid;
      }
    }
  }
  if (!cpu_state_updated) {
    // Parsing failure resets CPU sampling state to avoid mixing stale counters.
    // This keeps future samples self-healing when /proc becomes readable again.
    // Without reset, a later successful sample could compute an invalidly large delta.
    cpu_label_->setText("CPU: -");
    last_cpu_ticks_ = 0;
    last_cpu_check_time_ = {};
    last_cpu_pid_ = -1;
  }

  if (cpu_ok && ram_ok) {
    // Aggregate only complete samples so averages remain comparable across runs.
    // Partial samples (CPU or RAM missing) are intentionally excluded.
    cpu_sum_percent_ += cpu_percent;
    ram_sum_mb_ += rss_mb;
    ++resource_sample_count_;
  }
}

// Odom callback accumulates traveled distance only while an exploration run is active.
void ExplorationControlPanel::onOdometry(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  if (!exploration_running_ || !msg) {
    return;
  }

  std::lock_guard<std::mutex> lock(odom_mutex_);
  const double x = msg->pose.pose.position.x;
  const double y = msg->pose.pose.position.y;
  if (!has_last_odom_) {
    // First sample seeds baseline but does not contribute distance.
    has_last_odom_ = true;
    last_x_ = x;
    last_y_ = y;
    return;
  }

  const double dx = x - last_x_;
  const double dy = y - last_y_;
  const double step = std::hypot(dx, dy);
  last_x_ = x;
  last_y_ = y;

  // Guard against rare TF/odom jumps on respawn or reset.
  if (!std::isfinite(step) || step > 2.0) {
    return;
  }

  traveled_distance_m_ += step;
}

}  // namespace utils

PLUGINLIB_EXPORT_CLASS(utils::ExplorationControlPanel, rviz_common::Panel)
