#ifndef UTILS__EXPLORATION_CONTROL_PANEL_HPP_
#define UTILS__EXPLORATION_CONTROL_PANEL_HPP_

#include <QString>
#include <QVector>
#include <QWidget>
#include <QProcess>
#include <QTimer>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/empty.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class QLabel;
class QPushButton;

namespace utils
{

// RViz panel that orchestrates exploration launches and exposes run-time controls.
// The class keeps UI actions, process supervision, and ROS metric plumbing together.
class ExplorationControlPanel : public rviz_common::Panel
{
public:
  // Construction wires UI and ROS-side metric hooks; teardown stops worker threads/processes.
  explicit ExplorationControlPanel(QWidget * parent = nullptr);
  ~ExplorationControlPanel() override;

private:
  // User-triggered control callbacks for start/stop/reset/navigation actions.
  void onStartClicked();
  void onEndClicked();
  void onPathResetClicked();
  void onReturnStartClicked();

  // Process lifecycle callbacks keep UI state synchronized with external launch outcomes.
  void onProcessFinished(int exit_code, QProcess::ExitStatus exit_status);
  void onProcessError(QProcess::ProcessError error);

  // Periodic sampler that updates CPU/RAM/time/distance labels for the active explorer.
  void updateProcessStats();

  // UI rows are data-driven so adding a new explorer variant only needs a new entry.
  // Each row carries both launch metadata and optional runtime matching/activation hints.
  struct ExplorerEntry
  {
    // Human-facing explorer label shown in the panel.
    QString display_name;
    // Path-tracker channel name that should be marked active after launch.
    QString active_path_package_name;
    // ROS package used for `ros2 launch` or `ros2 run`.
    QString package_name;
    // Launch file name for launch-mode explorers.
    QString launch_file;
    // Extra launch arguments forwarded as-is.
    QString launch_arguments;
    // Executable name for run-mode explorers.
    QString executable_name;
    // Chooses `ros2 run` path when true, otherwise `ros2 launch`.
    bool use_ros2_run{false};
    // Optional post-start shell snippet for delayed actions (e.g. sending action goals).
    QString post_start_command;
    // Regex-like pattern used to resolve/monitor the true long-lived process.
    QString process_match_pattern;
    // Start button instance owned by panel layout.
    QPushButton * start_button{nullptr};
  };

  // Launch/termination primitives for the currently selected explorer entry.
  void startExplorer(const ExplorerEntry & entry);
  void stopExplorer();

  // UI and status text helpers centralize label/button updates.
  void setRunningUiState(bool running);
  void setStatusText(const QString & text);
  void postStatusText(const QString & text);

  // Background operation helper prevents UI-thread blocking for long service/action calls.
  void startBackgroundOperation(std::function<void()> task);

  // Shell/path helpers keep command construction and quoting consistent across call sites.
  QString projectRootPath() const;
  QString shellQuote(const QString & value) const;
  QString safeLogFileStem(const QString & value) const;
  QString runCommand(const QString & command) const;

  // Process tree signaling is explicit because some explorer launch paths spawn children.
  // We terminate descendants deterministically instead of relying on a single parent PID.
  void signalProcessTree(qint64 root_pid, const QString & signal_name) const;
  QVector<qint64> collectProcessTreePids(qint64 root_pid) const;
  void signalPids(const QVector<qint64> & pids, const QString & signal_name) const;

  // Process discovery helpers support resilient monitoring when launch wrappers fork.
  bool isPidAlive(qint64 pid) const;
  qint64 resolveMonitoredPid() const;
  QString formatElapsedTime(qint64 elapsed_seconds) const;

  // ROS callback hooks for incremental run metrics.
  void onOdometry(const nav_msgs::msg::Odometry::SharedPtr msg);
  void onInitialPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

  // Visualization/path reset utilities used by panel reset actions.
  void clearFrontierMarkers();
  void clearTraveledPath();
  void setActivePathPackage(const QString & package_name);
  void requestPathTrackerReset();

  // Service/action wrappers keep timeout/error handling uniform for reset workflows.
  bool callEmptyService(const QString & service_name, int timeout_ms);
  bool callSlamResetService(const QString & service_name, const QString & service_type, int timeout_ms);
  QVector<QPair<QString, QString>> discoverSlamResetServices() const;
  void onMapResetClicked();
  void sendReturnToStartGoal();

  // Core explorer launch/session bookkeeping and sampling state.
  QVector<ExplorerEntry> explorers_;
  // Long-lived shell process used to host the explorer launch command.
  std::unique_ptr<QProcess> process_;
  // Periodic timer driving process resource sampling.
  std::unique_ptr<QTimer> stats_timer_;
  // Dedicated executor for ROS callbacks owned by this panel.
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> metrics_executor_;
  // Active explorer identity and log routing metadata.
  QString active_explorer_name_;
  QString active_log_file_;
  QString active_process_pattern_;
  // Worker threads: one for ROS spinning and one for serialized background operations.
  std::thread ros_spin_thread_;
  std::thread operation_thread_;
  // Atomic guards used across UI/callback/worker contexts.
  std::atomic<bool> stop_ros_thread_{false};
  std::atomic<bool> exploration_running_{false};
  std::atomic<bool> operation_in_progress_{false};
  // Currently monitored process id for stats and shutdown.
  qint64 monitored_pid_{-1};
  // Aggregated resource counters for computing run averages.
  double cpu_sum_percent_{0.0};
  double ram_sum_mb_{0.0};
  qint64 resource_sample_count_{0};
  // CPU deltas require previous snapshot ticks/time/pid.
  uint64_t last_cpu_ticks_{0};
  std::chrono::steady_clock::time_point last_cpu_check_time_{};
  qint64 last_cpu_pid_{-1};

  // ROS interfaces used by the panel for metrics, path reset, marker clear, and return-to-start.
  rclcpp::Node::SharedPtr metrics_node_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr initial_pose_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traveled_path_clear_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr active_path_package_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr traveled_path_reset_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr frontier_clear_pub_;
  rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr navigate_to_pose_client_;

  // Odom-derived run metrics are protected because callbacks and UI updates run concurrently.
  mutable std::mutex odom_mutex_;
  // Track whether baseline/current odometry points are valid.
  bool has_last_odom_{false};
  bool has_initial_pose_{false};
  // Initial pose is reused for return-to-start requests.
  geometry_msgs::msg::PoseStamped initial_pose_;
  // Incremental distance accumulator state.
  double last_x_{0.0};
  double last_y_{0.0};
  double traveled_distance_m_{0.0};
  // Wall-clock run timing cache for elapsed-time display.
  std::chrono::steady_clock::time_point run_start_time_{};
  bool has_run_start_time_{false};

  // Cached widget pointers let callbacks update the panel without repeated lookup.
  QLabel * status_label_{nullptr};
  QLabel * pid_label_{nullptr};
  QLabel * cpu_label_{nullptr};
  QLabel * ram_label_{nullptr};
  QLabel * distance_label_{nullptr};
  QLabel * time_label_{nullptr};
  QPushButton * end_button_{nullptr};
  QPushButton * path_reset_button_{nullptr};
  QPushButton * map_reset_button_{nullptr};
  QPushButton * return_start_button_{nullptr};
};

}  // namespace utils

#endif  // UTILS__EXPLORATION_CONTROL_PANEL_HPP_
