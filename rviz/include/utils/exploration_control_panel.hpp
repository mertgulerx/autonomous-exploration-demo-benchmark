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

#ifndef UTILS_EXPLORATION_CONTROL_PANEL_HPP_
#define UTILS_EXPLORATION_CONTROL_PANEL_HPP_
#include <QPointer>
#include <QVector>
#include <atomic>
#include <exploration_benchmark_interfaces/msg/parameter_catalog.hpp>
#include <exploration_benchmark_interfaces/msg/profile_catalog.hpp>
#include <exploration_benchmark_interfaces/msg/robot_catalog.hpp>
#include <exploration_benchmark_interfaces/msg/run_metrics.hpp>
#include <exploration_benchmark_interfaces/msg/scenario_catalog.hpp>
#include <exploration_benchmark_interfaces/msg/session_state.hpp>
#include <exploration_benchmark_interfaces/srv/resolve_parameters.hpp>
#include <exploration_benchmark_interfaces/srv/return_to_configuration.hpp>
#include <exploration_benchmark_interfaces/srv/start_comparison.hpp>
#include <exploration_benchmark_interfaces/srv/start_single_run.hpp>
#include <exploration_benchmark_interfaces/srv/stop_comparison.hpp>
#include <exploration_benchmark_interfaces/srv/stop_run.hpp>
#include <functional>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <mutex>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav_msgs/msg/path.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/empty.hpp>
#include <thread>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>
class QLabel;
class QPushButton;
class QComboBox;
class QLineEdit;
class QSpinBox;
class QDoubleSpinBox;
class QStackedWidget;
class QMainWindow;
class QGridLayout;
class QVBoxLayout;
class QDockWidget;
namespace rviz_common
{
class VisualizationManager;
}
namespace utils
{
class ComparisonView;
class ExplorationControlPanel : public rviz_common::Panel
{
 public:
  explicit ExplorationControlPanel(QWidget* parent = nullptr);
  ~ExplorationControlPanel() override;
  void onInitialize() override;

 private:
  struct Row
  {
    QLineEdit* label;
    QPushButton* editor;
    std::vector<rcl_interfaces::msg::Parameter> parameters;
    QWidget* widget;
  };
  void catalogReceived(const exploration_benchmark_interfaces::msg::ProfileCatalog&);
  void stateReceived(const exploration_benchmark_interfaces::msg::SessionState&);
  void metricsReceived(const exploration_benchmark_interfaces::msg::RunMetrics&);
  void startComparison();
  void startSingle();
  void editParameters(size_t index);
  void scenarioReceived(const exploration_benchmark_interfaces::msg::ScenarioCatalog&);
  void robotCatalogReceived(const exploration_benchmark_interfaces::msg::RobotCatalog&);
  void robotChanged();
  void parameterCatalogReceived(const exploration_benchmark_interfaces::msg::ParameterCatalog&);
  void stopRun(const std::string& id);
  void rebuildViews(const exploration_benchmark_interfaces::msg::SessionState&);
  void switchView(bool comparison);
  void onPathResetClicked();
  void onMapResetClicked();
  void onReturnStartClicked();
  void setStatusText(const QString&);
  void postStatusText(const QString&);
  void startBackgroundOperation(std::function<void()>);
  void onInitialPose(const geometry_msgs::msg::PoseStamped::SharedPtr);
  void clearFrontierMarkers();
  void clearTraveledPath();
  void requestPathTrackerReset();
  void sendReturnToStartGoal();
  bool callEmptyService(const QString&, int);
  bool callSlamResetService(const QString&, const QString&, int);
  QVector<QPair<QString, QString>> discoverSlamResetServices() const;
  template <class Service>
  void request(const std::shared_ptr<rclcpp::Client<Service>>& client,
               const std::shared_ptr<typename Service::Request>& message)
  {
    startBackgroundOperation(
        [this, client, message]()
        {
          if (!client->wait_for_service(std::chrono::seconds(2)))
          {
            postStatusText("Coordinator unavailable; start this session using launch.sh");
            return;
          }
          auto future = client->async_send_request(message);
          if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
          {
            client->remove_pending_request(future);
            postStatusText("Control response timed out; inspect session state");
            return;
          }
          auto response = future.get();
          postStatusText(QString::fromStdString(response->message));
        });
  }
  std::string session_root_;
  std::string single_view_run_;
  bool single_has_metrics_{false};
  exploration_benchmark_interfaces::msg::ParameterCatalog parameter_catalog_;
  QComboBox* world_;
  QComboBox* robot_;
  QLabel* robot_details_;
  exploration_benchmark_interfaces::msg::RobotCatalog robot_catalog_;
  exploration_benchmark_interfaces::msg::ParameterCatalog baseline_parameter_catalog_;
  rclcpp::Subscription<exploration_benchmark_interfaces::msg::RobotCatalog>::SharedPtr robot_sub_;
  QComboBox* mode_;
  QPushButton* start_single_button_;
  QPointer<QWidget> idle_widget_;
  rclcpp::Client<exploration_benchmark_interfaces::srv::ResolveParameters>::SharedPtr
      parameter_client_;
  rclcpp::Subscription<exploration_benchmark_interfaces::msg::ScenarioCatalog>::SharedPtr
      scenario_sub_;
  rclcpp::Subscription<exploration_benchmark_interfaces::msg::ParameterCatalog>::SharedPtr
      parameter_sub_;
  exploration_benchmark_interfaces::msg::SessionState session_state_;
  std::shared_ptr<rclcpp::Node> metrics_node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> metrics_executor_;
  std::thread ros_spin_thread_, operation_thread_;
  std::atomic<bool> stop_ros_thread_{false}, operation_in_progress_{false},
      exploration_running_{false};
  std::mutex odom_mutex_;
  bool has_initial_pose_{false};
  geometry_msgs::msg::PoseStamped initial_pose_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr initial_pose_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traveled_path_clear_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr traveled_path_reset_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr frontier_clear_pub_;
  rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr navigate_to_pose_client_;
  rclcpp::Subscription<exploration_benchmark_interfaces::msg::ProfileCatalog>::SharedPtr
      catalog_sub_;
  rclcpp::Subscription<exploration_benchmark_interfaces::msg::SessionState>::SharedPtr state_sub_;
  rclcpp::Subscription<exploration_benchmark_interfaces::msg::RunMetrics>::SharedPtr metrics_sub_;
  rclcpp::Client<exploration_benchmark_interfaces::srv::StartSingleRun>::SharedPtr single_client_;
  rclcpp::Client<exploration_benchmark_interfaces::srv::StartComparison>::SharedPtr
      comparison_client_;
  rclcpp::Client<exploration_benchmark_interfaces::srv::StopRun>::SharedPtr stop_client_;
  rclcpp::Client<exploration_benchmark_interfaces::srv::StopComparison>::SharedPtr stop_all_client_;
  rclcpp::Client<exploration_benchmark_interfaces::srv::ReturnToConfiguration>::SharedPtr
      return_client_;
  QWidget* single_group_;
  QWidget* single_controls_widget_;
  std::vector<Row> rows_;
  QLabel* status_label_;
  QLabel* single_metrics_;
  QPushButton* end_button_;
  QPushButton* path_reset_button_;
  QPushButton* map_reset_button_;
  QPushButton* return_start_button_;
  QPushButton* comparison_button_;
  QPushButton* stop_all_button_;
  QPushButton* single_mode_button_;
  QSpinBox* count_;
  QDoubleSpinBox* limit_;
  QPointer<QMainWindow> window_;
  QPointer<QStackedWidget> stack_;
  QPointer<QWidget> original_central_;
  QPointer<QWidget> grid_widget_;
  rviz_common::VisualizationManager* original_manager_{nullptr};
  std::vector<ComparisonView*> views_;
  std::vector<std::string> view_ids_;
  std::vector<QPointer<QDockWidget>> hidden_docks_;
  bool comparison_mode_{false};
  bool building_views_{false};
  std::optional<exploration_benchmark_interfaces::msg::SessionState> deferred_state_;
};
}  // namespace utils
#endif
