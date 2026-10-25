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

#include <QApplication>
#include <QComboBox>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/visualization_manager.hpp>
#include <rviz_common/window_manager_interface.hpp>
#include <slam_toolbox/srv/clear.hpp>
#include <slam_toolbox/srv/clear_queue.hpp>
#include <slam_toolbox/srv/reset.hpp>
#include <std_srvs/srv/empty.hpp>

#include "utils/comparison_view.hpp"
#include "utils/parameter_editor.hpp"

namespace utils
{
namespace interfaces = exploration_benchmark_interfaces;
ExplorationControlPanel::ExplorationControlPanel(QWidget* parent) : rviz_common::Panel(parent)
{
  auto* outer = new QVBoxLayout(this);
  outer->setContentsMargins(0, 0, 0, 0);
  auto* scroll = new QScrollArea;
  scroll->setWidgetResizable(true);
  auto* content = new QWidget;
  auto* root = new QVBoxLayout(content);
  scroll->setWidget(content);
  outer->addWidget(scroll);
  auto* scenario_box = new QGroupBox("Scenario");
  auto* scenario_layout = new QVBoxLayout(scenario_box);
  world_ = new QComboBox;
  world_->setObjectName("benchmark_world");
  world_->addItem("Select a world…", "");
  scenario_layout->addWidget(world_);
  robot_ = new QComboBox;
  robot_->setObjectName("benchmark_robot");
  robot_->addItem("Loading robot catalog…", "");
  scenario_layout->addWidget(robot_);
  robot_details_ = new QLabel("Waiting for robot catalog");
  robot_details_->setWordWrap(true);
  scenario_layout->addWidget(robot_details_);
  mode_ = new QComboBox;
  mode_->setObjectName("benchmark_mode");
  mode_->addItem("Single Run");
  mode_->addItem("Comparison");
  scenario_layout->addWidget(mode_);
  root->addWidget(scenario_box);
  auto* single = new QGroupBox("Single Run Controls");
  single_group_ = single;
  auto* single_layout = new QVBoxLayout(single);
  root->addWidget(single);
  auto* controls = new QHBoxLayout;
  end_button_ = new QPushButton("Stop Run");
  end_button_->setObjectName("benchmark_stop_single");
  controls->addWidget(end_button_);
  path_reset_button_ = new QPushButton("Path Reset");
  controls->addWidget(path_reset_button_);
  map_reset_button_ = new QPushButton("Map Reset");
  controls->addWidget(map_reset_button_);
  return_start_button_ = new QPushButton("Return Start");
  controls->addWidget(return_start_button_);
  single_controls_widget_ = new QWidget;
  single_controls_widget_->setLayout(controls);
  single_layout->addWidget(single_controls_widget_);
  single_metrics_ = new QLabel("CPU: –    RAM: –\nDistance: –    Time: –");
  single_metrics_->setObjectName("benchmark_single_metrics");
  single_metrics_->setWordWrap(true);
  single_layout->addWidget(single_metrics_);
  connect(path_reset_button_, &QPushButton::clicked, this,
          &ExplorationControlPanel::onPathResetClicked);
  connect(map_reset_button_, &QPushButton::clicked, this,
          &ExplorationControlPanel::onMapResetClicked);
  connect(return_start_button_, &QPushButton::clicked, this,
          &ExplorationControlPanel::onReturnStartClicked);
  connect(end_button_, &QPushButton::clicked, this,
          [this]()
          {
            if (!session_state_.runs.empty())
            {
              stopRun(session_state_.runs.front().run_id);
            }
          });
  auto* comparison = new QGroupBox("Configurations");
  auto* settings = new QVBoxLayout(comparison);
  auto* shared = new QHBoxLayout;
  shared->addWidget(new QLabel("Runs"));
  count_ = new QSpinBox;
  count_->setRange(1, 1);
  count_->setValue(1);
  count_->setObjectName("benchmark_count");
  shared->addWidget(count_);
  settings->addLayout(shared);
  for (int i = 0; i < 4; ++i)
  {
    Row row;
    row.widget = new QWidget;
    auto* layout = new QGridLayout(row.widget);
    layout->setContentsMargins(0, 0, 0, 0);
    row.label = new QLineEdit(i == 0 ? "Default" : QString("Variant %1").arg(i));
    row.label->setPlaceholderText("Configuration label");
    row.label->setObjectName(QString("benchmark_label_%1").arg(i));
    row.editor = new QPushButton("Edit Parameters (Default)");
    row.editor->setObjectName(QString("benchmark_edit_%1").arg(i));
    layout->addWidget(row.label, 0, 0);
    layout->addWidget(row.editor, 1, 0);
    connect(row.editor, &QPushButton::clicked, this,
            [this, i]() { editParameters(static_cast<size_t>(i)); });
    row.widget->setVisible(i == 0);
    settings->addWidget(row.widget);
    rows_.push_back(row);
  }
  connect(count_, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int n)
          {
            for (size_t i = 0; i < rows_.size(); ++i)
            {
              rows_[i].widget->setVisible(static_cast<int>(i) < n);
            }
          });
  auto* time = new QHBoxLayout;
  time->addWidget(new QLabel("Simulation limit (s; 0 disables)"));
  limit_ = new QDoubleSpinBox;
  limit_->setRange(0, 1e7);
  limit_->setDecimals(1);
  limit_->setValue(1800);
  time->addWidget(limit_);
  settings->addLayout(time);
  start_single_button_ = new QPushButton("Start Single Run");
  settings->addWidget(start_single_button_);
  comparison_button_ = new QPushButton("Run Comparison");
  comparison_button_->setVisible(false);
  settings->addWidget(comparison_button_);
  stop_all_button_ = new QPushButton("Stop All");
  settings->addWidget(stop_all_button_);
  single_mode_button_ = new QPushButton("Back to Configuration");
  settings->addWidget(single_mode_button_);
  root->addWidget(comparison);
  status_label_ = new QLabel("Connecting to session coordinator");
  status_label_->setWordWrap(true);
  root->addWidget(status_label_);
  root->addStretch();
  connect(start_single_button_, &QPushButton::clicked, this, &ExplorationControlPanel::startSingle);
  connect(mode_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [this](int index)
          {
            count_->setRange(index == 0 ? 1 : 2, index == 0 ? 1 : 4);
            start_single_button_->setVisible(index == 0);
            comparison_button_->setVisible(index != 0);
            stateReceived(session_state_);
          });
  connect(world_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [this]() { stateReceived(session_state_); });
  connect(robot_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [this]() { robotChanged(); });
  connect(comparison_button_, &QPushButton::clicked, this,
          &ExplorationControlPanel::startComparison);
  connect(stop_all_button_, &QPushButton::clicked, this,
          [this]()
          {
            request<interfaces::srv::StopComparison>(
                stop_all_client_, std::make_shared<interfaces::srv::StopComparison::Request>());
          });
  connect(single_mode_button_, &QPushButton::clicked, this,
          [this]()
          {
            request<interfaces::srv::ReturnToConfiguration>(
                return_client_,
                std::make_shared<interfaces::srv::ReturnToConfiguration::Request>());
          });
  const auto session = qEnvironmentVariable("BENCHMARK_SESSION_ID", "unattached");
  session_root_ = "/benchmark/" + session.toStdString();
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  rclcpp::NodeOptions options;
  options.use_global_arguments(false);
  options.parameter_overrides({rclcpp::Parameter("use_sim_time", true)});
  metrics_node_ =
      std::make_shared<rclcpp::Node>("benchmark_panel_" + std::to_string(stamp), options);
  auto retained = rclcpp::QoS(1).reliable().transient_local();
  traveled_path_clear_pub_ =
      metrics_node_->create_publisher<nav_msgs::msg::Path>("/explore/traversed_path", retained);
  traveled_path_reset_pub_ = metrics_node_->create_publisher<std_msgs::msg::Empty>(
      "/explore/reset_traveled_path", retained);
  frontier_clear_pub_ = metrics_node_->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/explore/frontiers", 10);
  navigate_to_pose_client_ = rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(
      metrics_node_, "/navigate_to_pose");
  initial_pose_sub_ = metrics_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/explore/path_tracker/initial_pose", retained,
      [this](geometry_msgs::msg::PoseStamped::SharedPtr m) { onInitialPose(m); });
  single_client_ = metrics_node_->create_client<interfaces::srv::StartSingleRun>(
      session_root_ + "/start_single_run");
  comparison_client_ = metrics_node_->create_client<interfaces::srv::StartComparison>(
      session_root_ + "/start_comparison");
  stop_client_ =
      metrics_node_->create_client<interfaces::srv::StopRun>(session_root_ + "/stop_run");
  stop_all_client_ = metrics_node_->create_client<interfaces::srv::StopComparison>(
      session_root_ + "/stop_comparison");
  return_client_ = metrics_node_->create_client<interfaces::srv::ReturnToConfiguration>(
      session_root_ + "/return_to_configuration");
  parameter_client_ = metrics_node_->create_client<interfaces::srv::ResolveParameters>(
      session_root_ + "/resolve_parameters");
  scenario_sub_ = metrics_node_->create_subscription<interfaces::msg::ScenarioCatalog>(
      session_root_ + "/scenarios", retained,
      [this](interfaces::msg::ScenarioCatalog::SharedPtr message)
      {
        QMetaObject::invokeMethod(
            this, [this, message]() { scenarioReceived(*message); }, Qt::QueuedConnection);
      });
  parameter_sub_ = metrics_node_->create_subscription<interfaces::msg::ParameterCatalog>(
      session_root_ + "/parameters", retained,
      [this](interfaces::msg::ParameterCatalog::SharedPtr message)
      {
        QMetaObject::invokeMethod(
            this, [this, message]() { parameterCatalogReceived(*message); }, Qt::QueuedConnection);
      });
  robot_sub_ = metrics_node_->create_subscription<interfaces::msg::RobotCatalog>(
      session_root_ + "/robots", retained,
      [this](interfaces::msg::RobotCatalog::SharedPtr message)
      {
        QMetaObject::invokeMethod(this, [this, message]() { robotCatalogReceived(*message); },
                                  Qt::QueuedConnection);
      });
  catalog_sub_ = metrics_node_->create_subscription<interfaces::msg::ProfileCatalog>(
      session_root_ + "/catalog", retained,
      [this](interfaces::msg::ProfileCatalog::SharedPtr m) {
        QMetaObject::invokeMethod(this, [this, m]() { catalogReceived(*m); }, Qt::QueuedConnection);
      });
  state_sub_ = metrics_node_->create_subscription<interfaces::msg::SessionState>(
      session_root_ + "/state", retained,
      [this](interfaces::msg::SessionState::SharedPtr m) {
        QMetaObject::invokeMethod(this, [this, m]() { stateReceived(*m); }, Qt::QueuedConnection);
      });
  metrics_sub_ = metrics_node_->create_subscription<interfaces::msg::RunMetrics>(
      session_root_ + "/metrics", 20,
      [this](interfaces::msg::RunMetrics::SharedPtr m) {
        QMetaObject::invokeMethod(this, [this, m]() { metricsReceived(*m); }, Qt::QueuedConnection);
      });
  metrics_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  metrics_executor_->add_node(metrics_node_);
  ros_spin_thread_ = std::thread([this]() { metrics_executor_->spin(); });
  comparison_button_->setEnabled(false);
  start_single_button_->setEnabled(false);
  end_button_->setEnabled(false);
  stop_all_button_->setEnabled(false);
}
ExplorationControlPanel::~ExplorationControlPanel()
{
  stop_ros_thread_ = true;
  if (metrics_executor_)
  {
    metrics_executor_->cancel();
  }
  if (operation_thread_.joinable())
  {
    operation_thread_.join();
  }
  if (ros_spin_thread_.joinable())
  {
    ros_spin_thread_.join();
  }
  if (grid_widget_)
  {
    delete grid_widget_;
  }
}
void ExplorationControlPanel::onInitialize()
{
  window_ = qobject_cast<QMainWindow*>(getDisplayContext()->getWindowManager()->getParentWindow());
  original_manager_ = dynamic_cast<rviz_common::VisualizationManager*>(getDisplayContext());
  connect(qApp, &QCoreApplication::aboutToQuit, this,
          [this]()
          {
            // VisualizationFrame deletes its main render window before its
            // custom docks. Destroy embedded scenes while that GL context is
            // still available, and reject any queued ROS presentation updates.
            stop_ros_thread_ = true;
            if (metrics_executor_)
            {
              metrics_executor_->cancel();
            }
            if (grid_widget_)
            {
              delete grid_widget_;
            }
            views_.clear();
            view_ids_.clear();
            deferred_state_.reset();
          });
  QTimer::singleShot(
      0, this,
      [this]()
      {
        if (!window_ || stack_)
        {
          return;
        }
        original_central_ = window_->takeCentralWidget();
        stack_ = new QStackedWidget(window_);
        stack_->addWidget(original_central_);
        auto* idle = new QLabel(
            "Select a world and configure explorer parameters on the left.\nStart a Single Run or "
            "a Comparison when ready.");
        idle->setAlignment(Qt::AlignCenter);
        idle_widget_ = idle;
        stack_->addWidget(idle);
        stack_->setCurrentWidget(idle);
        window_->setCentralWidget(stack_);
        const auto available = window_->screen()->availableGeometry();
        if (window_->width() > available.width() || window_->height() > available.height())
        {
          window_->resize(std::min(window_->width(), available.width()),
                          std::min(window_->height(), available.height()));
          window_->move(available.topLeft());
        }
        stateReceived(session_state_);
      });
}
void ExplorationControlPanel::catalogReceived(const interfaces::msg::ProfileCatalog& catalog)
{
  if (catalog.profiles.size() != 1 || catalog.profiles.front().id != "default")
    setStatusText("Rebuild the workspace: expected the Default frontier configuration");
}
void ExplorationControlPanel::scenarioReceived(const interfaces::msg::ScenarioCatalog& catalog)
{
  const auto selected = world_->currentData().toString();
  world_->blockSignals(true);
  world_->clear();
  world_->addItem("Select a world…", "");
  for (const auto& item : catalog.scenarios)
    world_->addItem(QString::fromStdString(item.label), QString::fromStdString(item.id));
  const auto preferred =
      selected.isEmpty() ? QString::fromStdString(catalog.preselected_world) : selected;
  world_->setCurrentIndex(std::max(0, world_->findData(preferred)));
  world_->blockSignals(false);
  stateReceived(session_state_);
}
void ExplorationControlPanel::parameterCatalogReceived(
    const interfaces::msg::ParameterCatalog& catalog)
{
  baseline_parameter_catalog_ = catalog;
  parameter_catalog_ = catalog;
  robotChanged();
  stateReceived(session_state_);
}
void ExplorationControlPanel::robotCatalogReceived(const interfaces::msg::RobotCatalog& catalog)
{
  const auto selected = robot_->currentData().toString();
  robot_catalog_ = catalog;
  robot_->blockSignals(true);
  robot_->clear();
  for (const auto& profile : catalog.robots)
  {
    robot_->addItem(QString::fromStdString(profile.label) +
                        (profile.available ? QString() : " — unavailable"),
                    QString::fromStdString(profile.id));
    robot_->setItemData(robot_->count() - 1, QString::fromStdString(profile.reason), Qt::ToolTipRole);
  }
  robot_->setCurrentIndex(std::max(0, robot_->findData(
      selected.isEmpty() ? QString::fromStdString(catalog.default_robot) : selected)));
  robot_->blockSignals(false);
  robotChanged();
}
void ExplorationControlPanel::robotChanged()
{
  const auto selected = robot_->currentData().toString().toStdString();
  auto found = std::find_if(robot_catalog_.robots.begin(), robot_catalog_.robots.end(),
                            [&selected](const auto& profile) { return profile.id == selected; });
  if (found == robot_catalog_.robots.end()) return;
  robot_details_->setText(QString::fromStdString(found->lidar_label) + "\n" +
      (found->available ? QString("Limits: %1 m/s · %2 rad/s · physics %3 ms\n")
                          .arg(found->max_linear_speed).arg(found->max_angular_speed)
                          .arg(found->physics_step * 1000.0)
                       : QString()) + QString::fromStdString(found->reason));
  auto previous = parameter_catalog_;
  parameter_catalog_ = baseline_parameter_catalog_;
  parameter_catalog_.robot_id = found->id;
  for (auto& definition : parameter_catalog_.parameters)
  {
    // Robot-derived Defaults, as the coordinator's robot_parameter_schema sets them.
    if (definition.name == "max_linear_speed_vmax")
      definition.default_value.double_value = found->max_linear_speed;
    else if (definition.name == "max_angular_speed_wmax")
      definition.default_value.double_value = found->max_angular_speed;
    else if (definition.name == "goal_preemption_lidar_range_m")
      definition.default_value.double_value = found->lidar_range;
    else
      continue;
    for (auto& row : rows_)
      for (auto& parameter : row.parameters)
        for (const auto& old : previous.parameters)
          if (parameter.name == definition.name && old.name == definition.name &&
              parameter.value == old.default_value)
            parameter.value = definition.default_value;
  }
  stateReceived(session_state_);
}
void ExplorationControlPanel::editParameters(size_t index)
{
  ParameterEditor editor(parameter_catalog_, rows_.at(index).parameters, parameter_client_, this);
  if (editor.exec() != QDialog::Accepted) return;
  rows_[index].parameters = editor.parameters();
  size_t changed = 0;
  for (const auto& parameter : rows_[index].parameters)
    for (const auto& definition : parameter_catalog_.parameters)
      if (parameter.name == definition.name && parameter.value != definition.default_value)
        ++changed;
  rows_[index].editor->setText(changed ? QString("Edit Parameters (%1 changed)").arg(changed)
                                       : "Edit Parameters (Default)");
}
void ExplorationControlPanel::startSingle()
{
  auto message = std::make_shared<interfaces::srv::StartSingleRun::Request>();
  message->world = world_->currentData().toString().toStdString();
  message->robot_id = robot_->currentData().toString().toStdString();
  message->simulation_time_limit = limit_->value();
  message->spec.profile = "default";
  message->spec.label = rows_[0].label->text().toStdString();
  message->spec.overrides = rows_[0].parameters;
  request<interfaces::srv::StartSingleRun>(single_client_, message);
}
void ExplorationControlPanel::startComparison()
{
  auto message = std::make_shared<interfaces::srv::StartComparison::Request>();
  message->robot_id = robot_->currentData().toString().toStdString();
  message->world = world_->currentData().toString().toStdString();
  message->simulation_time_limit = limit_->value();
  for (int i = 0; i < count_->value(); ++i)
  {
    interfaces::msg::RunSpec spec;
    spec.profile = "default";
    spec.label = rows_[i].label->text().toStdString();
    spec.overrides = rows_[i].parameters;
    message->runs.push_back(spec);
  }
  request<interfaces::srv::StartComparison>(comparison_client_, message);
}
void ExplorationControlPanel::stopRun(const std::string& id)
{
  auto message = std::make_shared<interfaces::srv::StopRun::Request>();
  message->run_id = id;
  request<interfaces::srv::StopRun>(stop_client_, message);
}
void ExplorationControlPanel::switchView(bool comparison)
{
  if (!stack_ || comparison == comparison_mode_)
  {
    return;
  }
  comparison_mode_ = comparison;
  single_group_->setVisible(!comparison);
  single_controls_widget_->setVisible(!comparison);
  single_metrics_->setVisible(!comparison);
  if (comparison)
  {
    original_manager_->stopUpdate();
    for (auto* dock : window_->findChildren<QDockWidget*>())
    {
      if (dock->isVisible() && !dock->isAncestorOf(this))
      {
        hidden_docks_.push_back(dock);
        dock->hide();
      }
    }
  }
  else
  {
    stack_->setCurrentWidget(original_central_);
    original_manager_->resetTime();
    original_manager_->startUpdate();
    for (auto dock : hidden_docks_)
    {
      if (dock)
      {
        dock->show();
      }
    }
    hidden_docks_.clear();
    if (grid_widget_)
    {
      delete grid_widget_;
    }
    views_.clear();
    view_ids_.clear();
    has_initial_pose_ = false;
  }
  for (auto* toolbar : window_->findChildren<QToolBar*>())
  {
    toolbar->setEnabled(!comparison);
  }
  for (auto* button : window_->findChildren<QToolButton*>())
  {
    if (button->text() == "Reset")
    {
      button->setEnabled(!comparison);
    }
  }
}
void ExplorationControlPanel::rebuildViews(const interfaces::msg::SessionState& state)
{
  if (!stack_ || state.runs.empty())
  {
    return;
  }
  std::vector<std::string> ids;
  for (const auto& run : state.runs)
  {
    ids.push_back(run.run_id);
  }
  if (ids == view_ids_)
  {
    return;
  }
  // Realizing native render windows pumps Qt events. Retained ROS callbacks
  // must not replace a grid whose ComparisonView constructors are still active.
  building_views_ = true;
  if (grid_widget_)
  {
    delete grid_widget_;
  }
  views_.clear();
  view_ids_ = ids;
  grid_widget_ = new QWidget(stack_);
  auto* layout = new QGridLayout(grid_widget_);
  stack_->addWidget(grid_widget_);
  for (size_t i = 0; i < state.runs.size(); ++i)
  {
    const auto& run = state.runs[i];
    auto* view = new ComparisonView(run.run_id, run.topic_root,
                                    QString::fromStdString(run.spec.label), grid_widget_);
    layout->addWidget(view, state.runs.size() == 2 ? 0 : static_cast<int>(i) / 2,
                      static_cast<int>(i) % 2);
    views_.push_back(view);
    connect(view->stop_button, &QPushButton::clicked, this,
            [this, id = run.run_id]() { stopRun(id); });
  }
  if (state.runs.size() == 3)
  {
    layout->addWidget(new QLabel("", grid_widget_), 1, 1);
  }
  linkCameras(views_, grid_widget_);
  stack_->setCurrentWidget(grid_widget_);
  building_views_ = false;
  if (deferred_state_)
  {
    QTimer::singleShot(0, this,
                       [this]()
                       {
                         if (!deferred_state_ || stop_ros_thread_)
                         {
                           return;
                         }
                         auto state = std::move(*deferred_state_);
                         deferred_state_.reset();
                         stateReceived(state);
                       });
  }
  // RViz may restore saved dock geometry after panel initialization. Refit the
  // comparison once those events have settled, including on smaller monitors.
  QTimer::singleShot(100, this,
                     [this]()
                     {
                       if (!window_ || !comparison_mode_ || stop_ros_thread_)
                       {
                         return;
                       }
                       for (auto* dock : window_->findChildren<QDockWidget*>())
                       {
                         if (dock->isAncestorOf(this))
                         {
                           window_->resizeDocks({dock}, {380}, Qt::Horizontal);
                         }
                       }
                       const auto available = window_->screen()->availableGeometry();
                       window_->resize(std::min(window_->width(), available.width()),
                                       std::min(window_->height(), available.height()));
                     });
}
void ExplorationControlPanel::stateReceived(const interfaces::msg::SessionState& state)
{
  if (stop_ros_thread_)
  {
    return;
  }
  if (building_views_ || deferred_state_)
  {
    deferred_state_ = state;
    return;
  }
  session_state_ = state;
  const bool single = state.mode == "SINGLE";
  const bool comparison = state.mode == "COMPARISON";
  const bool live = single && !state.runs.empty() && state.runs.front().phase == "RUNNING";
  const bool stoppable = single && !state.runs.empty() && state.runs.front().phase != "FINISHED" &&
                         state.runs.front().phase != "STOPPING";
  exploration_running_ = live;
  end_button_->setEnabled(stoppable && stop_client_->service_is_ready());
  path_reset_button_->setEnabled(false);
  map_reset_button_->setEnabled(false);
  return_start_button_->setEnabled(live && navigate_to_pose_client_->action_server_is_ready());
  const bool configured = !state.session_id.empty() &&
                          std::any_of(robot_catalog_.robots.begin(), robot_catalog_.robots.end(),
                            [this](const auto& p) {
                              return p.available && p.id == robot_->currentData().toString().toStdString() &&
                                  std::find(p.worlds.begin(), p.worlds.end(), world_->currentData().toString().toStdString()) != p.worlds.end();
                            }) &&
                          !world_->currentData().toString().isEmpty() &&
                          !parameter_catalog_.parameters.empty() && state.mode != "SWITCHING" &&
                          state.mode != "STOPPING";
  comparison_button_->setEnabled(configured && !(comparison && state.busy));
  start_single_button_->setEnabled(configured && !(single && state.busy));
  for (auto& row : rows_) row.editor->setEnabled(!parameter_catalog_.parameters.empty());
  stop_all_button_->setEnabled(comparison && state.busy);
  single_mode_button_->setEnabled(state.mode != "SWITCHING" && state.mode != "IDLE" &&
                                  state.mode != "STOPPING");
  if (comparison)
  {
    switchView(true);
    rebuildViews(state);
  }
  else if (single || state.mode == "IDLE")
  {
    switchView(false);
    if (single && !state.runs.empty() && single_view_run_ != state.runs.front().run_id)
    {
      single_view_run_ = state.runs.front().run_id;
      single_has_metrics_ = false;
      has_initial_pose_ = false;
      if (original_manager_) original_manager_->resetTime();
    }
    if (single && !state.runs.empty())
    {
      const auto& run = state.runs.front();
      if (!single_has_metrics_)
        single_metrics_->setText(QString::fromStdString(
            run.phase + (run.outcome.empty() ? "" : " / " + run.outcome) + "\n" + run.detail));
    }
    if (stack_) stack_->setCurrentWidget(single ? original_central_.data() : idle_widget_.data());
  }
  if (window_)
  {
    for (auto* toolbar : window_->findChildren<QToolBar*>()) toolbar->setEnabled(live);
    for (auto* button : window_->findChildren<QToolButton*>())
      if (button->text() == "Reset") button->setEnabled(live);
  }
  single_group_->setVisible(single);
  for (size_t i = 0; i < views_.size() && i < state.runs.size(); ++i)
  {
    const auto& run = state.runs[i];
    views_[i]->setStatus(QString::fromStdString(run.spec.label + " — " + run.phase +
                                                (run.outcome.empty() ? "" : " / " + run.outcome)));
    views_[i]->stop_button->setEnabled(run.phase != "FINISHED" && run.phase != "STOPPING");
  }
  QString status = QString::fromStdString(state.detail);
  if (single && !state.runs.empty())
  {
    status = QString::fromStdString(state.runs.front().phase + " " + state.runs.front().outcome +
                                    " " + state.runs.front().detail);
  }
  setStatusText(status);
}
void ExplorationControlPanel::metricsReceived(const interfaces::msg::RunMetrics& m)
{
  if (stop_ros_thread_)
  {
    return;
  }
  const auto cpu = m.frozen ? m.cpu_average_percent : m.cpu_percent;
  const auto rss = m.frozen ? m.rss_average_mib : m.rss_mib;
  // Unmeasured values arrive as NaN (manual result schema 6) and show as a dash, never as 0.
  const auto measured = [](double value, int decimals, const QString& unit)
  { return std::isfinite(value) ? QString::number(value, 'f', decimals) + unit : QString("—"); };
  QStringList flags;
  for (const auto& flag : m.quality_flags)
  {
    flags << QString::fromStdString(flag);
  }
  const auto health = !m.motion_started && !m.frozen
                          ? QString("Waiting for first movement") +
                                (flags.empty() ? QString() : QString("; ") + flags.join(", "))
                          : (flags.empty() ? QString("OK") : flags.join(", "));
  QString outcome;
  for (const auto& run : session_state_.runs)
  {
    if (run.run_id == m.run_id)
    {
      outcome = QString::fromStdString(run.outcome.empty() ? run.phase : run.outcome);
      break;
    }
  }
  QStringList milestones;
  for (const auto& milestone : m.coverage_milestones)
  {
    const QString value = milestone.reached
                              ? QString::number(milestone.simulation_elapsed, 'f', 1) + " s"
                              : (!m.motion_started ? (m.frozen ? "not started" : "pending")
                                                   : (m.frozen ? "unreached" : "pending"));
    // Whole percentages print without decimals, 99.9 % with one.
    milestones << QString("%1%: %2").arg(QString::number(milestone.percent, 'g', 4)).arg(value);
  }
  QString coverage = "Coverage: ";
  if ((m.coverage_status == "valid" || m.coverage_status == "stale") &&
      std::isfinite(m.coverage_percent))
  {
    coverage += QString::number(m.coverage_percent, 'f', 1) + "%";
    if (m.coverage_status == "stale") coverage += " (stale)";
  }
  else
  {
    coverage += m.coverage_status == "pending" ? "pending" : "unavailable";
  }
  if (!m.coverage_quality_flags.empty())
  {
    coverage += " (flagged)";
  }
  QStringList coverage_flags;
  for (const auto& flag : m.coverage_quality_flags)
  {
    coverage_flags << QString::fromStdString(flag);
  }
  const QString coverage_tooltip =
      QString("Coverage: current SLAM free space against the fixed world reference.\n"
              "Covered: %1 / Target: %2 m²\nMap source age: %3 simulation seconds\n"
              "First reached times use the first robot movement, without interpolation.\n"
              "Coverage quality: %4")
          .arg(measured(m.coverage_covered_m2, 2, " m²"))
          .arg(m.coverage_target_m2, 0, 'f', 2)
          .arg(m.coverage_map_age, 0, 'f', 2)
          .arg(coverage_flags.empty() ? QString::fromStdString(m.coverage_status)
                                      : coverage_flags.join(", "));
  const QString text =
      QString(
          "CPU%1: %2    RAM%1: %3    Distance: %4 m\nSimulation (from first move): %5 s    "
          "RTF: %6\nTiming: %7    Outcome: %8")
          .arg(m.frozen ? " (avg)" : "")
          .arg(measured(cpu, 1, "%"))
          .arg(measured(rss, 1, " MiB"))
          .arg(m.distance_m, 0, 'f', 2)
          .arg(m.simulation_elapsed, 0, 'f', 1)
          .arg(m.rtf, 0, 'f', 2)
          .arg(health)
          .arg(outcome) + "\n" + coverage + "\nFirst reached (simulation): " +
      (milestones.empty() ? QString("pending") : milestones.join("    "));
  for (size_t i = 0; i < views_.size() && i < view_ids_.size(); ++i)
  {
    if (view_ids_[i] == m.run_id)
    {
      views_[i]->setMetrics(text, !m.frozen, coverage_tooltip);
    }
  }
  if (session_state_.mode == "SINGLE" && !session_state_.runs.empty() &&
      session_state_.runs.front().run_id == m.run_id)
  {
    single_has_metrics_ = true;
    single_metrics_->setText(text);
    single_metrics_->setToolTip(coverage_tooltip);
  }
}
void ExplorationControlPanel::onPathResetClicked()
{
  if (session_state_.mode != "SINGLE" || session_state_.busy)
  {
    return;
  }
  requestPathTrackerReset();
  clearTraveledPath();
  setStatusText("Path reset requested");
}
void ExplorationControlPanel::onReturnStartClicked()
{
  if (session_state_.mode != "SINGLE" || operation_in_progress_)
  {
    return;
  }
  sendReturnToStartGoal();
}
void ExplorationControlPanel::onMapResetClicked()
{
  if (session_state_.mode != "SINGLE" || session_state_.busy)
  {
    setStatusText("Status: map reset blocked (exploration is running)");
    return;
  }
  if (operation_in_progress_)
  {
    setStatusText("Status: map reset ignored (another request is still running)");
    return;
  }

  clearFrontierMarkers();

  auto discovered = discoverSlamResetServices();
  if (discovered.isEmpty())
  {
    // Conservative fallback if graph query is late on startup.
    discovered.push_back({"/slam_toolbox/reset", "slam_toolbox/srv/Reset"});
    discovered.push_back({"/slam_toolbox/clear", "slam_toolbox/srv/Clear"});
    discovered.push_back({"/slam_toolbox/clear_queue", "slam_toolbox/srv/ClearQueue"});
    discovered.push_back({"/slam_toolbox/reset", "std_srvs/srv/Empty"});
    discovered.push_back({"/slam_toolbox/clear", "std_srvs/srv/Empty"});
    discovered.push_back({"/slam_toolbox/clear_queue", "std_srvs/srv/Empty"});
  }

  setStatusText(
      QString("Status: map reset requested (%1 candidate service(s))").arg(discovered.size()));
  startBackgroundOperation(
      [this, discovered]()
      {
        for (const auto& candidate : discovered)
        {
          if (stop_ros_thread_)
          {
            return;
          }
          if (callSlamResetService(candidate.first, candidate.second, 2000))
          {
            postStatusText(
                QString("Status: map reset OK (%1, %2)").arg(candidate.first, candidate.second));
            return;
          }
        }

        postStatusText(QString("Status: map reset failed (checked %1 candidate service(s))")
                           .arg(discovered.size()));
      });
}

void ExplorationControlPanel::setStatusText(const QString& text) { status_label_->setText(text); }

void ExplorationControlPanel::postStatusText(const QString& text)
{
  QMetaObject::invokeMethod(this, [this, text]() { setStatusText(text); }, Qt::QueuedConnection);
}

void ExplorationControlPanel::startBackgroundOperation(std::function<void()> task)
{
  if (operation_in_progress_.exchange(true))
  {
    return;
  }

  if (operation_thread_.joinable())
  {
    operation_thread_.join();
  }

  operation_thread_ = std::thread(
      [this, task = std::move(task)]() mutable
      {
        try
        {
          task();
        }
        catch (const std::exception& ex)
        {
          postStatusText(QString("Status: background request failed (%1)").arg(ex.what()));
        }
        catch (...)
        {
          postStatusText("Status: background request failed");
        }
        operation_in_progress_ = false;
      });
}

void ExplorationControlPanel::onInitialPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!msg)
  {
    return;
  }
  std::lock_guard<std::mutex> lock(odom_mutex_);
  initial_pose_ = *msg;
  has_initial_pose_ = true;
}

void ExplorationControlPanel::clearFrontierMarkers()
{
  if (!frontier_clear_pub_ || !metrics_node_)
  {
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

void ExplorationControlPanel::clearTraveledPath()
{
  if (!traveled_path_clear_pub_ || !metrics_node_)
  {
    return;
  }

  nav_msgs::msg::Path clear_path;
  clear_path.header.stamp = metrics_node_->now();
  clear_path.header.frame_id = "map";
  clear_path.poses.clear();
  traveled_path_clear_pub_->publish(clear_path);
}

void ExplorationControlPanel::requestPathTrackerReset()
{
  if (!traveled_path_reset_pub_)
  {
    return;
  }
  std_msgs::msg::Empty msg;
  traveled_path_reset_pub_->publish(msg);
}

void ExplorationControlPanel::sendReturnToStartGoal()
{
  if (!metrics_node_ || !navigate_to_pose_client_)
  {
    setStatusText("Status: return start failed (client unavailable)");
    return;
  }

  geometry_msgs::msg::PoseStamped initial_pose;
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (!has_initial_pose_)
    {
      setStatusText("Status: return start failed (initial pose not available yet)");
      return;
    }
    initial_pose = initial_pose_;
  }

  setStatusText("Status: return start requested");
  startBackgroundOperation(
      [this, initial_pose]()
      {
        if (!navigate_to_pose_client_->wait_for_action_server(std::chrono::milliseconds(1200)))
        {
          postStatusText("Status: return start failed (navigate_to_pose unavailable)");
          return;
        }

        nav2_msgs::action::NavigateToPose::Goal goal;
        // Stamp at send time so Nav2 receives a current header even for old latched poses.
        goal.pose.header.stamp = metrics_node_->now();
        goal.pose.header.frame_id =
            initial_pose.header.frame_id.empty() ? "map" : initial_pose.header.frame_id;
        goal.pose.pose = initial_pose.pose;

        rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions options;
        options.goal_response_callback =
            [this](
                const rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>::SharedPtr&
                    handle)
        {
          if (!handle)
          {
            postStatusText("Status: return start rejected");
          }
          else
          {
            postStatusText("Status: return start accepted");
          }
        };
        options.result_callback = [this](const rclcpp_action::ClientGoalHandle<
                                         nav2_msgs::action::NavigateToPose>::WrappedResult& result)
        {
          if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
          {
            postStatusText("Status: return start completed");
          }
          else if (result.code == rclcpp_action::ResultCode::ABORTED)
          {
            postStatusText("Status: return start aborted");
          }
          else if (result.code == rclcpp_action::ResultCode::CANCELED)
          {
            postStatusText("Status: return start canceled");
          }
          else
          {
            postStatusText("Status: return start failed");
          }
        };
        navigate_to_pose_client_->async_send_goal(goal, options);
      });
}

bool ExplorationControlPanel::callEmptyService(const QString& service_name, int timeout_ms)
{
  if (!metrics_node_ || stop_ros_thread_)
  {
    return false;
  }

  auto client = metrics_node_->create_client<std_srvs::srv::Empty>(service_name.toStdString());
  if (!client->wait_for_service(std::chrono::milliseconds(timeout_ms)))
  {
    return false;
  }

  auto request = std::make_shared<std_srvs::srv::Empty::Request>();
  auto future = client->async_send_request(request);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  // Poll in short waits so shutdown can interrupt promptly.
  while (!stop_ros_thread_ && std::chrono::steady_clock::now() < deadline)
  {
    if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready)
    {
      (void)future.get();
      return true;
    }
  }
  return false;
}

QVector<QPair<QString, QString>> ExplorationControlPanel::discoverSlamResetServices() const
{
  QVector<QPair<QString, QString>> services;
  if (!metrics_node_)
  {
    return services;
  }

  const auto graph = metrics_node_->get_service_names_and_types();
  for (const auto& item : graph)
  {
    const auto& service_name = item.first;
    const auto& types = item.second;
    const QString qname = QString::fromStdString(service_name);
    const QString lname = qname.toLower();
    for (const auto& t : types)
    {
      const QString qtype = QString::fromStdString(t);
      const bool supported_type =
          (qtype == "slam_toolbox/srv/Reset") || (qtype == "slam_toolbox/srv/Clear") ||
          (qtype == "slam_toolbox/srv/ClearQueue") || (qtype == "std_srvs/srv/Empty");
      if (!supported_type)
      {
        continue;
      }

      // Require both "slam-like" and "reset-like" signals to reduce false positives.
      const QString ltype = qtype.toLower();
      const bool reset_like_name = lname.contains("reset") || lname.contains("clear");
      const bool reset_like_type = ltype.contains("reset") || ltype.contains("clear");
      const bool slam_like =
          lname.contains("slam") || lname.contains("toolbox") || ltype.contains("slam_toolbox");
      if (slam_like && (reset_like_name || reset_like_type))
      {
        services.push_back({qname, qtype});
      }
    }
  }

  std::sort(services.begin(), services.end(),
            [](const QPair<QString, QString>& a, const QPair<QString, QString>& b)
            {
              const QString na = a.first.toLower();
              const QString nb = b.first.toLower();
              const QString ta = a.second.toLower();
              const QString tb = b.second.toLower();
              // Prefer semantic reset endpoints before broader clear/empty fallbacks.
              auto score = [](const QString& name, const QString& type)
              {
                if (type.contains("slam_toolbox/srv/reset"))
                {
                  return 0;
                }
                if (name.contains("reset"))
                {
                  return 1;
                }
                if (type.contains("slam_toolbox/srv/clear") && !type.contains("queue"))
                {
                  return 2;
                }
                if (name.contains("clear") && !name.contains("queue"))
                {
                  return 3;
                }
                if (type.contains("clearqueue"))
                {
                  return 4;
                }
                if (name.contains("clear_queue"))
                {
                  return 5;
                }
                if (type.contains("std_srvs/srv/empty"))
                {
                  return 6;
                }
                return 3;
              };
              const int sa = score(na, ta);
              const int sb = score(nb, tb);
              if (sa != sb)
              {
                return sa < sb;
              }
              if (na != nb)
              {
                return na < nb;
              }
              return ta < tb;
            });

  return services;
}

bool ExplorationControlPanel::callSlamResetService(const QString& service_name,
                                                   const QString& service_type, int timeout_ms)
{
  if (service_type == "std_srvs/srv/Empty")
  {
    // Legacy/bridged deployments may only expose Empty services.
    return callEmptyService(service_name, timeout_ms);
  }

  if (!metrics_node_ || stop_ros_thread_)
  {
    return false;
  }

  if (service_type == "slam_toolbox/srv/Reset")
  {
    auto client =
        metrics_node_->create_client<slam_toolbox::srv::Reset>(service_name.toStdString());
    if (!client->wait_for_service(std::chrono::milliseconds(timeout_ms)))
    {
      return false;
    }
    auto request = std::make_shared<slam_toolbox::srv::Reset::Request>();
    // Keep measurements flowing after reset so exploration can resume immediately.
    request->pause_new_measurements = false;
    auto future = client->async_send_request(request);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!stop_ros_thread_ && std::chrono::steady_clock::now() < deadline)
    {
      if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready)
      {
        auto response = future.get();
        return response && response->result == 0;
      }
    }
    return false;
  }

  if (service_type == "slam_toolbox/srv/Clear")
  {
    auto client =
        metrics_node_->create_client<slam_toolbox::srv::Clear>(service_name.toStdString());
    if (!client->wait_for_service(std::chrono::milliseconds(timeout_ms)))
    {
      return false;
    }
    auto request = std::make_shared<slam_toolbox::srv::Clear::Request>();
    auto future = client->async_send_request(request);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!stop_ros_thread_ && std::chrono::steady_clock::now() < deadline)
    {
      if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready)
      {
        (void)future.get();
        return true;
      }
    }
    return false;
  }

  if (service_type == "slam_toolbox/srv/ClearQueue")
  {
    auto client =
        metrics_node_->create_client<slam_toolbox::srv::ClearQueue>(service_name.toStdString());
    if (!client->wait_for_service(std::chrono::milliseconds(timeout_ms)))
    {
      return false;
    }
    auto request = std::make_shared<slam_toolbox::srv::ClearQueue::Request>();
    auto future = client->async_send_request(request);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!stop_ros_thread_ && std::chrono::steady_clock::now() < deadline)
    {
      if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready)
      {
        auto response = future.get();
        return response && response->status;
      }
    }
    return false;
  }

  return false;
}

}  // namespace utils
PLUGINLIB_EXPORT_CLASS(utils::ExplorationControlPanel, rviz_common::Panel)
