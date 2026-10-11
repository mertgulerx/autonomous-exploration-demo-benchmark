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

#include "utils/comparison_view.hpp"

#include <QCoreApplication>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display.hpp>
#include <rviz_common/display_group.hpp>
#include <rviz_common/render_panel.hpp>
#include <rviz_common/tool_manager.hpp>
#include <rviz_common/transformation/transformation_manager.hpp>
#include <rviz_common/view_controller.hpp>
#include <rviz_common/view_manager.hpp>
#include <rviz_common/visualization_manager.hpp>
#include <rviz_rendering/render_window.hpp>

namespace utils
{
ViewNode::ViewNode(const std::string& name, const std::string& root)
{
  rclcpp::NodeOptions options;
  options.use_global_arguments(false);
  options.arguments({"--ros-args", "-r", "/clock:=" + root + "/clock", "-r", "/tf:=" + root + "/tf",
                     "-r", "/tf_static:=" + root + "/tf_static"});
  options.parameter_overrides(
      {rclcpp::Parameter("use_sim_time", true), rclcpp::Parameter("benchmark_topic_root", root)});
  options.automatically_declare_parameters_from_overrides(true);
  node_ = std::make_shared<rclcpp::Node>(name, options);
}
std::string ViewNode::get_node_name() const
{
  return node_->get_name();
}
std::map<std::string, std::vector<std::string>> ViewNode::get_topic_names_and_types() const
{
  return node_->get_topic_names_and_types();
}
rclcpp::Node::SharedPtr ViewNode::get_raw_node()
{
  return node_;
}

RunFrameTransformer::RunFrameTransformer()
    : RunFrameTransformer(std::make_shared<rviz_default_plugins::transformation::TFWrapper>())
{
}
RunFrameTransformer::RunFrameTransformer(
    std::shared_ptr<rviz_default_plugins::transformation::TFWrapper> wrapper)
    : TFFrameTransformer(wrapper), wrapper_(std::move(wrapper))
{
}
RunFrameTransformer::~RunFrameTransformer()
{
  executor_.cancel();
  if (thread_.joinable())
  {
    thread_.join();
  }
}
void RunFrameTransformer::initialize(
    rviz_common::ros_integration::RosNodeAbstractionIface::WeakPtr abstraction,
    rclcpp::Clock::SharedPtr clock)
{
  auto node = abstraction.lock()->get_raw_node();
  if (!node->has_parameter("benchmark_topic_root"))
  {
    // The plugin can also be selected from the main RViz transformer menu.
    // Native single-run nodes should retain the standard TF behavior.
    TFFrameTransformer::initialize(abstraction, clock);
    return;
  }
  const auto root = node->get_parameter("benchmark_topic_root").as_string();
  tf_node_ = std::make_shared<ViewNode>(std::string(node->get_name()) + "_tf", root);
  wrapper_->initialize(clock, tf_node_, false);
  wrapper_->getBuffer()->setUsingDedicatedThread(true);
  executor_.add_node(tf_node_->get_raw_node());
  thread_ = std::thread([this]() { executor_.spin(); });
}

ComparisonView::ComparisonView(const std::string& id, const std::string& root, const QString& label,
                               QWidget* parent)
    : QWidget(parent)
{
  auto* layout = new QVBoxLayout(this);
  title = new QLabel(label);
  title->setWordWrap(true);
  title->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  title->setStyleSheet("font-weight: bold");
  layout->addWidget(title);
  panel_ = new rviz_common::RenderPanel(this);
  panel_->setMinimumSize(240, 180);
  layout->addWidget(panel_, 1);
  metrics_ = new QLabel("Waiting for measurements");
  metrics_->setWordWrap(true);
  layout->addWidget(metrics_);
  stop_button = new QPushButton("Stop Run");
  layout->addWidget(stop_button);
  node_ = std::make_shared<ViewNode>("benchmark_view_" + id, root);
  // Each RenderWindow owns a distinct Ogre scene. Never use the RViz main scene.
  // Realize the Qt container before Ogre binds the X11 child window.
  winId();
  panel_->winId();
  QCoreApplication::processEvents();
  panel_->getRenderWindow()->initialize();
  manager_ = std::make_unique<rviz_common::VisualizationManager>(
      panel_, node_, nullptr, node_->get_raw_node()->get_clock());
  panel_->initialize(manager_.get(), false);
  manager_->initialize();
  rviz_common::PluginInfo transformer;
  transformer.id = "rviz_autonomous_exploration_benchmark/RunTF";
  manager_->getTransformationManager()->setTransformer(transformer);
  manager_->setFixedFrame(QString::fromStdString(id + "/map"));
  manager_->getRootDisplayGroup()->subProp("Global Options")->subProp("Frame Rate")->setValue(10);
  auto add = [this, &root](const QString& type, const QString& name, const std::string& topic,
                           bool latched = false)
  {
    auto* display = manager_->createDisplay("rviz_default_plugins/" + type, name, true);
    display->subProp("Topic")->setValue(QString::fromStdString(root + "/" + topic));
    display->subProp("Topic")->subProp("Reliability Policy")->setValue("Reliable");
    if (latched)
    {
      display->subProp("Topic")->subProp("Durability Policy")->setValue("Transient Local");
    }
    return display;
  };
  add("Map", "Map", "map", true);
  auto* robot = manager_->createDisplay("rviz_default_plugins/RobotModel", "Robot", true);
  robot->subProp("Description Topic")
      ->setValue(QString::fromStdString(root + "/robot_description"));
  robot->subProp("TF Prefix")->setValue(QString::fromStdString(id));
  add("Path", "Measured path", "path", true);
  auto* scan = add("LaserScan", "Laser", "scan", true);
  scan->subProp("Size (m)")->setValue(0.025);
  scan->setEnabled(false);
  add("MarkerArray", "Frontiers", "frontiers", true);
  add("Path", "Navigation plan", "plan", true);
  auto* views = manager_->getViewManager();
  views->setCurrentViewControllerType("rviz_default_plugins/TopDownOrtho");
  views->getCurrent()->subProp("Scale")->setValue(20.0);
  auto* tools = manager_->getToolManager();
  tools->removeAll();
  auto* camera = tools->addTool("rviz_default_plugins/MoveCamera");
  tools->setDefaultTool(camera);
  tools->setCurrentTool(camera);
  manager_->startUpdate();
}
ComparisonView::~ComparisonView()
{
  manager_->stopUpdate();
  manager_.reset();
}
void ComparisonView::setMetrics(const QString& text, bool running, const QString& tooltip)
{
  metrics_->setText(text);
  metrics_->setToolTip(tooltip);
  stop_button->setEnabled(running);
}
void ComparisonView::setStatus(const QString& text)
{
  title->setText(text);
  title->setToolTip(text);
}
void linkCameras(const std::vector<ComparisonView*>& views, QObject* owner)
{
  auto propagating = std::make_shared<bool>(false);
  for (auto* view : views)
  {
    for (const QString name : {"Scale", "X", "Y", "Angle"})
    {
      auto* source = view->manager()->getViewManager()->getCurrent()->subProp(name);
      QObject::connect(
          source, &rviz_common::properties::Property::changed, owner,
          [source, name, view, views, propagating]()
          {
            if (*propagating)
            {
              return;
            }
            *propagating = true;
            for (auto* target : views)
            {
              if (target != view)
              {
                target->manager()->getViewManager()->getCurrent()->subProp(name)->setValue(
                    source->getValue());
              }
            }
            *propagating = false;
          });
    }
  }
}
}  // namespace utils
PLUGINLIB_EXPORT_CLASS(utils::RunFrameTransformer, rviz_common::transformation::FrameTransformer)
