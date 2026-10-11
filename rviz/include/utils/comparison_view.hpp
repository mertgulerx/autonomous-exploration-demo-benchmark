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

#ifndef UTILS_COMPARISON_VIEW_HPP_
#define UTILS_COMPARISON_VIEW_HPP_

#include <QWidget>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_default_plugins/transformation/tf_frame_transformer.hpp>
#include <string>
#include <vector>

class QLabel;
class QPushButton;
namespace rviz_common
{
class RenderPanel;
class VisualizationManager;
class Display;
}  // namespace rviz_common
namespace utils
{
// NodeOptions are essential: a view must follow only its own clock and TF topics.
class ViewNode : public rviz_common::ros_integration::RosNodeAbstractionIface
{
 public:
  ViewNode(const std::string& name, const std::string& topic_root);
  std::string get_node_name() const override;
  std::map<std::string, std::vector<std::string>> get_topic_names_and_types() const override;
  rclcpp::Node::SharedPtr get_raw_node() override;

 private:
  rclcpp::Node::SharedPtr node_;
};

// The upstream dedicated TF listener constructs a different node and drops remaps.
// Pump a remapped listener explicitly, retaining the standard TF plugin connector.
class RunFrameTransformer : public rviz_default_plugins::transformation::TFFrameTransformer
{
 public:
  RunFrameTransformer();
  ~RunFrameTransformer() override;
  void initialize(rviz_common::ros_integration::RosNodeAbstractionIface::WeakPtr,
                  rclcpp::Clock::SharedPtr) override;

 private:
  explicit RunFrameTransformer(std::shared_ptr<rviz_default_plugins::transformation::TFWrapper>);
  std::shared_ptr<rviz_default_plugins::transformation::TFWrapper> wrapper_;
  std::shared_ptr<ViewNode> tf_node_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread thread_;
};

class ComparisonView : public QWidget
{
 public:
  ComparisonView(const std::string& run_id, const std::string& topic_root, const QString& title,
                 QWidget* parent = nullptr);
  ~ComparisonView() override;
  rviz_common::VisualizationManager* manager() const
  {
    return manager_.get();
  }
  void setMetrics(const QString& text, bool running, const QString& tooltip = {});
  void setStatus(const QString& text);
  QLabel* title;
  QPushButton* stop_button;

 private:
  std::shared_ptr<ViewNode> node_;
  rviz_common::RenderPanel* panel_;
  std::unique_ptr<rviz_common::VisualizationManager> manager_;
  QLabel* metrics_;
};

void linkCameras(const std::vector<ComparisonView*>& views, QObject* owner);
}  // namespace utils
#endif
