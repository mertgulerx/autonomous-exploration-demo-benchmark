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

#ifndef UTILS_PARAMETER_EDITOR_HPP_
#define UTILS_PARAMETER_EDITOR_HPP_

#include <QDialog>
#include <exploration_benchmark_interfaces/msg/parameter_catalog.hpp>
#include <exploration_benchmark_interfaces/srv/resolve_parameters.hpp>
#include <map>
#include <rclcpp/rclcpp.hpp>

class QLabel;

namespace utils
{
class ParameterEditor : public QDialog
{
 public:
  using Service = exploration_benchmark_interfaces::srv::ResolveParameters;
  ParameterEditor(const exploration_benchmark_interfaces::msg::ParameterCatalog& catalog,
                  const std::vector<rcl_interfaces::msg::Parameter>& values,
                  rclcpp::Client<Service>::SharedPtr client, QWidget* parent = nullptr);
  std::vector<rcl_interfaces::msg::Parameter> parameters() const { return accepted_values_; }
  bool importYaml(const QString& text);
  QString exportYaml();

 private:
  void populate(const std::vector<rcl_interfaces::msg::Parameter>& values);
  std::vector<rcl_interfaces::msg::Parameter> collect() const;
  std::shared_ptr<Service::Response> resolve(const QString& yaml = QString());
  void changed();
  exploration_benchmark_interfaces::msg::ParameterCatalog catalog_;
  exploration_benchmark_interfaces::msg::ParameterCatalog full_catalog_;
  std::vector<rcl_interfaces::msg::Parameter> hidden_values_;
  rclcpp::Client<Service>::SharedPtr client_;
  std::map<std::string, QWidget*> fields_;
  std::map<std::string, QLabel*> labels_;
  QLabel* status_;
  std::vector<rcl_interfaces::msg::Parameter> accepted_values_;
};
}  // namespace utils
#endif
