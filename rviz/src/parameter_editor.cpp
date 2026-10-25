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

#include "utils/parameter_editor.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleValidator>
#include <QFile>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QPushButton>
#include <QRegularExpressionValidator>
#include <QScrollArea>
#include <QVBoxLayout>
#include <algorithm>
#include <rclcpp/parameter.hpp>
#include <stdexcept>

namespace utils
{
namespace interfaces = exploration_benchmark_interfaces;
ParameterEditor::ParameterEditor(const interfaces::msg::ParameterCatalog& catalog,
                                 const std::vector<rcl_interfaces::msg::Parameter>& values,
                                 rclcpp::Client<Service>::SharedPtr client, QWidget* parent)
    : QDialog(parent), catalog_(catalog), full_catalog_(catalog), client_(std::move(client))
{
  // Limit the form to research controls. Retain omitted values in the snapshot
  // so opening/saving an imported configuration never silently changes it.
  auto& definitions = catalog_.parameters;
  definitions.erase(std::remove_if(definitions.begin(), definitions.end(),
                                   [](const auto& definition)
                                   {
                                     return definition.managed ||
                                            definition.group == "Suppression" ||
                                            definition.group == "QoS" ||
                                            definition.group == "Visualization" ||
                                            definition.group == "Benchmark Integration";
                                   }),
                    definitions.end());
  setWindowTitle("Explorer Parameters");
  resize(760, 650);
  auto* layout = new QVBoxLayout(this);
  layout->addWidget(new QLabel("* marks a change from Default. Settings apply to the next run."));
  auto* contents = new QWidget;
  auto* groups = new QVBoxLayout(contents);
  std::map<std::string, QFormLayout*> forms;
  auto display_order = catalog_.parameters;
  const std::map<std::string, int> priorities = {{"Behavior", 0},
                                                 {"MRTSP Scoring", 1},
                                                 {"Navigation Safety", 2},
                                                 {"Map Optimization", 3},
                                                 {"Preemption", 4}};
  auto priority = [&priorities](const auto& definition)
  {
    auto found = priorities.find(definition.group);
    return found == priorities.end() ? 5 : found->second;
  };
  std::stable_sort(display_order.begin(), display_order.end(),
                   [&priority](const auto& a, const auto& b) { return priority(a) < priority(b); });
  for (const auto& definition : display_order)
  {
    const auto& group = definition.group;
    if (!forms.count(group))
    {
      auto* box = new QGroupBox(QString::fromStdString(group));
      forms[group] = new QFormLayout(box);
      forms[group]->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
      groups->addWidget(box);
    }
    QWidget* field;
    if (definition.default_value.type == rcl_interfaces::msg::ParameterType::PARAMETER_BOOL)
    {
      field = new QCheckBox;
      connect(static_cast<QCheckBox*>(field), &QCheckBox::toggled, this, [this]() { changed(); });
    }
    else if (!definition.choices.empty())
    {
      auto* combo = new QComboBox;
      for (const auto& choice : definition.choices) combo->addItem(QString::fromStdString(choice));
      field = combo;
      connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
              [this]() { changed(); });
    }
    else
    {
      auto* text = new QLineEdit;
      if (definition.default_value.type == rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER)
        text->setValidator(
            new QRegularExpressionValidator(QRegularExpression("-?[0-9]{1,19}"), text));
      if (definition.default_value.type == rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE)
      {
        auto* validator = new QDoubleValidator(text);
        validator->setLocale(QLocale::c());
        text->setValidator(validator);
      }
      field = text;
      connect(text, &QLineEdit::textChanged, this, [this]() { changed(); });
    }
    field->setObjectName(QString::fromStdString("parameter_" + definition.name));
    field->setEnabled(!definition.managed);
    field->setToolTip(QString::fromStdString(definition.description));
    auto* label = new QLabel(QString::fromStdString(definition.name));
    label->setToolTip(field->toolTip());
    forms[group]->addRow(label, field);
    fields_[definition.name] = field;
    labels_[definition.name] = label;
  }
  auto* scroll = new QScrollArea;
  scroll->setWidgetResizable(true);
  scroll->setWidget(contents);
  layout->addWidget(scroll);
  status_ = new QLabel;
  status_->setWordWrap(true);
  layout->addWidget(status_);
  auto* actions = new QHBoxLayout;
  auto* reset = new QPushButton("Reset to Default");
  auto* import = new QPushButton("Import YAML");
  auto* export_button = new QPushButton("Export YAML");
  actions->addWidget(reset);
  actions->addWidget(import);
  actions->addWidget(export_button);
  layout->addLayout(actions);
  auto* buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
  layout->addWidget(buttons);
  connect(reset, &QPushButton::clicked, this, [this]() { populate({}); });
  connect(import, &QPushButton::clicked, this,
          [this]()
          {
            auto path = QFileDialog::getOpenFileName(this, "Import Explorer Parameters", {},
                                                     "YAML (*.yaml *.yml)");
            if (path.isEmpty()) return;
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly))
            {
              status_->setText(file.errorString());
              return;
            }
            importYaml(QString::fromUtf8(file.readAll()));
          });
  connect(export_button, &QPushButton::clicked, this,
          [this]()
          {
            auto yaml = exportYaml();
            if (yaml.isEmpty()) return;
            auto path = QFileDialog::getSaveFileName(this, "Export Explorer Parameters", {},
                                                     "YAML (*.yaml *.yml)");
            if (path.isEmpty()) return;
            QFile file(path);
            if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
            {
              status_->setText(file.errorString());
              return;
            }
            auto bytes = yaml.toUtf8();
            if (file.write(bytes) != bytes.size())
              status_->setText(file.errorString());
            else
              status_->setText("Exported validated parameters");
          });
  connect(buttons, &QDialogButtonBox::accepted, this,
          [this]()
          {
            auto response = resolve();
            if (response)
            {
              accepted_values_ = response->parameters;
              accept();
            }
          });
  connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
  populate(values);
}

bool ParameterEditor::importYaml(const QString& text)
{
  if (text.isEmpty())
  {
    status_->setText("Parameter YAML is empty");
    return false;
  }
  auto response = resolve(text);
  if (!response) return false;
  populate(response->parameters);
  return true;
}

QString ParameterEditor::exportYaml()
{
  auto response = resolve();
  return response ? QString::fromStdString(response->yaml) : QString();
}

void ParameterEditor::populate(const std::vector<rcl_interfaces::msg::Parameter>& values)
{
  std::map<std::string, rcl_interfaces::msg::ParameterValue> supplied;
  for (const auto& value : values) supplied[value.name] = value.value;
  hidden_values_.clear();
  for (const auto& definition : full_catalog_.parameters)
  {
    if (fields_.count(definition.name)) continue;
    rcl_interfaces::msg::Parameter parameter;
    parameter.name = definition.name;
    parameter.value =
        supplied.count(definition.name) ? supplied.at(definition.name) : definition.default_value;
    hidden_values_.push_back(parameter);
  }
  for (const auto& definition : catalog_.parameters)
  {
    const auto value =
        supplied.count(definition.name) ? supplied[definition.name] : definition.default_value;
    auto* field = fields_.at(definition.name);
    if (auto* check = qobject_cast<QCheckBox*>(field))
      check->setChecked(value.bool_value);
    else if (auto* combo = qobject_cast<QComboBox*>(field))
      combo->setCurrentText(QString::fromStdString(value.string_value));
    else
    {
      auto* line = static_cast<QLineEdit*>(field);
      if (value.type == rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER)
        line->setText(QString::number(value.integer_value));
      else if (value.type == rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE)
        line->setText(QString::number(value.double_value, 'g', 17));
      else
        line->setText(QString::fromStdString(value.string_value));
    }
  }
  changed();
}

std::vector<rcl_interfaces::msg::Parameter> ParameterEditor::collect() const
{
  std::vector<rcl_interfaces::msg::Parameter> result;
  for (const auto& definition : catalog_.parameters)
  {
    auto value = definition.default_value;
    auto* field = fields_.at(definition.name);
    if (auto* check = qobject_cast<QCheckBox*>(field))
      value.bool_value = check->isChecked();
    else if (auto* combo = qobject_cast<QComboBox*>(field))
      value.string_value = combo->currentText().toStdString();
    else
    {
      auto text = static_cast<QLineEdit*>(field)->text();
      bool valid = true;
      if (value.type == rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER)
        value.integer_value = text.toLongLong(&valid);
      else if (value.type == rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE)
        value.double_value = QLocale::c().toDouble(text, &valid);
      else
        value.string_value = text.toStdString();
      if (!valid) throw std::runtime_error(definition.name + ": invalid numeric value");
    }
    rcl_interfaces::msg::Parameter parameter;
    parameter.name = definition.name;
    parameter.value = value;
    result.push_back(parameter);
  }
  result.insert(result.end(), hidden_values_.begin(), hidden_values_.end());
  return result;
}

void ParameterEditor::changed()
{
  // Signals may arrive while the form is still being constructed/populated.
  if (fields_.size() != catalog_.parameters.size()) return;
  try
  {
    auto values = collect();
    for (size_t i = 0; i < catalog_.parameters.size(); ++i)
    {
      const auto& definition = catalog_.parameters[i];
      auto* label = labels_.at(definition.name);
      bool modified = values[i].value != definition.default_value;
      label->setText(QString::fromStdString(definition.name) + (modified ? " *" : ""));
      auto font = label->font();
      font.setBold(modified);
      label->setFont(font);
    }
  }
  catch (const std::exception&)
  { /* Validation errors are reported on Save. */
  }
}

std::shared_ptr<ParameterEditor::Service::Response> ParameterEditor::resolve(const QString& yaml)
{
  try
  {
    auto request = std::make_shared<Service::Request>();
    request->yaml = yaml.toStdString();
    request->robot_id = full_catalog_.robot_id;
    if (yaml.isEmpty()) request->overrides = collect();
    if (!client_->wait_for_service(std::chrono::seconds(2)))
      throw std::runtime_error("Coordinator unavailable");
    auto future = client_->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
    {
      client_->remove_pending_request(future);
      throw std::runtime_error("Parameter validation timed out");
    }
    auto response = future.get();
    status_->setText(QString::fromStdString(response->message));
    if (response->accepted) return response;
  }
  catch (const std::exception& error)
  {
    status_->setText(error.what());
  }
  return nullptr;
}
}  // namespace utils
