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

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "exploration_benchmark_telemetry/msg/navigation_event.hpp"
#include "nav2_bt_navigator/navigators/navigate_to_pose.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace exploration_benchmark_telemetry
{
class Navigator : public nav2_bt_navigator::NavigateToPoseNavigator
{
public:
  using Event = exploration_benchmark_telemetry::msg::NavigationEvent;
  ~Navigator() override { stop(); }

  bool configure(rclcpp_lifecycle::LifecycleNode::WeakPtr parent,
    std::shared_ptr<nav2_util::OdomSmoother> smoother) override
  {
    const bool result = NavigateToPoseNavigator::configure(parent, smoother);
    if (!result) { return result; }
    auto node = parent.lock();
    publisher_ = node->create_publisher<Event>("/benchmark/navigation_events",
      rclcpp::QoS(256).reliable());
    publisher_->on_activate();
    closing_ = false;
    thread_ = std::thread([this]() { publish(); });
    return result;
  }

  bool cleanup() override
  {
    const bool result = NavigateToPoseNavigator::cleanup();
    stop();
    publisher_.reset();
    return result;
  }

protected:
  bool goalReceived(ActionT::Goal::ConstSharedPtr goal) override
  {
    const auto sequence = ++next_goal_;
    enqueue("received", sequence, &goal->pose);
    const bool accepted = NavigateToPoseNavigator::goalReceived(goal);
    if (accepted) {
      current_goal_ = sequence;
      initialized(sequence);
    } else {
      enqueue("rejected", sequence, &goal->pose);
    }
    return accepted;
  }

  void onLoop() override
  {
    NavigateToPoseNavigator::onLoop();
  }

  void onPreempt(ActionT::Goal::ConstSharedPtr goal) override
  {
    const auto sequence = ++next_goal_;
    enqueue("preempt_requested", sequence, &goal->pose);
    geometry_msgs::msg::PoseStamped previous;
    auto blackboard = bt_action_server_->getBlackboard();
    const auto before = blackboard->getStamped(goal_blackboard_id_, previous);
    NavigateToPoseNavigator::onPreempt(goal);
    geometry_msgs::msg::PoseStamped updated;
    const auto after = blackboard->getStamped(goal_blackboard_id_, updated);
    if (after && (!before || after.value().seq != before.value().seq)) {
      enqueue("preempted", current_goal_);
      current_goal_ = sequence;
      enqueue("initialized", sequence, &updated);
    } else {
      enqueue("preempt_rejected", sequence, &goal->pose);
    }
  }

  void goalCompleted(ActionT::Result::SharedPtr result,
    const nav2_behavior_tree::BtStatus status) override
  {
    NavigateToPoseNavigator::goalCompleted(result, status);
    const char * terminal = status == nav2_behavior_tree::BtStatus::SUCCEEDED ? "succeeded" :
      status == nav2_behavior_tree::BtStatus::CANCELED ? "canceled" : "failed";
    enqueue("terminal", current_goal_, nullptr, terminal, result->error_code);
  }

private:
  void initialized(uint64_t sequence)
  {
    geometry_msgs::msg::PoseStamped pose;
    const bool valid = bt_action_server_->getBlackboard()->get(goal_blackboard_id_, pose);
    enqueue(valid ? "initialized" : "initialization_unavailable", sequence,
      valid ? &pose : nullptr);
  }

  void enqueue(const char * name, uint64_t goal,
    const geometry_msgs::msg::PoseStamped * pose = nullptr,
    const char * terminal = "", uint16_t code = 0)
  {
    Event event;
    event.header.stamp = clock_->now();
    event.event_sequence = ++next_event_;
    event.goal_sequence = goal;
    event.event = name;
    event.terminal_status = terminal;
    event.native_result_code = code;
    event.goal_valid = pose != nullptr;
    if (pose) { event.goal = *pose; }
    std::lock_guard<std::mutex> guard(mutex_);
    if (queue_.size() == 256) { queue_.pop_front(); ++lost_; }
    event.lost_events = lost_;
    queue_.push_back(std::move(event));
    condition_.notify_one();
  }

  void publish()
  {
    for (;;) {
      Event event;
      {
        std::unique_lock<std::mutex> guard(mutex_);
        condition_.wait(guard, [this]() { return closing_ || !queue_.empty(); });
        if (queue_.empty() && closing_) { return; }
        event = std::move(queue_.front());
        queue_.pop_front();
        event.lost_events = lost_;
      }
      try {
        publisher_->publish(event);
      } catch (const std::exception & error) {
        std::lock_guard<std::mutex> guard(mutex_);
        ++lost_;
        RCLCPP_ERROR(logger_, "Navigation telemetry publication failed: %s", error.what());
      }
    }
  }

  void stop()
  {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      closing_ = true;
    }
    condition_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
  }

  rclcpp_lifecycle::LifecyclePublisher<Event>::SharedPtr publisher_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Event> queue_;
  std::thread thread_;
  bool closing_ = false;
  uint64_t next_goal_ = 0, current_goal_ = 0, next_event_ = 0, lost_ = 0;
};
}  // namespace exploration_benchmark_telemetry

PLUGINLIB_EXPORT_CLASS(exploration_benchmark_telemetry::Navigator, nav2_core::NavigatorBase)
