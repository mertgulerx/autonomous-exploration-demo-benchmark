// Copyright 2026 Mert Güler
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Read-only frontier analysis for the suite Monitor. Every tick re-runs the explorer's own debug
// analysis (frontier_exploration_ros2::debug::analyze_frontier_debug_snapshot, compiled from the
// explorer's unchanged source) on the latest map, costmaps and robot pose, and publishes the result
// as one typed message. The node subscribes and publishes only; nothing it publishes is read by the
// explorer, Nav2 or SLAM.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "exploration_benchmark_insight/msg/frontier_insight.hpp"
#include "exploration_benchmark_insight/msg/frontier_insight_candidate.hpp"
#include "frontier_exploration_ros2/debug/debug_analyzer.hpp"
#include "frontier_exploration_ros2/decision_map.hpp"
#include "frontier_exploration_ros2/frontier_types.hpp"
#include "frontier_exploration_ros2/mrtsp_ordering.hpp"
#include "frontier_exploration_ros2/qos_utils.hpp"

namespace exploration_benchmark_insight
{

namespace
{

namespace fe = frontier_exploration_ros2;
namespace fed = frontier_exploration_ros2::debug;
using InsightMsg = exploration_benchmark_insight::msg::FrontierInsight;
using CandidateMsg = exploration_benchmark_insight::msg::FrontierInsightCandidate;

// The explorer's decision-map dirty detection works on 32-cell chunks (kRawDiffChunkSizeCells in
// decision_map.cpp, not exported); the observer's chunk overlay uses the same value.
constexpr int kChunkSizeCells = 32;
constexpr int kWarnThrottleMs = 5000;
constexpr uint16_t kNoRank = 0;

std::size_t positive_size_parameter(
  const rclcpp::Node & node, const std::string & name,
  int fallback)
{
  // Same clamp as the debug observer, so both analyse identical DP pools for the same parameters.
  const int value = node.get_parameter(name).as_int();
  if (value > 0) {
    return static_cast<std::size_t>(value);
  }
  return static_cast<std::size_t>(std::max(1, fallback));
}

uint16_t saturate_u16(std::size_t value)
{
  return static_cast<uint16_t>(std::min<std::size_t>(value, std::numeric_limits<uint16_t>::max()));
}

uint16_t rank_or_none(const std::optional<std::size_t> & rank)
{
  return rank.has_value() ? saturate_u16(*rank) : kNoRank;
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

const char * transition_name(fe::DecisionMapGeometryTransition transition)
{
  switch (transition) {
    case fe::DecisionMapGeometryTransition::SameGeometry:
      return "same";
    case fe::DecisionMapGeometryTransition::OverlapReuse:
      return "overlap_reuse";
    case fe::DecisionMapGeometryTransition::FullRebuildFallback:
      return "full_rebuild";
  }
  return "full_rebuild";
}

// The observer's score-label order: finite start costs first, by cost, ties by index. Non-finite
// costs keep index order here (the observer leaves their relative order to std::sort).
std::vector<std::size_t> score_order(const fed::FrontierDebugSnapshot & snapshot)
{
  std::vector<std::size_t> order(snapshot.candidates.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  std::sort(
    order.begin(), order.end(), [&snapshot](std::size_t lhs, std::size_t rhs) {
      const double lhs_cost = snapshot.candidates[lhs].mrtsp_start_cost;
      const double rhs_cost = snapshot.candidates[rhs].mrtsp_start_cost;
      const bool lhs_finite = std::isfinite(lhs_cost);
      const bool rhs_finite = std::isfinite(rhs_cost);
      if (lhs_finite != rhs_finite) {
        return lhs_finite;
      }
      if (lhs_finite && lhs_cost != rhs_cost) {
        return lhs_cost < rhs_cost;
      }
      return lhs < rhs;
    });
  return order;
}

// One analysis input set, copied in the timer callback and analysed on the worker thread.
struct AnalysisJob
{
  rclcpp::Time stamp;
  geometry_msgs::msg::Pose pose;
  fe::OccupancyGrid2d map;
  fe::OccupancyGrid2d costmap;
  std::optional<fe::OccupancyGrid2d> local_costmap;
};

}  // namespace

class FrontierInsightNode : public rclcpp::Node
{
public:
  explicit FrontierInsightNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("frontier_insight", options), steady_clock_(RCL_STEADY_TIME)
  {
    declare_parameters();
    read_parameters();
    create_ros_interfaces();
    worker_ = std::thread(&FrontierInsightNode::worker_loop, this);
    RCLCPP_INFO(
      get_logger(),
        "Frontier insight ready: map='%s', costmap='%s', rate=%.2f Hz (ROS time), topic='%s'",
      map_topic_.c_str(), costmap_topic_.c_str(), rate_hz_, insight_topic_.c_str());
  }

  ~FrontierInsightNode() override
  {
    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      stopping_ = true;
    }
    job_ready_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  FrontierInsightNode(const FrontierInsightNode &) = delete;
  FrontierInsightNode & operator=(const FrontierInsightNode &) = delete;

private:
  void declare_parameters()
  {
    // Inputs and analysis parameters carry the debug observer's names and defaults, so one explorer
    // parameter file configures both and they analyse the same candidate set.
    declare_parameter<std::string>("map_topic", "map");
    declare_parameter<std::string>("costmap_topic", "global_costmap/costmap");
    declare_parameter<std::string>("local_costmap_topic", "local_costmap/costmap");
    declare_parameter<std::string>("global_frame", "map");
    declare_parameter<std::string>("robot_base_frame", "base_footprint");
    declare_parameter<std::string>("map_qos_durability", "transient_local");
    declare_parameter<std::string>("map_qos_reliability", "reliable");
    declare_parameter<int>("map_qos_depth", 1);
    declare_parameter<std::string>("costmap_qos_reliability", "reliable");
    declare_parameter<int>("costmap_qos_depth", 10);
    declare_parameter<std::string>("local_costmap_qos_reliability", "inherit");
    declare_parameter<int>("local_costmap_qos_depth", -1);

    declare_parameter<bool>("frontier_map_optimization_enabled", true);
    declare_parameter<double>("sigma_s", 2.0);
    declare_parameter<double>("sigma_r", 30.0);
    declare_parameter<int>("dilation_kernel_radius_cells", 1);
    declare_parameter<double>("sensor_effective_range_m", 1.5);
    declare_parameter<double>("weight_distance_wd", 1.0);
    declare_parameter<double>("weight_gain_ws", 1.0);
    declare_parameter<double>("max_linear_speed_vmax", 0.5);
    declare_parameter<double>("max_angular_speed_wmax", 1.0);
    declare_parameter<std::string>("mrtsp_solver", "dp");
    declare_parameter<int>("dp_solver_candidate_limit", 15);
    declare_parameter<int>("dp_planning_horizon", 10);
    declare_parameter<int>("occ_threshold", fe::OCC_THRESHOLD);
    declare_parameter<int>("min_frontier_size_cells", fe::MIN_FRONTIER_SIZE);
    declare_parameter<double>("frontier_candidate_min_goal_distance_m", 0.0);
    declare_parameter<double>("frontier_selection_min_distance", 0.5);
    declare_parameter<double>("frontier_visit_tolerance", 0.30);

    declare_parameter<double>("insight_rate_hz", 1.0);
    declare_parameter<std::string>("insight_topic", "/benchmark/frontier_insight");
  }

  void read_parameters()
  {
    map_topic_ = get_parameter("map_topic").as_string();
    costmap_topic_ = get_parameter("costmap_topic").as_string();
    local_costmap_topic_ = get_parameter("local_costmap_topic").as_string();
    global_frame_ = get_parameter("global_frame").as_string();
    robot_base_frame_ = get_parameter("robot_base_frame").as_string();

    config_.frontier_map_optimization_enabled =
      get_parameter("frontier_map_optimization_enabled").as_bool();
    config_.sigma_s = get_parameter("sigma_s").as_double();
    config_.sigma_r = get_parameter("sigma_r").as_double();
    config_.dilation_kernel_radius_cells = get_parameter("dilation_kernel_radius_cells").as_int();
    config_.sensor_effective_range_m = get_parameter("sensor_effective_range_m").as_double();
    config_.weight_distance_wd = get_parameter("weight_distance_wd").as_double();
    config_.weight_gain_ws = get_parameter("weight_gain_ws").as_double();
    config_.max_linear_speed_vmax = get_parameter("max_linear_speed_vmax").as_double();
    config_.max_angular_speed_wmax = get_parameter("max_angular_speed_wmax").as_double();
    config_.mrtsp_solver = get_parameter("mrtsp_solver").as_string();
    config_.dp_solver_candidate_limit = positive_size_parameter(*this, "dp_solver_candidate_limit",
        15);
    config_.dp_planning_horizon = positive_size_parameter(*this, "dp_planning_horizon", 10);
    config_.occ_threshold = get_parameter("occ_threshold").as_int();
    config_.min_frontier_size_cells = get_parameter("min_frontier_size_cells").as_int();
    config_.frontier_candidate_min_goal_distance_m =
      get_parameter("frontier_candidate_min_goal_distance_m").as_double();
    config_.frontier_selection_min_distance =
      get_parameter("frontier_selection_min_distance").as_double();
    config_.frontier_visit_tolerance = get_parameter("frontier_visit_tolerance").as_double();
    // Scores and the DP pool are the purpose of this node, so both analyses always run.
    config_.analyze_mrtsp_scores = true;
    config_.analyze_dp_pruning = true;

    rate_hz_ = std::clamp(get_parameter("insight_rate_hz").as_double(), 0.1, 10.0);
    if (!std::isfinite(rate_hz_)) {
      rate_hz_ = 1.0;
    }
    insight_topic_ = get_parameter("insight_topic").as_string();

    qos_profiles_ = fe::resolve_topic_qos_profiles(
      get_parameter("map_qos_durability").as_string(),
      get_parameter("map_qos_reliability").as_string(),
      get_parameter("map_qos_depth").as_int(),
      get_parameter("costmap_qos_reliability").as_string(),
      get_parameter("costmap_qos_depth").as_int(),
      get_parameter("local_costmap_qos_reliability").as_string(),
      get_parameter("local_costmap_qos_depth").as_int());
  }

  void create_ros_interfaces()
  {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Subscriptions only keep the latest grids; analysis never runs in a subscription callback.
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      map_topic_, qos_profiles_.make_map_qos(qos_profiles_.map_durability),
      [this](const nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        map_ = fe::OccupancyGrid2d(msg);
      });
    costmap_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      costmap_topic_, qos_profiles_.make_costmap_qos(),
      [this](const nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        costmap_ = fe::OccupancyGrid2d(msg);
      });
    local_costmap_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      local_costmap_topic_, qos_profiles_.make_local_costmap_qos(),
      [this](const nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        local_costmap_ = fe::OccupancyGrid2d(msg);
      });

    publisher_ = create_publisher<InsightMsg>(insight_topic_, rclcpp::QoS(10).reliable());

    // ROS time (simulation time in the suite): ticks follow simulated progress, so a slow
    // simulation gets the same analysis density per simulated second and a paused clock produces
    // no ticks.
    timer_ = rclcpp::create_timer(
      this, get_clock(),
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 /
        rate_hz_)),
      std::bind(&FrontierInsightNode::timer_callback, this));
  }

  std::optional<geometry_msgs::msg::Pose> current_pose()
  {
    try {
      // Latest transform without waiting: the timer callback never blocks.
      const auto transform = tf_buffer_->lookupTransform(global_frame_, robot_base_frame_,
          tf2::TimePointZero);
      geometry_msgs::msg::Pose pose;
      pose.position.x = transform.transform.translation.x;
      pose.position.y = transform.transform.translation.y;
      pose.position.z = transform.transform.translation.z;
      pose.orientation = transform.transform.rotation;
      return pose;
    } catch (const tf2::TransformException & exc) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_clock_, kWarnThrottleMs, "Could not transform %s -> %s: %s",
        robot_base_frame_.c_str(), global_frame_.c_str(), exc.what());
      return std::nullopt;
    }
  }

  void timer_callback()
  {
    AnalysisJob job;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!map_.has_value() || !costmap_.has_value()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), steady_clock_, kWarnThrottleMs, "Waiting for map and global costmap");
        return;
      }
      job.map = *map_;
      job.costmap = *costmap_;
      job.local_costmap = local_costmap_;
    }
    const auto pose = current_pose();
    if (!pose.has_value()) {
      return;
    }
    job.pose = *pose;
    job.stamp = now();

    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      // Never queue behind a running analysis: a late tick would describe an older state.
      if (busy_ || pending_.has_value()) {
        ++skipped_ticks_;
        return;
      }
      pending_ = std::move(job);
    }
    job_ready_.notify_one();
  }

  void worker_loop()
  {
    while (true) {
      AnalysisJob job;
      {
        std::unique_lock<std::mutex> lock(job_mutex_);
        job_ready_.wait(lock, [this] {return stopping_ || pending_.has_value();});
        if (stopping_) {
          return;
        }
        job = std::move(*pending_);
        pending_.reset();
        busy_ = true;
      }
      try {
        analyse(job);
      } catch (const std::exception & exc) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), steady_clock_, kWarnThrottleMs, "Frontier insight analysis failed: %s",
            exc.what());
      }
      std::lock_guard<std::mutex> lock(job_mutex_);
      busy_ = false;
    }
  }

  void analyse(const AnalysisJob & job)
  {
    const auto started = std::chrono::steady_clock::now();
    // The workspace is the analysis chunk cache of this node; only this thread touches it.
    const fed::FrontierDebugSnapshot snapshot = fed::analyze_frontier_debug_snapshot(
      job.pose, job.map, job.costmap, job.local_costmap, config_, workspace_);

    InsightMsg msg;
    msg.header.stamp = job.stamp;
    msg.header.frame_id = global_frame_;
    msg.map_stamp = job.map.map().header.stamp;
    msg.robot_x = job.pose.position.x;
    msg.robot_y = job.pose.position.y;
    msg.robot_yaw = yaw_from_quaternion(job.pose.orientation);
    msg.mode = snapshot.active_selection_mode;
    msg.raw_frontier_count = static_cast<uint32_t>(
      std::min<std::size_t>(snapshot.raw_frontiers.size(), std::numeric_limits<uint32_t>::max()));

    const std::vector<std::size_t> order = score_order(snapshot);
    std::vector<uint16_t> score_rank(snapshot.candidates.size(), kNoRank);
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
      score_rank[order[rank]] = saturate_u16(rank + 1U);
    }
    std::vector<uint16_t> route_rank(snapshot.candidates.size(), kNoRank);
    for (std::size_t rank = 0; rank < snapshot.active_order.size(); ++rank) {
      if (snapshot.active_order[rank] < route_rank.size()) {
        route_rank[snapshot.active_order[rank]] = saturate_u16(rank + 1U);
      }
    }

    msg.candidates.reserve(snapshot.candidates.size());
    for (const auto & item : snapshot.candidates) {
      CandidateMsg candidate;
      candidate.id = saturate_u16(item.id);
      candidate.x = item.dispatch_point.first;
      candidate.y = item.dispatch_point.second;
      candidate.centroid_x = item.reference_point.first;
      candidate.centroid_y = item.reference_point.second;
      candidate.size_cells = item.candidate.size;
      candidate.gain = item.mrtsp_gain;
      candidate.path_cost = item.mrtsp_initial_path_cost;
      candidate.time_cost = item.mrtsp_motion_time_cost;
      candidate.start_cost = item.mrtsp_start_cost;
      candidate.distance_m = std::hypot(
        item.dispatch_point.first - msg.robot_x, item.dispatch_point.second - msg.robot_y);
      candidate.score_order = score_rank[item.id];
      candidate.pool_rank = rank_or_none(item.dp_prune_rank);
      candidate.route_position = route_rank[item.id];
      candidate.greedy_position = rank_or_none(item.mrtsp_greedy_rank);
      msg.candidates.push_back(candidate);
    }

    msg.route.reserve(snapshot.active_order.size());
    std::vector<fe::FrontierCandidate> route_candidates;
    route_candidates.reserve(snapshot.active_order.size());
    for (const auto index : snapshot.active_order) {
      if (index < snapshot.optimized_frontiers.size()) {
        msg.route.push_back(saturate_u16(index));
        route_candidates.push_back(snapshot.optimized_frontiers[index]);
      }
    }
    msg.route_cost = route_cost(job.pose, route_candidates);

    fill_chunks(snapshot, msg);
    msg.analysis_wall_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      msg.skipped_ticks = skipped_ticks_;
    }
    msg.tick = ++published_ticks_;
    publisher_->publish(msg);
  }

  // M(0, r_1) + sum of M(r_k, r_k+1) with the explorer's cost matrix over the route candidates
  // (row 0 is the robot). Matrix entries depend only on the robot and the two candidates, so a
  // matrix over the route gives the same entries as the solver's pool matrix.
  double route_cost(
    const geometry_msgs::msg::Pose & pose,
    const std::vector<fe::FrontierCandidate> & route)
  {
    if (route.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    fe::RobotState robot;
    robot.position = {pose.position.x, pose.position.y};
    robot.yaw = yaw_from_quaternion(pose.orientation);
    fe::CostWeights weights;
    weights.distance_wd = config_.weight_distance_wd;
    weights.gain_ws = config_.weight_gain_ws;
    const fe::MrtspCostMatrix matrix = fe::build_cost_matrix(
      route, robot, weights, config_.sensor_effective_range_m, config_.max_linear_speed_vmax,
      config_.max_angular_speed_wmax);
    double total = matrix.at(0, 1);
    for (std::size_t row = 1; row < route.size(); ++row) {
      total += matrix.at(row, row + 1);
    }
    return total;
  }

  void fill_chunks(const fed::FrontierDebugSnapshot & snapshot, InsightMsg & msg)
  {
    const auto & info = snapshot.decision_map_msg.info;
    msg.chunk_size_cells = static_cast<uint16_t>(kChunkSizeCells);
    msg.chunk_resolution = info.resolution;
    msg.chunk_origin_x = info.origin.position.x;
    msg.chunk_origin_y = info.origin.position.y;
    msg.chunk_map_width = info.width;
    msg.chunk_map_height = info.height;
    msg.chunk_columns = saturate_u16(static_cast<std::size_t>(std::max(0, workspace_.chunk_cols)));
    msg.chunk_rows = saturate_u16(static_cast<std::size_t>(std::max(0, workspace_.chunk_rows)));
    const std::size_t chunk = static_cast<std::size_t>(kChunkSizeCells);
    const std::size_t expected_columns = (static_cast<std::size_t>(info.width) + chunk - 1U) /
      chunk;
    const std::size_t expected_rows = (static_cast<std::size_t>(info.height) + chunk - 1U) / chunk;
    const std::size_t count = static_cast<std::size_t>(msg.chunk_columns) * msg.chunk_rows;
    if (
      static_cast<std::size_t>(msg.chunk_columns) != expected_columns ||
      static_cast<std::size_t>(msg.chunk_rows) != expected_rows ||
      snapshot.decision_map_chunks.size() != count)
    {
      // The chunk size is copied from the explorer; a mismatch means the explorer changed it.
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_clock_, kWarnThrottleMs,
        "Decision-map chunk grid %ux%u does not match %d-cell chunks over %ux%u cells; "
        "chunks omitted",
        msg.chunk_columns, msg.chunk_rows, kChunkSizeCells, info.width, info.height);
      msg.chunk_columns = 0;
      msg.chunk_rows = 0;
    } else {
      msg.chunk_states.reserve(count);
      // The analyzer lists chunks row by row from the map origin, which is the message's order.
      for (const auto & chunk : snapshot.decision_map_chunks) {
        switch (chunk.state) {
          case fed::DecisionMapChunkDebugState::CacheHit:
            msg.chunk_states.push_back(InsightMsg::CHUNK_HIT);
            ++msg.chunk_hits;
            break;
          case fed::DecisionMapChunkDebugState::DirtyRebuild:
            msg.chunk_states.push_back(InsightMsg::CHUNK_REBUILT);
            ++msg.chunk_rebuilt;
            break;
          case fed::DecisionMapChunkDebugState::GeometryReset:
            msg.chunk_states.push_back(InsightMsg::CHUNK_RESET);
            ++msg.chunk_resets;
            break;
        }
      }
    }
    msg.geometry_transition = snapshot.decision_map_geometry_changed ?
      transition_name(snapshot.decision_map_geometry_transition) : "same";
    msg.output_reused = snapshot.decision_map_output_reused;
    msg.output_changed = snapshot.decision_map_output_changed;
  }

  fed::DebugAnalyzerConfig config_;
  fe::TopicQosProfiles qos_profiles_;
  std::string map_topic_;
  std::string costmap_topic_;
  std::string local_costmap_topic_;
  std::string global_frame_;
  std::string robot_base_frame_;
  std::string insight_topic_;
  double rate_hz_{1.0};

  // Warnings throttle on wall time so they keep appearing while the simulation clock stands still.
  rclcpp::Clock steady_clock_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr local_costmap_sub_;
  rclcpp::Publisher<InsightMsg>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex data_mutex_;
  std::optional<fe::OccupancyGrid2d> map_;
  std::optional<fe::OccupancyGrid2d> costmap_;
  std::optional<fe::OccupancyGrid2d> local_costmap_;

  std::mutex job_mutex_;
  std::condition_variable job_ready_;
  std::optional<AnalysisJob> pending_;
  bool busy_{false};
  bool stopping_{false};
  uint64_t skipped_ticks_{0};

  // Worker-thread state.
  fe::DecisionMapWorkspace workspace_;
  uint64_t published_ticks_{0};
  std::thread worker_;
};

}  // namespace exploration_benchmark_insight

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  {
    auto node = std::make_shared<exploration_benchmark_insight::FrontierInsightNode>();
    rclcpp::spin(node);
  }
  rclcpp::shutdown();
  return 0;
}
