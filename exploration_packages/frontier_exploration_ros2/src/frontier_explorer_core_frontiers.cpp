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

#include "frontier_exploration_ros2/frontier_explorer_core.hpp"

#include "frontier_explorer_core_detail.hpp"
#include "frontier_exploration_ros2/mrtsp_solver.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace frontier_exploration_ros2
{

FrontierSequence FrontierExplorerCore::build_mrtsp_frontier_sequence(
  const FrontierSequence & frontiers,
  const geometry_msgs::msg::Pose & current_pose) const
{
  if (frontiers.empty()) {
    return {};
  }

  const double pose_quantum = std::max(params.frontier_visit_tolerance, 0.1);
  const int pose_x_bucket = detail::quantize_bucket(current_pose.position.x, pose_quantum);
  const int pose_y_bucket = detail::quantize_bucket(current_pose.position.y, pose_quantum);
  const int yaw_bucket = detail::quantize_bucket(
    detail::yaw_from_quaternion(current_pose.orientation),
    detail::kPi / 12.0);
  const FrontierSignature signature = frontier_signature(frontiers);

  // Solver mode and DP bounds participate in the cache key because the same frontier
  // geometry can yield a different order when the route horizon or candidate pool changes.
  if (mrtsp_order_cache.has_value() &&
    mrtsp_order_cache->frontier_signature == signature &&
    mrtsp_order_cache->pose_x_bucket == pose_x_bucket &&
    mrtsp_order_cache->pose_y_bucket == pose_y_bucket &&
    mrtsp_order_cache->yaw_bucket == yaw_bucket &&
    mrtsp_order_cache->sensor_effective_range_m == params.sensor_effective_range_m &&
    mrtsp_order_cache->weight_distance_wd == params.weight_distance_wd &&
    mrtsp_order_cache->weight_gain_ws == params.weight_gain_ws &&
    mrtsp_order_cache->max_linear_speed_vmax == params.max_linear_speed_vmax &&
    mrtsp_order_cache->max_angular_speed_wmax == params.max_angular_speed_wmax &&
    mrtsp_order_cache->mrtsp_solver == params.mrtsp_solver &&
    mrtsp_order_cache->dp_solver_candidate_limit == params.dp_solver_candidate_limit &&
    mrtsp_order_cache->dp_planning_horizon == params.dp_planning_horizon)
  {
    const_cast<FrontierExplorerCore *>(this)->mrtsp_order_cache_hits += 1;
    if (debug_outputs_enabled()) {
      callbacks.log_debug(
        "mrtsp_order_cache: hit, frontiers=" + std::to_string(frontiers.size()));
    }
    return mrtsp_order_cache->frontier_sequence;
  }

  const std::vector<FrontierCandidate> & candidates = frontiers;

  RobotState robot_state;
  robot_state.position = {current_pose.position.x, current_pose.position.y};
  robot_state.yaw = detail::yaw_from_quaternion(current_pose.orientation);

  CostWeights weights;
  weights.distance_wd = params.weight_distance_wd;
  weights.gain_ws = params.weight_gain_ws;

  std::vector<std::size_t> order;
  if (params.mrtsp_solver == "dp") {
    // DP mode keeps the matrix compact by scoring all candidates with the MRTSP start
    // cost, retaining the best pool, and evaluating only bounded-horizon sequences.
    const auto pruned = prune_mrtsp_candidates(
      candidates,
      robot_state,
      weights,
      params.sensor_effective_range_m,
      params.max_linear_speed_vmax,
      params.max_angular_speed_wmax,
      MrtspSolverConfig{
        params.dp_solver_candidate_limit,
        params.dp_planning_horizon});

    std::vector<FrontierCandidate> pruned_candidates;
    pruned_candidates.reserve(pruned.size());
    // build_cost_matrix() owns the canonical pairwise cost logic, so the core extracts
    // a compact candidate vector instead of asking the solver to duplicate matrix rules.
    for (const auto & item : pruned) {
      pruned_candidates.push_back(item.candidate);
    }

    const MrtspCostMatrix cost_matrix = build_cost_matrix(
      pruned_candidates,
      robot_state,
      weights,
      params.sensor_effective_range_m,
      params.max_linear_speed_vmax,
      params.max_angular_speed_wmax);
    std::vector<std::size_t> pruned_order = solve_bounded_horizon_mrtsp_order(
      cost_matrix,
      params.dp_planning_horizon);

    // If the bounded solver cannot form a finite route, the pruned matrix can still
    // provide a valid single-step ordering through the standard greedy traversal.
    if (pruned_order.empty()) {
      pruned_order = greedy_mrtsp_order(cost_matrix);
    }

    order.reserve(pruned_order.size());
    // Convert pruned-vector indices back to the candidate vector used by the core.
    for (const std::size_t pruned_index : pruned_order) {
      if (pruned_index < pruned.size()) {
        order.push_back(pruned[pruned_index].original_index);
      }
    }
  } else {
    // Greedy mode intentionally keeps the full candidate set to preserve the simple
    // MRTSP traversal behavior selected by the user-facing solver parameter.
    if (params.mrtsp_solver != "greedy") {
      callbacks.log_warn(
        "Unknown mrtsp_solver='" + params.mrtsp_solver + "'; falling back to greedy");
    }
    const MrtspCostMatrix cost_matrix = build_cost_matrix(
      candidates,
      robot_state,
      weights,
      params.sensor_effective_range_m,
      params.max_linear_speed_vmax,
      params.max_angular_speed_wmax);
    order = greedy_mrtsp_order(cost_matrix);
  }

  FrontierSequence ordered_frontiers;
  ordered_frontiers.reserve(order.size());
  for (const std::size_t index : order) {
    if (index < frontiers.size()) {
      ordered_frontiers.push_back(frontiers[index]);
    }
  }

  auto & mutable_self = const_cast<FrontierExplorerCore &>(*this);
  // Cache stores the already mapped FrontierSequence rather than solver indices, which
  // keeps later dispatch code independent from candidate/pruned-vector bookkeeping.
  mutable_self.mrtsp_order_cache = MrtspOrderCacheEntry{
    signature,
    pose_x_bucket,
    pose_y_bucket,
    yaw_bucket,
    params.sensor_effective_range_m,
    params.weight_distance_wd,
    params.weight_gain_ws,
    params.max_linear_speed_vmax,
    params.max_angular_speed_wmax,
    params.mrtsp_solver,
    params.dp_solver_candidate_limit,
    params.dp_planning_horizon,
    ordered_frontiers,
  };
  mutable_self.mrtsp_order_cache_misses += 1;
  if (debug_outputs_enabled()) {
    callbacks.log_debug(
      "mrtsp_order_cache: miss, frontiers=" + std::to_string(frontiers.size()) +
      ", ordered=" + std::to_string(ordered_frontiers.size()) +
      ", solver=" + params.mrtsp_solver);
  }
  return ordered_frontiers;
}

std::pair<double, double> FrontierExplorerCore::frontier_position(const FrontierLike & frontier) const
{
  return frontier_exploration_ros2::frontier_position(frontier);
}

std::pair<double, double> FrontierExplorerCore::frontier_reference_point(const FrontierLike & frontier) const
{
  return frontier_exploration_ros2::frontier_reference_point(frontier);
}

int FrontierExplorerCore::frontier_size(const FrontierLike & frontier) const
{
  return frontier_exploration_ros2::frontier_size(frontier);
}

std::string FrontierExplorerCore::describe_frontier(const FrontierLike & frontier) const
{
  return frontier_exploration_ros2::describe_frontier(frontier);
}

FrontierSignature FrontierExplorerCore::frontier_signature(const FrontierSequence & frontiers) const
{
  return frontier_exploration_ros2::frontier_signature(frontiers, params.frontier_visit_tolerance);
}

bool FrontierExplorerCore::frontier_snapshot_matches(
  const std::optional<FrontierSnapshot> & snapshot,
  const std::pair<int, int> & robot_map_cell,
  double min_goal_distance) const
{
  // Snapshot reuse follows the actual frontier-search inputs. Raw map generation may advance
  // without changing decision_map output, so decision_map_generation is the map-side key.
  return (
    snapshot.has_value() &&
    snapshot->decision_map_generation == decision_map_generation &&
    snapshot->costmap_generation == costmap_generation &&
    snapshot->local_costmap_generation == local_costmap_generation &&
    snapshot->robot_map_cell == robot_map_cell &&
    snapshot->min_goal_distance == min_goal_distance);
}

void FrontierExplorerCore::throttled_debug(const std::string & message)
{
  const int64_t now_ns = callbacks.now_ns();
  // Shared throttle avoids flooding debug logs in rapid map/costmap callback bursts.
  const int64_t throttle_ns = static_cast<int64_t>(frontier_stats_log_throttle_seconds * 1e9);
  if (!last_frontier_stats_log_time_ns.has_value() ||
    now_ns - *last_frontier_stats_log_time_ns >= throttle_ns)
  {
    // Move throttle window only when we actually emit a message.
    last_frontier_stats_log_time_ns = now_ns;
    callbacks.log_debug(message);
  }
}

void FrontierExplorerCore::log_frontier_snapshot_stats(
  const FrontierSequence & frontiers,
  double duration_ms,
  bool cache_hit)
{
  std::ostringstream oss;
  // Compact log line keeps p50/p95-style frontier timing inspection easy in runtime logs.
  oss << "frontier_snapshot: "
      << (cache_hit ? "hit" : "miss")
      << ", frontiers=" << frontiers.size()
      << ", duration_ms=" << std::fixed << std::setprecision(2) << duration_ms
      << ", hits=" << frontier_snapshot_cache_hits
      << ", misses=" << frontier_snapshot_cache_misses;
  throttled_debug(oss.str());
}

FrontierSnapshot FrontierExplorerCore::get_frontier_snapshot(
  const geometry_msgs::msg::Pose & current_pose,
  double min_goal_distance)
{
  if (!decision_map.has_value() || decision_map_dirty) {
    if (!map.has_value()) {
      throw std::logic_error("Decision map is not initialized");
    }
    refresh_decision_map();
  }
  // Snapshot cache is keyed by decision/costmap generations + robot cell + min_goal_distance.
  const auto robot_map_cell = decision_map->worldToMap(current_pose.position.x, current_pose.position.y);
  if (frontier_snapshot_matches(frontier_snapshot, robot_map_cell, min_goal_distance)) {
    // Cache hit: avoid repeating expensive frontier extraction.
    frontier_snapshot_cache_hits += 1;
    log_frontier_snapshot_stats(frontier_snapshot->frontiers, 0.0, true);
    return *frontier_snapshot;
  }

  const auto started_at = std::chrono::steady_clock::now();
  const auto search_result = callbacks.frontier_search(
    current_pose,
    *decision_map,
    *costmap,
    local_costmap,
    min_goal_distance,
    true);
  const auto finished_at = std::chrono::steady_clock::now();
  const double duration_ms = std::chrono::duration<double, std::milli>(finished_at - started_at).count();

  FrontierSnapshot snapshot;
  // Convert low-level search output into policy-facing representation + cache key metadata.
  snapshot.frontiers = to_frontier_sequence(search_result.frontiers);
  snapshot.signature = frontier_signature(snapshot.frontiers);
  snapshot.map_generation = map_generation;
  snapshot.decision_map_generation = decision_map_generation;
  snapshot.costmap_generation = costmap_generation;
  snapshot.local_costmap_generation = local_costmap_generation;
  snapshot.robot_map_cell = search_result.robot_map_cell;
  snapshot.min_goal_distance = min_goal_distance;

  frontier_snapshot = snapshot;
  frontier_snapshot_cache_misses += 1;
  if (debug_outputs_enabled() && map.has_value()) {
    std::size_t raw_frontier_count = snapshot.frontiers.size();
    if (frontier_map_optimization_enabled()) {
      const FrontierSearchOptions options = frontier_search_options();
      if (
        raw_frontier_debug_cache.has_value() &&
        raw_frontier_debug_cache->map_generation == map_generation &&
        raw_frontier_debug_cache->costmap_generation == costmap_generation &&
        raw_frontier_debug_cache->local_costmap_generation == local_costmap_generation &&
        raw_frontier_debug_cache->robot_map_cell == search_result.robot_map_cell &&
        raw_frontier_debug_cache->min_goal_distance == min_goal_distance &&
        raw_frontier_debug_cache->search_options.occ_threshold == options.occ_threshold &&
        raw_frontier_debug_cache->search_options.min_frontier_size_cells == options.min_frontier_size_cells &&
        raw_frontier_debug_cache->search_options.candidate_min_goal_distance_m == options.candidate_min_goal_distance_m)
      {
        raw_frontier_count = raw_frontier_debug_cache->frontier_count;
      } else {
        const auto raw_search_result = get_frontier(
          current_pose,
          *map,
          *costmap,
          local_costmap,
          min_goal_distance,
          false,
          options);
        raw_frontier_count = raw_search_result.frontiers.size();
        raw_frontier_debug_cache = RawFrontierDebugCacheEntry{
          map_generation,
          costmap_generation,
          local_costmap_generation,
          raw_search_result.robot_map_cell,
          min_goal_distance,
          options,
          raw_frontier_count,
        };
      }
    }
    callbacks.log_debug(
      "frontier_counts: raw=" + std::to_string(raw_frontier_count) +
      ", decision=" + std::to_string(snapshot.frontiers.size()));
  }
  log_frontier_snapshot_stats(snapshot.frontiers, duration_ms, false);
  return snapshot;
}

void FrontierExplorerCore::start_post_goal_settle()
{
  awaiting_map_refresh = true;
  post_goal_settle_active = true;
  post_goal_settle_started_at_ns = callbacks.now_ns();
}

void FrontierExplorerCore::wait_for_next_map_refresh()
{
   awaiting_map_refresh = true;

  if (params.post_goal_settle_enabled) {
    post_goal_settle_active = true;
    post_goal_settle_started_at_ns = callbacks.now_ns();
  } else {
    post_goal_settle_active = false;
    post_goal_settle_started_at_ns.reset();
  }
  start_post_goal_settle();
}

void FrontierExplorerCore::clear_post_goal_wait_state()
{
  awaiting_map_refresh = false;
  post_goal_settle_active = false;
  post_goal_settle_started_at_ns.reset();
}

bool FrontierExplorerCore::post_goal_settle_ready() const
{
  if (!awaiting_map_refresh) {
    return true;
  }

  if (!post_goal_settle_active) {
    return !params.post_goal_settle_enabled;
  }

  if (!post_goal_settle_started_at_ns.has_value()) {
    return false;
  }

  const double elapsed = static_cast<double>(callbacks.now_ns() - *post_goal_settle_started_at_ns) / 1e9;
  if (elapsed < params.post_goal_min_settle) {
    return false;
  }
  return true;
}

FrontierSelectionResult FrontierExplorerCore::select_frontier(
  const FrontierSequence & frontiers,
  const geometry_msgs::msg::Pose & current_pose) const
{
  const FrontierSequence ordered_frontiers = build_mrtsp_frontier_sequence(frontiers, current_pose);
  if (ordered_frontiers.empty()) {
    return {std::nullopt, ""};
  }
  return {ordered_frontiers.front(), "mrtsp"};
}

void FrontierExplorerCore::record_start_pose(const geometry_msgs::msg::Pose & current_pose)
{
  if (start_pose.has_value()) {
    // Start pose is recorded once per node lifetime and survives session stop/start cycles.
    return;
  }

  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = params.global_frame;
  pose.pose = current_pose;
  start_pose = pose;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2)
      << "Recorded exploration start pose: ("
      << current_pose.position.x << ", "
      << current_pose.position.y << ")";
  callbacks.log_info(oss.str());
}

bool FrontierExplorerCore::are_frontiers_equivalent(
  const std::optional<FrontierLike> & first_frontier,
  const std::optional<FrontierLike> & second_frontier) const
{
  return frontier_exploration_ros2::are_frontiers_equivalent(
    first_frontier,
    second_frontier,
    params.frontier_visit_tolerance);
}

bool FrontierExplorerCore::frontier_exists_in_set(
  const std::optional<FrontierLike> & frontier,
  const FrontierSequence & frontiers) const
{
  if (!frontier.has_value()) {
    return false;
  }

  for (const auto & candidate : frontiers) {
    // Tolerance-based check reuses shared frontier equivalence policy.
    if (are_frontiers_equivalent(frontier, candidate)) {
      return true;
    }
  }

  return false;
}

std::optional<std::string> FrontierExplorerCore::frontier_cost_status(
  const std::optional<FrontierLike> & frontier) const
{
  if (!frontier.has_value()) {
    return std::nullopt;
  }

  const auto goal_point = frontier_position(*frontier);

  const auto local_cost = world_point_cost(local_costmap, goal_point);
  if (local_cost.has_value() && *local_cost >= params.occ_threshold) {
    // Local map blocks have priority because they are most immediate for controller safety.
    return std::string(
      "Current frontier target is blocked in local costmap (cost=") +
      std::to_string(*local_cost) + ")";
  }

  const auto global_cost = world_point_cost(costmap, goal_point);
  if (global_cost.has_value() && *global_cost >= params.occ_threshold) {
    return std::string(
      "Current frontier target is blocked in global costmap (cost=") +
      std::to_string(*global_cost) + ")";
  }

  return std::nullopt;
}

geometry_msgs::msg::PoseStamped FrontierExplorerCore::build_goal_pose(
  const FrontierLike & target_frontier,
  const geometry_msgs::msg::Pose & current_pose) const
{
  const auto [target_x, target_y] = frontier_position(target_frontier);
  geometry_msgs::msg::PoseStamped goal_pose;
  goal_pose.header.frame_id = params.global_frame;
  goal_pose.pose.position.x = target_x;
  goal_pose.pose.position.y = target_y;
  // Default heading follows the travel vector toward the selected target frontier.
  // This avoids forcing a stale "current heading" orientation at goal completion.
  goal_pose.pose.orientation = current_pose.orientation;
  const double to_target_dx = target_x - current_pose.position.x;
  const double to_target_dy = target_y - current_pose.position.y;
  if (std::hypot(to_target_dx, to_target_dy) > 0.05) {
    goal_pose.pose.orientation = detail::quaternion_from_yaw(std::atan2(to_target_dy, to_target_dx));
  }
  return goal_pose;
}

geometry_msgs::msg::PoseStamped FrontierExplorerCore::build_dispatch_goal_pose(
  const FrontierLike & target_frontier,
  const geometry_msgs::msg::Pose & current_pose,
  bool bypass_min_distance_dispatch) const
{
  const auto goal_pose_for_point =
    [this, &current_pose](const std::pair<double, double> & target_point) {
      geometry_msgs::msg::PoseStamped goal_pose;
      goal_pose.header.frame_id = params.global_frame;
      goal_pose.pose.position.x = target_point.first;
      goal_pose.pose.position.y = target_point.second;
      goal_pose.pose.orientation = current_pose.orientation;
      const double to_target_dx = target_point.first - current_pose.position.x;
      const double to_target_dy = target_point.second - current_pose.position.y;
      if (std::hypot(to_target_dx, to_target_dy) > 0.05) {
        goal_pose.pose.orientation = detail::quaternion_from_yaw(std::atan2(to_target_dy, to_target_dx));
      }
      return goal_pose;
    };

  const auto fallback_goal_pose = build_goal_pose(target_frontier, current_pose);
  if (bypass_min_distance_dispatch ||
    params.frontier_selection_min_distance <= 0.0 ||
    !map.has_value())
  {
    return fallback_goal_pose;
  }

  const auto target_point = frontier_position(target_frontier);
  const double distance_to_robot = std::hypot(
    target_point.first - current_pose.position.x,
    target_point.second - current_pose.position.y);
  if (distance_to_robot >= params.frontier_selection_min_distance) {
    return fallback_goal_pose;
  }

  int target_map_x = 0;
  int target_map_y = 0;
  if (!map->worldToMapNoThrow(target_point.first, target_point.second, target_map_x, target_map_y)) {
    return fallback_goal_pose;
  }

  const auto is_dispatch_cell_eligible = [this, &current_pose, &target_point](int map_x, int map_y) {
      if (map->getCost(map_x, map_y) != static_cast<int>(OccupancyGrid2d::CostValues::FreeSpace)) {
        return false;
      }

      const auto world_point = map->mapToWorld(map_x, map_y);
      const double robot_distance = std::hypot(
        world_point.first - current_pose.position.x,
        world_point.second - current_pose.position.y);
      if (robot_distance < params.frontier_selection_min_distance) {
        return false;
      }

      const double target_distance = std::hypot(
        world_point.first - target_point.first,
        world_point.second - target_point.second);
      if (target_distance < params.frontier_selection_min_distance) {
        return false;
      }

      const auto local_cost = world_point_cost(local_costmap, world_point);
      if (local_cost.has_value() && *local_cost >= params.occ_threshold) {
        return false;
      }

      const auto global_cost = world_point_cost(costmap, world_point);
      if (global_cost.has_value() && *global_cost >= params.occ_threshold) {
        return false;
      }

      return true;
    };

  const int max_radius = std::max(map->getSizeX(), map->getSizeY());
  for (int radius = 0; radius < max_radius; ++radius) {
    std::optional<std::pair<double, double>> best_world_point;
    double best_distance_sq = std::numeric_limits<double>::infinity();

    const auto consider_cell = [&](int map_x, int map_y) {
        if (map_x < 0 || map_y < 0 || map_x >= map->getSizeX() || map_y >= map->getSizeY()) {
          return;
        }
        if (!is_dispatch_cell_eligible(map_x, map_y)) {
          return;
        }

        const auto world_point = map->mapToWorld(map_x, map_y);
        const double target_distance_sq = squared_distance(world_point, target_point);
        if (target_distance_sq < best_distance_sq) {
          best_distance_sq = target_distance_sq;
          best_world_point = world_point;
        }
      };

    if (radius == 0) {
      consider_cell(target_map_x, target_map_y);
    } else {
      for (int map_x = target_map_x - radius; map_x <= target_map_x + radius; ++map_x) {
        consider_cell(map_x, target_map_y - radius);
        consider_cell(map_x, target_map_y + radius);
      }
      for (int map_y = target_map_y - radius + 1; map_y <= target_map_y + radius - 1; ++map_y) {
        consider_cell(target_map_x - radius, map_y);
        consider_cell(target_map_x + radius, map_y);
      }
    }

    if (best_world_point.has_value()) {
      return goal_pose_for_point(*best_world_point);
    }
  }

  return fallback_goal_pose;
}

std::vector<geometry_msgs::msg::PoseStamped> FrontierExplorerCore::build_goal_pose_sequence(
  const FrontierSequence & target_frontiers,
  const geometry_msgs::msg::Pose & current_pose) const
{
  std::vector<geometry_msgs::msg::PoseStamped> goal_sequence;
  // Reserve once to keep goal sequence creation allocation-free for steady-state single-frontier mode.
  goal_sequence.reserve(target_frontiers.size());
  for (std::size_t i = 0; i < target_frontiers.size(); ++i) {
    goal_sequence.push_back(build_goal_pose(target_frontiers[i], current_pose));
  }
  return goal_sequence;
}

FrontierSequence FrontierExplorerCore::select_frontier_sequence(
  const FrontierSequence & frontiers,
  const geometry_msgs::msg::Pose & current_pose,
  const std::optional<FrontierLike> & initial_frontier) const
{
  (void)initial_frontier;
  return build_mrtsp_frontier_sequence(frontiers, current_pose);
}

bool FrontierExplorerCore::are_frontier_sequences_equivalent(
  const FrontierSequence & first_frontier_sequence,
  const FrontierSequence & second_frontier_sequence) const
{
  return frontier_exploration_ros2::are_frontier_sequences_equivalent(
    first_frontier_sequence,
    second_frontier_sequence,
    params.frontier_visit_tolerance);
}

}  // namespace frontier_exploration_ros2
