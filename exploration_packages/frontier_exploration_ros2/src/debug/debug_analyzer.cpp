/*
Copyright 2026 Mert Guler

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

#include "frontier_exploration_ros2/debug/debug_analyzer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "frontier_exploration_ros2/decision_map.hpp"
#include "frontier_exploration_ros2/mrtsp_ordering.hpp"
#include "frontier_exploration_ros2/mrtsp_solver.hpp"

namespace frontier_exploration_ros2::debug
{

namespace
{

FrontierSearchOptions make_search_options(const DebugAnalyzerConfig & config)
{
  FrontierSearchOptions options;
  options.occ_threshold = config.occ_threshold;
  options.min_frontier_size_cells = config.min_frontier_size_cells;
  options.candidate_min_goal_distance_m = config.frontier_candidate_min_goal_distance_m;
  return options;
}

DecisionMapConfig make_decision_map_config(const DebugAnalyzerConfig & config)
{
  DecisionMapConfig map_config;
  map_config.optimization_enabled = config.frontier_map_optimization_enabled;
  map_config.occ_threshold = config.occ_threshold;
  map_config.sigma_s = config.sigma_s;
  map_config.sigma_r = config.sigma_r;
  map_config.dilation_kernel_radius_cells = config.dilation_kernel_radius_cells;
  return map_config;
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & orientation)
{
  const double siny_cosp =
    2.0 * ((orientation.w * orientation.z) + (orientation.x * orientation.y));
  const double cosy_cosp =
    1.0 - (2.0 * ((orientation.y * orientation.y) + (orientation.z * orientation.z)));
  return std::atan2(siny_cosp, cosy_cosp);
}

void annotate_mrtsp_candidates(
  const geometry_msgs::msg::Pose & current_pose,
  const DebugAnalyzerConfig & config,
  FrontierDebugSnapshot & snapshot)
{
  if (snapshot.optimized_frontiers.empty()) {
    return;
  }

  RobotState robot_state;
  robot_state.position = {current_pose.position.x, current_pose.position.y};
  robot_state.yaw = yaw_from_quaternion(current_pose.orientation);

  CostWeights weights;
  weights.distance_wd = config.weight_distance_wd;
  weights.gain_ws = config.weight_gain_ws;

  for (auto & debug_candidate : snapshot.candidates) {
    // Store the same start-row components used by compute_mrtsp_start_cost().
    // This makes the score visible without introducing a second scoring model.
    const auto & candidate = debug_candidate.candidate;
    debug_candidate.mrtsp_gain = frontier_information_gain(candidate);
    debug_candidate.mrtsp_initial_path_cost = initial_frontier_path_cost(
      robot_state.position,
      candidate,
      candidate.start_world_point,
      config.sensor_effective_range_m);
    debug_candidate.mrtsp_motion_time_cost = lower_bound_time_cost(
      robot_state,
      candidate.center_point,
      candidate.robot_center_distance_m,
      config.max_linear_speed_vmax,
      config.max_angular_speed_wmax);
    debug_candidate.mrtsp_start_cost = compute_mrtsp_start_cost(
      candidate,
      robot_state,
      weights,
      config.sensor_effective_range_m,
      config.max_linear_speed_vmax,
      config.max_angular_speed_wmax);
  }

  const MrtspCostMatrix full_matrix = build_cost_matrix(
    snapshot.optimized_frontiers,
    robot_state,
    weights,
    config.sensor_effective_range_m,
    config.max_linear_speed_vmax,
    config.max_angular_speed_wmax);
  snapshot.mrtsp_greedy_order = greedy_mrtsp_order(full_matrix);
  // Greedy rank is always computed for comparison, even when DP is the active
  // solver. This lets the overlay show how far DP reordered the candidate pool.
  for (std::size_t rank = 0; rank < snapshot.mrtsp_greedy_order.size(); ++rank) {
    const std::size_t candidate_index = snapshot.mrtsp_greedy_order[rank];
    if (candidate_index < snapshot.candidates.size()) {
      snapshot.candidates[candidate_index].mrtsp_greedy_rank = rank + 1U;
    }
  }

  if (config.mrtsp_solver == "dp") {
    // DP uses a pruned candidate pool. The analyzer keeps both index spaces:
    // pruned indices for the solver and original indices for RViz labels.
    const auto pruned = prune_mrtsp_candidates(
      snapshot.optimized_frontiers,
      robot_state,
      weights,
      config.sensor_effective_range_m,
      config.max_linear_speed_vmax,
      config.max_angular_speed_wmax,
      MrtspSolverConfig{
        config.dp_solver_candidate_limit,
        config.dp_planning_horizon});

    std::vector<FrontierCandidate> pruned_candidates;
    pruned_candidates.reserve(pruned.size());
    for (std::size_t rank = 0; rank < pruned.size(); ++rank) {
      const auto & item = pruned[rank];
      snapshot.dp_pruned_indices.push_back(item.original_index);
      pruned_candidates.push_back(item.candidate);
      if (item.original_index < snapshot.candidates.size()) {
        snapshot.candidates[item.original_index].dp_pruned = true;
        snapshot.candidates[item.original_index].dp_prune_rank = rank + 1U;
      }
    }

    const MrtspCostMatrix pruned_matrix = build_cost_matrix(
      pruned_candidates,
      robot_state,
      weights,
      config.sensor_effective_range_m,
      config.max_linear_speed_vmax,
      config.max_angular_speed_wmax);
    std::vector<std::size_t> pruned_order = solve_bounded_horizon_mrtsp_order(
      pruned_matrix,
      config.dp_planning_horizon);
    if (pruned_order.empty()) {
      // An empty DP result means no valid bounded sequence was found. The debug
      // snapshot mirrors the explorer's fallback path instead of hiding it.
      pruned_order = greedy_mrtsp_order(pruned_matrix);
      snapshot.active_selection_mode = "mrtsp/dp-greedy-fallback";
    } else {
      snapshot.active_selection_mode = "mrtsp/dp";
    }

    for (std::size_t rank = 0; rank < pruned_order.size(); ++rank) {
      // Convert solver-relative indices back to the optimized frontier list so
      // every overlay can reference the same candidate ids.
      const std::size_t pruned_index = pruned_order[rank];
      if (pruned_index >= pruned.size()) {
        continue;
      }
      const std::size_t original_index = pruned[pruned_index].original_index;
      snapshot.dp_order.push_back(original_index);
      if (original_index < snapshot.candidates.size()) {
        snapshot.candidates[original_index].dp_order_rank = rank + 1U;
      }
    }
  } else {
    snapshot.active_selection_mode = config.mrtsp_solver == "greedy" ?
      "mrtsp/greedy" : "mrtsp/greedy-fallback";
  }

  snapshot.active_order = config.mrtsp_solver == "dp" ?
    snapshot.dp_order : snapshot.mrtsp_greedy_order;
  // active_order_selected is a display hint used by marker builders; it does not
  // influence scoring or any future analysis tick.
  for (const auto candidate_index : snapshot.active_order) {
    if (candidate_index < snapshot.candidates.size()) {
      snapshot.candidates[candidate_index].active_order_selected = true;
    }
  }
}

void annotate_decision_map_cache(
  const DecisionMapWorkspace & decision_map_workspace,
  const DecisionMapBuildStatus & decision_map_status,
  FrontierDebugSnapshot & snapshot)
{
  snapshot.decision_map_total_chunks = decision_map_status.total_chunks;
  snapshot.decision_map_dirty_chunks = decision_map_status.dirty_chunks;
  snapshot.decision_map_geometry_changed = decision_map_status.geometry_changed;
  snapshot.decision_map_geometry_transition = decision_map_status.geometry_transition;
  snapshot.decision_map_config_changed = decision_map_status.config_changed;
  snapshot.decision_map_output_reused = decision_map_status.reused_existing_output;
  snapshot.decision_map_output_changed = decision_map_status.output_changed;
  snapshot.decision_map_chunks.reserve(decision_map_status.total_chunks);

  for (int chunk_y = 0; chunk_y < decision_map_workspace.chunk_rows; ++chunk_y) {
    for (int chunk_x = 0; chunk_x < decision_map_workspace.chunk_cols; ++chunk_x) {
      const std::size_t chunk_index =
        static_cast<std::size_t>(chunk_y) *
        static_cast<std::size_t>(decision_map_workspace.chunk_cols) +
        static_cast<std::size_t>(chunk_x);

      DecisionMapChunkDebugCell cell;
      cell.chunk_x = chunk_x;
      cell.chunk_y = chunk_y;
      if (
        decision_map_status.geometry_changed &&
        decision_map_status.geometry_transition == DecisionMapGeometryTransition::FullRebuildFallback)
      {
        cell.state = DecisionMapChunkDebugState::GeometryReset;
      } else if (
        chunk_index < decision_map_workspace.dirty_chunk_flags.size() &&
        decision_map_workspace.dirty_chunk_flags[chunk_index] != 0U)
      {
        cell.state = DecisionMapChunkDebugState::DirtyRebuild;
      } else {
        cell.state = DecisionMapChunkDebugState::CacheHit;
      }
      snapshot.decision_map_chunks.push_back(cell);
    }
  }
}

}  // namespace

FrontierDebugSnapshot analyze_frontier_debug_snapshot(
  const geometry_msgs::msg::Pose & current_pose,
  const OccupancyGrid2d & map,
  const OccupancyGrid2d & costmap,
  const std::optional<OccupancyGrid2d> & local_costmap,
  const DebugAnalyzerConfig & config,
  DecisionMapWorkspace & decision_map_workspace)
{
  FrontierDebugSnapshot snapshot;

  // Build both raw and decision-map frontier sets in the same tick. Seeing both
  // side by side makes map optimization effects visible without changing the
  // explorer node or publishing any navigation command.
  const FrontierSearchOptions search_options = make_search_options(config);
  const DecisionMapBuildStatus decision_map_status = build_decision_map(
    map,
    make_decision_map_config(config),
    decision_map_workspace);
  snapshot.decision_map_msg = *decision_map_workspace.optimized_map_msg;
  annotate_decision_map_cache(decision_map_workspace, decision_map_status, snapshot);
  const OccupancyGrid2d decision_map(snapshot.decision_map_msg);

  // Raw frontiers are extracted from the incoming occupancy map. They are shown
  // as context and are not used for MRTSP score labels.
  const auto raw_search_result = get_frontier(
    current_pose,
    map,
    costmap,
    local_costmap,
    config.frontier_candidate_min_goal_distance_m,
    true,
    search_options);
  snapshot.raw_frontiers = raw_search_result.frontiers;

  // Optimized frontiers are the candidate set used by the debug score analysis.
  // The decision map can be identical to the raw map when optimization is off.
  const auto optimized_search_result = get_frontier(
    current_pose,
    decision_map,
    costmap,
    local_costmap,
    config.frontier_candidate_min_goal_distance_m,
    true,
    search_options);
  snapshot.optimized_frontiers = optimized_search_result.frontiers;

  snapshot.candidates.reserve(snapshot.optimized_frontiers.size());
  for (std::size_t index = 0; index < snapshot.optimized_frontiers.size(); ++index) {
    // Candidate ids are stable within one snapshot and match marker labels.
    // They intentionally reset on each tick because the frontier set can change.
    FrontierDebugCandidate debug_candidate;
    debug_candidate.id = index;
    debug_candidate.candidate = snapshot.optimized_frontiers[index];
    debug_candidate.reference_point = debug_candidate.candidate.centroid;
    debug_candidate.dispatch_point =
      debug_candidate.candidate.goal_point.value_or(debug_candidate.candidate.center_point);
    snapshot.candidates.push_back(std::move(debug_candidate));
  }

  if (config.analyze_mrtsp_scores || config.analyze_dp_pruning) {
    // MRTSP analysis is also needed for DP pruning because pruning uses the same
    // start-row score as the MRTSP cost matrix.
    annotate_mrtsp_candidates(current_pose, config, snapshot);
  }

  return snapshot;
}

}  // namespace frontier_exploration_ros2::debug
