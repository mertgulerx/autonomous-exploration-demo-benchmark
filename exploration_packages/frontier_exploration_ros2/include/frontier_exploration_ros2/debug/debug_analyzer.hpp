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

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include "frontier_exploration_ros2/decision_map.hpp"
#include "frontier_exploration_ros2/frontier_search.hpp"
#include "frontier_exploration_ros2/frontier_types.hpp"

namespace frontier_exploration_ros2::debug
{

// Configuration mirrored by the passive debug observer. The analyzer uses these
// values to reproduce frontier extraction and scoring decisions without owning
// navigation state, sending goals, or changing the running explorer behavior.
struct DebugAnalyzerConfig
{
  // Frontier extraction and map optimization settings must match the explorer
  // node so raw and optimized overlays explain the same candidate set.
  bool frontier_map_optimization_enabled{true};
  double sigma_s{2.0};
  double sigma_r{30.0};
  int dilation_kernel_radius_cells{1};
  int occ_threshold{OCC_THRESHOLD};
  int min_frontier_size_cells{MIN_FRONTIER_SIZE};
  double frontier_candidate_min_goal_distance_m{0.0};
  double frontier_selection_min_distance{0.5};
  double frontier_visit_tolerance{0.30};

  // MRTSP and DP parameters are only used for analysis. They build the same
  // cost matrix, pruning pool, and bounded-horizon sequence shown in RViz.
  std::string mrtsp_solver{"dp"};
  std::size_t dp_solver_candidate_limit{15};
  std::size_t dp_planning_horizon{10};
  double sensor_effective_range_m{1.5};
  double weight_distance_wd{1.0};
  double weight_gain_ws{1.0};
  double max_linear_speed_vmax{0.5};
  double max_angular_speed_wmax{1.0};
  bool analyze_mrtsp_scores{true};
  bool analyze_dp_pruning{true};
};

// Per-candidate explanation data used by the RViz marker layer.
struct FrontierDebugCandidate
{
  std::size_t id{};
  FrontierCandidate candidate;

  std::pair<double, double> reference_point{0.0, 0.0};
  std::pair<double, double> dispatch_point{0.0, 0.0};

  // MRTSP score breakdown mirrors the start-row cost used by the matrix. Keeping
  // the parts separate makes weight, gain, path, and time effects visible.
  double mrtsp_gain{0.0};
  double mrtsp_initial_path_cost{0.0};
  double mrtsp_motion_time_cost{0.0};
  double mrtsp_start_cost{0.0};
  std::optional<std::size_t> mrtsp_greedy_rank;

  // DP annotations preserve the distinction between pruning rank and route rank:
  // a candidate can be in the DP pool without appearing in the selected sequence.
  bool dp_pruned{false};
  std::optional<std::size_t> dp_prune_rank;
  std::optional<std::size_t> dp_order_rank;
  bool active_order_selected{false};
};

enum class DecisionMapChunkDebugState : uint8_t
{
  CacheHit = 0,
  DirtyRebuild = 1,
  GeometryReset = 2,
};

struct DecisionMapChunkDebugCell
{
  int chunk_x{0};
  int chunk_y{0};
  DecisionMapChunkDebugState state{DecisionMapChunkDebugState::CacheHit};
};

// Full analysis result for one observer tick. Marker publishers consume this
// snapshot directly, keeping visualization formatting separate from scoring.
struct FrontierDebugSnapshot
{
  std::vector<FrontierCandidate> raw_frontiers;
  std::vector<FrontierCandidate> optimized_frontiers;
  std::vector<FrontierDebugCandidate> candidates;
  std::vector<std::size_t> mrtsp_greedy_order;
  std::vector<std::size_t> dp_pruned_indices;
  std::vector<std::size_t> dp_order;
  std::vector<std::size_t> active_order;
  std::vector<DecisionMapChunkDebugCell> decision_map_chunks;
  std::size_t decision_map_total_chunks{0};
  std::size_t decision_map_dirty_chunks{0};
  bool decision_map_geometry_changed{false};
  DecisionMapGeometryTransition decision_map_geometry_transition{
    DecisionMapGeometryTransition::FullRebuildFallback};
  bool decision_map_config_changed{false};
  bool decision_map_output_reused{false};
  bool decision_map_output_changed{false};
  nav_msgs::msg::OccupancyGrid decision_map_msg;
  std::string active_selection_mode;
};

// Builds a read-only debug snapshot from map, costmap, pose, and parameters.
// The function intentionally returns data only; it does not publish, dispatch,
// cancel, suppress, or mutate exploration state.
FrontierDebugSnapshot analyze_frontier_debug_snapshot(
  const geometry_msgs::msg::Pose & current_pose,
  const OccupancyGrid2d & map,
  const OccupancyGrid2d & costmap,
  const std::optional<OccupancyGrid2d> & local_costmap,
  const DebugAnalyzerConfig & config,
  DecisionMapWorkspace & decision_map_workspace);

}  // namespace frontier_exploration_ros2::debug
