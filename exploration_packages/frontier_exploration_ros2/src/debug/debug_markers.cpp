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

#include "frontier_exploration_ros2/debug/debug_markers.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace frontier_exploration_ros2::debug
{

namespace
{

std_msgs::msg::ColorRGBA rgba(float r, float g, float b, float a = 1.0F)
{
  std_msgs::msg::ColorRGBA color;
  color.r = r;
  color.g = g;
  color.b = b;
  color.a = a;
  return color;
}

geometry_msgs::msg::Point point(double x, double y, double z)
{
  geometry_msgs::msg::Point marker_point;
  marker_point.x = x;
  marker_point.y = y;
  marker_point.z = z;
  return marker_point;
}

geometry_msgs::msg::Point candidate_point(
  const FrontierCandidate & candidate,
  double z_offset)
{
  // Markers are placed on the dispatchable goal point when available. Falling
  // back to the center point keeps MRTSP-only candidates visible before a
  // separate navigation goal point is built.
  const auto target = candidate.goal_point.value_or(candidate.center_point);
  return point(target.first, target.second, z_offset);
}

geometry_msgs::msg::Point offset_point(
  geometry_msgs::msg::Point marker_point,
  double dx,
  double dy,
  double dz = 0.0)
{
  marker_point.x += dx;
  marker_point.y += dy;
  marker_point.z += dz;
  return marker_point;
}

visualization_msgs::msg::Marker clear_marker(const DebugMarkerConfig & config)
{
  // Each debug topic owns its marker namespace. DELETEALL prevents stale labels
  // and edges from remaining in RViz when frontier counts shrink between ticks.
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = config.frame_id;
  marker.action = visualization_msgs::msg::Marker::DELETEALL;
  return marker;
}

visualization_msgs::msg::Marker point_marker(
  const DebugMarkerConfig & config,
  const std::string & ns,
  int id,
  const std_msgs::msg::ColorRGBA & color,
  double scale)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = config.frame_id;
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::POINTS;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.color = color;
  return marker;
}

void append_rectangle_outline(
  visualization_msgs::msg::Marker & marker,
  double x_min,
  double y_min,
  double x_max,
  double y_max,
  double z)
{
  const auto bottom_left = point(x_min, y_min, z);
  const auto bottom_right = point(x_max, y_min, z);
  const auto top_right = point(x_max, y_max, z);
  const auto top_left = point(x_min, y_max, z);

  marker.points.push_back(bottom_left);
  marker.points.push_back(bottom_right);
  marker.points.push_back(bottom_right);
  marker.points.push_back(top_right);
  marker.points.push_back(top_right);
  marker.points.push_back(top_left);
  marker.points.push_back(top_left);
  marker.points.push_back(bottom_left);
}

visualization_msgs::msg::Marker sphere_marker(
  const DebugMarkerConfig & config,
  const std::string & ns,
  int id,
  const geometry_msgs::msg::Point & marker_point,
  const std_msgs::msg::ColorRGBA & color,
  double scale)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = config.frame_id;
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position = marker_point;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;
  marker.color = color;
  return marker;
}

visualization_msgs::msg::Marker text_marker(
  const DebugMarkerConfig & config,
  const std::string & ns,
  int id,
  const geometry_msgs::msg::Point & marker_point,
  const std_msgs::msg::ColorRGBA & color,
  const std::string & text)
{
  // TEXT_VIEW_FACING is used for all score labels so the numeric breakdown stays
  // readable while the RViz camera moves around the map.
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = config.frame_id;
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position = marker_point;
  marker.pose.position.z += config.text_scale * 1.5;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = config.text_scale;
  marker.color = color;
  marker.text = text;
  return marker;
}

visualization_msgs::msg::Marker line_strip_marker(
  const DebugMarkerConfig & config,
  const std::string & ns,
  int id,
  const std_msgs::msg::ColorRGBA & color,
  double line_width)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = config.frame_id;
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = line_width;
  marker.pose.orientation.w = 1.0;
  marker.color = color;
  return marker;
}

visualization_msgs::msg::Marker line_list_marker(
  const DebugMarkerConfig & config,
  const std::string & ns,
  int id,
  const std_msgs::msg::ColorRGBA & color,
  double line_width)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = config.frame_id;
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = line_width;
  marker.pose.orientation.w = 1.0;
  marker.color = color;
  return marker;
}

std::string fixed(double value, int precision = 2)
{
  if (!std::isfinite(value)) {
    return "inf";
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

std::vector<std::size_t> mrtsp_score_order(const FrontierDebugSnapshot & snapshot)
{
  // Score labels are ranked from the same MRTSP start-row cost used for pruning.
  // Finite scores are shown before invalid ones, and index ordering keeps ties
  // deterministic for repeated snapshots.
  std::vector<std::size_t> order;
  order.reserve(snapshot.candidates.size());
  for (std::size_t index = 0; index < snapshot.candidates.size(); ++index) {
    order.push_back(index);
  }
  std::sort(
    order.begin(),
    order.end(),
    [&snapshot](std::size_t lhs, std::size_t rhs) {
      const auto lhs_score = snapshot.candidates[lhs].mrtsp_start_cost;
      const auto rhs_score = snapshot.candidates[rhs].mrtsp_start_cost;
      const bool lhs_finite = std::isfinite(lhs_score);
      const bool rhs_finite = std::isfinite(rhs_score);
      if (lhs_finite != rhs_finite) {
        return lhs_finite;
      }
      if (lhs_score != rhs_score) {
        return lhs_score < rhs_score;
      }
      return lhs < rhs;
    });
  return order;
}

std_msgs::msg::ColorRGBA mrtsp_rank_color(std::size_t rank, std::size_t total)
{
  // MRTSP candidates use a green-to-warm gradient so low-cost candidates stand
  // out without needing every label to be enabled.
  if (total <= 1U) {
    return rgba(0.15F, 0.9F, 0.25F, 0.95F);
  }
  const float t = static_cast<float>(rank) / static_cast<float>(total - 1U);
  return rgba(0.15F + (0.85F * t), 0.9F - (0.55F * t), 0.25F, 0.9F);
}

std::string geometry_transition_label(DecisionMapGeometryTransition transition)
{
  switch (transition) {
    case DecisionMapGeometryTransition::SameGeometry:
      return "same";
    case DecisionMapGeometryTransition::OverlapReuse:
      return "overlap_reuse";
    case DecisionMapGeometryTransition::FullRebuildFallback:
      return "full_rebuild";
  }

  return "unknown";
}

}  // namespace

visualization_msgs::msg::MarkerArray make_raw_frontier_markers(
  const FrontierDebugSnapshot & snapshot,
  const DebugMarkerConfig & config)
{
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(clear_marker(config));

  // Raw frontiers are drawn as a single POINTS marker for low RViz overhead.
  // They show the frontier detector input before decision-map optimization.
  auto raw_points = point_marker(
    config,
    "debug_raw_frontiers",
    0,
    rgba(1.0F, 0.48F, 0.05F, 0.9F),
    config.point_scale);
  for (const auto & frontier : snapshot.raw_frontiers) {
    raw_points.points.push_back(candidate_point(frontier, config.z_offset));
  }
  markers.markers.push_back(raw_points);
  return markers;
}

visualization_msgs::msg::MarkerArray make_optimized_frontier_markers(
  const FrontierDebugSnapshot & snapshot,
  const DebugMarkerConfig & config)
{
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(clear_marker(config));

  // Optimized candidates are the frontiers actually used by the debug scoring
  // layers. The active first target is drawn separately as a white sphere.
  auto optimized_points = point_marker(
    config,
    "debug_optimized_frontiers",
    0,
    rgba(0.15F, 0.9F, 0.25F, 0.95F),
    config.point_scale);
  for (const auto & candidate : snapshot.candidates) {
    optimized_points.points.push_back(candidate_point(candidate.candidate, config.z_offset));
  }
  markers.markers.push_back(optimized_points);

  if (!snapshot.active_order.empty()) {
    const std::size_t selected_index = snapshot.active_order.front();
    if (selected_index < snapshot.candidates.size()) {
      markers.markers.push_back(
        sphere_marker(
          config,
          "debug_selected_frontier",
          1,
          candidate_point(snapshot.candidates[selected_index].candidate, config.z_offset + 0.02),
          rgba(1.0F, 1.0F, 1.0F, 1.0F),
          config.selected_scale));
    }
  }
  return markers;
}

visualization_msgs::msg::MarkerArray make_mrtsp_score_markers(
  const FrontierDebugSnapshot & snapshot,
  const DebugMarkerConfig & config)
{
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(clear_marker(config));

  // Rank colors are computed for every optimized candidate. Text labels remain
  // limited to the top-N scores so dense maps stay usable in RViz.
  const auto score_order = mrtsp_score_order(snapshot);
  std::vector<std::size_t> rank_by_index(snapshot.candidates.size(), snapshot.candidates.size());
  for (std::size_t rank = 0; rank < score_order.size(); ++rank) {
    rank_by_index[score_order[rank]] = rank;
  }

  for (std::size_t index = 0; index < snapshot.candidates.size(); ++index) {
    const auto & candidate = snapshot.candidates[index];
    markers.markers.push_back(
      sphere_marker(
        config,
        "debug_mrtsp_candidates",
        static_cast<int>(index),
        candidate_point(candidate.candidate, config.z_offset),
        mrtsp_rank_color(rank_by_index[index], std::max<std::size_t>(1U, snapshot.candidates.size())),
        candidate.active_order_selected ? config.selected_scale : config.point_scale));
  }

  if (!config.labels_enabled) {
    return markers;
  }

  const std::size_t label_count = std::min(config.label_top_n, score_order.size());
  for (std::size_t rank = 0; rank < label_count; ++rank) {
    const std::size_t candidate_index = score_order[rank];
    if (candidate_index >= snapshot.candidates.size()) {
      continue;
    }
    const auto & candidate = snapshot.candidates[candidate_index];
    const auto base_point = candidate_point(candidate.candidate, config.z_offset);
    const double label_offset = std::max(config.selected_scale, config.point_scale) +
      (config.text_scale * 3.0);

    // The three-part label layout keeps the marker visible while still showing
    // the MRTSP cost breakdown that explains the rank.
    std::ostringstream left_label;
    left_label << "#" << candidate.id
               << "\ncost=" << fixed(candidate.mrtsp_start_cost, 3)
               << "\ngain=" << fixed(candidate.mrtsp_gain, 0);
    markers.markers.push_back(
      text_marker(
        config,
        "debug_mrtsp_labels",
        static_cast<int>(rank * 3U),
        offset_point(base_point, 0.0, label_offset),
        rgba(0.0F, 0.0F, 0.0F, 1.0F),
        left_label.str()));

    std::ostringstream rank_label;
    rank_label << "rank=" << (rank + 1U);
    markers.markers.push_back(
      text_marker(
        config,
        "debug_mrtsp_labels",
        static_cast<int>((rank * 3U) + 1U),
        base_point,
        rgba(0.0F, 0.0F, 0.0F, 1.0F),
        rank_label.str() + "\n \n "));

    std::ostringstream right_label;
    right_label << "path=" << fixed(candidate.mrtsp_initial_path_cost)
                << "\ntime=" << fixed(candidate.mrtsp_motion_time_cost);
    markers.markers.push_back(
      text_marker(
        config,
        "debug_mrtsp_labels",
        static_cast<int>((rank * 3U) + 2U),
        offset_point(base_point, 0.0, -label_offset),
        rgba(0.0F, 0.0F, 0.0F, 1.0F),
        right_label.str()));
  }
  return markers;
}

visualization_msgs::msg::MarkerArray make_mrtsp_order_markers(
  const FrontierDebugSnapshot & snapshot,
  const geometry_msgs::msg::Pose & current_pose,
  const DebugMarkerConfig & config)
{
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(clear_marker(config));

  // Faint start edges show the low-cost candidate neighborhood considered by
  // MRTSP without drawing the full matrix, which would overwhelm RViz.
  auto start_edges = line_list_marker(
    config,
    "debug_mrtsp_start_edges",
    0,
    rgba(0.35F, 0.75F, 1.0F, 0.25F),
    config.line_width * 0.5);
  const geometry_msgs::msg::Point robot_point = point(
    current_pose.position.x,
    current_pose.position.y,
    config.z_offset);
  const auto score_order = mrtsp_score_order(snapshot);
  const std::size_t edge_count = std::min(config.edge_top_n, score_order.size());
  for (std::size_t rank = 0; rank < edge_count; ++rank) {
    const std::size_t candidate_index = score_order[rank];
    if (candidate_index >= snapshot.candidates.size()) {
      continue;
    }
    start_edges.points.push_back(robot_point);
    start_edges.points.push_back(candidate_point(snapshot.candidates[candidate_index].candidate, config.z_offset));
  }
  markers.markers.push_back(start_edges);

  auto selected_order = line_strip_marker(
    config,
    "debug_mrtsp_active_order",
    1,
    rgba(0.0F, 0.95F, 1.0F, 1.0F),
    config.line_width);
  // The active order starts at the robot pose and then follows the analyzed
  // greedy or DP sequence. Only the first target is dispatched by the explorer.
  selected_order.points.push_back(robot_point);
  for (const auto candidate_index : snapshot.active_order) {
    if (candidate_index < snapshot.candidates.size()) {
      selected_order.points.push_back(candidate_point(snapshot.candidates[candidate_index].candidate, config.z_offset));
    }
  }
  if (selected_order.points.size() > 1U) {
    markers.markers.push_back(selected_order);
  }

  if (!config.labels_enabled || snapshot.active_order.empty()) {
    return markers;
  }

  const std::size_t label_count = std::min(config.label_top_n, snapshot.active_order.size());
  for (std::size_t rank = 0; rank < label_count; ++rank) {
    const std::size_t candidate_index = snapshot.active_order[rank];
    if (candidate_index >= snapshot.candidates.size()) {
      continue;
    }
    std::ostringstream label;
    label << "order=" << (rank + 1U);
    auto order_label = text_marker(
      config,
      "debug_mrtsp_order_labels",
      static_cast<int>(rank),
      candidate_point(snapshot.candidates[candidate_index].candidate, config.z_offset + 0.08),
      rgba(0.0F, 0.02F, 0.18F, 1.0F),
      label.str());
    order_label.scale.z = config.text_scale * 1.35;
    markers.markers.push_back(order_label);
  }
  return markers;
}

visualization_msgs::msg::MarkerArray make_dp_pruning_markers(
  const FrontierDebugSnapshot & snapshot,
  const DebugMarkerConfig & config)
{
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(clear_marker(config));

  // Orange points are the candidate pool passed into bounded-horizon DP. Grey
  // points are valid optimized frontiers that were not selected by pruning.
  auto pruned_points = point_marker(
    config,
    "debug_dp_pruned",
    0,
    rgba(1.0F, 0.48F, 0.05F, 0.9F),
    config.point_scale * 1.45);
  auto rejected_points = point_marker(
    config,
    "debug_dp_rejected",
    1,
    rgba(0.55F, 0.55F, 0.55F, 0.35F),
    config.point_scale);

  for (const auto & candidate : snapshot.candidates) {
    if (candidate.dp_pruned) {
      pruned_points.points.push_back(candidate_point(candidate.candidate, config.z_offset));
    } else {
      rejected_points.points.push_back(candidate_point(candidate.candidate, config.z_offset));
    }
  }
  markers.markers.push_back(rejected_points);
  markers.markers.push_back(pruned_points);

  if (!config.labels_enabled) {
    return markers;
  }

  std::vector<std::size_t> label_indices;
  label_indices.reserve(snapshot.dp_pruned_indices.size());
  // Label only pruned candidates because they are the inputs that can affect the
  // DP sequence. The order follows prune rank, not map insertion order.
  for (const auto candidate_index : snapshot.dp_pruned_indices) {
    if (candidate_index < snapshot.candidates.size()) {
      label_indices.push_back(candidate_index);
    }
  }
  const std::size_t label_count = std::min(config.label_top_n, label_indices.size());
  for (std::size_t index = 0; index < label_count; ++index) {
    const auto & candidate = snapshot.candidates[label_indices[index]];
    const auto base_point = candidate_point(candidate.candidate, config.z_offset);
    const double label_offset = std::max(config.selected_scale, config.point_scale) +
      (config.text_scale * 3.0);

    // DP labels stay on one side of the frontier marker. This keeps the pruning
    // overlay compact while exposing prune rank, route rank, and pruning score.
    std::ostringstream label;
    if (candidate.dp_prune_rank.has_value()) {
      label << "prune=" << *candidate.dp_prune_rank;
    } else {
      label << "prune=-";
    }
    if (candidate.dp_order_rank.has_value()) {
      label << "\ndp_order=" << *candidate.dp_order_rank;
    } else {
      label << "\ndp_order=-";
    }
    label << "\ncost=" << fixed(candidate.mrtsp_start_cost, 3);

    auto label_text = text_marker(
      config,
      "debug_dp_labels",
      static_cast<int>(index),
      offset_point(base_point, 0.0, label_offset),
      rgba(0.64F, 0.24F, 0.0F, 1.0F),
      label.str());
    label_text.scale.z = config.text_scale * 1.25;
    markers.markers.push_back(label_text);
  }
  return markers;
}

visualization_msgs::msg::MarkerArray make_decision_map_chunk_cache_markers(
  const FrontierDebugSnapshot & snapshot,
  const DebugMarkerConfig & config)
{
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(clear_marker(config));

  if (
    snapshot.decision_map_msg.info.width == 0U ||
    snapshot.decision_map_msg.info.height == 0U ||
    snapshot.decision_map_chunks.empty())
  {
    return markers;
  }

  const double resolution = snapshot.decision_map_msg.info.resolution;
  const double origin_x = snapshot.decision_map_msg.info.origin.position.x;
  const double origin_y = snapshot.decision_map_msg.info.origin.position.y;
  const int width = static_cast<int>(snapshot.decision_map_msg.info.width);
  const int height = static_cast<int>(snapshot.decision_map_msg.info.height);
  constexpr int kChunkSizeCells = 32;

  auto hit_borders = line_list_marker(
    config,
    "debug_chunk_cache_hit",
    0,
    rgba(0.12F, 0.82F, 0.25F, 0.9F),
    config.line_width * 0.55);
  auto dirty_borders = line_list_marker(
    config,
    "debug_chunk_cache_dirty",
    1,
    rgba(0.95F, 0.12F, 0.12F, 0.95F),
    config.line_width * 0.75);
  auto reset_borders = line_list_marker(
    config,
    "debug_chunk_cache_reset",
    2,
    rgba(1.0F, 0.85F, 0.15F, 0.95F),
    config.line_width * 0.75);

  std::size_t hit_count = 0;
  std::size_t dirty_count = 0;
  std::size_t reset_count = 0;
  for (const auto & chunk : snapshot.decision_map_chunks) {
    const int cell_x_min = chunk.chunk_x * kChunkSizeCells;
    const int cell_y_min = chunk.chunk_y * kChunkSizeCells;
    const int cell_x_max = std::min(width, cell_x_min + kChunkSizeCells);
    const int cell_y_max = std::min(height, cell_y_min + kChunkSizeCells);

    const double x_min = origin_x + (static_cast<double>(cell_x_min) * resolution);
    const double y_min = origin_y + (static_cast<double>(cell_y_min) * resolution);
    const double x_max = origin_x + (static_cast<double>(cell_x_max) * resolution);
    const double y_max = origin_y + (static_cast<double>(cell_y_max) * resolution);

    switch (chunk.state) {
      case DecisionMapChunkDebugState::CacheHit:
        append_rectangle_outline(hit_borders, x_min, y_min, x_max, y_max, config.z_offset + 0.02);
        hit_count += 1U;
        break;
      case DecisionMapChunkDebugState::DirtyRebuild:
        append_rectangle_outline(dirty_borders, x_min, y_min, x_max, y_max, config.z_offset + 0.03);
        dirty_count += 1U;
        break;
      case DecisionMapChunkDebugState::GeometryReset:
        append_rectangle_outline(reset_borders, x_min, y_min, x_max, y_max, config.z_offset + 0.04);
        reset_count += 1U;
        break;
    }
  }

  if (!hit_borders.points.empty()) {
    markers.markers.push_back(hit_borders);
  }
  if (!dirty_borders.points.empty()) {
    markers.markers.push_back(dirty_borders);
  }
  if (!reset_borders.points.empty()) {
    markers.markers.push_back(reset_borders);
  }

  std::ostringstream summary;
  summary << "chunk cache"
          << "\nhit=" << hit_count
          << " dirty=" << dirty_count
          << " reset=" << reset_count
          << "\ngeometry=" << geometry_transition_label(snapshot.decision_map_geometry_transition)
          << "\noutput=" << (snapshot.decision_map_output_reused ? "hit" : "miss")
          << " changed=" << (snapshot.decision_map_output_changed ? "yes" : "no");
  if (
    snapshot.decision_map_geometry_changed &&
    snapshot.decision_map_geometry_transition == DecisionMapGeometryTransition::FullRebuildFallback)
  {
    summary << "\ngeometry_reset=yes";
  } else if (snapshot.decision_map_config_changed) {
    summary << "\nconfig_rebuild=yes";
  }

  const double map_top_y = origin_y + (static_cast<double>(height) * resolution);
  auto summary_marker = text_marker(
    config,
    "debug_chunk_cache_summary",
    3,
    point(origin_x + (resolution * 0.5), map_top_y + (resolution * 0.5), config.z_offset + 0.08),
    rgba(0.05F, 0.05F, 0.05F, 1.0F),
    summary.str());
  summary_marker.scale.z = config.text_scale * 1.1;
  markers.markers.push_back(summary_marker);

  return markers;
}

}  // namespace frontier_exploration_ros2::debug
