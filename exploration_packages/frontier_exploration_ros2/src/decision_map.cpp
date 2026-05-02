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

#include "frontier_exploration_ros2/decision_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace frontier_exploration_ros2
{

namespace
{

constexpr double kGeometryEpsilon = 1e-9;
constexpr double kGeometryResolutionToleranceRatio = 1e-3;
constexpr double kGeometryCellAlignmentToleranceCells = 0.35;
constexpr double kGeometryReuseCellAlignmentToleranceCells = 0.75;
constexpr double kMinimumReuseOverlapRatio = 0.20;
constexpr double kMaximumKnownConflictRatio = 0.35;
constexpr double kMaximumKnownLossRatio = 0.60;
constexpr std::size_t kMinimumKnownEvidenceCells = 64U;
constexpr double kChunkUnknownObstacleUnknownRatioDeltaThreshold = 0.20;
constexpr double kChunkUnknownObstacleOccupiedRatioDeltaThreshold = 0.20;
constexpr double kChunkUnknownObstacleSemanticChangeRatioThreshold = 0.35;
constexpr double kChunkKnownAreaRatioDeltaThreshold = 0.25;
constexpr double kChunkKnownAreaSemanticChangeRatioThreshold = 0.40;
constexpr int kRawDiffChunkSizeCells = 32;
// Threshold between UNKNOWN and FREE after filtering the paper-domain intensities.
constexpr float kFreeThreshold =
  static_cast<float>((static_cast<int>(PAPER_FREE) + static_cast<int>(PAPER_UNKNOWN)) / 2.0);

using RowSpan = std::pair<int, int>;
using RowSpanRows = std::vector<std::vector<RowSpan>>;

struct GeometryTransitionPlan
{
  DecisionMapGeometryTransition transition{DecisionMapGeometryTransition::FullRebuildFallback};
  int x_offset_cells{0};
  int y_offset_cells{0};
  int old_overlap_x{0};
  int old_overlap_y{0};
  int new_overlap_x{0};
  int new_overlap_y{0};
  int overlap_width{0};
  int overlap_height{0};
};

struct ForcedDirtyChunks
{
  std::vector<uint8_t> chunk_flags;
};

struct ChunkSemanticStats
{
  std::size_t cell_count{0U};
  std::size_t previous_unknown_count{0U};
  std::size_t previous_free_count{0U};
  std::size_t previous_occupied_count{0U};
  std::size_t next_unknown_count{0U};
  std::size_t next_free_count{0U};
  std::size_t next_occupied_count{0U};
  std::size_t semantic_changed_count{0U};
};

// Returns true when no row contains a dirty interval.
[[nodiscard]] bool row_spans_empty(const RowSpanRows & row_spans)
{
  for (const auto & row : row_spans) {
    if (!row.empty()) {
      return false;
    }
  }
  return true;
}

void clear_row_spans(RowSpanRows & row_spans, int height)
{
  row_spans.resize(static_cast<std::size_t>(std::max(0, height)));
  for (auto & row : row_spans) {
    row.clear();
  }
}

// Marks every row as dirty so downstream stages fully recompute.
void assign_full_row_spans(RowSpanRows & row_spans, int width, int height)
{
  clear_row_spans(row_spans, height);
  if (width <= 0) {
    return;
  }

  for (auto & row : row_spans) {
    row.emplace_back(0, width - 1);
  }
}

// Geometry changes invalidate the reusable buffers and linear-offset caches.
[[nodiscard]] bool same_geometry(
  const DecisionMapWorkspace & workspace,
  const nav_msgs::msg::OccupancyGrid & reference)
{
  const double resolution_tolerance =
    std::max(kGeometryEpsilon, reference.info.resolution * kGeometryResolutionToleranceRatio);
  const double origin_tolerance =
    std::max(kGeometryEpsilon, reference.info.resolution * kGeometryCellAlignmentToleranceCells);
  return (
    workspace.width == static_cast<int>(reference.info.width) &&
    workspace.height == static_cast<int>(reference.info.height) &&
    std::abs(workspace.resolution - reference.info.resolution) <= resolution_tolerance &&
    std::abs(workspace.origin_x - reference.info.origin.position.x) <= origin_tolerance &&
    std::abs(workspace.origin_y - reference.info.origin.position.y) <= origin_tolerance);
}

[[nodiscard]] std::optional<int> aligned_cell_offset(
  double old_origin,
  double new_origin,
  double resolution,
  double tolerance_cells)
{
  const double safe_resolution = std::max(resolution, kGeometryEpsilon);
  const double offset_cells = (old_origin - new_origin) / safe_resolution;
  const double rounded = std::round(offset_cells);
  if (std::abs(offset_cells - rounded) > tolerance_cells) {
    return std::nullopt;
  }
  return static_cast<int>(rounded);
}

enum class OccupancySemantic : uint8_t
{
  Unknown = 0,
  Free = 1,
  Occupied = 2,
};

[[nodiscard]] OccupancySemantic classify_occupancy_semantic(int8_t value, int occ_threshold)
{
  if (value < 0) {
    return OccupancySemantic::Unknown;
  }
  if (static_cast<int>(value) >= occ_threshold) {
    return OccupancySemantic::Occupied;
  }
  return OccupancySemantic::Free;
}

[[nodiscard]] bool overlap_semantics_support_reuse(
  const DecisionMapWorkspace & workspace,
  const nav_msgs::msg::OccupancyGrid & reference,
  const GeometryTransitionPlan & plan,
  int occ_threshold)
{
  const std::size_t overlap_cells =
    static_cast<std::size_t>(std::max(0, plan.overlap_width)) *
    static_cast<std::size_t>(std::max(0, plan.overlap_height));
  const std::size_t min_map_cells = std::min(
    static_cast<std::size_t>(std::max(0, workspace.width * workspace.height)),
    static_cast<std::size_t>(reference.info.width) * static_cast<std::size_t>(reference.info.height));
  if (min_map_cells == 0U) {
    return false;
  }

  const double overlap_ratio =
    static_cast<double>(overlap_cells) / static_cast<double>(min_map_cells);
  if (overlap_ratio < kMinimumReuseOverlapRatio) {
    return false;
  }

  std::size_t old_known = 0U;
  std::size_t both_known = 0U;
  std::size_t known_conflicts = 0U;
  std::size_t known_to_unknown = 0U;
  const auto & new_map_data = reference.data;
  for (int row = 0; row < plan.overlap_height; ++row) {
    const std::size_t old_row_offset =
      static_cast<std::size_t>(plan.old_overlap_y + row) *
      static_cast<std::size_t>(workspace.width) +
      static_cast<std::size_t>(plan.old_overlap_x);
    const std::size_t new_row_offset =
      static_cast<std::size_t>(plan.new_overlap_y + row) *
      static_cast<std::size_t>(reference.info.width) +
      static_cast<std::size_t>(plan.new_overlap_x);
    for (int column = 0; column < plan.overlap_width; ++column) {
      const OccupancySemantic old_semantic = classify_occupancy_semantic(
        workspace.raw_occupancy_cache[old_row_offset + static_cast<std::size_t>(column)],
        occ_threshold);
      const OccupancySemantic new_semantic = classify_occupancy_semantic(
        new_map_data[new_row_offset + static_cast<std::size_t>(column)],
        occ_threshold);

      if (old_semantic != OccupancySemantic::Unknown) {
        old_known += 1U;
      }
      if (
        old_semantic != OccupancySemantic::Unknown &&
        new_semantic != OccupancySemantic::Unknown)
      {
        both_known += 1U;
        if (old_semantic != new_semantic) {
          known_conflicts += 1U;
        }
      } else if (
        old_semantic != OccupancySemantic::Unknown &&
        new_semantic == OccupancySemantic::Unknown)
      {
        known_to_unknown += 1U;
      }
    }
  }

  if (both_known >= kMinimumKnownEvidenceCells) {
    const double conflict_ratio =
      static_cast<double>(known_conflicts) / static_cast<double>(both_known);
    if (conflict_ratio > kMaximumKnownConflictRatio) {
      return false;
    }
  }

  if (old_known >= kMinimumKnownEvidenceCells) {
    const double known_loss_ratio =
      static_cast<double>(known_to_unknown) / static_cast<double>(old_known);
    if (known_loss_ratio > kMaximumKnownLossRatio) {
      return false;
    }
  }

  return true;
}

[[nodiscard]] ChunkSemanticStats analyze_chunk_semantics(
  const std::vector<int8_t> & previous_data,
  const std::vector<int8_t> & next_data,
  int width,
  int chunk_x,
  int chunk_y,
  int map_width,
  int map_height,
  int occ_threshold)
{
  ChunkSemanticStats stats;
  const int x_min = chunk_x * kRawDiffChunkSizeCells;
  const int y_min = chunk_y * kRawDiffChunkSizeCells;
  const int x_max = std::min(map_width, x_min + kRawDiffChunkSizeCells);
  const int y_max = std::min(map_height, y_min + kRawDiffChunkSizeCells);

  for (int y = y_min; y < y_max; ++y) {
    const std::size_t row_offset =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
    for (int x = x_min; x < x_max; ++x) {
      const std::size_t idx = row_offset + static_cast<std::size_t>(x);
      const OccupancySemantic previous_semantic =
        classify_occupancy_semantic(previous_data[idx], occ_threshold);
      const OccupancySemantic next_semantic =
        classify_occupancy_semantic(next_data[idx], occ_threshold);

      stats.cell_count += 1U;
      switch (previous_semantic) {
        case OccupancySemantic::Unknown:
          stats.previous_unknown_count += 1U;
          break;
        case OccupancySemantic::Free:
          stats.previous_free_count += 1U;
          break;
        case OccupancySemantic::Occupied:
          stats.previous_occupied_count += 1U;
          break;
      }
      switch (next_semantic) {
        case OccupancySemantic::Unknown:
          stats.next_unknown_count += 1U;
          break;
        case OccupancySemantic::Free:
          stats.next_free_count += 1U;
          break;
        case OccupancySemantic::Occupied:
          stats.next_occupied_count += 1U;
          break;
      }
      if (previous_semantic != next_semantic) {
        stats.semantic_changed_count += 1U;
      }
    }
  }

  return stats;
}

[[nodiscard]] bool chunk_semantic_change_requires_rebuild(
  const ChunkSemanticStats & stats)
{
  if (stats.semantic_changed_count == 0U || stats.cell_count == 0U) {
    return false;
  }

  const double total_cells = static_cast<double>(stats.cell_count);
  const double previous_unknown_ratio =
    static_cast<double>(stats.previous_unknown_count) / total_cells;
  const double previous_free_ratio =
    static_cast<double>(stats.previous_free_count) / total_cells;
  const double previous_occupied_ratio =
    static_cast<double>(stats.previous_occupied_count) / total_cells;
  const double next_unknown_ratio =
    static_cast<double>(stats.next_unknown_count) / total_cells;
  const double next_free_ratio =
    static_cast<double>(stats.next_free_count) / total_cells;
  const double next_occupied_ratio =
    static_cast<double>(stats.next_occupied_count) / total_cells;

  const bool has_unknown = stats.previous_unknown_count > 0U || stats.next_unknown_count > 0U;
  const bool has_free = stats.previous_free_count > 0U || stats.next_free_count > 0U;
  const bool has_occupied =
    stats.previous_occupied_count > 0U || stats.next_occupied_count > 0U;
  const double semantic_change_ratio =
    static_cast<double>(stats.semantic_changed_count) / total_cells;

  if (has_unknown && has_free) {
    return true;
  }

  if (has_unknown && has_occupied && !has_free) {
    const double unknown_ratio_delta =
      std::abs(previous_unknown_ratio - next_unknown_ratio);
    const double occupied_ratio_delta =
      std::abs(previous_occupied_ratio - next_occupied_ratio);
    return (
      unknown_ratio_delta > kChunkUnknownObstacleUnknownRatioDeltaThreshold ||
      occupied_ratio_delta > kChunkUnknownObstacleOccupiedRatioDeltaThreshold ||
      semantic_change_ratio > kChunkUnknownObstacleSemanticChangeRatioThreshold);
  }

  const double free_ratio_delta = std::abs(previous_free_ratio - next_free_ratio);
  const double occupied_ratio_delta =
    std::abs(previous_occupied_ratio - next_occupied_ratio);
  return (
    free_ratio_delta > kChunkKnownAreaRatioDeltaThreshold ||
    occupied_ratio_delta > kChunkKnownAreaRatioDeltaThreshold ||
    semantic_change_ratio > kChunkKnownAreaSemanticChangeRatioThreshold);
}

[[nodiscard]] int filter_radius_for_sigma(double sigma_s)
{
  const double safe_sigma_s = std::max(sigma_s, 1e-6);
  return std::max(1, static_cast<int>(std::ceil(2.0 * safe_sigma_s)));
}

[[nodiscard]] GeometryTransitionPlan classify_geometry_transition(
  const DecisionMapWorkspace & workspace,
  const nav_msgs::msg::OccupancyGrid & reference,
  int occ_threshold)
{
  GeometryTransitionPlan plan;
  if (!workspace.initialized || workspace.width <= 0 || workspace.height <= 0) {
    return plan;
  }

  if (same_geometry(workspace, reference)) {
    plan.transition = DecisionMapGeometryTransition::SameGeometry;
    return plan;
  }

  const double resolution_tolerance =
    std::max(kGeometryEpsilon, reference.info.resolution * kGeometryResolutionToleranceRatio);
  if (std::abs(workspace.resolution - reference.info.resolution) > resolution_tolerance) {
    return plan;
  }

  const auto x_offset = aligned_cell_offset(
    workspace.origin_x,
    reference.info.origin.position.x,
    reference.info.resolution,
    kGeometryReuseCellAlignmentToleranceCells);
  const auto y_offset = aligned_cell_offset(
    workspace.origin_y,
    reference.info.origin.position.y,
    reference.info.resolution,
    kGeometryReuseCellAlignmentToleranceCells);
  if (!x_offset.has_value() || !y_offset.has_value()) {
    return plan;
  }

  const int safe_x_offset = *x_offset;
  const int safe_y_offset = *y_offset;
  const int new_width = static_cast<int>(reference.info.width);
  const int new_height = static_cast<int>(reference.info.height);
  const int new_overlap_x = std::max(0, safe_x_offset);
  const int new_overlap_y = std::max(0, safe_y_offset);
  const int old_overlap_x = std::max(0, -safe_x_offset);
  const int old_overlap_y = std::max(0, -safe_y_offset);
  const int overlap_width = std::min(workspace.width - old_overlap_x, new_width - new_overlap_x);
  const int overlap_height = std::min(workspace.height - old_overlap_y, new_height - new_overlap_y);
  if (overlap_width <= 0 || overlap_height <= 0) {
    return plan;
  }

  plan.x_offset_cells = safe_x_offset;
  plan.y_offset_cells = safe_y_offset;
  plan.old_overlap_x = old_overlap_x;
  plan.old_overlap_y = old_overlap_y;
  plan.new_overlap_x = new_overlap_x;
  plan.new_overlap_y = new_overlap_y;
  plan.overlap_width = overlap_width;
  plan.overlap_height = overlap_height;
  if (!overlap_semantics_support_reuse(workspace, reference, plan, occ_threshold)) {
    return plan;
  }

  plan.transition = DecisionMapGeometryTransition::OverlapReuse;
  return plan;
}

// Resizes image storage without changing the existing semantic contents.
void assign_image_geometry(PaperImage & image, int width, int height)
{
  image.width = width;
  image.height = height;
  image.data.resize(static_cast<std::size_t>(width * height));
}

// Rebuilds workspace-owned buffers whenever map geometry changes.
void resize_workspace_for_geometry(
  DecisionMapWorkspace & workspace,
  const nav_msgs::msg::OccupancyGrid & reference)
{
  const int width = static_cast<int>(reference.info.width);
  const int height = static_cast<int>(reference.info.height);
  const std::size_t cell_count = static_cast<std::size_t>(width * height);
  const int chunk_cols = (width + kRawDiffChunkSizeCells - 1) / kRawDiffChunkSizeCells;
  const int chunk_rows = (height + kRawDiffChunkSizeCells - 1) / kRawDiffChunkSizeCells;
  const std::size_t chunk_count =
    static_cast<std::size_t>(std::max(0, chunk_cols * chunk_rows));

  workspace.width = width;
  workspace.height = height;
  workspace.resolution = reference.info.resolution;
  workspace.origin_x = reference.info.origin.position.x;
  workspace.origin_y = reference.info.origin.position.y;
  workspace.chunk_cols = chunk_cols;
  workspace.chunk_rows = chunk_rows;

  assign_image_geometry(workspace.raw_image, width, height);
  assign_image_geometry(workspace.threshold_image, width, height);
  workspace.dilation_scratch.width = height;
  workspace.dilation_scratch.height = 1;
  workspace.dilation_scratch.data.resize(static_cast<std::size_t>(height), 0U);

  clear_row_spans(workspace.dirty_spans_by_row, height);
  clear_row_spans(workspace.filter_spans_by_row, height);
  clear_row_spans(workspace.dilation_spans_by_row, height);
  workspace.raw_occupancy_cache.assign(cell_count, static_cast<int8_t>(-1));
  workspace.dirty_chunk_flags.assign(chunk_count, 0U);

  workspace.optimized_map_msg->header = reference.header;
  workspace.optimized_map_msg->info = reference.info;
  workspace.optimized_map_msg->data.resize(cell_count, -1);
}

template<typename T>
void copy_buffer_region(
  const std::vector<T> & source,
  int source_width,
  int source_x,
  int source_y,
  std::vector<T> & target,
  int target_width,
  int target_x_offset,
  int target_y_offset,
  int copy_width,
  int copy_height)
{
  static_assert(!std::is_same_v<T, bool>);
  for (int row = 0; row < copy_height; ++row) {
    const std::size_t source_offset =
      static_cast<std::size_t>(row + source_y) * static_cast<std::size_t>(source_width) +
      static_cast<std::size_t>(source_x);
    const std::size_t target_offset =
      static_cast<std::size_t>(row + target_y_offset) * static_cast<std::size_t>(target_width) +
      static_cast<std::size_t>(target_x_offset);
    std::copy_n(
      source.begin() + static_cast<std::ptrdiff_t>(source_offset),
      copy_width,
      target.begin() + static_cast<std::ptrdiff_t>(target_offset));
  }
}

void copy_image_region(
  const PaperImage & source,
  PaperImage & target,
  int source_x,
  int source_y,
  int target_x_offset,
  int target_y_offset,
  int copy_width,
  int copy_height)
{
  copy_buffer_region(
    source.data,
    source.width,
    source_x,
    source_y,
    target.data,
    target.width,
    target_x_offset,
    target_y_offset,
    copy_width,
    copy_height);
}

void prepare_workspace_for_overlap_reuse(
  DecisionMapWorkspace & workspace,
  const nav_msgs::msg::OccupancyGrid & reference,
  const GeometryTransitionPlan & plan)
{
  DecisionMapWorkspace previous_workspace = std::move(workspace);

  DecisionMapWorkspace grown_workspace;
  resize_workspace_for_geometry(grown_workspace, reference);
  grown_workspace.initialized = previous_workspace.initialized;
  grown_workspace.last_optimization_enabled = previous_workspace.last_optimization_enabled;
  grown_workspace.last_occ_threshold = previous_workspace.last_occ_threshold;
  grown_workspace.last_build_sigma_s = previous_workspace.last_build_sigma_s;
  grown_workspace.last_build_sigma_r = previous_workspace.last_build_sigma_r;
  grown_workspace.last_build_dilation_radius_cells =
    previous_workspace.last_build_dilation_radius_cells;

  grown_workspace.cached_occ_threshold = previous_workspace.cached_occ_threshold;
  grown_workspace.occupancy_to_paper_lut = previous_workspace.occupancy_to_paper_lut;
  grown_workspace.cached_sigma_s = previous_workspace.cached_sigma_s;
  grown_workspace.cached_filter_radius = previous_workspace.cached_filter_radius;
  grown_workspace.spatial_kernel_samples = previous_workspace.spatial_kernel_samples;
  grown_workspace.cached_spatial_linear_width = -1;
  grown_workspace.spatial_linear_offsets.clear();
  grown_workspace.cached_sigma_r = previous_workspace.cached_sigma_r;
  grown_workspace.range_weight_lut = previous_workspace.range_weight_lut;
  grown_workspace.cached_dilation_radius = previous_workspace.cached_dilation_radius;
  grown_workspace.dilation_offsets = previous_workspace.dilation_offsets;
  grown_workspace.cached_dilation_linear_width = -1;
  grown_workspace.dilation_linear_offsets.clear();

  std::fill(grown_workspace.raw_image.data.begin(), grown_workspace.raw_image.data.end(), PAPER_UNKNOWN);
  std::fill(
    grown_workspace.threshold_image.data.begin(),
    grown_workspace.threshold_image.data.end(),
    PAPER_UNKNOWN);

  copy_buffer_region(
    previous_workspace.raw_occupancy_cache,
    previous_workspace.width,
    plan.old_overlap_x,
    plan.old_overlap_y,
    grown_workspace.raw_occupancy_cache,
    grown_workspace.width,
    plan.new_overlap_x,
    plan.new_overlap_y,
    plan.overlap_width,
    plan.overlap_height);
  copy_image_region(
    previous_workspace.raw_image,
    grown_workspace.raw_image,
    plan.old_overlap_x,
    plan.old_overlap_y,
    plan.new_overlap_x,
    plan.new_overlap_y,
    plan.overlap_width,
    plan.overlap_height);
  copy_image_region(
    previous_workspace.threshold_image,
    grown_workspace.threshold_image,
    plan.old_overlap_x,
    plan.old_overlap_y,
    plan.new_overlap_x,
    plan.new_overlap_y,
    plan.overlap_width,
    plan.overlap_height);
  copy_buffer_region(
    previous_workspace.optimized_map_msg->data,
    previous_workspace.width,
    plan.old_overlap_x,
    plan.old_overlap_y,
    grown_workspace.optimized_map_msg->data,
    grown_workspace.width,
    plan.new_overlap_x,
    plan.new_overlap_y,
    plan.overlap_width,
    plan.overlap_height);

  workspace = std::move(grown_workspace);
}

// Maps occupancy costs into the three paper-domain intensity classes.
[[nodiscard]] uint8_t occupancy_cost_to_paper_value(int cost, int occ_threshold)
{
  if (cost >= occ_threshold) {
    return PAPER_OCCUPIED;
  }
  if (cost >= 0) {
    return PAPER_FREE;
  }
  return PAPER_UNKNOWN;
}

// Precomputes all possible int8 occupancy conversions for the current threshold.
void fill_occupancy_to_paper_lut(std::array<uint8_t, 256> & lut, int occ_threshold)
{
  for (int i = 0; i < 256; ++i) {
    lut[static_cast<std::size_t>(i)] = occupancy_cost_to_paper_value(
      static_cast<int>(static_cast<int8_t>(i)),
      occ_threshold);
  }
}

// The LUT changes only when the occupancy threshold changes.
void ensure_occupancy_to_paper_lut_cache(DecisionMapWorkspace & workspace, int occ_threshold)
{
  if (workspace.cached_occ_threshold == occ_threshold) {
    return;
  }

  fill_occupancy_to_paper_lut(workspace.occupancy_to_paper_lut, occ_threshold);
  workspace.cached_occ_threshold = occ_threshold;
}

// Compacts paper intensities into three indices for the range-weight lookup table.
[[nodiscard]] int paper_value_index(uint8_t value)
{
  if (value == PAPER_OCCUPIED) {
    return 0;
  }
  if (value == PAPER_UNKNOWN) {
    return 1;
  }
  return 2;
}

// Precomputes the spatial Gaussian kernel used by the bilateral filter.
void ensure_spatial_kernel_cache(DecisionMapWorkspace & workspace, double sigma_s)
{
  const double safe_sigma_s = std::max(sigma_s, 1e-6);
  const int radius = std::max(1, static_cast<int>(std::ceil(2.0 * safe_sigma_s)));
  if (
    workspace.cached_filter_radius == radius &&
    std::abs(workspace.cached_sigma_s - safe_sigma_s) <= kGeometryEpsilon)
  {
    return;
  }

  workspace.cached_sigma_s = safe_sigma_s;
  workspace.cached_filter_radius = radius;
  workspace.cached_spatial_linear_width = -1;
  workspace.spatial_kernel_samples.clear();
  workspace.spatial_kernel_samples.reserve(
    static_cast<std::size_t>((radius * 2 + 1) * (radius * 2 + 1)));

  const double denominator = 2.0 * safe_sigma_s * safe_sigma_s;
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const double distance_sq = static_cast<double>((dx * dx) + (dy * dy));
      workspace.spatial_kernel_samples.push_back(
        WeightedKernelOffset{
          dx,
          dy,
          static_cast<float>(std::exp(-(distance_sq / denominator))),
        });
    }
  }
}

// Converts 2-D filter offsets into linear offsets for the current map width.
void ensure_spatial_linear_offset_cache(DecisionMapWorkspace & workspace)
{
  if (workspace.cached_spatial_linear_width == workspace.width) {
    return;
  }

  workspace.cached_spatial_linear_width = workspace.width;
  workspace.spatial_linear_offsets.clear();
  workspace.spatial_linear_offsets.reserve(workspace.spatial_kernel_samples.size());
  for (const auto & sample : workspace.spatial_kernel_samples) {
    workspace.spatial_linear_offsets.push_back(
      static_cast<std::ptrdiff_t>(sample.dy) * static_cast<std::ptrdiff_t>(workspace.width) +
      static_cast<std::ptrdiff_t>(sample.dx));
  }
}

// Precomputes bilateral range weights for occupied/unknown/free paper values.
void ensure_range_lut_cache(DecisionMapWorkspace & workspace, double sigma_r)
{
  const double safe_sigma_r = std::max(sigma_r, 1e-6);
  if (std::abs(workspace.cached_sigma_r - safe_sigma_r) <= kGeometryEpsilon) {
    return;
  }

  workspace.cached_sigma_r = safe_sigma_r;
  constexpr std::array<uint8_t, 3> paper_values{PAPER_OCCUPIED, PAPER_UNKNOWN, PAPER_FREE};
  const double denominator = 2.0 * safe_sigma_r * safe_sigma_r;
  for (std::size_t center_index = 0; center_index < paper_values.size(); ++center_index) {
    for (std::size_t shifted_index = 0; shifted_index < paper_values.size(); ++shifted_index) {
      const double diff =
        static_cast<double>(paper_values[center_index]) -
        static_cast<double>(paper_values[shifted_index]);
      workspace.range_weight_lut[center_index * paper_values.size() + shifted_index] =
        static_cast<float>(std::exp(-((diff * diff) / denominator)));
    }
  }
}

// Builds the circular dilation stencil used after thresholding.
void ensure_dilation_kernel_cache(DecisionMapWorkspace & workspace, int radius)
{
  const int safe_radius = std::max(0, radius);
  if (workspace.cached_dilation_radius == safe_radius) {
    return;
  }

  workspace.cached_dilation_radius = safe_radius;
  workspace.cached_dilation_linear_width = -1;
  workspace.dilation_offsets.clear();
  workspace.dilation_offsets.reserve(
    static_cast<std::size_t>((safe_radius * 2 + 1) * (safe_radius * 2 + 1)));
  for (int dy = -safe_radius; dy <= safe_radius; ++dy) {
    for (int dx = -safe_radius; dx <= safe_radius; ++dx) {
      if ((dx * dx) + (dy * dy) <= (safe_radius * safe_radius)) {
        workspace.dilation_offsets.push_back(KernelOffset{dx, dy});
      }
    }
  }
}

// Converts dilation offsets into linear form for fast interior-cell access.
void ensure_dilation_linear_offset_cache(DecisionMapWorkspace & workspace)
{
  if (workspace.cached_dilation_linear_width == workspace.width) {
    return;
  }

  workspace.cached_dilation_linear_width = workspace.width;
  workspace.dilation_linear_offsets.clear();
  workspace.dilation_linear_offsets.reserve(workspace.dilation_offsets.size());
  for (const auto & offset : workspace.dilation_offsets) {
    workspace.dilation_linear_offsets.push_back(
      static_cast<std::ptrdiff_t>(offset.dy) * static_cast<std::ptrdiff_t>(workspace.width) +
      static_cast<std::ptrdiff_t>(offset.dx));
  }
}

// Coalesces overlapping or adjacent dirty intervals within one row.
void normalize_row_spans(std::vector<RowSpan> & row_spans)
{
  if (row_spans.empty()) {
    return;
  }

  std::sort(row_spans.begin(), row_spans.end());

  std::size_t write_index = 0;
  for (std::size_t read_index = 1; read_index < row_spans.size(); ++read_index) {
    if (row_spans[read_index].first <= row_spans[write_index].second + 1) {
      row_spans[write_index].second =
        std::max(row_spans[write_index].second, row_spans[read_index].second);
      continue;
    }

    ++write_index;
    row_spans[write_index] = row_spans[read_index];
  }

  row_spans.resize(write_index + 1);
}

// Expands dirty spans by each stage's support radius so only affected cells are recomputed.
void build_expanded_row_spans(
  const RowSpanRows & source_spans_by_row,
  int horizontal_radius,
  int vertical_radius,
  int width,
  int height,
  DecisionMapWorkspace & workspace,
  RowSpanRows & target_spans_by_row)
{
  clear_row_spans(target_spans_by_row, height);
  if (width <= 0 || height <= 0) {
    return;
  }

  auto & touched_rows = workspace.dilation_scratch.data;
  if (touched_rows.size() < static_cast<std::size_t>(height)) {
    touched_rows.resize(static_cast<std::size_t>(height), 0U);
  }
  std::fill_n(touched_rows.begin(), height, 0U);

  const int safe_horizontal_radius = std::max(0, horizontal_radius);
  const int safe_vertical_radius = std::max(0, vertical_radius);

  for (int y = 0; y < height; ++y) {
    for (const auto & span : source_spans_by_row[static_cast<std::size_t>(y)]) {
      const int expanded_start = std::max(0, span.first - safe_horizontal_radius);
      const int expanded_end = std::min(width - 1, span.second + safe_horizontal_radius);
      const int row_min = std::max(0, y - safe_vertical_radius);
      const int row_max = std::min(height - 1, y + safe_vertical_radius);
      for (int target_y = row_min; target_y <= row_max; ++target_y) {
        target_spans_by_row[static_cast<std::size_t>(target_y)].emplace_back(
          expanded_start,
          expanded_end);
        touched_rows[static_cast<std::size_t>(target_y)] = 1U;
      }
    }
  }

  for (int y = 0; y < height; ++y) {
    if (touched_rows[static_cast<std::size_t>(y)] != 0U) {
      normalize_row_spans(target_spans_by_row[static_cast<std::size_t>(y)]);
    }
  }
}

void mark_chunk_dirty_rows(
  DecisionMapWorkspace & workspace,
  int chunk_x,
  int chunk_y)
{
  const int x_min = chunk_x * kRawDiffChunkSizeCells;
  const int y_min = chunk_y * kRawDiffChunkSizeCells;
  const int x_max = std::min(workspace.width, x_min + kRawDiffChunkSizeCells) - 1;
  const int y_max = std::min(workspace.height, y_min + kRawDiffChunkSizeCells) - 1;

  for (int y = y_min; y <= y_max; ++y) {
    workspace.dirty_spans_by_row[static_cast<std::size_t>(y)].emplace_back(x_min, x_max);
  }
}

void mark_chunk_dirty(
  DecisionMapWorkspace & workspace,
  std::size_t chunk_index,
  int chunk_x,
  int chunk_y)
{
  if (chunk_index < workspace.dirty_chunk_flags.size()) {
    workspace.dirty_chunk_flags[chunk_index] = 1U;
  }
  mark_chunk_dirty_rows(workspace, chunk_x, chunk_y);
}

void mark_forced_dirty_rect(
  ForcedDirtyChunks & forced_dirty,
  int chunk_cols,
  int chunk_rows,
  int x_min,
  int y_min,
  int x_max,
  int y_max)
{
  if (x_min > x_max || y_min > y_max || chunk_cols <= 0 || chunk_rows <= 0) {
    return;
  }

  const int chunk_x_min = std::max(0, x_min / kRawDiffChunkSizeCells);
  const int chunk_y_min = std::max(0, y_min / kRawDiffChunkSizeCells);
  const int chunk_x_max = std::min(chunk_cols - 1, x_max / kRawDiffChunkSizeCells);
  const int chunk_y_max = std::min(chunk_rows - 1, y_max / kRawDiffChunkSizeCells);
  for (int chunk_y = chunk_y_min; chunk_y <= chunk_y_max; ++chunk_y) {
    for (int chunk_x = chunk_x_min; chunk_x <= chunk_x_max; ++chunk_x) {
      const std::size_t chunk_index =
        static_cast<std::size_t>(chunk_y) * static_cast<std::size_t>(chunk_cols) +
        static_cast<std::size_t>(chunk_x);
      if (chunk_index < forced_dirty.chunk_flags.size()) {
        forced_dirty.chunk_flags[chunk_index] = 1U;
      }
    }
  }
}

[[nodiscard]] ForcedDirtyChunks build_overlap_reuse_forced_dirty_chunks(
  const DecisionMapWorkspace & workspace,
  const GeometryTransitionPlan & plan,
  int halo_radius)
{
  ForcedDirtyChunks forced_dirty;
  forced_dirty.chunk_flags.assign(
    static_cast<std::size_t>(std::max(0, workspace.chunk_cols * workspace.chunk_rows)),
    0U);

  const int overlap_x_min = plan.new_overlap_x;
  const int overlap_y_min = plan.new_overlap_y;
  const int overlap_x_max = plan.new_overlap_x + plan.overlap_width - 1;
  const int overlap_y_max = plan.new_overlap_y + plan.overlap_height - 1;
  const int right_dirty_start = overlap_x_max + 1;
  const int top_dirty_start = overlap_y_max + 1;

  if (overlap_x_min > 0) {
    mark_forced_dirty_rect(
      forced_dirty,
      workspace.chunk_cols,
      workspace.chunk_rows,
      0,
      0,
      overlap_x_min - 1,
      workspace.height - 1);
  }
  if (right_dirty_start < workspace.width) {
    mark_forced_dirty_rect(
      forced_dirty,
      workspace.chunk_cols,
      workspace.chunk_rows,
      right_dirty_start,
      0,
      workspace.width - 1,
      workspace.height - 1);
  }
  if (overlap_y_min > 0) {
    mark_forced_dirty_rect(
      forced_dirty,
      workspace.chunk_cols,
      workspace.chunk_rows,
      0,
      0,
      workspace.width - 1,
      overlap_y_min - 1);
  }
  if (top_dirty_start < workspace.height) {
    mark_forced_dirty_rect(
      forced_dirty,
      workspace.chunk_cols,
      workspace.chunk_rows,
      0,
      top_dirty_start,
      workspace.width - 1,
      workspace.height - 1);
  }

  if (halo_radius <= 0) {
    return forced_dirty;
  }

  if (overlap_x_min > 0) {
    mark_forced_dirty_rect(
      forced_dirty,
      workspace.chunk_cols,
      workspace.chunk_rows,
      overlap_x_min,
      overlap_y_min,
      std::min(overlap_x_max, overlap_x_min + halo_radius - 1),
      overlap_y_max);
  }
  if (right_dirty_start < workspace.width) {
    mark_forced_dirty_rect(
      forced_dirty,
      workspace.chunk_cols,
      workspace.chunk_rows,
      std::max(overlap_x_min, overlap_x_max - halo_radius + 1),
      overlap_y_min,
      overlap_x_max,
      overlap_y_max);
  }
  if (overlap_y_min > 0) {
    mark_forced_dirty_rect(
      forced_dirty,
      workspace.chunk_cols,
      workspace.chunk_rows,
      overlap_x_min,
      overlap_y_min,
      overlap_x_max,
      std::min(overlap_y_max, overlap_y_min + halo_radius - 1));
  }
  if (top_dirty_start < workspace.height) {
    mark_forced_dirty_rect(
      forced_dirty,
      workspace.chunk_cols,
      workspace.chunk_rows,
      overlap_x_min,
      std::max(overlap_y_min, overlap_y_max - halo_radius + 1),
      overlap_x_max,
      overlap_y_max);
  }

  return forced_dirty;
}

[[nodiscard]] bool chunk_raw_occupancy_changed(
  const std::vector<int8_t> & map_data,
  const DecisionMapWorkspace & workspace,
  int chunk_x,
  int chunk_y,
  int occ_threshold)
{
  bool raw_changed = false;
  const int x_min = chunk_x * kRawDiffChunkSizeCells;
  const int y_min = chunk_y * kRawDiffChunkSizeCells;
  const int x_max = std::min(workspace.width, x_min + kRawDiffChunkSizeCells);
  const int y_max = std::min(workspace.height, y_min + kRawDiffChunkSizeCells);

  for (int y = y_min; y < y_max; ++y) {
    const std::size_t row_offset =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(workspace.width);
    for (int x = x_min; x < x_max; ++x) {
      const std::size_t idx = row_offset + static_cast<std::size_t>(x);
      if (workspace.raw_occupancy_cache[idx] != map_data[idx]) {
        raw_changed = true;
        break;
      }
    }
    if (raw_changed) {
      break;
    }
  }

  if (!raw_changed) {
    return false;
  }

  const ChunkSemanticStats semantic_stats = analyze_chunk_semantics(
    workspace.raw_occupancy_cache,
    map_data,
    workspace.width,
    chunk_x,
    chunk_y,
    workspace.width,
    workspace.height,
    occ_threshold);
  return chunk_semantic_change_requires_rebuild(semantic_stats);
}

void rebuild_raw_chunk(
  const OccupancyGrid2d & occupancy_map,
  DecisionMapWorkspace & workspace,
  int chunk_x,
  int chunk_y)
{
  const auto & map_data = occupancy_map.map().data;
  const int x_min = chunk_x * kRawDiffChunkSizeCells;
  const int y_min = chunk_y * kRawDiffChunkSizeCells;
  const int x_max = std::min(workspace.width, x_min + kRawDiffChunkSizeCells);
  const int y_max = std::min(workspace.height, y_min + kRawDiffChunkSizeCells);

  for (int y = y_min; y < y_max; ++y) {
    const std::size_t row_offset =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(workspace.width);
    for (int x = x_min; x < x_max; ++x) {
      const std::size_t idx = row_offset + static_cast<std::size_t>(x);
      const int8_t occupancy_value = map_data[idx];
      workspace.raw_occupancy_cache[idx] = occupancy_value;
      workspace.raw_image.data[idx] =
        workspace.occupancy_to_paper_lut[static_cast<uint8_t>(occupancy_value)];
    }
  }
}

// Re-materializes only dirty raw chunks and records the chunk rectangles that changed.
void populate_chunked_raw_image_and_detect_dirty(
  const OccupancyGrid2d & occupancy_map,
  DecisionMapWorkspace & workspace,
  bool force_full_raw_rebuild,
  const ForcedDirtyChunks * forced_dirty_chunks = nullptr)
{
  clear_row_spans(workspace.dirty_spans_by_row, workspace.height);
  if (!workspace.dirty_chunk_flags.empty()) {
    std::fill(
      workspace.dirty_chunk_flags.begin(),
      workspace.dirty_chunk_flags.end(),
      0U);
  }

  const auto & map_data = occupancy_map.map().data;
  for (int chunk_y = 0; chunk_y < workspace.chunk_rows; ++chunk_y) {
    for (int chunk_x = 0; chunk_x < workspace.chunk_cols; ++chunk_x) {
      const std::size_t chunk_index =
        static_cast<std::size_t>(chunk_y) * static_cast<std::size_t>(workspace.chunk_cols) +
        static_cast<std::size_t>(chunk_x);
      const bool force_chunk_rebuild =
        forced_dirty_chunks != nullptr &&
        chunk_index < forced_dirty_chunks->chunk_flags.size() &&
        forced_dirty_chunks->chunk_flags[chunk_index] != 0U;
      const bool chunk_changed =
        force_full_raw_rebuild ||
        force_chunk_rebuild ||
        chunk_raw_occupancy_changed(
        map_data,
        workspace,
        chunk_x,
        chunk_y,
        workspace.cached_occ_threshold);
      if (!chunk_changed) {
        continue;
      }

      rebuild_raw_chunk(occupancy_map, workspace, chunk_x, chunk_y);
      mark_chunk_dirty(workspace, chunk_index, chunk_x, chunk_y);
    }
  }

  for (auto & row_spans : workspace.dirty_spans_by_row) {
    normalize_row_spans(row_spans);
  }
}

[[nodiscard]] std::size_t count_dirty_chunks(const DecisionMapWorkspace & workspace)
{
  return static_cast<std::size_t>(std::count(
    workspace.dirty_chunk_flags.begin(),
    workspace.dirty_chunk_flags.end(),
    static_cast<uint8_t>(1U)));
}

// Recomputes only the requested row spans for bilateral filtering and thresholding.
void recompute_filtered_threshold_row_spans(
  const PaperImage & raw_image,
  std::vector<float> * filtered_image,
  PaperImage * threshold_image,
  const RowSpanRows & row_spans_by_row,
  const DecisionMapWorkspace & workspace)
{
  const auto * raw_data = raw_image.data.data();
  auto * filtered_data = filtered_image != nullptr ? filtered_image->data() : nullptr;
  auto * threshold_data = threshold_image ? threshold_image->data.data() : nullptr;
  const auto * range_lut = workspace.range_weight_lut.data();
  const int radius = workspace.cached_filter_radius;
  const int width = raw_image.width;
  const int height = raw_image.height;

  // Border cells clamp neighborhood access back into image bounds.
  auto recompute_border_cell = [&](int x, int y, std::size_t idx) {
      const uint8_t center_value = raw_data[idx];
      const int center_index = paper_value_index(center_value);
      float weighted_sum = 0.0F;
      float normalization = 0.0F;

      for (const auto & sample : workspace.spatial_kernel_samples) {
        const int sample_x = std::clamp(x + sample.dx, 0, width - 1);
        const int sample_y = std::clamp(y + sample.dy, 0, height - 1);
        const uint8_t shifted_value =
          raw_data[static_cast<std::size_t>(sample_y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(sample_x)];
        const int shifted_index = paper_value_index(shifted_value);
        const float weight =
          sample.weight *
          range_lut[static_cast<std::size_t>(center_index * 3 + shifted_index)];
        weighted_sum += weight * static_cast<float>(shifted_value);
        normalization += weight;
      }

      const float filtered_value =
        normalization > std::numeric_limits<float>::epsilon() ?
        (weighted_sum / normalization) :
        static_cast<float>(center_value);
      if (filtered_data != nullptr) {
        filtered_data[idx] = filtered_value;
      }

      if (threshold_data != nullptr) {
        threshold_data[idx] =
          center_value == PAPER_OCCUPIED ?
          PAPER_OCCUPIED :
          (filtered_value >= kFreeThreshold ? PAPER_FREE : PAPER_UNKNOWN);
      }
    };

  // Interior cells can use precomputed linear offsets with no per-sample bounds checks.
  auto recompute_interior_cell = [&](std::size_t idx) {
      const uint8_t center_value = raw_data[idx];
      const int center_index = paper_value_index(center_value);
      float weighted_sum = 0.0F;
      float normalization = 0.0F;
      const auto base_index = static_cast<std::ptrdiff_t>(idx);

      for (std::size_t sample_index = 0; sample_index < workspace.spatial_kernel_samples.size();
        ++sample_index)
      {
        const uint8_t shifted_value =
          raw_data[static_cast<std::size_t>(base_index + workspace.spatial_linear_offsets[sample_index])];
        const int shifted_index = paper_value_index(shifted_value);
        const float weight =
          workspace.spatial_kernel_samples[sample_index].weight *
          range_lut[static_cast<std::size_t>(center_index * 3 + shifted_index)];
        weighted_sum += weight * static_cast<float>(shifted_value);
        normalization += weight;
      }

      const float filtered_value =
        normalization > std::numeric_limits<float>::epsilon() ?
        (weighted_sum / normalization) :
        static_cast<float>(center_value);
      if (filtered_data != nullptr) {
        filtered_data[idx] = filtered_value;
      }

      if (threshold_data != nullptr) {
        threshold_data[idx] =
          center_value == PAPER_OCCUPIED ?
          PAPER_OCCUPIED :
          (filtered_value >= kFreeThreshold ? PAPER_FREE : PAPER_UNKNOWN);
      }
    };

  for (int y = 0; y < height; ++y) {
    const auto & row_spans = row_spans_by_row[static_cast<std::size_t>(y)];
    if (row_spans.empty()) {
      continue;
    }

    const std::size_t row_offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
    const bool interior_y = y >= radius && y < (height - radius);

    for (const auto & span : row_spans) {
      const int border_left_end = interior_y ? std::min(span.second, radius - 1) : span.second;
      for (int x = span.first; x <= border_left_end; ++x) {
        recompute_border_cell(x, y, row_offset + static_cast<std::size_t>(x));
      }

      if (interior_y) {
        const int interior_start = std::max(span.first, radius);
        const int interior_end = std::min(span.second, width - radius - 1);
        for (int x = interior_start; x <= interior_end; ++x) {
          recompute_interior_cell(row_offset + static_cast<std::size_t>(x));
        }

        const int border_right_start = std::max(interior_end + 1, span.first);
        for (int x = border_right_start; x <= span.second; ++x) {
          if (x < radius || x >= width - radius) {
            recompute_border_cell(x, y, row_offset + static_cast<std::size_t>(x));
          }
        }
      }
    }
  }
}

[[nodiscard]] int8_t paper_value_to_occupancy_cost(uint8_t value)
{
  if (value == PAPER_FREE) {
    return 0;
  }
  if (value == PAPER_OCCUPIED) {
    return 100;
  }
  return -1;
}

[[nodiscard]] uint8_t occupancy_cost_to_paper_output_value(int8_t value)
{
  if (value == 0) {
    return PAPER_FREE;
  }
  if (value >= 100) {
    return PAPER_OCCUPIED;
  }
  return PAPER_UNKNOWN;
}

// Fast path used when optimization is disabled and raw classifications should be forwarded.
[[nodiscard]] bool copy_raw_to_output_row_spans(
  const PaperImage & raw_image,
  PaperImage & threshold_image,
  nav_msgs::msg::OccupancyGrid & optimized_map_msg,
  const RowSpanRows & row_spans_by_row,
  bool force_write_output)
{
  bool output_changed = false;
  for (int y = 0; y < raw_image.height; ++y) {
    const auto & row_spans = row_spans_by_row[static_cast<std::size_t>(y)];
    if (row_spans.empty()) {
      continue;
    }

    const std::size_t row_offset =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(raw_image.width);
    for (const auto & span : row_spans) {
      for (int x = span.first; x <= span.second; ++x) {
        const std::size_t idx = row_offset + static_cast<std::size_t>(x);
        const uint8_t raw_value = raw_image.data[idx];
        threshold_image.data[idx] = raw_value;
        const int8_t occupancy_value = paper_value_to_occupancy_cost(raw_value);
        if (optimized_map_msg.data[idx] != occupancy_value) {
          output_changed = true;
        }
        if (force_write_output || optimized_map_msg.data[idx] != occupancy_value) {
          optimized_map_msg.data[idx] = occupancy_value;
        }
      }
    }
  }

  return output_changed;
}

// Applies circular dilation over thresholded free-space support on the requested spans.
[[nodiscard]] bool recompute_dilation_row_spans(
  const PaperImage & raw_image,
  const PaperImage & threshold_image,
  nav_msgs::msg::OccupancyGrid & optimized_map_msg,
  const RowSpanRows & row_spans_by_row,
  const DecisionMapWorkspace & workspace,
  bool force_write_output)
{
  const auto * raw_data = raw_image.data.data();
  const auto * threshold_data = threshold_image.data.data();
  auto * optimized_msg_data = optimized_map_msg.data.data();
  const int radius = workspace.cached_dilation_radius;
  const int width = raw_image.width;
  const int height = raw_image.height;
  bool output_changed = false;

  // Border cells keep explicit bounds checks because the stencil may cross image edges.
  auto recompute_border_cell = [&](int x, int y, std::size_t idx) {
      uint8_t optimized_value = PAPER_UNKNOWN;
      if (raw_data[idx] == PAPER_OCCUPIED) {
        optimized_value = PAPER_OCCUPIED;
      } else {
        for (const auto & offset : workspace.dilation_offsets) {
          const int sample_x = x + offset.dx;
          const int sample_y = y + offset.dy;
          if (
            sample_x < 0 || sample_y < 0 ||
            sample_x >= width || sample_y >= height)
          {
            continue;
          }

          if (threshold_data[
              static_cast<std::size_t>(sample_y) * static_cast<std::size_t>(width) +
              static_cast<std::size_t>(sample_x)] == PAPER_FREE)
          {
            optimized_value = PAPER_FREE;
            break;
          }
        }
      }

      const int8_t occupancy_value = paper_value_to_occupancy_cost(optimized_value);
      if (optimized_msg_data[idx] != occupancy_value) {
        output_changed = true;
      }
      if (force_write_output || optimized_msg_data[idx] != occupancy_value) {
        optimized_msg_data[idx] = occupancy_value;
      }
    };

  // Interior cells reuse linear offsets and skip per-neighbor coordinate math.
  auto recompute_interior_cell = [&](std::size_t idx) {
      uint8_t optimized_value = PAPER_UNKNOWN;
      if (raw_data[idx] == PAPER_OCCUPIED) {
        optimized_value = PAPER_OCCUPIED;
      } else {
        const auto base_index = static_cast<std::ptrdiff_t>(idx);
        for (const auto linear_offset : workspace.dilation_linear_offsets) {
          if (threshold_data[static_cast<std::size_t>(base_index + linear_offset)] == PAPER_FREE) {
            optimized_value = PAPER_FREE;
            break;
          }
        }
      }

      const int8_t occupancy_value = paper_value_to_occupancy_cost(optimized_value);
      if (optimized_msg_data[idx] != occupancy_value) {
        output_changed = true;
      }
      if (force_write_output || optimized_msg_data[idx] != occupancy_value) {
        optimized_msg_data[idx] = occupancy_value;
      }
    };

  for (int y = 0; y < height; ++y) {
    const auto & row_spans = row_spans_by_row[static_cast<std::size_t>(y)];
    if (row_spans.empty()) {
      continue;
    }

    const std::size_t row_offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
    const bool interior_y = y >= radius && y < (height - radius);

    for (const auto & span : row_spans) {
      const int border_left_end = interior_y ? std::min(span.second, radius - 1) : span.second;
      for (int x = span.first; x <= border_left_end; ++x) {
        recompute_border_cell(x, y, row_offset + static_cast<std::size_t>(x));
      }

      if (interior_y) {
        const int interior_start = std::max(span.first, radius);
        const int interior_end = std::min(span.second, width - radius - 1);
        for (int x = interior_start; x <= interior_end; ++x) {
          recompute_interior_cell(row_offset + static_cast<std::size_t>(x));
        }

        const int border_right_start = std::max(interior_end + 1, span.first);
        for (int x = border_right_start; x <= span.second; ++x) {
          if (x < radius || x >= width - radius) {
            recompute_border_cell(x, y, row_offset + static_cast<std::size_t>(x));
          }
        }
      }
    }
  }

  return output_changed;
}

}  // namespace

PaperImage occupancy_grid_to_paper_image(
  const OccupancyGrid2d & occupancy_map,
  int occ_threshold)
{
  PaperImage image;
  const auto & grid = occupancy_map.map();
  image.width = static_cast<int>(grid.info.width);
  image.height = static_cast<int>(grid.info.height);
  image.data.resize(static_cast<std::size_t>(image.width * image.height), PAPER_UNKNOWN);

  std::array<uint8_t, 256> occupancy_to_paper_lut{};
  fill_occupancy_to_paper_lut(occupancy_to_paper_lut, occ_threshold);
  for (std::size_t i = 0; i < image.data.size(); ++i) {
    image.data[i] = occupancy_to_paper_lut[static_cast<uint8_t>(grid.data[i])];
  }

  return image;
}

nav_msgs::msg::OccupancyGrid paper_image_to_occupancy_grid(
  const PaperImage & image,
  const nav_msgs::msg::OccupancyGrid & reference)
{
  nav_msgs::msg::OccupancyGrid msg;
  msg.header = reference.header;
  msg.info = reference.info;
  msg.data.resize(image.data.size(), -1);

  for (std::size_t i = 0; i < image.data.size(); ++i) {
    msg.data[i] = paper_value_to_occupancy_cost(image.data[i]);
  }

  return msg;
}

std::vector<float> bilateral_filter(
  const PaperImage & image,
  double sigma_s,
  double sigma_r)
{
  // One-shot helper: build a minimal workspace and treat the whole image as dirty.
  DecisionMapWorkspace workspace;
  workspace.width = image.width;
  workspace.height = image.height;
  workspace.raw_image = image;
  std::vector<float> filtered_image(image.data.size(), 0.0F);
  clear_row_spans(workspace.filter_spans_by_row, image.height);
  assign_full_row_spans(workspace.filter_spans_by_row, image.width, image.height);

  ensure_spatial_kernel_cache(workspace, sigma_s);
  ensure_spatial_linear_offset_cache(workspace);
  ensure_range_lut_cache(workspace, sigma_r);
  recompute_filtered_threshold_row_spans(
    image,
    &filtered_image,
    nullptr,
    workspace.filter_spans_by_row,
    workspace);
  return filtered_image;
}

DecisionMapBuildStatus build_decision_map(
  const OccupancyGrid2d & raw_map,
  const DecisionMapConfig & config,
  DecisionMapWorkspace & workspace)
{
  DecisionMapBuildStatus status;
  const auto & reference = raw_map.map();
  const GeometryTransitionPlan geometry_plan =
    classify_geometry_transition(workspace, reference, config.occ_threshold);
  const bool geometry_changed =
    !workspace.initialized ||
    geometry_plan.transition != DecisionMapGeometryTransition::SameGeometry;
  const bool config_changed =
    !workspace.initialized ||
    workspace.last_optimization_enabled != config.optimization_enabled ||
    workspace.last_occ_threshold != config.occ_threshold ||
    std::abs(workspace.last_build_sigma_s - config.sigma_s) > kGeometryEpsilon ||
    std::abs(workspace.last_build_sigma_r - config.sigma_r) > kGeometryEpsilon ||
    workspace.last_build_dilation_radius_cells != config.dilation_kernel_radius_cells;
  status.config_changed = config_changed;
  status.geometry_transition = geometry_changed ?
    geometry_plan.transition :
    DecisionMapGeometryTransition::SameGeometry;
  const bool raw_classification_changed =
    !workspace.initialized ||
    geometry_plan.transition == DecisionMapGeometryTransition::FullRebuildFallback ||
    workspace.last_occ_threshold != config.occ_threshold;

  if (!workspace.initialized) {
    resize_workspace_for_geometry(workspace, reference);
    status.geometry_changed = true;
    status.geometry_transition = DecisionMapGeometryTransition::FullRebuildFallback;
  } else if (geometry_plan.transition == DecisionMapGeometryTransition::OverlapReuse) {
    prepare_workspace_for_overlap_reuse(workspace, reference, geometry_plan);
    status.geometry_changed = true;
  } else if (geometry_plan.transition == DecisionMapGeometryTransition::FullRebuildFallback) {
    resize_workspace_for_geometry(workspace, reference);
    status.geometry_changed = true;
  } else {
    workspace.optimized_map_msg->header = reference.header;
    workspace.optimized_map_msg->info = reference.info;
  }

  ensure_occupancy_to_paper_lut_cache(workspace, config.occ_threshold);
  std::optional<ForcedDirtyChunks> forced_dirty_chunks;
  if (geometry_plan.transition == DecisionMapGeometryTransition::OverlapReuse) {
    const int halo_radius =
      config.optimization_enabled ? filter_radius_for_sigma(config.sigma_s) : 0;
    forced_dirty_chunks = build_overlap_reuse_forced_dirty_chunks(
      workspace,
      geometry_plan,
      halo_radius);
  }
  populate_chunked_raw_image_and_detect_dirty(
    raw_map,
    workspace,
    raw_classification_changed,
    forced_dirty_chunks.has_value() ? &(*forced_dirty_chunks) : nullptr);
  status.total_chunks =
    static_cast<std::size_t>(std::max(0, workspace.chunk_cols * workspace.chunk_rows));
  status.dirty_chunks = count_dirty_chunks(workspace);

  // Any configuration change invalidates stage-local reuse, so mark the full image dirty.
  if (config_changed) {
    assign_full_row_spans(workspace.dirty_spans_by_row, workspace.width, workspace.height);
  }
  if (row_spans_empty(workspace.dirty_spans_by_row)) {
    status.reused_existing_output = true;
    status.output_changed = false;
    return status;
  }

  const bool force_full_output_write = status.geometry_changed || !workspace.initialized || config_changed;

  // Disabled optimization means the paper-domain raw classification becomes the final output.
  if (!config.optimization_enabled) {
    status.output_changed = status.geometry_changed;
    if (copy_raw_to_output_row_spans(
        workspace.raw_image,
        workspace.threshold_image,
        *workspace.optimized_map_msg,
        workspace.dirty_spans_by_row,
        force_full_output_write))
    {
      status.output_changed = true;
    }

    workspace.initialized = true;
    workspace.last_optimization_enabled = config.optimization_enabled;
    workspace.last_occ_threshold = config.occ_threshold;
    workspace.last_build_sigma_s = config.sigma_s;
    workspace.last_build_sigma_r = config.sigma_r;
    workspace.last_build_dilation_radius_cells = config.dilation_kernel_radius_cells;
    status.reused_existing_output = !status.output_changed;
    return status;
  }

  ensure_spatial_kernel_cache(workspace, config.sigma_s);
  ensure_spatial_linear_offset_cache(workspace);
  ensure_range_lut_cache(workspace, config.sigma_r);
  ensure_dilation_kernel_cache(workspace, config.dilation_kernel_radius_cells);
  ensure_dilation_linear_offset_cache(workspace);

  // Dirty spans expand by each stage's support so partial recomputation remains exact.
  if (config_changed) {
    assign_full_row_spans(workspace.filter_spans_by_row, workspace.width, workspace.height);
    assign_full_row_spans(workspace.dilation_spans_by_row, workspace.width, workspace.height);
  } else {
    build_expanded_row_spans(
      workspace.dirty_spans_by_row,
      workspace.cached_filter_radius,
      workspace.cached_filter_radius,
      workspace.width,
      workspace.height,
      workspace,
      workspace.filter_spans_by_row);
    build_expanded_row_spans(
      workspace.dirty_spans_by_row,
      workspace.cached_filter_radius + std::max(0, config.dilation_kernel_radius_cells),
      workspace.cached_filter_radius + std::max(0, config.dilation_kernel_radius_cells),
      workspace.width,
      workspace.height,
      workspace,
      workspace.dilation_spans_by_row);
  }

  recompute_filtered_threshold_row_spans(
    workspace.raw_image,
    nullptr,
    &workspace.threshold_image,
    workspace.filter_spans_by_row,
    workspace);

  status.output_changed = status.geometry_changed;
  if (recompute_dilation_row_spans(
      workspace.raw_image,
      workspace.threshold_image,
      *workspace.optimized_map_msg,
      workspace.dilation_spans_by_row,
      workspace,
      force_full_output_write))
  {
    status.output_changed = true;
  }

  workspace.initialized = true;
  workspace.last_optimization_enabled = config.optimization_enabled;
  workspace.last_occ_threshold = config.occ_threshold;
  workspace.last_build_sigma_s = config.sigma_s;
  workspace.last_build_sigma_r = config.sigma_r;
  workspace.last_build_dilation_radius_cells = config.dilation_kernel_radius_cells;
  status.reused_existing_output = !status.output_changed;
  return status;
}

DecisionMapResult build_decision_map(
  const OccupancyGrid2d & raw_map,
  const DecisionMapConfig & config)
{
  // Convenience wrapper for callers that do not need incremental workspace reuse.
  DecisionMapWorkspace workspace;
  (void)build_decision_map(raw_map, config, workspace);

  DecisionMapResult result;
  result.raw_image = workspace.raw_image;
  if (config.optimization_enabled) {
    result.filtered_image = bilateral_filter(result.raw_image, config.sigma_s, config.sigma_r);
  } else {
    result.filtered_image.resize(result.raw_image.data.size(), 0.0F);
    for (std::size_t i = 0; i < result.raw_image.data.size(); ++i) {
      result.filtered_image[i] = static_cast<float>(result.raw_image.data[i]);
    }
  }
  result.threshold_image = workspace.threshold_image;
  result.optimized_map_msg = *workspace.optimized_map_msg;
  result.optimized_image.width = result.raw_image.width;
  result.optimized_image.height = result.raw_image.height;
  result.optimized_image.data.resize(result.optimized_map_msg.data.size(), PAPER_UNKNOWN);
  for (std::size_t i = 0; i < result.optimized_map_msg.data.size(); ++i) {
    result.optimized_image.data[i] =
      occupancy_cost_to_paper_output_value(result.optimized_map_msg.data[i]);
  }
  result.decision_map = OccupancyGrid2d(result.optimized_map_msg);
  return result;
}

}  // namespace frontier_exploration_ros2
