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

#include <gtest/gtest.h>

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "frontier_exploration_ros2/decision_map.hpp"
#include "frontier_exploration_ros2/frontier_policy.hpp"
#include "frontier_exploration_ros2/frontier_search.hpp"
#include "frontier_exploration_ros2/mrtsp_ordering.hpp"

namespace frontier_exploration_ros2
{
namespace
{

geometry_msgs::msg::Pose make_pose(double x, double y)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.orientation.w = 1.0;
  return pose;
}

nav_msgs::msg::OccupancyGrid build_grid(int width, int height, int default_value)
{
  nav_msgs::msg::OccupancyGrid msg;
  msg.info.width = static_cast<uint32_t>(width);
  msg.info.height = static_cast<uint32_t>(height);
  msg.info.resolution = 1.0;
  msg.info.origin.position.x = 0.0;
  msg.info.origin.position.y = 0.0;
  msg.info.origin.orientation.w = 1.0;
  msg.data.assign(static_cast<std::size_t>(width * height), static_cast<int8_t>(default_value));
  return msg;
}

void set_cell(nav_msgs::msg::OccupancyGrid & msg, int x, int y, int value)
{
  const int width = static_cast<int>(msg.info.width);
  msg.data[static_cast<std::size_t>(y * width + x)] = static_cast<int8_t>(value);
}

void set_rect(nav_msgs::msg::OccupancyGrid & msg, int x0, int y0, int x1, int y1, int value)
{
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      set_cell(msg, x, y, value);
    }
  }
}

void set_origin(nav_msgs::msg::OccupancyGrid & msg, double x, double y)
{
  msg.info.origin.position.x = x;
  msg.info.origin.position.y = y;
}

void copy_grid_into(
  nav_msgs::msg::OccupancyGrid & target,
  const nav_msgs::msg::OccupancyGrid & source,
  int x_offset,
  int y_offset)
{
  const int source_width = static_cast<int>(source.info.width);
  const int source_height = static_cast<int>(source.info.height);
  const int target_width = static_cast<int>(target.info.width);
  for (int y = 0; y < source_height; ++y) {
    for (int x = 0; x < source_width; ++x) {
      target.data[static_cast<std::size_t>((y + y_offset) * target_width + (x + x_offset))] =
        source.data[static_cast<std::size_t>(y * source_width + x)];
    }
  }
}

std::vector<float> reference_bilateral_filter(
  const PaperImage & image,
  double sigma_s,
  double sigma_r)
{
  const double safe_sigma_s = std::max(sigma_s, 1e-6);
  const double safe_sigma_r = std::max(sigma_r, 1e-6);
  const int radius = std::max(1, static_cast<int>(std::ceil(2.0 * safe_sigma_s)));
  const double spatial_denominator = 2.0 * safe_sigma_s * safe_sigma_s;
  const double range_denominator = 2.0 * safe_sigma_r * safe_sigma_r;
  std::vector<float> filtered(image.data.size(), 0.0F);

  for (int y = 0; y < image.height; ++y) {
    const std::size_t row_offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width);
    for (int x = 0; x < image.width; ++x) {
      const std::size_t idx = row_offset + static_cast<std::size_t>(x);
      const uint8_t center_value = image.data[idx];
      float weighted_sum = 0.0F;
      float normalization = 0.0F;

      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          const int sample_x = std::clamp(x + dx, 0, image.width - 1);
          const int sample_y = std::clamp(y + dy, 0, image.height - 1);
          const uint8_t shifted_value = image.at(sample_x, sample_y);
          const double distance_sq = static_cast<double>((dx * dx) + (dy * dy));
          const double diff =
            static_cast<double>(center_value) - static_cast<double>(shifted_value);
          const float weight = static_cast<float>(
            std::exp(-(distance_sq / spatial_denominator)) *
            std::exp(-((diff * diff) / range_denominator)));
          weighted_sum += weight * static_cast<float>(shifted_value);
          normalization += weight;
        }
      }

      filtered[idx] =
        normalization > std::numeric_limits<float>::epsilon() ?
        (weighted_sum / normalization) :
        static_cast<float>(center_value);
    }
  }

  return filtered;
}

PaperImage reference_threshold_image(
  const PaperImage & raw_image,
  const std::vector<float> & filtered_image)
{
  PaperImage threshold_image;
  threshold_image.width = raw_image.width;
  threshold_image.height = raw_image.height;
  threshold_image.data.resize(raw_image.data.size(), PAPER_UNKNOWN);
  const float free_threshold =
    static_cast<float>((static_cast<int>(PAPER_FREE) + static_cast<int>(PAPER_UNKNOWN)) / 2.0);

  for (std::size_t i = 0; i < raw_image.data.size(); ++i) {
    threshold_image.data[i] =
      filtered_image[i] >= free_threshold ? PAPER_FREE : PAPER_UNKNOWN;
    if (raw_image.data[i] == PAPER_OCCUPIED) {
      threshold_image.data[i] = PAPER_OCCUPIED;
    }
  }

  return threshold_image;
}

PaperImage reference_dilated_image(
  const PaperImage & raw_image,
  const PaperImage & threshold_image,
  int dilation_radius)
{
  PaperImage optimized_image;
  optimized_image.width = raw_image.width;
  optimized_image.height = raw_image.height;
  optimized_image.data.resize(raw_image.data.size(), PAPER_UNKNOWN);

  const int safe_radius = std::max(0, dilation_radius);
  for (int y = 0; y < raw_image.height; ++y) {
    const std::size_t row_offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(raw_image.width);
    for (int x = 0; x < raw_image.width; ++x) {
      const std::size_t idx = row_offset + static_cast<std::size_t>(x);
      if (raw_image.data[idx] == PAPER_OCCUPIED) {
        optimized_image.data[idx] = PAPER_OCCUPIED;
        continue;
      }

      uint8_t optimized_value = PAPER_UNKNOWN;
      for (int dy = -safe_radius; dy <= safe_radius; ++dy) {
        for (int dx = -safe_radius; dx <= safe_radius; ++dx) {
          if ((dx * dx) + (dy * dy) > (safe_radius * safe_radius)) {
            continue;
          }

          const int sample_x = x + dx;
          const int sample_y = y + dy;
          if (
            sample_x < 0 || sample_y < 0 ||
            sample_x >= raw_image.width || sample_y >= raw_image.height)
          {
            continue;
          }

          if (threshold_image.at(sample_x, sample_y) == PAPER_FREE) {
            optimized_value = PAPER_FREE;
            break;
          }
        }
        if (optimized_value == PAPER_FREE) {
          break;
        }
      }

      optimized_image.data[idx] = optimized_value;
    }
  }

  return optimized_image;
}

std::vector<int8_t> paper_image_to_occupancy_costs(const PaperImage & image)
{
  std::vector<int8_t> data(image.data.size(), -1);
  for (std::size_t i = 0; i < image.data.size(); ++i) {
    if (image.data[i] == PAPER_FREE) {
      data[i] = 0;
    } else if (image.data[i] == PAPER_OCCUPIED) {
      data[i] = 100;
    }
  }
  return data;
}

void expect_workspace_matches_full_result(
  const DecisionMapWorkspace & workspace,
  const DecisionMapResult & full_result)
{
  EXPECT_EQ(workspace.raw_image.data, full_result.raw_image.data);
  EXPECT_EQ(workspace.threshold_image.data, full_result.threshold_image.data);
  EXPECT_EQ(workspace.optimized_map_msg->data, full_result.optimized_map_msg.data);
}

TEST(DecisionMapTests, OccupancyToPaperMappingPreservesThresholdSemantics)
{
  auto map_msg = build_grid(6, 1, -1);
  set_cell(map_msg, 0, 0, -1);
  set_cell(map_msg, 1, 0, 0);
  set_cell(map_msg, 2, 0, 49);
  set_cell(map_msg, 3, 0, 50);
  set_cell(map_msg, 4, 0, 100);
  set_cell(map_msg, 5, 0, -42);

  const auto paper_image = occupancy_grid_to_paper_image(OccupancyGrid2d(map_msg), 50);

  const std::vector<uint8_t> expected{
    PAPER_UNKNOWN,
    PAPER_FREE,
    PAPER_FREE,
    PAPER_OCCUPIED,
    PAPER_OCCUPIED,
    PAPER_UNKNOWN,
  };
  EXPECT_EQ(paper_image.data, expected);
}

TEST(DecisionMapTests, OptimizationReducesFrontierCountWhilePreservingOccupiedCells)
{
  auto map_msg = build_grid(12, 12, -1);
  set_rect(map_msg, 2, 2, 9, 9, 0);
  set_cell(map_msg, 4, 4, -1);
  set_cell(map_msg, 7, 4, -1);
  set_cell(map_msg, 4, 7, -1);
  set_cell(map_msg, 7, 7, -1);
  set_cell(map_msg, 6, 6, 100);

  auto costmap_msg = build_grid(12, 12, 0);
  const OccupancyGrid2d raw_map(map_msg);
  const OccupancyGrid2d costmap(costmap_msg);

  FrontierSearchOptions options;
  options.min_frontier_size_cells = 1;

  const auto raw_frontiers = get_frontier(
    make_pose(3.0, 3.0),
    raw_map,
    costmap,
    std::nullopt,
    0.0,
    false,
    options);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;
  const auto decision_map_result = build_decision_map(raw_map, config);

  const auto decision_frontiers = get_frontier(
    make_pose(3.0, 3.0),
    decision_map_result.decision_map,
    costmap,
    std::nullopt,
    0.0,
    false,
    options);

  EXPECT_GT(raw_frontiers.frontiers.size(), decision_frontiers.frontiers.size());
  EXPECT_EQ(decision_map_result.decision_map.getCost(6, 6), 100);
}

TEST(DecisionMapTests, OptimizationKeepsNarrowDoorwayTraversable)
{
  auto map_msg = build_grid(12, 12, -1);
  set_rect(map_msg, 2, 2, 9, 5, 0);
  for (int x = 2; x <= 9; ++x) {
    if (x == 6) {
      continue;
    }
    set_cell(map_msg, x, 6, 100);
  }
  set_cell(map_msg, 6, 6, 0);
  set_cell(map_msg, 6, 7, 0);

  auto costmap_msg = build_grid(12, 12, 0);
  const OccupancyGrid2d raw_map(map_msg);
  const OccupancyGrid2d costmap(costmap_msg);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;
  const auto decision_map_result = build_decision_map(raw_map, config);

  EXPECT_EQ(decision_map_result.decision_map.getCost(6, 6), 0);
  EXPECT_EQ(decision_map_result.decision_map.getCost(5, 6), 100);

  FrontierSearchOptions options;
  options.min_frontier_size_cells = 1;
  const auto decision_frontiers = get_frontier(
    make_pose(6.0, 3.0),
    decision_map_result.decision_map,
    costmap,
    std::nullopt,
    0.0,
    false,
    options);
  EXPECT_FALSE(decision_frontiers.frontiers.empty());
}

TEST(DecisionMapTests, WorkspaceReusesExistingOutputForIdenticalMapContent)
{
  auto map_msg = build_grid(10, 10, -1);
  set_rect(map_msg, 2, 2, 7, 7, 0);

  const OccupancyGrid2d raw_map(map_msg);
  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  const auto first = build_decision_map(raw_map, config, workspace);
  const auto second = build_decision_map(raw_map, config, workspace);

  EXPECT_TRUE(first.output_changed);
  EXPECT_FALSE(first.reused_existing_output);
  EXPECT_FALSE(second.output_changed);
  EXPECT_TRUE(second.reused_existing_output);
}

TEST(DecisionMapTests, FilterThresholdAndDilationMatchReferenceMathAtInteriorAndBorders)
{
  auto map_msg = build_grid(14, 14, -1);
  set_rect(map_msg, 2, 2, 11, 11, 0);
  set_rect(map_msg, 0, 0, 1, 3, 0);
  set_rect(map_msg, 11, 0, 13, 2, 0);
  set_cell(map_msg, 4, 4, -1);
  set_cell(map_msg, 7, 5, -1);
  set_cell(map_msg, 10, 10, -1);
  set_cell(map_msg, 0, 0, 100);
  set_cell(map_msg, 5, 5, 100);
  set_cell(map_msg, 12, 1, 100);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  const OccupancyGrid2d raw_map(map_msg);
  const auto raw_image = occupancy_grid_to_paper_image(raw_map, config.occ_threshold);
  const auto expected_filtered = reference_bilateral_filter(raw_image, config.sigma_s, config.sigma_r);
  const auto expected_threshold = reference_threshold_image(raw_image, expected_filtered);
  const auto expected_optimized =
    reference_dilated_image(raw_image, expected_threshold, config.dilation_kernel_radius_cells);
  const auto expected_costs = paper_image_to_occupancy_costs(expected_optimized);

  const auto result = build_decision_map(raw_map, config);

  EXPECT_EQ(result.raw_image.data, raw_image.data);
  ASSERT_EQ(result.filtered_image.size(), expected_filtered.size());
  for (std::size_t i = 0; i < result.filtered_image.size(); ++i) {
    EXPECT_NEAR(result.filtered_image[i], expected_filtered[i], 1e-4F);
  }
  EXPECT_EQ(result.threshold_image.data, expected_threshold.data);
  EXPECT_EQ(result.optimized_image.data, expected_optimized.data);
  EXPECT_EQ(result.optimized_map_msg.data, expected_costs);
}

TEST(DecisionMapTests, DirtyRegionWorkspaceRecomputeMatchesFullRecompute)
{
  auto first_map_msg = build_grid(12, 12, -1);
  set_rect(first_map_msg, 2, 2, 9, 9, 0);
  set_cell(first_map_msg, 4, 4, -1);
  set_cell(first_map_msg, 6, 6, 100);

  auto second_map_msg = first_map_msg;
  set_cell(second_map_msg, 4, 4, 0);
  set_cell(second_map_msg, 5, 4, 0);
  set_cell(second_map_msg, 5, 5, -1);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status = build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_FALSE(incremental_status.geometry_changed);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, DisjointDirtyRegionsRecomputeMatchesFullRecompute)
{
  auto first_map_msg = build_grid(20, 20, -1);
  set_rect(first_map_msg, 2, 2, 17, 17, 0);
  set_cell(first_map_msg, 4, 4, -1);
  set_cell(first_map_msg, 15, 15, -1);
  set_cell(first_map_msg, 10, 10, 100);

  auto second_map_msg = first_map_msg;
  set_cell(second_map_msg, 3, 4, -1);
  set_cell(second_map_msg, 4, 4, 0);
  set_cell(second_map_msg, 16, 15, 100);
  set_cell(second_map_msg, 15, 14, -1);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status = build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_FALSE(incremental_status.geometry_changed);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, BorderDirtySpansRecomputeMatchesFullRecompute)
{
  auto first_map_msg = build_grid(16, 16, -1);
  set_rect(first_map_msg, 1, 1, 14, 14, 0);
  set_cell(first_map_msg, 0, 0, 100);
  set_cell(first_map_msg, 15, 15, 100);

  auto second_map_msg = first_map_msg;
  set_cell(second_map_msg, 0, 1, 0);
  set_cell(second_map_msg, 1, 0, -1);
  set_cell(second_map_msg, 15, 14, 0);
  set_cell(second_map_msg, 14, 15, -1);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status = build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_FALSE(incremental_status.geometry_changed);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, LargeMapIdenticalContentReusesExistingOutputAcrossMultipleChunks)
{
  auto map_msg = build_grid(70, 70, -1);
  set_rect(map_msg, 3, 3, 66, 66, 0);
  set_cell(map_msg, 10, 10, 100);
  set_cell(map_msg, 40, 40, -1);
  set_cell(map_msg, 55, 20, 100);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  const auto first = build_decision_map(OccupancyGrid2d(map_msg), config, workspace);
  const auto second = build_decision_map(OccupancyGrid2d(map_msg), config, workspace);

  EXPECT_TRUE(first.output_changed);
  EXPECT_FALSE(first.reused_existing_output);
  EXPECT_FALSE(second.output_changed);
  EXPECT_TRUE(second.reused_existing_output);
  EXPECT_EQ(workspace.chunk_cols, 3);
  EXPECT_EQ(workspace.chunk_rows, 3);
}

TEST(DecisionMapTests, UnknownObstacleChunkIgnoresSmallObstacleJitter)
{
  auto first_map_msg = build_grid(32, 32, -1);

  auto second_map_msg = first_map_msg;
  set_rect(second_map_msg, 2, 2, 4, 4, 100);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto second_status =
    build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);

  EXPECT_FALSE(second_status.geometry_changed);
  EXPECT_EQ(
    second_status.geometry_transition,
    DecisionMapGeometryTransition::SameGeometry);
  EXPECT_TRUE(second_status.reused_existing_output);
  EXPECT_EQ(second_status.dirty_chunks, 0U);
}

TEST(DecisionMapTests, UnknownFreeChunkStillRebuildsOnSemanticChange)
{
  auto first_map_msg = build_grid(32, 32, -1);
  set_rect(first_map_msg, 4, 4, 20, 20, 0);

  auto second_map_msg = first_map_msg;
  set_cell(second_map_msg, 21, 10, 100);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto second_status =
    build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_FALSE(second_status.geometry_changed);
  EXPECT_FALSE(second_status.reused_existing_output);
  EXPECT_GT(second_status.dirty_chunks, 0U);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, SingleDirtyChunkOnLargeMapMatchesFullRecompute)
{
  auto first_map_msg = build_grid(64, 64, -1);
  set_rect(first_map_msg, 4, 4, 59, 59, 0);
  set_cell(first_map_msg, 12, 12, -1);
  set_cell(first_map_msg, 40, 40, 100);

  auto second_map_msg = first_map_msg;
  set_cell(second_map_msg, 10, 10, 100);
  set_cell(second_map_msg, 11, 10, -1);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status = build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_FALSE(incremental_status.geometry_changed);
  EXPECT_EQ(workspace.chunk_cols, 2);
  EXPECT_EQ(workspace.chunk_rows, 2);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, ChunkBoundaryChangesMatchFullRecompute)
{
  auto first_map_msg = build_grid(70, 70, -1);
  set_rect(first_map_msg, 2, 2, 67, 67, 0);
  set_cell(first_map_msg, 31, 20, -1);
  set_cell(first_map_msg, 32, 20, -1);
  set_cell(first_map_msg, 20, 31, -1);
  set_cell(first_map_msg, 20, 32, -1);

  auto second_map_msg = first_map_msg;
  set_cell(second_map_msg, 31, 20, 0);
  set_cell(second_map_msg, 32, 20, 100);
  set_cell(second_map_msg, 20, 31, 100);
  set_cell(second_map_msg, 20, 32, 0);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status = build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_FALSE(incremental_status.geometry_changed);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, SameGeometryMapResetMatchesFullRecompute)
{
  auto first_map_msg = build_grid(64, 64, -1);
  set_rect(first_map_msg, 3, 3, 60, 60, 0);
  set_cell(first_map_msg, 8, 8, 100);
  set_cell(first_map_msg, 48, 48, 100);

  auto second_map_msg = build_grid(64, 64, -1);
  set_rect(second_map_msg, 20, 20, 43, 43, 0);
  set_cell(second_map_msg, 30, 30, 100);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status = build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_FALSE(incremental_status.geometry_changed);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, SemanticallyInconsistentShiftFallsBackToFullRebuild)
{
  auto first_map_msg = build_grid(40, 40, -1);
  set_rect(first_map_msg, 4, 4, 18, 18, 0);
  set_rect(first_map_msg, 22, 22, 30, 30, 100);

  auto second_map_msg = build_grid(40, 40, -1);
  second_map_msg.info.origin.position.x = 5.0;
  second_map_msg.info.origin.position.y = -2.0;
  set_rect(second_map_msg, 20, 4, 35, 18, 0);
  set_rect(second_map_msg, 4, 22, 16, 34, 100);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status = build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_TRUE(incremental_status.geometry_changed);
  EXPECT_EQ(
    incremental_status.geometry_transition,
    DecisionMapGeometryTransition::FullRebuildFallback);
  EXPECT_DOUBLE_EQ(workspace.origin_x, 5.0);
  EXPECT_DOUBLE_EQ(workspace.origin_y, -2.0);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, MostlyUnknownShiftUsesOverlapReuseAndMatchesFullRecompute)
{
  auto first_map_msg = build_grid(40, 40, -1);
  set_rect(first_map_msg, 10, 10, 14, 14, 0);
  set_cell(first_map_msg, 12, 12, 100);

  auto second_map_msg = first_map_msg;
  second_map_msg.info.origin.position.x = 2.0;
  second_map_msg.info.origin.position.y = -1.0;
  set_cell(second_map_msg, 18, 16, 100);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status =
    build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_TRUE(incremental_status.geometry_changed);
  EXPECT_EQ(
    incremental_status.geometry_transition,
    DecisionMapGeometryTransition::OverlapReuse);
  EXPECT_GT(incremental_status.dirty_chunks, 0U);
  EXPECT_LT(incremental_status.dirty_chunks, incremental_status.total_chunks);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, SmallOriginJitterDoesNotForceGeometryReset)
{
  auto first_map_msg = build_grid(40, 40, -1);
  set_rect(first_map_msg, 4, 4, 35, 35, 0);
  set_cell(first_map_msg, 12, 12, 100);

  auto second_map_msg = first_map_msg;
  second_map_msg.info.origin.position.x = 0.2;
  second_map_msg.info.origin.position.y = -0.2;

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status =
    build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);

  EXPECT_FALSE(incremental_status.geometry_changed);
  EXPECT_EQ(
    incremental_status.geometry_transition,
    DecisionMapGeometryTransition::SameGeometry);
  EXPECT_TRUE(incremental_status.reused_existing_output);
}

TEST(DecisionMapTests, SameOriginGrowthReusesOverlapAndMatchesFullRecompute)
{
  auto first_map_msg = build_grid(40, 40, -1);
  set_rect(first_map_msg, 4, 4, 35, 35, 0);
  set_cell(first_map_msg, 12, 12, 100);
  set_cell(first_map_msg, 28, 18, 100);

  auto second_map_msg = build_grid(56, 52, -1);
  copy_grid_into(second_map_msg, first_map_msg, 0, 0);
  set_rect(second_map_msg, 42, 6, 51, 15, 0);
  set_rect(second_map_msg, 8, 44, 20, 49, 0);
  set_cell(second_map_msg, 48, 10, 100);
  set_cell(second_map_msg, 10, 46, 100);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status =
    build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_TRUE(incremental_status.geometry_changed);
  EXPECT_EQ(
    incremental_status.geometry_transition,
    DecisionMapGeometryTransition::GrowthReuse);
  EXPECT_LT(incremental_status.dirty_chunks, incremental_status.total_chunks);
  EXPECT_EQ(workspace.width, 56);
  EXPECT_EQ(workspace.height, 52);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, ShiftedGrowthReusesOverlapAndMatchesFullRecompute)
{
  auto first_map_msg = build_grid(40, 40, -1);
  set_rect(first_map_msg, 3, 3, 36, 34, 0);
  set_cell(first_map_msg, 14, 14, 100);
  set_cell(first_map_msg, 24, 18, -1);

  auto second_map_msg = build_grid(54, 50, -1);
  set_origin(second_map_msg, -4.0, -3.0);
  copy_grid_into(second_map_msg, first_map_msg, 4, 3);
  set_rect(second_map_msg, 0, 0, 2, 20, 0);
  set_rect(second_map_msg, 41, 42, 52, 48, 0);
  set_cell(second_map_msg, 1, 1, 100);
  set_cell(second_map_msg, 50, 46, 100);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status =
    build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_TRUE(incremental_status.geometry_changed);
  EXPECT_EQ(
    incremental_status.geometry_transition,
    DecisionMapGeometryTransition::GrowthReuse);
  EXPECT_DOUBLE_EQ(workspace.origin_x, -4.0);
  EXPECT_DOUBLE_EQ(workspace.origin_y, -3.0);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, GrowthReuseRecomputesPreviousBorderHaloExactly)
{
  auto first_map_msg = build_grid(32, 32, -1);
  set_rect(first_map_msg, 26, 12, 30, 18, 0);

  auto second_map_msg = build_grid(40, 32, -1);
  copy_grid_into(second_map_msg, first_map_msg, 0, 0);
  set_rect(second_map_msg, 32, 12, 35, 18, 0);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status =
    build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_TRUE(incremental_status.geometry_changed);
  EXPECT_EQ(
    incremental_status.geometry_transition,
    DecisionMapGeometryTransition::GrowthReuse);
  expect_workspace_matches_full_result(workspace, full_result);
  EXPECT_EQ(
    workspace.optimized_map_msg->data[static_cast<std::size_t>(15 * 40 + 31)],
    full_result.optimized_map_msg.data[static_cast<std::size_t>(15 * 40 + 31)]);
}

TEST(DecisionMapTests, GrowthReuseKeepsLaterOverlapEditsIncremental)
{
  auto first_map_msg = build_grid(40, 40, -1);
  set_rect(first_map_msg, 4, 4, 35, 35, 0);
  set_cell(first_map_msg, 12, 12, 100);

  auto grown_map_msg = build_grid(56, 52, -1);
  copy_grid_into(grown_map_msg, first_map_msg, 0, 0);
  set_rect(grown_map_msg, 42, 6, 51, 15, 0);
  set_cell(grown_map_msg, 48, 10, 100);

  auto edited_grown_map_msg = grown_map_msg;
  set_cell(edited_grown_map_msg, 14, 14, 100);
  set_cell(edited_grown_map_msg, 15, 14, -1);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  (void)build_decision_map(OccupancyGrid2d(grown_map_msg), config, workspace);
  const auto incremental_status =
    build_decision_map(OccupancyGrid2d(edited_grown_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(edited_grown_map_msg), config);

  EXPECT_FALSE(incremental_status.geometry_changed);
  EXPECT_EQ(
    incremental_status.geometry_transition,
    DecisionMapGeometryTransition::SameGeometry);
  EXPECT_GT(incremental_status.dirty_chunks, 0U);
  EXPECT_LT(incremental_status.dirty_chunks, incremental_status.total_chunks);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, ShrinkCanUseOverlapReuseWhenSemanticallyConsistent)
{
  auto first_map_msg = build_grid(40, 40, -1);
  set_rect(first_map_msg, 4, 4, 35, 35, 0);

  auto second_map_msg = build_grid(32, 32, -1);
  set_rect(second_map_msg, 2, 2, 29, 29, 0);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status =
    build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_TRUE(incremental_status.geometry_changed);
  EXPECT_EQ(
    incremental_status.geometry_transition,
    DecisionMapGeometryTransition::OverlapReuse);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, ResolutionChangeForcesFullRebuildFallback)
{
  auto first_map_msg = build_grid(40, 40, -1);
  set_rect(first_map_msg, 4, 4, 35, 35, 0);

  auto second_map_msg = first_map_msg;
  second_map_msg.info.resolution = 0.5;

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto incremental_status =
    build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(second_map_msg), config);

  EXPECT_TRUE(incremental_status.geometry_changed);
  EXPECT_EQ(
    incremental_status.geometry_transition,
    DecisionMapGeometryTransition::FullRebuildFallback);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, OccThresholdChangeForcesFullRawRebuildAndMatchesFullRecompute)
{
  auto map_msg = build_grid(48, 48, -1);
  set_rect(map_msg, 4, 4, 43, 43, 0);
  set_cell(map_msg, 10, 10, 55);
  set_cell(map_msg, 20, 20, 65);
  set_cell(map_msg, 30, 30, 75);

  DecisionMapConfig first_config;
  first_config.optimization_enabled = true;
  first_config.occ_threshold = 50;
  first_config.sigma_s = 2.0;
  first_config.sigma_r = 30.0;
  first_config.dilation_kernel_radius_cells = 1;

  DecisionMapConfig second_config = first_config;
  second_config.occ_threshold = 70;

  DecisionMapWorkspace workspace;
  (void)build_decision_map(OccupancyGrid2d(map_msg), first_config, workspace);
  const auto incremental_status = build_decision_map(OccupancyGrid2d(map_msg), second_config, workspace);
  const auto full_result = build_decision_map(OccupancyGrid2d(map_msg), second_config);

  EXPECT_FALSE(incremental_status.geometry_changed);
  expect_workspace_matches_full_result(workspace, full_result);
}

TEST(DecisionMapTests, ChunkStatusCountsReflectHitDirtyAndGeometryResetCases)
{
  auto first_map_msg = build_grid(64, 64, -1);
  set_rect(first_map_msg, 4, 4, 59, 59, 0);
  set_cell(first_map_msg, 12, 12, 100);
  set_cell(first_map_msg, 40, 40, -1);

  auto second_map_msg = first_map_msg;
  set_cell(second_map_msg, 10, 10, 100);

  auto third_map_msg = second_map_msg;
  third_map_msg.info.origin.position.x = 1.0;

  auto growth_map_msg = build_grid(72, 72, -1);
  copy_grid_into(growth_map_msg, second_map_msg, 0, 0);
  set_rect(growth_map_msg, 66, 4, 70, 12, 0);

  DecisionMapConfig config;
  config.optimization_enabled = true;
  config.occ_threshold = 50;
  config.sigma_s = 2.0;
  config.sigma_r = 30.0;
  config.dilation_kernel_radius_cells = 1;

  DecisionMapWorkspace workspace;
  const auto first_status = build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto second_status = build_decision_map(OccupancyGrid2d(first_map_msg), config, workspace);
  const auto third_status = build_decision_map(OccupancyGrid2d(second_map_msg), config, workspace);
  const auto fourth_status = build_decision_map(OccupancyGrid2d(third_map_msg), config, workspace);
  DecisionMapWorkspace growth_workspace;
  (void)build_decision_map(OccupancyGrid2d(second_map_msg), config, growth_workspace);
  const auto fifth_status =
    build_decision_map(OccupancyGrid2d(growth_map_msg), config, growth_workspace);

  EXPECT_EQ(first_status.total_chunks, 4U);
  EXPECT_EQ(first_status.dirty_chunks, 4U);
  EXPECT_TRUE(first_status.geometry_changed);
  EXPECT_EQ(
    first_status.geometry_transition,
    DecisionMapGeometryTransition::FullRebuildFallback);

  EXPECT_EQ(second_status.total_chunks, 4U);
  EXPECT_EQ(second_status.dirty_chunks, 0U);
  EXPECT_FALSE(second_status.geometry_changed);
  EXPECT_TRUE(second_status.reused_existing_output);
  EXPECT_EQ(
    second_status.geometry_transition,
    DecisionMapGeometryTransition::SameGeometry);

  EXPECT_EQ(third_status.total_chunks, 4U);
  EXPECT_EQ(third_status.dirty_chunks, 1U);
  EXPECT_FALSE(third_status.geometry_changed);
  EXPECT_EQ(
    third_status.geometry_transition,
    DecisionMapGeometryTransition::SameGeometry);

  EXPECT_EQ(fourth_status.total_chunks, 4U);
  EXPECT_EQ(fourth_status.dirty_chunks, 4U);
  EXPECT_TRUE(fourth_status.geometry_changed);
  EXPECT_EQ(
    fourth_status.geometry_transition,
    DecisionMapGeometryTransition::OverlapReuse);

  EXPECT_EQ(fifth_status.total_chunks, 9U);
  EXPECT_GT(fifth_status.dirty_chunks, 0U);
  EXPECT_LT(fifth_status.dirty_chunks, fifth_status.total_chunks);
  EXPECT_TRUE(fifth_status.geometry_changed);
  EXPECT_EQ(
    fifth_status.geometry_transition,
    DecisionMapGeometryTransition::GrowthReuse);
}

TEST(MrtspOrderingTests, GreedyOrderingCanPreferHighGainFrontierOverCloserCandidate)
{
  const FrontierCandidate near_small_gain{
    {1.0, 0.0},
    {1.0, 0.0},
    {1, 0},
    {1, 0},
    {1.0, 0.0},
    std::nullopt,
    1};
  const FrontierCandidate farther_high_gain{
    {2.0, 0.0},
    {2.0, 0.0},
    {2, 0},
    {2, 0},
    {2.0, 0.0},
    std::nullopt,
    10};

  RobotState robot_state;
  robot_state.position = {0.0, 0.0};
  robot_state.yaw = 0.0;

  CostWeights weights;
  weights.distance_wd = 1.0;
  weights.gain_ws = 1.0;

  const auto matrix = build_cost_matrix(
    {near_small_gain, farther_high_gain},
    robot_state,
    weights,
    0.0,
    100.0,
    1.0);
  const auto order_first = greedy_mrtsp_order(matrix);
  const auto order_second = greedy_mrtsp_order(matrix);

  ASSERT_EQ(order_first.size(), 2U);
  EXPECT_EQ(order_first, order_second);
  EXPECT_EQ(order_first.front(), 1U);
}

TEST(MrtspOrderingTests, FrontierDispatchPointFallsBackToCenterPointWithoutGoalPoint)
{
  const FrontierCandidate candidate_with_goal_point{
    {4.0, 4.0},
    {3.0, 4.0},
    {3, 4},
    {3, 4},
    {3.0, 4.0},
    std::make_optional(std::pair<double, double>{2.0, 4.0}),
    8};
  const FrontierCandidate mrtsp_candidate{
    {4.0, 4.0},
    {3.0, 4.0},
    {3, 4},
    {3, 4},
    {3.0, 4.0},
    std::nullopt,
    8};

  EXPECT_EQ(frontier_position(candidate_with_goal_point), (std::pair<double, double>{2.0, 4.0}));
  EXPECT_EQ(frontier_position(mrtsp_candidate), (std::pair<double, double>{3.0, 4.0}));
}

TEST(MrtspOrderingTests, MinGoalDistancePrefersFarEnoughGoalPointWhenAvailable)
{
  auto map_msg = build_grid(8, 8, 0);
  auto costmap_msg = build_grid(8, 8, 0);
  const OccupancyGrid2d occupancy_map(map_msg);
  const OccupancyGrid2d costmap(costmap_msg);

  FrontierCache frontier_cache(8, 8);
  std::vector<FrontierPoint *> frontier_points{
    frontier_cache.getPoint(1, 1),
    frontier_cache.getPoint(1, 2),
    frontier_cache.getPoint(2, 1),
    frontier_cache.getPoint(2, 2),
    frontier_cache.getPoint(3, 1),
    frontier_cache.getPoint(3, 2),
    frontier_cache.getPoint(4, 1),
    frontier_cache.getPoint(4, 2),
  };

  FrontierSearchOptions options;
  options.min_frontier_size_cells = 1;
  options.candidate_min_goal_distance_m = 2.0;

  const auto candidate = build_frontier_candidate(
    frontier_points,
    {1, 1},
    occupancy_map,
    costmap,
    std::nullopt,
    frontier_cache,
    make_pose(1.5, 1.5),
    0.0,
    options);
  ASSERT_TRUE(candidate.has_value());
  ASSERT_TRUE(candidate->goal_point.has_value());
  const double dx = candidate->goal_point->first - 1.5;
  const double dy = candidate->goal_point->second - 1.5;
  EXPECT_GE(std::hypot(dx, dy), 2.0);
}

TEST(MrtspOrderingTests, CenterPointTieBreakMatchesSortedCellOrder)
{
  auto map_msg = build_grid(8, 8, 0);
  auto costmap_msg = build_grid(8, 8, 0);
  const OccupancyGrid2d occupancy_map(map_msg);
  const OccupancyGrid2d costmap(costmap_msg);

  FrontierCache frontier_cache(8, 8);
  std::vector<FrontierPoint *> frontier_points{
    frontier_cache.getPoint(3, 2),
    frontier_cache.getPoint(2, 3),
    frontier_cache.getPoint(1, 2),
    frontier_cache.getPoint(2, 1),
  };

  FrontierSearchOptions options;
  options.min_frontier_size_cells = 1;

  const auto candidate = build_frontier_candidate(
    frontier_points,
    {3, 2},
    occupancy_map,
    costmap,
    std::nullopt,
    frontier_cache,
    make_pose(0.0, 0.0),
    0.0,
    options);

  ASSERT_TRUE(candidate.has_value());
  EXPECT_EQ(candidate->center_cell, (std::pair<int, int>{2, 1}));
  EXPECT_EQ(candidate->center_point, (std::pair<double, double>{2.5, 1.5}));
}

TEST(MrtspOrderingTests, CandidateCarriesApproxRobotCenterDistanceFromSelectedCenterFrontier)
{
  auto map_msg = build_grid(8, 8, 0);
  auto costmap_msg = build_grid(8, 8, 0);
  const OccupancyGrid2d occupancy_map(map_msg);
  const OccupancyGrid2d costmap(costmap_msg);

  FrontierCache frontier_cache(8, 8);
  FrontierPoint * point_a = frontier_cache.getPoint(3, 2);
  FrontierPoint * point_b = frontier_cache.getPoint(2, 3);
  FrontierPoint * point_c = frontier_cache.getPoint(1, 2);
  FrontierPoint * point_d = frontier_cache.getPoint(2, 1);
  point_a->robot_distance_m = 4.0;
  point_b->robot_distance_m = 3.0;
  point_c->robot_distance_m = 2.0;
  point_d->robot_distance_m = 1.0;

  std::vector<FrontierPoint *> frontier_points{point_a, point_b, point_c, point_d};

  FrontierSearchOptions options;
  options.min_frontier_size_cells = 1;

  const auto candidate = build_frontier_candidate(
    frontier_points,
    {3, 2},
    occupancy_map,
    costmap,
    std::nullopt,
    frontier_cache,
    make_pose(0.0, 0.0),
    0.0,
    options);

  ASSERT_TRUE(candidate.has_value());
  ASSERT_TRUE(candidate->robot_center_distance_m.has_value());
  EXPECT_EQ(candidate->center_cell, (std::pair<int, int>{2, 1}));
  EXPECT_DOUBLE_EQ(*candidate->robot_center_distance_m, 1.0);
}

TEST(MrtspOrderingTests, StartCostUsesApproxRobotCenterDistanceForPathAndTranslation)
{
  constexpr double kHalfPi = 1.57079632679489661923;
  const FrontierCandidate candidate{
    {0.0, 1.0},
    {0.0, 1.0},
    {0, 1},
    {0, 1},
    {0.0, 1.0},
    std::make_optional(std::pair<double, double>{0.0, 1.0}),
    4,
    std::nullopt,
    5.0};

  RobotState robot_state;
  robot_state.position = {0.0, 0.0};
  robot_state.yaw = 0.0;

  const double path_cost = initial_frontier_path_cost(
    robot_state.position,
    candidate,
    candidate.start_world_point,
    0.0);
  const double motion_time_cost = lower_bound_time_cost(
    robot_state,
    candidate.center_point,
    candidate.robot_center_distance_m,
    2.0,
    1.0);

  EXPECT_DOUBLE_EQ(path_cost, 5.0);
  EXPECT_NEAR(motion_time_cost, 2.5 + kHalfPi, 1e-9);
}

TEST(MrtspOrderingTests, StartCostFallsBackToEuclideanWithoutApproxRobotCenterDistance)
{
  constexpr double kHalfPi = 1.57079632679489661923;
  const FrontierCandidate candidate{
    {0.0, 1.0},
    {0.0, 1.0},
    {0, 1},
    {0, 1},
    {0.0, 1.0},
    std::make_optional(std::pair<double, double>{0.0, 1.0}),
    4};

  RobotState robot_state;
  robot_state.position = {0.0, 0.0};
  robot_state.yaw = 0.0;

  const double path_cost = initial_frontier_path_cost(
    robot_state.position,
    candidate,
    candidate.start_world_point,
    0.0);
  const double motion_time_cost = lower_bound_time_cost(
    robot_state,
    candidate.center_point,
    candidate.robot_center_distance_m,
    2.0,
    1.0);

  EXPECT_DOUBLE_EQ(path_cost, 1.0);
  EXPECT_NEAR(motion_time_cost, 0.5 + kHalfPi, 1e-9);
}

}  // namespace
}  // namespace frontier_exploration_ros2
