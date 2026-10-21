# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One analysis tick of the frontier insight node: a re-analysis of the explorer's inputs with the explorer's own
# analysis code, not the explorer's internal state.

uint8 CHUNK_HIT = 0
uint8 CHUNK_REBUILT = 1
uint8 CHUNK_RESET = 2

# stamp: simulation time when the inputs were taken; frame_id: global_frame.
std_msgs/Header header
# Analyses published since the node started (1-based).
uint64 tick
# Ticks skipped since the node started because the previous analysis was still running.
uint64 skipped_ticks
# Wall time of this analysis (ms).
float64 analysis_wall_ms
# Stamp of the SLAM map that was analysed.
builtin_interfaces/Time map_stamp
# Robot pose used (global_frame; m, rad).
float64 robot_x
float64 robot_y
float64 robot_yaw
# mrtsp/dp, mrtsp/dp-greedy-fallback, mrtsp/greedy or mrtsp/greedy-fallback; empty without candidates.
string mode
# Frontiers found on the SLAM map before decision-map optimization (a count only).
uint32 raw_frontier_count
FrontierInsightCandidate[] candidates
# Candidate ids in route order; its first entry is the analysis target.
uint16[] route
# M(0, r_1) + sum of M(r_k, r_k+1) over the route; NaN without a route.
float64 route_cost

# Decision-map chunk cache of this node (same algorithm as the explorer, its own cache history).
uint16 chunk_size_cells
float64 chunk_resolution
# Decision-map origin in global_frame (m; the explorer ignores origin rotation, so none is given). Chunk
# (column, row) covers cells from column * chunk_size_cells and row * chunk_size_cells, clipped to the map.
float64 chunk_origin_x
float64 chunk_origin_y
# Decision-map size (cells), for clipping the last column and row of chunks.
uint32 chunk_map_width
uint32 chunk_map_height
uint16 chunk_columns
uint16 chunk_rows
# One CHUNK_* value per chunk, row-major (row 0 at the origin).
uint8[] chunk_states
uint32 chunk_hits
uint32 chunk_rebuilt
uint32 chunk_resets
# same, overlap_reuse or full_rebuild.
string geometry_transition
bool output_reused
bool output_changed
