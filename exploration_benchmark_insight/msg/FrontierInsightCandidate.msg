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

# One frontier candidate of an analysis tick (the explorer's debug analysis on the optimized decision map).
# Positions are in the message's global_frame; ids restart at 0 every tick.

uint16 id
# Dispatch point: the candidate's goal point, else its centre point (m).
float64 x
float64 y
# Cluster centroid (m).
float64 centroid_x
float64 centroid_y
# Cluster size P_j (cells).
int32 size_cells
# G_j = P_j ^ weight_gain_ws.
float64 gain
# Start path term d_0j (m; negative inside sensor_effective_range_m).
float64 path_cost
# Start time term t_j (s).
float64 time_cost
# Start cost M(0, j) = weight_distance_wd * d_0j / G_j + t_j / sqrt(G_j).
float64 start_cost
# Straight-line distance from the analysed robot pose to the dispatch point (m).
float64 distance_m
# 1-based order by start cost (finite costs first, then cost, then id).
uint16 score_order
# 1-based rank in the DP candidate pool; 0 outside the pool or without DP.
uint16 pool_rank
# 1-based position on the analysed route (DP order, or the greedy order without DP); 0 off the route.
uint16 route_position
# 1-based position in the greedy order over all candidates.
uint16 greedy_position
