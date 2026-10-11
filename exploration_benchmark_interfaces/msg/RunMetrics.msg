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

string run_id
float64 cpu_percent
float64 rss_mib
float64 cpu_average_percent
float64 rss_average_mib
float64 cpu_seconds
float64 distance_m
# Elapsed wall/simulation time since first raw-odometry movement; zero before it.
float64 wall_elapsed
float64 simulation_elapsed
bool motion_started
# Explorer activation to first movement; increases while waiting, then freezes.
float64 pre_motion_wall_elapsed
float64 pre_motion_simulation_elapsed
float64 rtf
string[] stream_gaps
string[] quality_flags
bool frozen
# Original SLAM map against the selected world's fixed reference.
# Percent is NaN when unavailable; status is pending/valid/stale/unavailable.
float64 coverage_percent
float64 coverage_covered_m2
float64 coverage_target_m2
string coverage_status
float64 coverage_map_stamp
float64 coverage_map_age
string[] coverage_quality_flags
CoverageMilestone[] coverage_milestones
