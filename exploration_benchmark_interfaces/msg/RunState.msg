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
string robot_id
RunSpec spec
# STARTING, READY, RUNNING, STOPPING, FINISHED
string phase
# completed, time_limit, manual_stop, cancelled, startup_failure, runtime_failure
string outcome
string detail
int32 domain_id
string gazebo_partition
string topic_root
string result_directory
float64 simulation_start
float64 wall_start_unix
int32 native_result_code
