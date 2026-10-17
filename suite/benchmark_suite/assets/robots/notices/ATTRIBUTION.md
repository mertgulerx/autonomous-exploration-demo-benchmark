# Robot preview assets

These PNG previews were rendered from the benchmark robot descriptions and the
vendors' licensed visual meshes. Studio lighting, camera position and framing were
added for the campaign selector. Display finishes were compared with manufacturer
photographs; the review references and presentation adjustments are recorded in
`../sources.json`. Source meshes and reference photographs are not bundled with
the GUI. The model authors' licenses and notices continue to apply.

The ROSbot previews use Clearpath's coloured OAK-D Pro visual in place of the same camera's uncoloured
Luxonis STL, preserving its mounting frame. These presentation changes do not
modify the simulation descriptions, collision geometry or sensor configuration.

| Previews | Source | License |
|---|---|---|
| MOGI | BME MOGI ROS 2 navigation course, David Dudas and contributors | Apache-2.0 |
| TurtleBot3 Burger, Waffle Pi | ROBOTIS, `turtlebot3_simulations` | Apache-2.0 |
| TurtleBot4 Standard | Clearpath Robotics, `turtlebot4`; iRobot Corporation, `create3_sim` | Apache-2.0; BSD-3-Clause |
| Husarion ROSbot 3 PRO | Husarion, `rosbot_ros` and `husarion_components_description`; Luxonis, `depthai-ros`; Clearpath, `clearpath_common` | Apache-2.0; MIT; BSD-3-Clause |
| Husarion Lynx | Husarion, `husarion_ugv_ros` and `husarion_components_description` | Apache-2.0 |

Full license texts and installed-package copyright notices are in this directory.
`../sources.json` records package versions, source revisions, input mesh hashes,
description hashes and rendering settings for each preview.

Upstream projects:

- https://github.com/MOGI-ROS/Week-7-8-ROS2-Navigation
- https://github.com/ROBOTIS-GIT/turtlebot3_simulations
- https://github.com/turtlebot/turtlebot4
- https://github.com/iRobotEducation/create3_sim
- https://github.com/husarion/rosbot_ros
- https://github.com/husarion/husarion_components_description
- https://github.com/husarion/husarion_ugv_ros
- https://github.com/luxonis/depthai-ros
- https://github.com/clearpathrobotics/clearpath_common
