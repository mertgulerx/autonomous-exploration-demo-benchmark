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

"""Offline collision-envelope inspection for native URDF robot descriptions."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def transform(origin):
    from math import cos, sin

    matrix = np.eye(4)
    if origin is None:
        return matrix
    matrix[:3, 3] = list(map(float, origin.get("xyz", "0 0 0").split()))
    roll, pitch, yaw = map(float, origin.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = (
        cos(roll),
        sin(roll),
        cos(pitch),
        sin(pitch),
        cos(yaw),
        sin(yaw),
    )
    matrix[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return matrix


def collision_envelope(description, lookup):
    import trimesh

    root = ET.fromstring(description)
    joints = root.findall("joint")
    children = {joint.find("child").get("link") for joint in joints}
    roots = [
        link.get("name")
        for link in root.findall("link")
        if link.get("name") not in children
    ]
    if len(roots) != 1:
        raise ValueError(f"Native URDF must have one root, found {roots}")
    frames = {roots[0]: np.eye(4)}
    pending = list(joints)
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in frames]
        if not ready:
            raise ValueError("Native URDF contains a cycle or disconnected joints")
        for joint in ready:
            frames[joint.find("child").get("link")] = frames[
                joint.find("parent").get("link")
            ] @ transform(joint.find("origin"))
            pending.remove(joint)
    bounds = []
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            geometry = list(collision.find("geometry"))[0]
            if geometry.tag == "box":
                mesh = trimesh.creation.box(
                    extents=list(map(float, geometry.get("size").split()))
                )
            elif geometry.tag == "cylinder":
                mesh = trimesh.creation.cylinder(
                    radius=float(geometry.get("radius")),
                    height=float(geometry.get("length")),
                )
            elif geometry.tag == "sphere":
                mesh = trimesh.creation.icosphere(radius=float(geometry.get("radius")))
            elif geometry.tag == "mesh":
                uri = geometry.get("filename")
                if not uri.startswith("package://"):
                    raise ValueError(
                        f"Offline native mesh requires a package URI: {uri}"
                    )
                package, relative = uri[10:].split("/", 1)
                filename = Path(lookup(package)) / relative
                scene = trimesh.load(filename, force="scene", process=False)
                mesh = (
                    scene.to_mesh()
                    if hasattr(scene, "to_mesh")
                    else scene.dump(concatenate=True)
                )
                if filename.suffix == ".dae":
                    asset = ET.parse(filename).getroot().find("{*}asset")
                    unit = asset.find("{*}unit")
                    mesh.apply_scale(
                        float(unit.get("meter", "1")) if unit is not None else 1
                    )
                    if asset.findtext("{*}up_axis", "Z_UP") != "Z_UP":
                        raise ValueError(
                            f"Review non-Z_UP native collision mesh: {filename}"
                        )
                mesh.apply_scale(
                    list(map(float, geometry.get("scale", "1 1 1").split()))
                )
            else:
                raise ValueError(f"Unsupported native collision: {geometry.tag}")
            mesh.apply_transform(
                frames[link.get("name")] @ transform(collision.find("origin"))
            )
            bounds.append(mesh.bounds)
    if not bounds:
        raise ValueError("Native robot has no collision geometry")
    low = np.min([item[0] for item in bounds], axis=0)
    high = np.max([item[1] for item in bounds], axis=0)
    return dict(
        root_frame=roots[0],
        lower=low.tolist(),
        upper=high.tolist(),
        nominal_footprint=[
            [low[0], low[1]],
            [low[0], high[1]],
            [high[0], high[1]],
            [high[0], low[1]],
        ],
        joint_pose="Zero positions; review operating stance and moving-link envelope separately",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("description", type=Path)
    args = parser.parse_args()
    from ament_index_python.packages import get_package_share_directory

    print(
        json.dumps(
            collision_envelope(
                args.description.read_text(), get_package_share_directory
            ),
            indent=2,
        )
    )
