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

"""Offline reference generation and cache preparation; never run by a worker.

Supports the included worlds' SDF primitives, nested models and COLLADA/STL/OBJ
collision meshes. Unsupported frame semantics fail explicitly, rather than
silently creating an incorrect research denominator.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import quote, unquote, urlparse

import numpy as np
import yaml

from .configuration import scenario
from .coverage import (
    METRIC_VERSION,
    TARGET_MARGIN,
    Reference,
    erode,
    reference_directory,
    reference_file,
    sha256,
)


def pose(element):
    p = element.find("pose")
    if p is not None and (
        p.get("relative_to") or p.get("degrees") or p.get("rotation_format")
    ):
        raise ValueError(
            "Resolve relative_to/non-RPY poses before reference generation"
        )
    xyzrpy = [float(v) for v in p.text.split()] if p is not None and p.text else [0] * 6
    if len(xyzrpy) != 6 or not np.all(np.isfinite(xyzrpy)):
        raise ValueError("Invalid SDF pose")
    x, y, z, r, p, a = xyzrpy
    cr, sr, cp, sp, ca, sa = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(a),
        math.sin(a),
    )
    t = np.eye(4)
    t[:3, :3] = [
        [ca * cp, ca * sp * sr - sa * cr, ca * sp * cr + sa * sr],
        [sa * cp, sa * sp * sr + ca * cr, sa * sp * cr - ca * sr],
        [-sp, cp * sr, cp * cr],
    ]
    t[:3, 3] = [x, y, z]
    return t


class Geometry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.assets, self.meshes, self.records = {}, {}, []
        self.model_paths = list((self.root / "simulation/worlds").glob("*/models"))

    def asset(self, path, fuel=None):
        path = Path(path).resolve()
        identity = (
            ("fuel:" + fuel[0] + "#" + str(path.relative_to(fuel[1])))
            if fuel
            else str(path.relative_to(self.root))
        )
        self.assets[identity] = sha256(path)
        return path

    def resolve(self, uri, directory, fuel=None):
        if uri.startswith("model://"):
            relative = uri[8:]
            candidates = [base / relative for base in self.model_paths]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                raise ValueError(f"Missing model resource: {uri}")
            return path, None
        if uri.startswith("https://"):
            parsed = urlparse(uri)
            parts = [unquote(v) for v in parsed.path.split("/") if v]
            if len(parts) < 5 or not parts[4].isdigit():
                raise ValueError(
                    f"Pin the Fuel version before generating references: {uri}"
                )
            if len(parts) != 5 and not (len(parts) >= 7 and parts[5] == "files"):
                raise ValueError(f"Malformed versioned Fuel URI: {uri}")
            base_uri = "https://" + parsed.netloc + "/" + "/".join(parts[:5])
            cache = (
                Path.home() / ".gz/fuel" / parsed.netloc / parts[1].lower() / "models"
            )
            names = [parts[3].lower(), quote(parts[3].lower(), safe="")]
            folder = next(
                (
                    cache / name / parts[4]
                    for name in names
                    if (cache / name / parts[4]).is_dir()
                ),
                None,
            )
            if folder is None:
                raise ValueError(
                    f"Missing Fuel model. Run: gz fuel download -u '{base_uri}'"
                )
            relative = (
                "/".join(parts[6:]) if len(parts) > 5 and parts[5] == "files" else ""
            )
            return folder / relative, (base_uri, folder)
        return (directory / uri).resolve(), fuel

    def mesh(self, geometry, directory, fuel):
        import trimesh

        node = geometry[0]
        if node.tag == "plane":
            return None
        if node.tag == "box":
            return trimesh.creation.box(
                extents=[float(v) for v in node.findtext("size").split()]
            )
        if node.tag == "cylinder":
            return trimesh.creation.cylinder(
                radius=float(node.findtext("radius")),
                height=float(node.findtext("length")),
                sections=64,
            )
        if node.tag == "sphere":
            return trimesh.creation.icosphere(
                subdivisions=3, radius=float(node.findtext("radius"))
            )
        if node.tag != "mesh" or node.find("submesh") is not None:
            raise ValueError(f"Unsupported collision geometry: {node.tag}")
        path, mesh_fuel = self.resolve(node.findtext("uri").strip(), directory, fuel)
        self.asset(path, mesh_fuel)
        if path not in self.meshes:
            scene = trimesh.load(path, force="scene", process=False)
            mesh = (
                scene.to_mesh()
                if hasattr(scene, "to_mesh")
                else scene.dump(concatenate=True)
            )
            if path.suffix.lower() == ".dae":
                asset = ET.parse(path).getroot().find("{*}asset")
                unit = asset.find("{*}unit")
                mesh.apply_scale(
                    float(unit.get("meter", "1")) if unit is not None else 1
                )
                axis = asset.findtext("{*}up_axis", "Z_UP")
                if axis != "Z_UP":
                    raise ValueError(f"Convert non-Z_UP COLLADA mesh first: {path}")
            self.meshes[path] = mesh
        result = self.meshes[path].copy()
        result.apply_scale([float(v) for v in node.findtext("scale", "1 1 1").split()])
        return result

    def walk(self, node, parent, directory, fuel=None, inherited_static=False):
        if node.findall("frame") or node.get("placement_frame"):
            raise ValueError(
                "Resolve SDF named/placement frames before reference generation"
            )
        transform = parent @ pose(node)
        static = inherited_static or node.findtext("static", "false").strip() in (
            "1",
            "true",
        )
        for include in node.findall("include"):
            path, included_fuel = self.resolve(
                include.findtext("uri").strip(), directory, fuel
            )
            if path.is_dir():
                model_config = path / "model.config"
                self.asset(model_config, included_fuel)
                candidates = ET.parse(model_config).getroot().findall("sdf")
                candidates = [
                    s
                    for s in candidates
                    if tuple(map(int, s.get("version").split("."))) <= (1, 11)
                ]
                if not candidates:
                    raise ValueError(
                        f"No Harmonic-compatible model SDF in {model_config}"
                    )
                selected = max(
                    candidates,
                    key=lambda s: tuple(map(int, s.get("version").split("."))),
                )
                path = path / selected.text.strip()
            self.asset(path, included_fuel)
            model = ET.parse(path).getroot().find("model")
            yield from self.walk(
                model, transform @ pose(include), path.parent, included_fuel, static
            )
        for child in node.findall("model"):
            yield from self.walk(child, transform, directory, fuel, static)
        for link in node.findall("link"):
            for collision in link.findall("collision"):
                if not static:
                    self.records.append(
                        dict(
                            model=node.get("name"),
                            collision=collision.get("name"),
                            ignored="dynamic",
                        )
                    )
                    continue
                mesh = self.mesh(collision.find("geometry"), directory, fuel)
                if mesh is not None:
                    mesh.apply_transform(transform @ pose(link) @ pose(collision))
                    self.records.append(
                        dict(
                            model=node.get("name"),
                            collision=collision.get("name"),
                            bounds=mesh.bounds.tolist(),
                        )
                    )
                    yield mesh


def clip(vertices, level, above):
    result = []
    for previous, current in zip(vertices[-1:] + vertices[:-1], vertices):
        a, b = (previous[2] >= level), (current[2] >= level)
        if not above:
            a, b = not a, not b
        if a != b:
            result.append(
                previous
                + (current - previous)
                * ((level - previous[2]) / (current[2] - previous[2]))
            )
        if b:
            result.append(current)
    return result


def footprint(mesh, slab):
    """Project slab-clipped surfaces and fill section interiors by even/odd rings.

    Projecting only triangle surfaces misses the interior of tall boxes; sections
    fill those solids while retaining holes (e.g. the warehouse wall enclosure).
    """
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union, polygonize
    import trimesh

    low, high = slab
    if mesh.bounds[1, 2] <= low or mesh.bounds[0, 2] >= high:
        return Polygon()
    polygons = []
    for triangle in mesh.triangles:
        vertices = clip(list(triangle), low, True)
        if vertices:
            vertices = clip(vertices, high, False)
        if len(vertices) >= 3:
            polygon = Polygon(np.asarray(vertices)[:, :2])
            if polygon.area > 1e-12:
                polygons.append(polygon.buffer(0))
    levels = np.unique(np.round(mesh.vertices[:, 2], 4))
    levels = np.concatenate(([low], levels[(levels > low) & (levels < high)], [high]))
    for height in (levels[1:] + levels[:-1]) / 2:
        segments = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, height])[
            :, :, :2
        ]
        if not len(segments):
            continue
        # Resolve all crossings before polygonization. Following mesh paths at
        # shared vertices can connect unrelated shells and fill whole rooms.
        lines = [LineString(np.round(segment, 8)) for segment in segments]
        for region in polygonize(unary_union(lines)):
            point = region.representative_point()
            a, b = segments[:, 0], segments[:, 1]
            crossing = (a[:, 1] > point.y) != (b[:, 1] > point.y)
            a, b = a[crossing], b[crossing]
            intersections = a[:, 0] + (point.y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (
                b[:, 1] - a[:, 1]
            )
            if len(np.unique(np.round(intersections[intersections > point.x], 8))) % 2:
                polygons.append(region)
    return unary_union(polygons)


def generate(root, world, directory):
    import scipy
    import shapely
    import trimesh
    from scipy.ndimage import label
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely import contains_xy

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings_file = directory / "scenarios.yaml"
    settings = yaml.safe_load(settings_file.read_text())
    config, res, slab = (
        settings["scenarios"][world],
        settings["resolution"],
        settings["body_height_slab"],
    )
    selected = scenario(root, world)
    geometry = Geometry(root)
    world_file = geometry.asset(selected["world_file"])
    geometry.asset(settings_file)
    urdf = (
        Path(root)
        / "simulation/Week-7-8-ROS2-Navigation/bme_ros2_navigation/urdf/mogi_bot.urdf"
    )
    geometry.asset(urdf)
    polygons = [
        footprint(mesh, slab)
        for mesh in geometry.walk(
            ET.parse(world_file).getroot().find("world"), np.eye(4), world_file.parent
        )
    ]
    obstacles = unary_union(polygons)
    x0, y0, x1, y1 = config["bounds"]
    width, height = round((x1 - x0) / res), round((y1 - y0) / res)
    x, y = np.meshgrid(
        x0 + (np.arange(width) + 0.5) * res, y0 + (np.arange(height) + 0.5) * res
    )
    collision = contains_xy(obstacles.buffer(1e-9), x, y)
    roi = unary_union([Polygon(p) for p in config["inclusion_polygons"]])
    exclusions = unary_union([Polygon(p) for p in config["exclusion_polygons"]])
    inclusion = contains_xy(roi.difference(exclusions).buffer(1e-9), x, y)
    components, _ = label(inclusion & ~collision)
    sx, sy = selected["spawn"][:2]
    component = components[int((sy - y0) / res), int((sx - x0) / res)]
    if not component:
        raise ValueError("Spawn lies outside reference free space; review geometry/ROI")
    unreachable = inclusion & ~collision & (components != component)
    inclusion &= ~unreachable
    target = erode(inclusion & ~collision, TARGET_MARGIN / res)
    count = int(target.sum())
    grid_file = directory / (world + ".npz")
    np.savez_compressed(
        grid_file,
        collision=collision.astype(np.uint8),
        inclusion=inclusion.astype(np.uint8),
    )
    metadata = dict(
        id=f"{world}_coverage_v{METRIC_VERSION}",
        metric_version=METRIC_VERSION,
        target_margin_m=TARGET_MARGIN,
        world=world,
        spawn=selected["spawn"],
        resolution=res,
        origin=[x0, y0, 0.0],
        width=width,
        height=height,
        target_cells=count,
        target_area_m2=count * res**2,
        grid_file=grid_file.name,
        grid_sha256=sha256(grid_file),
        source_sha256=geometry.assets,
        generation=dict(
            body_height_slab=slab,
            settings=config,
            connectivity="4-connected free floor from spawn; not a footprint planner",
            unreachable_free_cells=int(unreachable.sum()),
            collision_records=geometry.records,
            generator_sha256=sha256(__file__),
            tool_versions=dict(
                numpy=np.__version__,
                scipy=scipy.__version__,
                shapely=shapely.__version__,
                trimesh=trimesh.__version__,
            ),
        ),
    )
    reference_file(directory, world).write_text(json.dumps(metadata, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    image = np.full((height, width, 3), 0.9)
    image[collision] = [0.15, 0.15, 0.15]
    image[target] = [0.55, 0.85, 0.65]
    image[unreachable] = [0.9, 0.6, 0.35]
    ax.imshow(image, origin="lower", extent=[x0, x1, y0, y1], interpolation="nearest")
    ax.plot(sx, sy, "r+", markersize=12)
    ax.set(
        title=f"{world.title()}: {count*res**2:.2f} m² target (green)",
        xlabel="Gazebo world x (m)",
        ylabel="Gazebo world y (m)",
    )
    fig.tight_layout()
    fig.savefig(directory / (world + "_reference.png"), dpi=130)
    plt.close(fig)
    print(f"{world}: {count} cells, {count*res**2:.2f} m²")


def overlay(run_directory):
    """Overlay the unchanged SLAM map on its saved world reference for review."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.transforms import Affine2D

    run = Path(run_directory)
    manifest = json.loads((run / "manifest.json").read_text())
    coverage = manifest["coverage"]
    reference = Reference(
        run / "coverage_reference" / Path(coverage["reference"]["metadata_file"]).name
    )
    mapping = json.loads((run / "final_map.json").read_text())
    info, pose = mapping["info"], mapping["info"]["origin"]
    q = pose["orientation"]
    origin_yaw = math.atan2(
        2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2)
    )
    alignment = coverage["alignment"]["world_from_map"]
    r, x0, y0 = reference.resolution, *reference.origin[:2]
    height, width = reference.target.shape
    background = np.full((height, width, 3), 0.93)
    background[reference.target] = [0.55, 0.85, 0.65]
    background[reference.collision] = [0.15, 0.15, 0.15]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        background,
        origin="lower",
        extent=[x0, x0 + width * r, y0, y0 + height * r],
        interpolation="nearest",
    )
    values = np.asarray(mapping["data"]).reshape(info["height"], info["width"])
    image = np.zeros((*values.shape, 4))
    image[values == 0] = [0.1, 0.35, 1.0, 0.35]
    image[values == 100] = [1.0, 0.1, 0.1, 0.7]
    transform = (
        Affine2D()
        .rotate(origin_yaw)
        .translate(pose["position"]["x"], pose["position"]["y"])
        .rotate(alignment[2])
        .translate(alignment[0], alignment[1])
    )
    ax.imshow(
        image,
        origin="lower",
        extent=[
            0,
            info["width"] * info["resolution"],
            0,
            info["height"] * info["resolution"],
        ],
        transform=transform + ax.transData,
        interpolation="nearest",
    )
    ax.set(
        xlim=(x0, x0 + width * r),
        ylim=(y0, y0 + height * r),
        xlabel="Gazebo world x (m)",
        ylabel="Gazebo world y (m)",
        title=f"{reference.metadata['world']}: reference green; SLAM free blue / occupied red",
    )
    fig.tight_layout()
    destination = run / "coverage_alignment.png"
    fig.savefig(destination, dpi=130)
    plt.close(fig)
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "check", "fetch", "overlay"])
    parser.add_argument("--run", help="Saved run directory for overlay")
    parser.add_argument(
        "--world", choices=["bookstore", "corridor", "warehouse"], action="append"
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--directory")
    args = parser.parse_args()
    if args.command == "overlay":
        if not args.run:
            parser.error("overlay requires --run")
        overlay(args.run)
        return
    directory = Path(args.directory) if args.directory else reference_directory()
    try:
        for world in args.world or ["bookstore", "corridor", "warehouse"]:
            if args.command == "generate":
                generate(args.project_root, world, directory)
                continue
            reference = Reference(reference_file(directory, world))
            if args.command == "fetch":
                uris = {
                    key[5:].split("#", 1)[0]
                    for key in reference.metadata["source_sha256"]
                    if key.startswith("fuel:")
                }
                for uri in sorted(uris):
                    subprocess.run(["gz", "fuel", "download", "-u", uri], check=True)
            reference.validate_scenario(
                args.project_root, scenario(args.project_root, world)
            )
            print(f"Verified {world}: {reference.area:.2f} m² ({reference.identity})")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
