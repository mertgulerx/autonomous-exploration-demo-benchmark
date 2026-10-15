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

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any
import xml.etree.ElementTree as ET

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector


def origin(element: Any) -> Any:
    if element is None:
        return Matrix.Identity(4)
    translation = Vector(tuple(map(float, element.get("xyz", "0 0 0").split())))
    rotation = Euler(tuple(map(float, element.get("rpy", "0 0 0").split())), "XYZ")
    return Matrix.Translation(translation) @ rotation.to_matrix().to_4x4()


def material(name: str, rgba: list[float]) -> Any:
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = rgba
    shader.inputs["Roughness"].default_value = 0.48
    return result


def import_mesh(path: Path) -> list[Any]:
    before = set(bpy.data.objects)
    if path.suffix.lower() == ".dae":
        bpy.ops.wm.collada_import(filepath=str(path))
    elif path.suffix.lower() == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    elif path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif path.suffix.lower() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported preview mesh: {path}")
    bpy.context.view_layer.update()
    imported = set(bpy.data.objects) - before
    meshes = [obj for obj in imported if obj.type == "MESH"]
    transforms = {obj: obj.matrix_world.copy() for obj in meshes}
    for obj in meshes:
        obj.parent = None
        obj.matrix_world = transforms[obj]
    for obj in imported - set(meshes):
        bpy.data.objects.remove(obj, do_unlink=True)
    return meshes


def load_robot(description: Path, sources: Path) -> list[Any]:
    root = ET.parse(description).getroot()
    children = {joint.find("child").get("link") for joint in root.findall("joint")}
    transforms = {
        link.get("name"): Matrix.Identity(4)
        for link in root.findall("link")
        if link.get("name") not in children
    }
    pending = list(root.findall("joint"))
    while pending:
        advanced = []
        for joint in pending:
            parent = joint.find("parent").get("link")
            if parent in transforms:
                transforms[joint.find("child").get("link")] = transforms[parent] @ origin(
                    joint.find("origin")
                )
                advanced.append(joint)
        if not advanced:
            raise ValueError("Robot description contains an unresolved joint tree")
        pending = [joint for joint in pending if joint not in advanced]
    materials = {
        entry.get("name"): material(
            entry.get("name"), list(map(float, entry.find("color").get("rgba").split()))
        )
        for entry in root.findall("material")
        if entry.find("color") is not None
    }
    objects = []
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh")
            matrix = transforms[link.get("name")] @ origin(visual.find("origin"))
            if mesh is not None:
                imported = import_mesh(sources / mesh.get("filename"))
                scale = tuple(map(float, mesh.get("scale", "1 1 1").split()))
                matrix = matrix @ Matrix.Diagonal((*scale, 1.0))
            else:
                if geometry.find("box") is not None:
                    bpy.ops.mesh.primitive_cube_add(size=1)
                    bpy.context.object.scale = tuple(
                        map(float, geometry.find("box").get("size").split())
                    )
                elif geometry.find("cylinder") is not None:
                    cylinder = geometry.find("cylinder")
                    bpy.ops.mesh.primitive_cylinder_add(
                        vertices=64,
                        radius=float(cylinder.get("radius")),
                        depth=float(cylinder.get("length")),
                    )
                elif geometry.find("sphere") is not None:
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        segments=32,
                        ring_count=16,
                        radius=float(geometry.find("sphere").get("radius")),
                    )
                else:
                    raise ValueError(f"Unsupported visual geometry on {link.get('name')}")
                imported = [bpy.context.object]
                bpy.context.view_layer.update()
            entry = visual.find("material")
            selected = None
            if entry is not None:
                selected = materials.get(entry.get("name"))
                if entry.find("color") is not None:
                    selected = material(
                        link.get("name"), list(map(float, entry.find("color").get("rgba").split()))
                    )
            for obj in imported:
                obj.matrix_world = matrix @ obj.matrix_world
                if selected is not None:
                    obj.data.materials.clear()
                    obj.data.materials.append(selected)
                if not obj.data.materials:
                    obj.data.materials.append(material("Neutral", [0.35, 0.37, 0.38, 1.0]))
                objects.append(obj)
    bpy.context.view_layer.update()
    return objects


def aim(obj: Any, target: Any) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def render(description: Path, sources: Path, target: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 100
    scene.view_settings.view_transform = "AgX"
    objects = load_robot(description, sources)
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    size = maximum - minimum
    print(
        description.stem, "native dimensions", tuple(round(value, 4) for value in size), flush=True
    )
    scale = 2 / max(size)
    normalization = Matrix.Scale(scale, 4) @ Matrix.Translation(
        Vector((-(minimum.x + maximum.x) / 2, -(minimum.y + maximum.y) / 2, -minimum.z))
    )
    for obj in objects:
        obj.matrix_world = normalization @ obj.matrix_world
    height = size.z * scale
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.004))
    floor = bpy.context.object
    floor.is_shadow_catcher = True
    floor.data.materials.append(material("Ground", [0.85, 0.81, 0.75, 1.0]))
    scene.world = bpy.data.worlds.new("Studio")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.72, 0.76, 0.82, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45
    for location, energy, size in [
        ((3, -4, 6), 700, 4),
        ((-3, -2, 4), 400, 4),
        ((1, 4, 5), 600, 3),
    ]:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        aim(light, (0, 0, height / 2))
    bpy.ops.object.camera_add(location=(3.5, -4.5, 3.0 + height / 2))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 3.5
    aim(camera, (0, 0, height / 2))
    scene.camera = camera
    bpy.context.view_layer.update()
    projected = [
        world_to_camera_view(scene, camera, obj.matrix_world @ Vector(corner))
        for obj in objects
        for corner in obj.bound_box
    ]
    camera.data.ortho_scale *= max(
        (max(p.x for p in projected) - min(p.x for p in projected)) / 0.84,
        (max(p.y for p in projected) - min(p.y for p in projected)) / 0.82,
    )
    scene.render.filepath = str(target)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render packaged native robot thumbnails with Blender."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(args.source.glob("*.urdf")):
        if args.robot and source.stem != args.robot:
            continue
        target = args.output / (source.stem + ".png")
        render(source, args.source, target)
        metadata = json.loads(source.with_suffix(".json").read_text())
        metadata.update(
            preview_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            renderer="Blender " + bpy.app.version_string,
            camera="Orthographic, front three-quarter",
            resolution=[640, 480],
            samples=32,
        )
        records.append(metadata)
    if (args.source / "notices").exists():
        shutil.copytree(args.source / "notices", args.output / "notices", dirs_exist_ok=True)
    (args.output / "sources.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
