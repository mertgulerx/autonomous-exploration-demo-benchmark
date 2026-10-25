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

"""``benchmark_reference import``: a confirmed world_collection world enters the
registry.

Offline, from a checkout (docs/plans/worlds/I2-references-and-scenarios.md §3.3-§3.9):
it checks the world's files against ``world.yaml``, writes the per-world coverage
settings file ``<id>_scenario.yaml``, builds the reference, validates every spawn,
runs the descriptor adapter (``tools/world_integration/reference_descriptors.py``, in
the suite's development Python) and prints the registry entry. Without ``--write``
every file goes to ``--directory`` only; with it the files are staged in the system
temporary directory, copied into the checkout's coverage directory (never over an
existing file), the would-be registry is loaded with every pin, and only then is the
entry appended to ``worlds.yaml`` (``_write``). A failing spawn is reported, never
moved.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .coverage import Reference, reference_file
from .parameters import UniqueKeyLoader
from .worlds import (
    DESCRIPTORS_SUFFIX,
    REFERENCE_DIRECTORY,
    REGISTRY_FILE,
    ROBOT_CATALOG_FILE,
    Descriptors,
    WorldRegistry,
    file_sha256,
)

# T0's grid and slab (worldgen raster.RESOLUTION, MARGIN_M, SLAB), equal to
# scenarios.yaml.
RESOLUTION = 0.05
SLAB = [0.03, 0.35]
GRID_MARGIN_M = 1.0
SPAWN_Z = 0.10  # the scenario height of Corridor and Warehouse (floor top at z = 0)
CLEARANCE_M = 0.5  # T0 check 3
MAX_SDF_VERSION = (1, 11)
GEOMETRY_TAGS = ("box", "cylinder", "sphere", "plane", "mesh", "polyline")
DESCRIPTOR_TOOL = "tools/world_integration/reference_descriptors.py"
DESCRIPTOR_PYTHON = ".venv-suite/bin/python"
DESCRIPTOR_TIMEOUT_S = 7200
# The registry carries only the mover keys the runtime checks against world.yaml (I5).
MOVING_KEYS = ("system", "format_version", "velocity_mode", "topic", "count")
# Shorter display labels for long family titles
# (docs/suite-gui-design/WORLD_SELECTION.md).
LABEL_SHORTER = (
    ("recursive backtracker", "backtracker"),
    (" per room", ""),
    ("Warehouse aisles ", "Aisles "),
    ("Walled forest ", "Forest "),
    (" trunks per m²", " trunks/m²"),
    ("Walled urban blocks", "Urban blocks"),
    (" nominal blocks", " blocks"),
)


def label_rule(title):
    """The registry label of a ``world.yaml`` title: comma-separated parts joined by
    `` · ``, long family words shortened (``--label`` overrides it)."""
    label = " · ".join(part.strip() for part in title.split(","))
    for old, new in LABEL_SHORTER:
        label = label.replace(old, new)
    return label


def t0_bounds(bounds):
    """``[x0, y0, x1, y1]`` of T0's grid around ``review/world.json`` ``bounds``
    (``[xmin, xmax, ymin, ymax]``): worldgen ``raster.Grid.around``, so the reference's
    cell centres coincide with T0's."""
    xmin, xmax, ymin, ymax = (float(v) for v in bounds)
    h, margin = RESOLUTION, GRID_MARGIN_M
    x0 = round(math.floor(round((xmin - margin) / h, 6)) * h, 6)
    y0 = round(math.floor(round((ymin - margin) / h, 6)) * h, 6)
    nx = math.ceil(round((xmax + margin - x0) / h, 6))
    ny = math.ceil(round((ymax + margin - y0) / h, 6))
    return [x0, y0, round(x0 + nx * h, 6), round(y0 + ny * h, 6)]


def _yaml_float(value):
    """A YAML float with a decimal point and no exponent (the registry's rule)."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value {value}")
    text = repr(value)
    return text if "e" not in text and "." in text else f"{value:.10f}".rstrip("0")


class CollectionWorld:
    """A world folder of ``world_collection/worlds``, hash-checked against its
    world.yaml."""

    def __init__(self, root, folder):
        self.root = Path(root).resolve()
        self.folder = (self.root / folder).resolve()
        if not self.folder.is_relative_to(self.root) or not self.folder.is_dir():
            raise ValueError(f"No world folder in the checkout: {folder}")
        self.name = self.folder.name
        self.id = self.name.lower()
        self.yaml_path = self.folder / "world.yaml"
        try:
            self._read()
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            # A malformed world names the file and the missing or wrong key.
            raise ValueError(f"{self.folder}: malformed world files: {exc!r}") from exc

    def _read(self):
        try:
            self.document = yaml.load(
                self.yaml_path.read_bytes(), Loader=UniqueKeyLoader
            )
        except yaml.YAMLError as exc:
            raise ValueError(f"{self.yaml_path}: {exc}") from exc
        doc = self.document
        if not isinstance(doc, dict):
            raise ValueError(f"{self.yaml_path}: expected a mapping")
        if doc.get("status") != "confirmed":
            raise ValueError(
                f"{self.name}: status {doc.get('status')!r}, not confirmed"
            )
        self.sdf = self.folder / f"{self.name}.sdf"
        hashes = doc.get("hashes") or {}
        if file_sha256(self.sdf) != hashes.get("sdf"):
            raise ValueError(f"{self.name}: the SDF differs from world.yaml hashes.sdf")
        world_json = self.folder / "review/world.json"
        data = world_json.read_bytes()
        if hashlib.sha256(data).hexdigest() != hashes.get("world_json"):
            raise ValueError(
                f"{self.name}: review/world.json differs from hashes.world_json"
            )
        self.bounds = t0_bounds(json.loads(data)["bounds"])
        element = ET.parse(self.sdf).getroot().find("world")
        if element is None:
            raise ValueError(f"{self.sdf}: no <world> element")
        name = element.get("name")
        if name != self.id:
            raise ValueError(f"{self.name}: SDF world name {name!r} is not {self.id!r}")
        self.tier = doc["tier"]
        # R-2: a family only for tier F (MO1/MO2 carry their own id as family).
        self.family = doc.get("family") if self.tier == "F" else None
        self.spawns = [
            dict(id=str(s["id"]), x=float(s["x"]), y=float(s["y"]), yaw=float(s["yaw"]))
            for s in doc["spawns"]
        ]
        if not self.spawns:
            raise ValueError(f"{self.name}: world.yaml has no spawns")
        moving = doc.get("moving_objects")
        self.movers = [o["id"] for o in moving["objects"]] if moving else []
        self.moving = {key: moving[key] for key in MOVING_KEYS} if moving else None
        if moving and moving["count"] != len(self.movers):
            raise ValueError(
                f"{self.name}: moving_objects count differs from its objects"
            )

    def relative(self, path):
        return str(Path(path).relative_to(self.root))

    def scenario(self, file_name):
        """The per-world coverage settings file (I2 §3.3): one keyed entry, so the
        builder's parser is unchanged."""
        doc = self.document
        polygons = doc.get("inclusion")
        if not polygons:
            raise ValueError(f"{self.name}: world.yaml has no inclusion polygon")
        entry = dict(
            bounds=self.bounds,
            inclusion_polygons=[[[float(x), float(y)] for x, y in p] for p in polygons],
            exclusion_polygons=[],
            review=f"{doc['id']} {doc['title']}: inclusion polygon of world.yaml; "
            "bounds = T0 grid extent",
            source=dict(
                world_yaml=self.relative(self.yaml_path),
                worldgen_version=doc.get("worldgen_version"),
            ),
        )
        header = (
            f"# Coverage settings of {self.id}, written by benchmark_reference import\n"
            "# (exploration_benchmark_runner/world_import.py); hashed into "
            f"{file_name.replace('_scenario.yaml', '_v2.json')}.\n"
        )
        body = yaml.safe_dump(
            dict(
                resolution=RESOLUTION, body_height_slab=SLAB, scenarios={self.id: entry}
            ),
            sort_keys=False,
            default_flow_style=None,
            width=4096,
            allow_unicode=True,
        )
        return header + body, entry


def check_sdf(path, movers):
    """I2 §3.4 checks before generation; each failure names the offending element.

    Collection worlds pin no Fuel models (T0 writes no includes), so every
    ``<include>`` is refused. A mover must be non-static, and so must every model
    around it (the builder inherits ``static`` from the parents)."""
    root = ET.parse(path).getroot()
    version = tuple(int(v) for v in root.get("version", "0").split("."))
    if version > MAX_SDF_VERSION:
        raise ValueError(f"{path.name}: SDF version {root.get('version')} > 1.11")
    for collision in root.iter("collision"):
        geometry = collision.find("geometry")
        tags = [child.tag for child in geometry] if geometry is not None else []
        if not tags or any(tag not in GEOMETRY_TAGS for tag in tags):
            raise ValueError(
                f"{path.name}: collision {collision.get('name')} has geometry {tags}"
            )
    for include in root.iter("include"):
        uri = (include.findtext("uri") or "").strip()
        raise ValueError(f"{path.name}: include {uri!r}; collection worlds have none")
    parents = {child: parent for parent in root.iter() for child in parent}
    models = {model.get("name"): model for model in root.iter("model")}
    for mover in movers:
        model = models.get(mover)
        if model is None:
            raise ValueError(f"{path.name}: moving object {mover} has no model")
        node = model
        while node is not None:
            if node.tag == "model" and node.findtext("static", "false").strip() in (
                "1",
                "true",
            ):
                raise ValueError(
                    f"{path.name}: moving object {mover} is static or inside the "
                    f"static model {node.get('name')}"
                )
            node = parents.get(node)


def check_floor(metadata, config):
    """The floor's top is at z = 0 under the whole inclusion region (catalog spawn
    heights assume it): a collision below the slab with top 0 covers the polygons."""
    xs = [x for polygon in config["inclusion_polygons"] for x, _ in polygon]
    ys = [y for polygon in config["inclusion_polygons"] for _, y in polygon]
    for record in metadata["generation"]["collision_records"]:
        bounds = record.get("bounds")
        if bounds is None or abs(bounds[1][2]) > 1e-6:
            continue
        (x0, y0, _), (x1, y1, _) = bounds
        if x0 <= min(xs) and y0 <= min(ys) and x1 >= max(xs) and y1 >= max(ys):
            return record["collision"]
    raise ValueError("No floor collision with its top at z = 0 under the inclusion")


def validate_spawns(reference_path, spawns, profiles):
    """I2 §3.6 for every spawn and robot; returns one report row per spawn."""
    from scipy.ndimage import distance_transform_edt, label
    from exploration_benchmark_robots.preflight import validate_spawn

    reference = Reference(reference_path)
    free = reference.inclusion & ~reference.collision
    # Cells beyond the grid count as outside F (T0's clearance; the adapter pads too).
    clearance = (
        distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * reference.resolution
    )
    components, _ = label(free)
    x0, y0, _ = reference.origin
    sx, sy = reference.metadata["spawn"][:2]
    home = components[
        int(math.floor((sy - y0) / reference.resolution)),
        int(math.floor((sx - x0) / reference.resolution)),
    ]
    rows, failures = [], []
    for spawn in spawns:
        row = int(math.floor((spawn["y"] - y0) / reference.resolution))
        column = int(math.floor((spawn["x"] - x0) / reference.resolution))
        inside = 0 <= row < free.shape[0] and 0 <= column < free.shape[1]
        component = inside and components[row, column] == home and home != 0
        clear = float(clearance[row, column]) if inside else 0.0
        problems = []
        if not component:
            problems.append("outside the reference's free component")
        if clear < CLEARANCE_M - 1e-9:
            problems.append(f"clearance {clear:.3f} m < {CLEARANCE_M} m")
        for robot, profile in profiles.items():
            try:
                validate_spawn(
                    profile, [spawn["x"], spawn["y"], 0.0, spawn["yaw"]], reference
                )
            except ValueError as exc:
                problems.append(f"{robot}: {exc}")
        rows.append(
            dict(spawn=spawn["id"], clearance_m=round(clear, 3), problems=problems)
        )
        failures += [f"{spawn['id']}: {p}" for p in problems]
    if failures:
        raise ValueError("Spawn validation failed: " + "; ".join(failures))
    return rows


def run_descriptors(root, python, reference_path, spawns, output):
    """The descriptor adapter in its own process (I2 O-2 (a)); returns the paths of
    ``descriptors.json`` and ``regions.png``."""
    command = [
        str(python),
        str(Path(root) / DESCRIPTOR_TOOL),
        "--reference",
        str(reference_path),
        "--output",
        str(output),
    ]
    for spawn in spawns:
        command += ["--spawn", spawn["id"], repr(spawn["x"]), repr(spawn["y"])]
    subprocess.run(command, check=True, cwd=root, timeout=DESCRIPTOR_TIMEOUT_S)
    return output / "descriptors.json", output / "regions.png"


def compare_base(
    reference_path, base_path, descriptors_path, base_descriptors_path, movers
):
    """I2 §3.7: a moving-object world's grids, target and static descriptors equal its
    base world's, and every mover is recorded as ``ignored: dynamic``."""
    world, base = Reference(reference_path), Reference(base_path)
    if world.collision.shape != base.collision.shape or world.origin != base.origin:
        raise ValueError("The grid differs from the base world's")
    for name in ("collision", "inclusion", "target"):
        if not np.array_equal(getattr(world, name), getattr(base, name)):
            raise ValueError(f"The {name} mask differs from the base world's")
    ignored = {
        record["model"]
        for record in world.metadata["generation"]["collision_records"]
        if record.get("ignored") == "dynamic"
    }
    missing = sorted(set(movers) - ignored)
    if missing:
        raise ValueError(
            f"Movers not recorded as ignored: dynamic: {', '.join(missing)}"
        )
    own = json.loads(Path(descriptors_path).read_text())
    other = json.loads(Path(base_descriptors_path).read_text())
    for key in ("grid", "spawns", "software", "moving_objects"):
        own.pop(key, None)
        other.pop(key, None)
    if own != other:
        keys = sorted(
            k for k in own.keys() | other.keys() if own.get(k) != other.get(k)
        )
        raise ValueError(f"Descriptors differ from the base world's: {', '.join(keys)}")


def entry_text(
    world, label, reference_metadata_sha256, descriptors_file, descriptors_sha, robots
):
    """The registry entry in worlds.yaml's style (every field, I1); ``robots`` are the
    catalog robot IDs, each fit with its catalog spawn height. The text is assembled
    here; the registry loader in ``_write`` is its check."""
    doc = world.document
    first = world.spawns[0]
    coverage = REFERENCE_DIRECTORY
    lines = [
        f"  {world.id}:",
        f"    label: {json.dumps(label, ensure_ascii=False)}",
        f"    tier: {world.tier}",
        f"    family: {world.family or 'null'}",
        "    role: evaluation",
        f"    world_file: {world.relative(world.sdf)}",
        "    source:",
        f"      world_yaml: {world.relative(world.yaml_path)}",
        f"      world_yaml_sha256: {file_sha256(world.yaml_path)}",
        f"      sdf_sha256: {file_sha256(world.sdf)}",
        f"      worldgen_version: {json.dumps(str(doc['worldgen_version']))}",
        "    scenario:",
        f"      file: {coverage}/{world.id}_scenario.yaml",
        f"      key: {world.id}",
        "    reference:",
        "      metric_version: 2",
        f"      id: {world.id}_coverage_v2",
        f"      metadata_sha256: {reference_metadata_sha256}",
        f"    default_spawn: {{id: default-{world.id}, name: Default, "
        f"x: {_yaml_float(first['x'])}, y: {_yaml_float(first['y'])}, "
        f"z: {_yaml_float(SPAWN_Z)}, "
        f"yaw: {_yaml_float(first['yaw'])}}}",
        "    spawns:",
        *(
            f"      - {{id: {s['id']}, name: {s['id']}, x: {_yaml_float(s['x'])}, "
            f"y: {_yaml_float(s['y'])}, yaw: {_yaml_float(s['yaw'])}}}"
            for s in world.spawns
        ),
        "    robots:",
        *(f"      {robot}: {{}}" for robot in robots),
        "    fuel_models: []",
        "    resource_paths: []",
    ]
    if world.moving is None:
        lines.append("    moving_objects: null")
    else:
        m = world.moving
        lines.append(
            f"    moving_objects: {{system: {m['system']}, format_version: "
            f"{int(m['format_version'])}, velocity_mode: {m['velocity_mode']}, "
            f"topic: {m['topic']}, count: {int(m['count'])}}}"
        )
    lines += [
        "    descriptors:",
        f"      file: {descriptors_file}",
        f"      sha256: {descriptors_sha}",
    ]
    return "\n".join(lines) + "\n"


def _place(source, destination):
    """Copy a staged file into place (mode from the umask); never over an existing
    file."""
    data = Path(source).read_bytes()
    try:
        with Path(destination).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError(f"Refusing to overwrite {destination}") from exc


def _write(root, coverage, staging, names, entry, world_id):
    """Place the staged files and append the entry, never leaving a broken registry.

    The loader needs the entry's files under the checkout, so the six new files are
    placed first; the would-be registry (current text plus the entry, staged) must
    then load with every pin before ``worlds.yaml`` is appended. On any failure the
    files this call placed are removed again and the registry is unchanged."""
    registry_file = root / REGISTRY_FILE
    current = registry_file.read_text(encoding="utf-8")
    if not current.endswith("\n"):
        raise ValueError(f"{registry_file} does not end with a newline")
    staged = staging / "worlds.yaml"
    staged.write_text(current + entry, encoding="utf-8")
    placed = []
    try:
        for name in names.values():
            _place(staging / name, coverage / name)
            placed.append(coverage / name)
        WorldRegistry.load(root, path=staged, references=coverage).world(world_id)
    except BaseException:
        for path in placed:
            path.unlink(missing_ok=True)
        raise
    if registry_file.read_text(encoding="utf-8") != current:
        for path in placed:
            path.unlink(missing_ok=True)
        raise ValueError(f"{registry_file} changed during the import; nothing appended")
    with registry_file.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def import_world(
    root,
    world_dir,
    *,
    directory=None,
    write=False,
    label=None,
    base_world=None,
    descriptor_python=None,
):
    """Import one world (I2 §3.5). Returns the registry entry text."""
    from .reference_builder import build, collection_provenance

    started = time.monotonic()
    root = Path(root).resolve()
    if write == (directory is not None):
        raise ValueError("Give --directory for a dry run, or --write (not both)")
    registry = WorldRegistry.load(root)
    world = CollectionWorld(root, world_dir)
    if world.id in registry.ids():
        raise ValueError(f"{world.id} is already in the registry")
    if (world.tier == "MO") != (base_world is not None):
        raise ValueError(
            "Moving-object worlds need --base-world; other worlds take none"
        )
    python = Path(descriptor_python or root / DESCRIPTOR_PYTHON)
    check_sdf(world.sdf, world.movers)
    coverage = root / REFERENCE_DIRECTORY
    pin = Descriptors(f"{REFERENCE_DIRECTORY}/{world.id}{DESCRIPTORS_SUFFIX}", "")
    names = dict(
        scenario=f"{world.id}_scenario.yaml",
        grid=f"{world.id}.npz",
        reference=reference_file(Path(), world.id).name,
        image=f"{world.id}_reference.png",
        descriptors=Path(pin.file).name,
        regions=Path(pin.regions_file).name,
    )
    if write:
        existing = [n for n in names.values() if (coverage / n).exists()]
        if existing:
            raise ValueError(
                f"Refusing to overwrite {', '.join(existing)} in {coverage}"
            )
    # --write stages in the system temporary directory; a dry run writes to --directory.
    with tempfile.TemporaryDirectory(prefix=f"import-{world.id}-") as temporary:
        staging = Path(temporary) if write else Path(directory)
        staging.mkdir(parents=True, exist_ok=True)
        text, config = world.scenario(names["scenario"])
        settings_file = staging / names["scenario"]
        if settings_file.exists():
            raise ValueError(f"Refusing to overwrite {settings_file}")
        settings_file.write_text(text, encoding="utf-8")
        first = world.spawns[0]
        metadata = build(
            root,
            world.id,
            world.sdf,
            [first["x"], first["y"], SPAWN_Z, first["yaw"]],
            (settings_file, config, RESOLUTION, SLAB),
            staging,
            provenance=collection_provenance(
                world.relative(world.yaml_path), world.document, world.id
            ),
            settings_identity=f"{REFERENCE_DIRECTORY}/{names['scenario']}",
        )
        floor = check_floor(metadata, config)
        reference_path = staging / names["reference"]
        catalog = yaml.load(
            (root / ROBOT_CATALOG_FILE).read_bytes(), Loader=UniqueKeyLoader
        )
        spawn_rows = validate_spawns(reference_path, world.spawns, catalog["robots"])
        with tempfile.TemporaryDirectory(dir=staging, prefix=".descriptors-") as output:
            descriptors, regions = run_descriptors(
                root, python, reference_path, world.spawns, Path(output)
            )
            document = descriptors.read_text(encoding="utf-8")
            if world.moving is not None:
                # T1 §3: base-world static descriptors plus the object count and
                # density.
                values = json.loads(document)
                block = world.document["moving_objects"]
                values["moving_objects"] = dict(
                    count=block["count"],
                    density=block["density"],
                    area_per_object_m2=block["area_per_object_m2"],
                )
                document = (
                    json.dumps(values, indent=1, allow_nan=False, ensure_ascii=False)
                    + "\n"
                )
            (staging / names["descriptors"]).write_text(document, encoding="utf-8")
            _place(regions, staging / names["regions"])
        if base_world is not None:
            base = registry.world(base_world)
            if base.descriptors is None:
                raise ValueError(f"Base world {base_world} has no descriptor file")
            compare_base(
                reference_path,
                base.reference.metadata_file,
                staging / names["descriptors"],
                registry.path(base.descriptors.file),
                world.movers,
            )
        descriptors_sha = file_sha256(staging / names["descriptors"])
        entry = entry_text(
            world,
            label or label_rule(world.document["title"]),
            file_sha256(reference_path),
            pin.file,
            descriptors_sha,
            list(catalog["robots"]),
        )
        if write:
            _write(root, coverage, staging, names, entry, world.id)
    # Peak resident memory (KiB on Linux) of the import and its descriptor process.
    peak = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    print(entry, end="")
    print(
        f"# {world.id}: {metadata['target_cells']} target cells, "
        f"{metadata['target_area_m2']:.2f} m²; floor {floor}; spawns "
        + ", ".join(f"{r['spawn']} clearance {r['clearance_m']} m" for r in spawn_rows)
        + f"; {time.monotonic() - started:.1f} s; peak RSS {peak / 1024:.0f} MiB"
        + ("" if write else f"; dry run in {directory}")
    )
    return entry
