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

"""The automatable checks of FIGURE_STYLE §11 for every figure (FP-9).

``check_pdf`` reads a saved PDF with poppler's ``pdfinfo``, ``pdffonts`` and ``pdfimages``: page width against
the preset (item 2), PDF version (item 2), every font embedded and no Type 3 (item 3), raster resolution
(item 4). ``contact_sheet`` renders the figure as seen with deuteranopia and protanopia (Viénot, Brettel and
Mollon 1999), tritanopia (Brettel, Viénot and Mollon 1997) and in greyscale (Rec. 709 luma), for review (items
8 and 9). The checks report; the researcher judges.
"""

from pathlib import Path
import subprocess
from typing import Any

from daltonlens import simulate
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

MIN_PPI = 300  # FIGURE_STYLE §2: above 300 ppi for colour and greyscale rasters (600 for line art)
WIDTH_TOLERANCE_PT = 0.5


def _run(*command: str) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _table(text: str) -> list[dict[str, str]]:
    """A poppler listing: a header line, a line of dashes giving the column spans, then rows."""
    lines = text.splitlines()
    if len(lines) < 2:
        return []
    spans, start = [], None
    for index, character in enumerate(lines[1] + " "):
        if character == "-" and start is None:
            start = index
        elif character != "-" and start is not None:
            spans.append((start, index))
            start = None
    names = [lines[0][a:b].strip() or lines[0][a:].split()[0] for a, b in spans]
    rows = []
    for line in lines[2:]:
        cells = [
            line[a : b if i < len(spans) - 1 else None].strip() for i, (a, b) in enumerate(spans)
        ]
        rows.append(dict(zip(names, cells, strict=True)))
    return rows


def check_pdf(path: Path, expected_width_pt: float) -> dict[str, Any]:
    info = _run("pdfinfo", str(path))
    fonts = _run("pdffonts", str(path))
    images = _run("pdfimages", "-list", str(path))
    if info is None or fonts is None or images is None:
        return dict(available=False, note="poppler-utils (pdfinfo, pdffonts, pdfimages) not found")
    fields = dict(line.split(":", 1) for line in info.splitlines() if ":" in line)
    size = fields.get("Page size", "").split()
    width, height = (float(size[0]), float(size[2])) if len(size) >= 3 else (None, None)
    font_rows: list[dict[str, Any]] = [
        dict(
            name=row["name"],
            type=row["type"],
            embedded=row["emb"] == "yes",
            subset=row["sub"] == "yes",
        )
        for row in _table(fonts)
    ]
    # pdfimages prints one unbroken line of dashes: its rows split on whitespace into page, num, type, width,
    # height, color, comp, bpc, enc, interp, object, ID, x-ppi, y-ppi, size, ratio
    image_rows: list[dict[str, Any]] = [
        dict(
            width=int(cells[3]),
            height=int(cells[4]),
            encoding=cells[8],
            x_ppi=int(cells[12]),
            y_ppi=int(cells[13]),
        )
        for cells in (line.split() for line in images.splitlines()[2:])
        if len(cells) >= 14
    ]
    problems = []
    if width is None or abs(width - expected_width_pt) > WIDTH_TOLERANCE_PT:
        problems.append(f"page width {width} pt, expected {expected_width_pt:.1f} pt")
    version = fields.get("PDF version", "").strip()
    if not version or tuple(map(int, version.split("."))) < (1, 4):
        problems.append(f"PDF version {version or 'unknown'} below 1.4")
    if any(row["type"].startswith("Type 3") for row in font_rows):
        problems.append("Type 3 fonts present")
    if not all(row["embedded"] for row in font_rows):
        problems.append("a font is not embedded")
    low = [row for row in image_rows if min(row["x_ppi"], row["y_ppi"]) <= MIN_PPI]
    if low:
        problems.append(f"{len(low)} raster(s) at or below {MIN_PPI} ppi")
    if any(row["encoding"] in ("jpeg", "jpx") for row in image_rows):
        problems.append("JPEG-encoded raster")
    return dict(
        available=True,
        page_width_pt=width,
        page_height_pt=height,
        expected_width_pt=round(expected_width_pt, 2),
        pdf_version=version,
        fonts=font_rows,
        images=image_rows,
        problems=problems,
        ok=not problems,
    )


def _pixels(figure: Figure, dpi: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    figure.set_dpi(dpi)
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())  # type: ignore[attr-defined]
    return np.ascontiguousarray(rgba[..., :3])


def contact_sheet(figure: Figure, path: Path, *, dpi: int = 200) -> list[str]:
    """Write the figure as seen under each simulation, side by side, to ``path`` (PNG)."""
    image = _pixels(figure, dpi)
    vienot, brettel = simulate.Simulator_Vienot1999(), simulate.Simulator_Brettel1997()
    luma = image.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    grey = np.repeat(np.clip(np.rint(luma), 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    views = [
        ("as drawn", image),
        ("deuteranopia (Viénot 1999)", vienot.simulate_cvd(image, simulate.Deficiency.DEUTAN, 1.0)),
        ("protanopia (Viénot 1999)", vienot.simulate_cvd(image, simulate.Deficiency.PROTAN, 1.0)),
        ("tritanopia (Brettel 1997)", brettel.simulate_cvd(image, simulate.Deficiency.TRITAN, 1.0)),
        ("greyscale (Rec. 709 luma)", grey),
    ]
    height, width = image.shape[:2]
    sheet = plt.figure(
        figsize=(len(views) * width / dpi, height / dpi + 0.3), dpi=dpi, layout="none"
    )
    for index, (label, view) in enumerate(views):
        axis = sheet.add_axes(
            (index / len(views), 0, 1 / len(views), height / (height + 0.3 * dpi))
        )
        axis.imshow(view, interpolation="nearest")
        axis.set_axis_off()
        sheet.text((index + 0.5) / len(views), 0.99, label, ha="center", va="top", fontsize=8)
    sheet.savefig(path, dpi=dpi, metadata={"Software": None})
    plt.close(sheet)
    return [label for label, _ in views]
