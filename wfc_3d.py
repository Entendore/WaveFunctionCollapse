"""
WFC 3D — Wave Function Collapse (3D Isometric Edition)
=========================================================
A standalone 3D procedural generation tool built with PySide6.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import sys
import time
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import (
    Qt, QTimer, QRectF, Signal, QObject, QThread, QPoint, QSize, QPointF
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QKeySequence, QShortcut,
    QPainterPath, QPixmap, QImage, QPolygonF, QVector3D, QMatrix4x4,
    QSurfaceFormat
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSpinBox, QLabel, QSlider, QGroupBox, QMessageBox,
    QComboBox, QCheckBox, QStatusBar, QProgressBar, QFileDialog,
    QLineEdit, QScrollArea, QColorDialog, QTableWidget, QTableWidgetItem,
    QDialog, QDialogButtonBox, QFormLayout, QToolTip, QMenuBar, QMenu,
    QSizePolicy, QRadioButton
)

# ---------------------------------------------------------------------------
# Performance Utilities
# ---------------------------------------------------------------------------

POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def get_popcount(n: int) -> int:
    return (POPCOUNT_TABLE[n & 0xff] + POPCOUNT_TABLE[(n >> 8) & 0xff] +
            POPCOUNT_TABLE[(n >> 16) & 0xff] + POPCOUNT_TABLE[(n >> 24) & 0xff] +
            POPCOUNT_TABLE[(n >> 32) & 0xff] + POPCOUNT_TABLE[(n >> 40) & 0xff] +
            POPCOUNT_TABLE[(n >> 48) & 0xff] + POPCOUNT_TABLE[(n >> 56) & 0xff])

# ---------------------------------------------------------------------------
# Core 3D Logic
# ---------------------------------------------------------------------------


class Direction(IntEnum):
    UP = 0; DOWN = 1; LEFT = 2; RIGHT = 3; FORWARD = 4; BACKWARD = 5

    @property
    def delta(self) -> Tuple[int, int, int]:
        return {0: (0, -1, 0), 1: (0, 1, 0), 2: (-1, 0, 0),
                3: (1, 0, 0), 4: (0, 0, 1), 5: (0, 0, -1)}[self]

    @property
    def opposite(self) -> Direction:
        return Direction({0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}[self])


@dataclass
class TilesetConfig:
    name: str
    tiles: List[str]
    colors: Dict[str, str]
    solid: Dict[str, bool] = field(default_factory=dict)  # True = opaque voxel
    rules: List[Tuple[str, str, str]] = field(default_factory=list)  # (tile, dir_name, neighbor)
    weights: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name, "tiles": self.tiles, "colors": self.colors,
            "solid": self.solid, "rules": [list(r) for r in self.rules],
            "weights": self.weights
        }, indent=4)

    @staticmethod
    def from_json(data_str: str) -> TilesetConfig:
        d = json.loads(data_str)
        return TilesetConfig(
            name=d['name'], tiles=d['tiles'], colors=d['colors'],
            solid=d.get('solid', {}), rules=[tuple(r) for r in d['rules']],
            weights=d.get('weights', {}))


DIR_NAMES = {"UP": Direction.UP, "DOWN": Direction.DOWN,
             "LEFT": Direction.LEFT, "RIGHT": Direction.RIGHT,
             "FORWARD": Direction.FORWARD, "BACKWARD": Direction.BACKWARD}
DIR_LIST_2D = ["UP", "DOWN", "LEFT", "RIGHT"]
DIR_LIST_3D = ["UP", "DOWN", "LEFT", "RIGHT", "FORWARD", "BACKWARD"]


def _make_sym_rules(pairs: List[Tuple[str, str]],
                    directions: List[str] = DIR_LIST_3D) -> List[Tuple[str, str, str]]:
    result = []
    for a, b in pairs:
        for d in directions:
            result.append((a, d, b))
    return result


def _make_vertical_sym_rules(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """Rules for cross-layer (Z-axis) stacking using FORWARD/BACKWARD.
    Each pair (a, b) means 'a is above b' (a at higher Z)."""
    result = []
    for a, b in pairs:
        result.append((b, "FORWARD", a))   # below has above in +Z direction
        result.append((a, "BACKWARD", b))  # above has below in -Z direction
    return result

# ---------------------------------------------------------------------------
# 3D Tileset Presets
# ---------------------------------------------------------------------------


def _get_voxel_terrain() -> TilesetConfig:
    tiles = ["AIR", "WATER", "SAND", "GRASS", "DIRT", "STONE", "SNOW"]
    colors = {"AIR": "#1e1e2e", "WATER": "#3498db", "SAND": "#f1c40f",
              "GRASS": "#2ecc71", "DIRT": "#8b6914", "STONE": "#7f8c8d",
              "SNOW": "#ecf0f1"}
    solid = {"AIR": False, "WATER": True, "SAND": True, "GRASS": True,
             "DIRT": True, "STONE": True, "SNOW": True}
    # Horizontal adjacency (same-layer neighbors)
    horiz = [("WATER", "WATER"), ("SAND", "SAND"), ("GRASS", "GRASS"),
             ("DIRT", "DIRT"), ("STONE", "STONE"), ("SNOW", "SNOW"),
             ("WATER", "SAND"), ("SAND", "GRASS"), ("GRASS", "DIRT"),
             ("DIRT", "STONE"), ("STONE", "SNOW"),
             ("AIR", "AIR"), ("AIR", "WATER"), ("AIR", "SAND"),
             ("AIR", "GRASS"), ("AIR", "DIRT"), ("AIR", "STONE")]
    # Vertical stacking (layer above/below)
    vert = [("AIR", "SNOW"), ("SNOW", "STONE"), ("STONE", "DIRT"),
            ("DIRT", "GRASS"), ("GRASS", "SAND"), ("SAND", "WATER"),
            ("AIR", "STONE"), ("AIR", "DIRT"), ("AIR", "GRASS"),
            ("AIR", "SAND"), ("AIR", "WATER"), ("WATER", "WATER"),
            ("STONE", "STONE"), ("DIRT", "DIRT"), ("SAND", "SAND"),
            ("GRASS", "GRASS")]
    rules = _make_sym_rules(horiz, DIR_LIST_2D) + _make_vertical_sym_rules(vert)
    weights = {"AIR": 4.0, "WATER": 1.5, "SAND": 1.2, "GRASS": 2.0,
               "DIRT": 2.5, "STONE": 3.0, "SNOW": 0.5}
    return TilesetConfig("Voxel Terrain", tiles, colors, solid, rules, weights)


def _get_voxel_dungeon() -> TilesetConfig:
    tiles = ["AIR", "FLOOR", "WALL", "PILLAR", "DOOR", "TORCH", "CEILING"]
    colors = {"AIR": "#1e1e2e", "FLOOR": "#d4c5a9", "WALL": "#2c3e50",
              "PILLAR": "#5d6d7e", "DOOR": "#e67e22", "TORCH": "#f39c12",
              "CEILING": "#34495e"}
    solid = {"AIR": False, "FLOOR": True, "WALL": True, "PILLAR": True,
             "DOOR": True, "TORCH": False, "CEILING": True}
    horiz = [("FLOOR", "FLOOR"), ("WALL", "WALL"), ("PILLAR", "PILLAR"),
             ("AIR", "AIR"), ("AIR", "FLOOR"), ("FLOOR", "DOOR"),
             ("DOOR", "WALL"), ("FLOOR", "PILLAR"), ("PILLAR", "WALL"),
             ("CEILING", "CEILING"), ("AIR", "TORCH"), ("TORCH", "WALL")]
    vert = [("AIR", "AIR"), ("CEILING", "AIR"), ("CEILING", "TORCH"),
            ("AIR", "FLOOR"), ("FLOOR", "AIR"), ("WALL", "WALL"),
            ("WALL", "CEILING"), ("PILLAR", "PILLAR"), ("PILLAR", "CEILING"),
            ("DOOR", "AIR"), ("AIR", "DOOR")]
    rules = _make_sym_rules(horiz, DIR_LIST_2D) + _make_vertical_sym_rules(vert)
    weights = {"AIR": 4.0, "FLOOR": 2.5, "WALL": 3.0, "PILLAR": 0.8,
               "DOOR": 0.6, "TORCH": 0.3, "CEILING": 2.0}
    return TilesetConfig("Voxel Dungeon", tiles, colors, solid, rules, weights)


def _get_voxel_cave() -> TilesetConfig:
    tiles = ["AIR", "STONE", "CRYSTAL", "WATER", "LAVA", "MOSS", "GEM"]
    colors = {"AIR": "#1e1e2e", "STONE": "#616a6b", "CRYSTAL": "#9b59b6",
              "WATER": "#2471a3", "LAVA": "#e74c3c", "MOSS": "#27ae60",
              "GEM": "#00d2ff"}
    solid = {"AIR": False, "STONE": True, "CRYSTAL": True, "WATER": True,
             "LAVA": True, "MOSS": False, "GEM": True}
    horiz = [("STONE", "STONE"), ("AIR", "AIR"), ("CRYSTAL", "CRYSTAL"),
             ("WATER", "WATER"), ("LAVA", "LAVA"), ("GEM", "GEM"),
             ("STONE", "AIR"), ("AIR", "STONE"), ("AIR", "CRYSTAL"),
             ("CRYSTAL", "AIR"), ("AIR", "WATER"), ("WATER", "AIR"),
             ("STONE", "CRYSTAL"), ("CRYSTAL", "STONE"), ("STONE", "LAVA"),
             ("LAVA", "STONE"), ("STONE", "GEM"), ("GEM", "STONE"),
             ("AIR", "MOSS"), ("MOSS", "AIR"), ("MOSS", "STONE")]
    vert = [("STONE", "STONE"), ("AIR", "AIR"), ("AIR", "WATER"),
            ("WATER", "AIR"), ("STONE", "CRYSTAL"), ("CRYSTAL", "AIR"),
            ("STONE", "LAVA"), ("LAVA", "AIR"), ("AIR", "AIR"),
            ("STONE", "MOSS"), ("MOSS", "AIR"), ("CRYSTAL", "GEM"),
            ("GEM", "CRYSTAL")]
    rules = _make_sym_rules(horiz, DIR_LIST_2D) + _make_vertical_sym_rules(vert)
    weights = {"AIR": 3.5, "STONE": 4.0, "CRYSTAL": 0.6, "WATER": 1.0,
               "LAVA": 0.3, "MOSS": 0.8, "GEM": 0.2}
    return TilesetConfig("Voxel Cave", tiles, colors, solid, rules, weights)


def _get_crystal_cavern() -> TilesetConfig:
    tiles = ["AIR", "ROCK", "AMETHYST", "EMERALD", "RUBY", "QUARTZ", "POOL"]
    colors = {"AIR": "#1e1e2e", "ROCK": "#4a4a4a", "AMETHYST": "#8e44ad",
              "EMERALD": "#1abc9c", "RUBY": "#e74c3c", "QUARTZ": "#f5f5f5",
              "POOL": "#2e86c1"}
    solid = {"AIR": False, "ROCK": True, "AMETHYST": True, "EMERALD": True,
             "RUBY": True, "QUARTZ": True, "POOL": True}
    horiz = [("ROCK", "ROCK"), ("AIR", "AIR"), ("AMETHYST", "AMETHYST"),
             ("EMERALD", "EMERALD"), ("RUBY", "RUBY"), ("QUARTZ", "QUARTZ"),
             ("POOL", "POOL"), ("ROCK", "AIR"), ("AIR", "ROCK"),
             ("AIR", "AMETHYST"), ("AMETHYST", "AIR"), ("AIR", "EMERALD"),
             ("EMERALD", "AIR"), ("AIR", "QUARTZ"), ("QUARTZ", "AIR"),
             ("ROCK", "AMETHYST"), ("AMETHYST", "ROCK"), ("ROCK", "EMERALD"),
             ("ROCK", "QUARTZ"), ("QUARTZ", "ROCK"), ("AMETHYST", "EMERALD"),
             ("EMERALD", "RUBY"), ("RUBY", "QUARTZ"), ("AIR", "POOL"),
             ("POOL", "AIR"), ("POOL", "ROCK")]
    vert = [("ROCK", "ROCK"), ("AIR", "AIR"), ("ROCK", "AMETHYST"),
            ("AMETHYST", "AIR"), ("ROCK", "EMERALD"), ("EMERALD", "AIR"),
            ("ROCK", "RUBY"), ("RUBY", "AIR"), ("ROCK", "QUARTZ"),
            ("QUARTZ", "AIR"), ("AIR", "POOL"), ("POOL", "AIR"),
            ("AMETHYST", "QUARTZ"), ("QUARTZ", "AMETHYST"),
            ("EMERALD", "RUBY"), ("RUBY", "EMERALD")]
    rules = _make_sym_rules(horiz, DIR_LIST_2D) + _make_vertical_sym_rules(vert)
    weights = {"AIR": 4.0, "ROCK": 4.0, "AMETHYST": 0.8, "EMERALD": 0.7,
               "RUBY": 0.5, "QUARTZ": 1.0, "POOL": 0.6}
    return TilesetConfig("Crystal Cavern", tiles, colors, solid, rules, weights)


def _get_maze() -> TilesetConfig:
    tiles = ["AIR", "WALL", "PATH", "EXIT"]
    colors = {"AIR": "#1e1e2e", "WALL": "#2c3e50", "PATH": "#f0e68c",
              "EXIT": "#e74c3c"}
    solid = {"AIR": False, "WALL": True, "PATH": True, "EXIT": True}
    horiz = [("WALL", "WALL"), ("AIR", "AIR"), ("PATH", "PATH"),
             ("WALL", "AIR"), ("AIR", "WALL"), ("PATH", "AIR"),
             ("AIR", "PATH"), ("PATH", "EXIT"), ("EXIT", "PATH"),
             ("WALL", "PATH"), ("PATH", "WALL")]
    vert = [("WALL", "WALL"), ("AIR", "AIR"), ("PATH", "PATH"),
            ("WALL", "AIR"), ("AIR", "WALL"), ("PATH", "AIR"),
            ("AIR", "PATH"), ("PATH", "EXIT"), ("EXIT", "AIR")]
    rules = _make_sym_rules(horiz, DIR_LIST_2D) + _make_vertical_sym_rules(vert)
    weights = {"AIR": 2.0, "WALL": 4.0, "PATH": 2.5, "EXIT": 0.3}
    return TilesetConfig("Maze", tiles, colors, solid, rules, weights)

# ---------------------------------------------------------------------------
# WFC Worker (3D)
# ---------------------------------------------------------------------------


class WFCWorker3D(QObject):
    progress = Signal(float)
    cell_collapsed = Signal(int, int, int, int)  # z, y, x, tile_idx
    finished = Signal(bool)
    contradiction_pos = Signal(int, int, int)
    step_done = Signal()

    def __init__(self, size_x: int, size_y: int, size_z: int,
                 config: TilesetConfig, seed: int, wrap: bool):
        super().__init__()
        if len(config.tiles) > 64:
            raise ValueError(
                f"Tileset has {len(config.tiles)} tiles, but bitmask supports at most 64.")
        self.sx, self.sy, self.sz = size_x, size_y, size_z
        self.wrap = wrap
        self.tiles = config.tiles
        self.n_tiles = len(self.tiles)
        self._tile_to_idx = {t: i for i, t in enumerate(self.tiles)}
        self.rng = np.random.default_rng(seed)
        self.weights = np.array(
            [config.weights.get(t, 1.0) for t in self.tiles], dtype=np.float64)
        self._heap_counter = 0

        self.rules = np.zeros((6, self.n_tiles), dtype=np.uint64)
        for tile_a, dir_name, tile_b in config.rules:
            if tile_a not in self._tile_to_idx or tile_b not in self._tile_to_idx:
                continue
            ia, ib = self._tile_to_idx[tile_a], self._tile_to_idx[tile_b]
            di = int(DIR_NAMES[dir_name])
            self.rules[di, ia] |= (1 << ib)
            self.rules[int(DIR_NAMES[dir_name].opposite), ib] |= (1 << ia)

        self.shape = (size_z, size_y, size_x)
        self.full_mask = (1 << self.n_tiles) - 1
        self._stop_requested = False
        self.step_count = 0
        self.reset()

    def reset(self):
        self.wave = np.full(self.shape, self.full_mask, dtype=np.uint64)
        self.collapsed = np.zeros(self.shape, dtype=np.uint8)
        self.step_count = 0
        self._heap = []
        self._in_heap = np.zeros(self.shape, dtype=bool)
        self._init_heap()

    def _init_heap(self):
        self._heap = []
        self._in_heap[:] = False
        for pos in np.ndindex(self.shape):
            self._push(pos, get_popcount(self.wave[pos]))

    def _push(self, pos, entropy):
        self._heap_counter += 1
        heapq.heappush(self._heap, (entropy, self._heap_counter, pos))
        self._in_heap[pos] = True

    def _pop_lowest(self):
        while self._heap:
            ent, _, pos = heapq.heappop(self._heap)
            if self._in_heap[pos]:
                self._in_heap[pos] = False
                if self.collapsed[pos]:
                    continue  # skip already-collapsed cells
                return pos, ent
        return None, -1

    def stop(self):
        self._stop_requested = True

    def get_snapshot(self) -> dict:
        return {'wave': self.wave.copy(), 'collapsed': self.collapsed.copy(),
                'step_count': self.step_count}

    def restore_snapshot(self, snap):
        self.wave = snap['wave'].copy()
        self.collapsed = snap['collapsed'].copy()
        self.step_count = snap['step_count']
        self._init_heap()

    def run_step(self) -> bool:
        if self._stop_requested:
            return False
        pos, ent = self._pop_lowest()
        if pos is None:
            return False
        if ent == 0:
            self.contradiction_pos.emit(*pos)
            return False
        mask = self.wave[pos]
        poss = [i for i in range(self.n_tiles) if (mask >> i) & 1]
        w = self.weights[poss]
        ws = np.sum(w)
        if ws == 0:
            self.contradiction_pos.emit(*pos)
            return False
        chosen = self.rng.choice(poss, p=w / ws)
        self.wave[pos] = (1 << chosen)
        self.collapsed[pos] = 1
        self.step_count += 1
        if not self._propagate(pos):
            self.contradiction_pos.emit(*pos)
            return False
        self.cell_collapsed.emit(*pos, chosen)
        self.step_done.emit()
        return True

    def _propagate(self, start_pos) -> bool:
        stack = [start_pos]
        shape = self.shape
        while stack:
            curr = stack.pop()
            cm = self.wave[curr]
            for d in Direction:
                di = int(d)
                dx, dy, dz = d.delta
                nz, ny, nx = curr[0] + dz, curr[1] + dy, curr[2] + dx
                if self.wrap:
                    nz, ny, nx = nz % shape[0], ny % shape[1], nx % shape[2]
                elif not (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]):
                    continue
                nb = (nz, ny, nx)
                allowed = np.uint64(0)
                for i in range(self.n_tiles):
                    if (cm >> i) & 1:
                        allowed |= self.rules[di, i]
                om = self.wave[nb]
                nm = om & allowed
                if nm != om:
                    if nm == 0:
                        return False
                    self.wave[nb] = nm
                    # Auto-collapse if only one possibility remains
                    if get_popcount(nm) == 1 and not self.collapsed[nb]:
                        self.collapsed[nb] = 1
                    self._in_heap[nb] = False
                    self._push(nb, get_popcount(nm))
                    if not self.collapsed[nb]:
                        stack.append(nb)
        return True

    def is_complete(self) -> bool:
        return bool(np.all(self.collapsed))


class InstantFinishWorker(QObject):
    progress = Signal(float)
    finished = Signal(bool, float)

    def __init__(self, worker: WFCWorker3D):
        super().__init__()
        self.worker = worker

    def run(self):
        t0 = time.perf_counter()
        total = int(np.prod(self.worker.shape))
        while True:
            if self.worker._stop_requested:
                self.finished.emit(False, time.perf_counter() - t0)
                return
            if not self.worker.run_step():
                break
            self.progress.emit(int(np.sum(self.worker.collapsed)) / total)
        self.finished.emit(self.worker.is_complete(), time.perf_counter() - t0)

# ---------------------------------------------------------------------------
# Isometric Voxel Canvas
# ---------------------------------------------------------------------------


class IsometricCanvas(QWidget):
    """3D isometric voxel renderer with orbit camera, transparency, and slicing."""

    cell_hovered = Signal(int, int, int, str)  # z, y, x, tile_name
    cell_painted = Signal(int, int, int, str)   # z, y, x, tile_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[WFCWorker3D] = None
        self._colors: Dict[str, str] = {}
        self._solid: Dict[str, bool] = {}
        self.setMinimumSize(500, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        # Camera
        self.azimuth = 45.0      # degrees around Y axis
        self.elevation = 35.0    # degrees above horizon
        self._zoom = 1.0
        self._pan = QPoint(0, 0)

        # Interaction
        self._rotating = False
        self._panning = False
        self._last_mouse = QPoint()
        self._paint_mode = False
        self._paint_tile = ""

        # Rendering options
        self.show_transparency = True
        self.show_grid_lines = True
        self.show_entropy = False
        self.slice_axis = "None"   # "None", "X", "Y", "Z"
        self.slice_pos = 0
        self.alpha_non_solid = 80  # 0-255

        # Hit-testing cache (filled during paintEvent)
        self._cell_screen_centers: Dict[Tuple[int, int, int], Tuple[float, float]] = {}

    def set_worker(self, worker, colors, solid):
        self.worker = worker
        self._colors = colors
        self._solid = solid
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._cell_screen_centers.clear()
        self.update()

    def reset_view(self):
        self.azimuth = 45.0; self.elevation = 35.0
        self._zoom = 1.0; self._pan = QPoint(0, 0)
        self.update()

    # ---- Isometric Projection ----
    def _project(self, x, y, z, cx, cy, scale):
        """Project 3D point to 2D isometric screen coordinates."""
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        cos_az, sin_az = math.cos(az), math.sin(az)
        cos_el, sin_el = math.cos(el), math.sin(el)
        # Rotate around Y (azimuth)
        rx = x * cos_az - z * sin_az
        rz = x * sin_az + z * cos_az
        ry = y
        # Rotate around X (elevation)
        ry2 = ry * cos_el - rz * sin_el
        rz2 = ry * sin_el + rz * cos_el
        # Orthographic projection
        sx = rx * scale + cx + self._pan.x()
        sy = -ry2 * scale + cy + self._pan.y()
        return sx, sy, rz2

    # ---- Face geometry for a unit cube at (gx, gy, gz) ----
    def _cube_faces(self, gx, gy, gz, cx, cy, scale):
        """Return list of (polygon_points, face_normal_z, color) for visible faces.
        Uses painter's algorithm: sort by depth later.
        Includes backface culling based on screen-space winding order."""
        x, y, z = gx, -gy, gz  # Flip Y so +Y is up
        s = scale

        # 8 vertices of unit cube
        v = [
            self._project(x, y, z, cx, cy, s),
            self._project(x + 1, y, z, cx, cy, s),
            self._project(x + 1, y + 1, z, cx, cy, s),
            self._project(x, y + 1, z, cx, cy, s),
            self._project(x, y, z + 1, cx, cy, s),
            self._project(x + 1, y, z + 1, cx, cy, s),
            self._project(x + 1, y + 1, z + 1, cx, cy, s),
            self._project(x, y + 1, z + 1, cx, cy, s),
        ]
        # Faces: (vertex indices, name, brightness factor)
        # Winding is CCW when viewed from outside the cube.
        faces = [
            ([3, 2, 1, 0], "front", 0.70),   # Z- (outward normal points toward viewer)
            ([4, 5, 6, 7], "back", 1.00),     # Z+
            ([7, 3, 0, 4], "left", 0.55),     # X-
            ([1, 2, 6, 5], "right", 0.85),    # X+
            ([7, 6, 2, 3], "top", 0.95),      # Y+ (up)
            ([0, 1, 5, 4], "bottom", 0.40),   # Y-
        ]
        result = []
        for indices, name, brightness in faces:
            # Backface culling: compute screen-space cross product
            p0, p1, p2 = v[indices[0]], v[indices[1]], v[indices[2]]
            e1x, e1y = p1[0] - p0[0], p1[1] - p0[1]
            e2x, e2y = p2[0] - p0[0], p2[1] - p0[1]
            cross = e1x * e2y - e1y * e2x
            if cross >= 0:
                continue  # back-facing, skip
            pts = [QPointF(v[i][0], v[i][1]) for i in indices]
            avg_depth = sum(v[i][2] for i in indices) / 4
            result.append((pts, avg_depth, brightness, name))
        return result

    def _get_tile_color(self, z, y, x) -> Optional[Tuple[QColor, bool]]:
        """Get color and solidity for a cell. Returns None for uncollapsed."""
        if not self.worker:
            return None
        mask = self.worker.wave[z, y, x]
        ent = get_popcount(mask)
        if self.show_entropy:
            norm = (ent - 1) / max(self.worker.n_tiles - 1, 1)
            return QColor.fromHslF(0.6 * (1.0 - norm), 0.7, 0.5), True
        if ent == 1:
            idx = int(mask).bit_length() - 1
            tname = self.worker.tiles[idx]
            base = QColor(self._colors.get(tname, "#ffffff"))
            return base, self._solid.get(tname, True)
        # Blended
        r, g, b, cnt = 0, 0, 0, 0
        for i in range(self.worker.n_tiles):
            if (mask >> i) & 1:
                c = QColor(self._colors.get(self.worker.tiles[i], "#ffffff"))
                r += c.red(); g += c.green(); b += c.blue(); cnt += 1
        if cnt == 0:
            return None
        return QColor(r // cnt, g // cnt, b // cnt), False

    def _cell_visible(self, z, y, x) -> bool:
        """Check if cell passes the slice filter."""
        if self.slice_axis == "X":
            return x == self.slice_pos
        elif self.slice_axis == "Y":
            return y == self.slice_pos
        elif self.slice_axis == "Z":
            return z == self.slice_pos
        return True

    def paintEvent(self, event):
        if not self.worker:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#11111b"))

        sz, sy, sx = self.worker.shape
        cx = self.width() / 2
        cy = self.height() / 2
        scale = min(self.width(), self.height()) / max(sz, sy, sx) * 0.45 * self._zoom

        # Collect all visible faces
        all_faces = []  # (depth, polygon, color, alpha)
        self._cell_screen_centers.clear()
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    if not self._cell_visible(z, y, x):
                        continue
                    # Store projected cell center for hit-testing
                    scx, scy, _ = self._project(x + 0.5, -(y + 0.5), z + 0.5, cx, cy, scale)
                    self._cell_screen_centers[(z, y, x)] = (scx, scy)
                    result = self._get_tile_color(z, y, x)
                    if result is None:
                        continue
                    base_color, is_solid = result
                    faces = self._cube_faces(x, y, z, cx, cy, scale)
                    for pts, depth, brightness, face_name in faces:
                        alpha = 255
                        if self.show_transparency and not is_solid:
                            alpha = self.alpha_non_solid
                        color = QColor(
                            int(base_color.red() * brightness),
                            int(base_color.green() * brightness),
                            int(base_color.blue() * brightness),
                            alpha)
                        all_faces.append((depth, pts, color))

        # Painter's algorithm: sort back-to-front
        all_faces.sort(key=lambda f: f[0])

        # Draw all faces
        for depth, pts, color in all_faces:
            poly = QPolygonF(pts)
            p.setBrush(QBrush(color))
            outline = QColor(255, 255, 255, 20) if self.show_grid_lines else QColor(0, 0, 0, 0)
            p.setPen(QPen(outline, 0.5))
            p.drawPolygon(poly)

        # HUD info
        p.setPen(QColor(200, 200, 200, 180))
        p.setFont(QFont("Monospace", 9))
        p.drawText(10, 20, f"Azimuth: {self.azimuth:.0f}°  Elevation: {self.elevation:.0f}°  Zoom: {self._zoom:.2f}x")
        if self.slice_axis != "None":
            p.drawText(10, 36, f"Slice: {self.slice_axis} = {self.slice_pos}")
        if self._paint_mode:
            p.setPen(QColor(243, 139, 168, 200))
            p.drawText(10, 52 if self.slice_axis != "None" else 36, f"PAINT MODE: {self._paint_tile}")
        p.end()

    # ---- Hit Testing ----
    def hit_test(self, screen_x: float, screen_y: float) -> Optional[Tuple[int, int, int]]:
        """Find the nearest grid cell to the given screen coordinates."""
        if not self._cell_screen_centers:
            return None
        best_pos = None
        best_dist = float('inf')
        threshold = min(self.width(), self.height()) / max(
            self.worker.shape[0], self.worker.shape[1], self.worker.shape[2]) * 0.6 \
            if self.worker else 999
        for pos, (sx, sy) in self._cell_screen_centers.items():
            d = (screen_x - sx) ** 2 + (screen_y - sy) ** 2
            if d < best_dist:
                best_dist = d
                best_pos = pos
        if best_pos and best_dist < threshold * threshold:
            return best_pos
        return None

    def _tile_name_at(self, z, y, x) -> str:
        """Get the tile name at a cell (or '?(n)' for uncollapsed)."""
        if not self.worker:
            return ""
        mask = self.worker.wave[z, y, x]
        ent = get_popcount(mask)
        if ent == 1:
            return self.worker.tiles[int(mask).bit_length() - 1]
        return f"?({ent})"

    # ---- Mouse Interaction ----
    def wheelEvent(self, event):
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        self._zoom = max(0.05, min(30.0, self._zoom * factor))
        self.update()

    def mousePressEvent(self, event):
        if self._paint_mode and event.button() == Qt.MouseButton.LeftButton:
            cell = self.hit_test(event.pos().x(), event.pos().y())
            if cell and self.worker and self._paint_tile:
                z, y, x = cell
                tname = self._paint_tile
                if tname in self.worker._tile_to_idx:
                    self.cell_painted.emit(z, y, x, tname)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._rotating = True; self._last_mouse = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True; self._last_mouse = event.pos()
            self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mouseMoveEvent(self, event):
        if self._rotating:
            dx = event.pos().x() - self._last_mouse.x()
            dy = event.pos().y() - self._last_mouse.y()
            self.azimuth += dx * 0.5
            self.elevation = max(-89, min(89, self.elevation + dy * 0.5))
            self._last_mouse = event.pos()
            self.update()
        elif self._panning:
            self._pan += event.pos() - self._last_mouse
            self._last_mouse = event.pos()
            self.update()
        else:
            # Hover info
            cell = self.hit_test(event.pos().x(), event.pos().y())
            if cell:
                z, y, x = cell
                tname = self._tile_name_at(z, y, x)
                self.cell_hovered.emit(z, y, x, tname)
            else:
                self.cell_hovered.emit(-1, -1, -1, "")

    def mouseReleaseEvent(self, event):
        self._rotating = False; self._panning = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def contextMenuEvent(self, event):
        """Suppress context menu so right-click can be used for panning."""
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        """Reset view on double-click."""
        self.reset_view()

    # ---- Export ----
    def export_obj(self, path: str):
        """Export the 3D grid as a Wavefront OBJ file."""
        if not self.worker:
            return
        sz, sy, sx = self.worker.shape
        vertices = []
        faces = []
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    mask = self.worker.wave[z, y, x]
                    ent = get_popcount(mask)
                    if ent != 1:
                        continue
                    tname = self.worker.tiles[int(mask).bit_length() - 1]
                    if not self._solid.get(tname, True) and tname == "AIR":
                        continue
                    # 8 vertices of a unit cube
                    base = len(vertices) + 1
                    corners = [
                        (x, y, z), (x+1, y, z), (x+1, y+1, z), (x, y+1, z),
                        (x, y, z+1), (x+1, y, z+1), (x+1, y+1, z+1), (x, y+1, z+1),
                    ]
                    vertices.extend(corners)
                    # 6 faces with consistent CCW winding (viewed from outside)
                    face_defs = [
                        (3, 2, 1, 0), (4, 5, 6, 7),  # Z- Z+
                        (7, 3, 0, 4), (1, 2, 6, 5),  # X- X+
                        (7, 6, 2, 3), (0, 1, 5, 4),  # Y+ Y-
                    ]
                    for f in face_defs:
                        a, b, c, d = [base + i for i in f]
                        faces.append((a, b, c, d, tname))

        # Write MTL file first (so mtllib reference in OBJ is valid)
        mtl_path = path.replace(".obj", ".mtl")
        with open(mtl_path, 'w') as mf:
            written = set()
            for _, _, _, _, tname in faces:
                if tname in written:
                    continue
                written.add(tname)
                hex_c = self._colors.get(tname, "#ffffff")
                r, g, b_ = int(hex_c[1:3], 16)/255, int(hex_c[3:5], 16)/255, int(hex_c[5:7], 16)/255
                mf.write(f"newmtl mat_{tname}\n")
                mf.write(f"Kd {r} {g} {b_}\n")
                mf.write(f"Ks 0.1 0.1 0.1\n")
                mf.write(f"Ka {r*0.2} {g*0.2} {b_*0.2}\n\n")

        with open(path, 'w') as f:
            mtl_basename = os.path.basename(mtl_path)
            f.write(f"# WFC 3D Export\n")
            f.write(f"# Grid: {sx}x{sy}x{sz}\n")
            f.write(f"mtllib {mtl_basename}\n\n")
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            f.write("\n")
            # Group faces by tile for materials
            current_tile = None
            for a, b, c, d, tname in faces:
                if tname != current_tile:
                    f.write(f"\no {tname}\n")
                    f.write(f"usemtl mat_{tname}\n")
                    current_tile = tname
                f.write(f"f {a} {b} {c} {d}\n")

    def export_layer_pngs(self, dir_path: str):
        """Export each Z-layer as a separate PNG."""
        if not self.worker:
            return
        sz, sy, sx = self.worker.shape
        scale = max(1, 256 // max(sx, sy))
        for z in range(sz):
            img = QImage(sx * scale, sy * scale, QImage.Format.Format_RGB32)
            img.fill(QColor("#11111b"))
            p = QPainter(img)
            for y in range(sy):
                for x in range(sx):
                    result = self._get_tile_color(z, y, x)
                    if result is None:
                        continue
                    color, _ = result
                    p.setBrush(QBrush(color))
                    p.setPen(Qt.PenStyle.NoPen)
                    p.drawRect(x * scale, y * scale, scale, scale)
            p.end()
            fname = os.path.join(dir_path, f"layer_{z:03d}.png")
            img.save(fname)


# ---------------------------------------------------------------------------
# Tileset Editor Dialog (3D-aware)
# ---------------------------------------------------------------------------


class TilesetEditor3DDialog(QDialog):
    def __init__(self, config: Optional[TilesetConfig] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Tileset Editor")
        self.setMinimumSize(650, 550)
        self._colors_map: Dict[str, str] = {}
        self._setup_ui(config)

    def _setup_ui(self, config):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(config.name if config else "New Tileset")
        form.addRow("Name:", self.name_edit); layout.addLayout(form)

        tg = QGroupBox("Tiles (name, color, solid, weight)")
        tl = QVBoxLayout(tg)
        self.tiles_table = QTableWidget(0, 4)
        self.tiles_table.setHorizontalHeaderLabels(["Name", "Color", "Solid", "Weight"])
        self.tiles_table.horizontalHeader().setStretchLastSection(True)
        tl.addWidget(self.tiles_table)
        br = QHBoxLayout()
        ba = QPushButton("Add Tile"); ba.clicked.connect(self._add_tile_row)
        bd = QPushButton("Remove"); bd.clicked.connect(self._del_tile_row)
        br.addWidget(ba); br.addWidget(bd); tl.addLayout(br)
        layout.addWidget(tg)

        rg = QGroupBox("Horizontal Rules (same-layer: tile_a <-> tile_b)")
        rl = QVBoxLayout(rg)
        self.hrules_table = QTableWidget(0, 2)
        self.hrules_table.setHorizontalHeaderLabels(["Tile A", "Tile B"])
        self.hrules_table.horizontalHeader().setStretchLastSection(True)
        rl.addWidget(self.hrules_table)
        br2 = QHBoxLayout()
        ba2 = QPushButton("Add"); ba2.clicked.connect(self._add_hrule)
        bd2 = QPushButton("Remove"); bd2.clicked.connect(self._del_hrule)
        br2.addWidget(ba2); br2.addWidget(bd2); rl.addLayout(br2)
        layout.addWidget(rg)

        vg = QGroupBox("Vertical Rules (layer stacking: above <-> below)")
        vl = QVBoxLayout(vg)
        self.vrules_table = QTableWidget(0, 2)
        self.vrules_table.setHorizontalHeaderLabels(["Above", "Below"])
        self.vrules_table.horizontalHeader().setStretchLastSection(True)
        vl.addWidget(self.vrules_table)
        br3 = QHBoxLayout()
        ba3 = QPushButton("Add"); ba3.clicked.connect(self._add_vrule)
        bd3 = QPushButton("Remove"); bd3.clicked.connect(self._del_vrule)
        br3.addWidget(ba3); br3.addWidget(bd3); vl.addLayout(br3)
        layout.addWidget(vg)

        if config:
            for t in config.tiles:
                row = self.tiles_table.rowCount(); self.tiles_table.insertRow(row)
                self.tiles_table.setItem(row, 0, QTableWidgetItem(t))
                ci = QTableWidgetItem(config.colors.get(t, "#ffffff"))
                ci.setBackground(QColor(config.colors.get(t, "#ffffff")))
                self.tiles_table.setItem(row, 1, ci)
                solid_item = QTableWidgetItem("Yes" if config.solid.get(t, True) else "No")
                self.tiles_table.setItem(row, 2, solid_item)
                self.tiles_table.setItem(row, 3, QTableWidgetItem(str(config.weights.get(t, 1.0))))
                self._colors_map[t] = config.colors.get(t, "#ffffff")

            # Parse rules into horizontal and vertical
            h_seen, v_seen = set(), set()
            for a, d, b in config.rules:
                if d in ("UP", "DOWN"):
                    key = tuple(sorted([a, b]))
                    if key not in v_seen:
                        v_seen.add(key)
                        row = self.vrules_table.rowCount(); self.vrules_table.insertRow(row)
                        self.vrules_table.setItem(row, 0, QTableWidgetItem(a))
                        self.vrules_table.setItem(row, 1, QTableWidgetItem(b))
                else:
                    key = tuple(sorted([a, b]))
                    if key not in h_seen:
                        h_seen.add(key)
                        row = self.hrules_table.rowCount(); self.hrules_table.insertRow(row)
                        self.hrules_table.setItem(row, 0, QTableWidgetItem(a))
                        self.hrules_table.setItem(row, 1, QTableWidgetItem(b))

        self.tiles_table.cellDoubleClicked.connect(self._on_color_dblclick)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _add_tile_row(self):
        r = self.tiles_table.rowCount(); self.tiles_table.insertRow(r)
        self.tiles_table.setItem(r, 0, QTableWidgetItem(f"TILE_{r}"))
        ci = QTableWidgetItem("#ffffff"); ci.setBackground(QColor("#ffffff"))
        self.tiles_table.setItem(r, 1, ci)
        self.tiles_table.setItem(r, 2, QTableWidgetItem("Yes"))
        self.tiles_table.setItem(r, 3, QTableWidgetItem("1.0"))

    def _del_tile_row(self):
        for r in sorted(set(i.row() for i in self.tiles_table.selectedItems()), reverse=True):
            self.tiles_table.removeRow(r)

    def _add_hrule(self):
        r = self.hrules_table.rowCount(); self.hrules_table.insertRow(r)
        self.hrules_table.setItem(r, 0, QTableWidgetItem(""))
        self.hrules_table.setItem(r, 1, QTableWidgetItem(""))

    def _del_hrule(self):
        for r in sorted(set(i.row() for i in self.hrules_table.selectedItems()), reverse=True):
            self.hrules_table.removeRow(r)

    def _add_vrule(self):
        r = self.vrules_table.rowCount(); self.vrules_table.insertRow(r)
        self.vrules_table.setItem(r, 0, QTableWidgetItem(""))
        self.vrules_table.setItem(r, 1, QTableWidgetItem(""))

    def _del_vrule(self):
        for r in sorted(set(i.row() for i in self.vrules_table.selectedItems()), reverse=True):
            self.vrules_table.removeRow(r)

    def _on_color_dblclick(self, row, col):
        if col == 1:
            name = self.tiles_table.item(row, 0)
            cur = self._colors_map.get(name.text() if name else "", "#ffffff")
            c = QColorDialog.getColor(QColor(cur), self)
            if c.isValid():
                item = self.tiles_table.item(row, 1)
                if item: item.setText(c.name()); item.setBackground(c)
                if name: self._colors_map[name.text()] = c.name()

    def _read_pairs(self, table):
        pairs = []
        for r in range(table.rowCount()):
            a, b = table.item(r, 0), table.item(r, 1)
            if a and b:
                aa, bb = a.text().strip(), b.text().strip()
                if aa and bb: pairs.append((aa, bb))
        return pairs

    def get_config(self) -> TilesetConfig:
        tiles, colors, weights, solid = [], {}, {}, {}
        for r in range(self.tiles_table.rowCount()):
            ni = self.tiles_table.item(r, 0)
            if not ni: continue
            t = ni.text().strip()
            if not t: continue
            tiles.append(t)
            ci = self.tiles_table.item(r, 1)
            colors[t] = ci.text().strip() if ci else "#ffffff"
            si = self.tiles_table.item(r, 2)
            solid[t] = si.text().strip().lower() != "no" if si else True
            wi = self.tiles_table.item(r, 3)
            try: weights[t] = float(wi.text().strip()) if wi else 1.0
            except ValueError: weights[t] = 1.0
        h_pairs = self._read_pairs(self.hrules_table)
        v_pairs = self._read_pairs(self.vrules_table)
        rules = _make_sym_rules(h_pairs, DIR_LIST_2D) + _make_vertical_sym_rules(v_pairs)
        return TilesetConfig(self.name_edit.text().strip() or "Untitled",
                             tiles, colors, solid, rules, weights)

# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

DARK_QSS = """
QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px;
            margin-top: 8px; padding-top: 14px; font-weight: bold; color: #cdd6f4; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background-color: #313244; border: 1px solid #45475a;
              border-radius: 5px; padding: 5px 12px; color: #cdd6f4; }
QPushButton:hover { background-color: #45475a; }
QPushButton:pressed { background-color: #585b70; }
QComboBox { background-color: #313244; border: 1px solid #45475a;
            border-radius: 5px; padding: 4px 8px; color: #cdd6f4; }
QComboBox QAbstractItemView { background-color: #313244; color: #cdd6f4;
                              selection-background-color: #45475a; }
QSpinBox { background-color: #313244; border: 1px solid #45475a;
           border-radius: 5px; padding: 4px; color: #cdd6f4; }
QSlider::groove:horizontal { background: #313244; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #89b4fa; width: 16px; height: 16px;
                             margin: -5px 0; border-radius: 8px; }
QProgressBar { background-color: #313244; border: 1px solid #45475a;
               border-radius: 5px; text-align: center; color: #cdd6f4; }
QProgressBar::chunk { background-color: #89b4fa; border-radius: 4px; }
QCheckBox { color: #cdd6f4; spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px;
                       border: 1px solid #45475a; background: #313244; }
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QRadioButton { color: #cdd6f4; spacing: 6px; }
QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px;
                          border: 1px solid #45475a; background: #313244; }
QRadioButton::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QLabel { color: #cdd6f4; }
QStatusBar { background-color: #181825; color: #a6adc8; }
QMenuBar { background-color: #181825; color: #cdd6f4; }
QMenuBar::item:selected { background-color: #313244; }
QMenu { background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; }
QMenu::item:selected { background-color: #313244; }
QTableWidget { background-color: #1e1e2e; color: #cdd6f4;
               gridline-color: #45475a; border: 1px solid #45475a; }
QHeaderView::section { background-color: #313244; color: #cdd6f4;
                       border: 1px solid #45475a; padding: 4px; }
QToolTip { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; padding: 4px; }
QScrollArea { border: none; }
QLineEdit { background-color: #313244; border: 1px solid #45475a;
            border-radius: 5px; padding: 4px; color: #cdd6f4; }
QDialog { background-color: #1e1e2e; color: #cdd6f4; }
"""


class MainWindow3D(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WFC 3D — Isometric Voxel Edition")
        self.resize(1400, 950)
        self._tilesets: Dict[str, TilesetConfig] = {
            "Voxel Terrain": _get_voxel_terrain(),
            "Voxel Dungeon": _get_voxel_dungeon(),
            "Voxel Cave": _get_voxel_cave(),
            "Crystal Cavern": _get_crystal_cavern(),
            "Maze": _get_maze(),
        }
        self.worker: Optional[WFCWorker3D] = None
        self._undo_stack: List[dict] = []
        self._redo_stack: List[dict] = []
        self._max_undo = 30
        self._is_running = False
        self._finish_thread: Optional[QThread] = None
        self._t_start = 0.0
        self._layer_anim_timer = QTimer()
        self._layer_anim_timer.timeout.connect(self._animate_layer)
        self._layer_anim_dir = 1

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self.setStyleSheet(DARK_QSS)
        self._init_grid()

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction("Export OBJ...", self._export_obj)
        fm.addAction("Export Layer PNGs...", self._export_layer_pngs)
        fm.addAction("Export Grid JSON...", self._export_json)
        fm.addAction("Save Screenshot...", self._export_screenshot)
        fm.addSeparator()
        fm.addAction("Save Tileset...", self._save_tileset)
        fm.addAction("Load Tileset...", self._load_tileset)
        fm.addSeparator()
        fm.addAction("Save State...", self._save_state)
        fm.addAction("Load State...", self._load_state)
        fm.addSeparator()
        fm.addAction("Exit", self.close, QKeySequence("Ctrl+Q"))
        em = mb.addMenu("Edit")
        em.addAction("Undo", self._undo, QKeySequence.Undo)
        em.addAction("Redo", self._redo, QKeySequence.Redo)
        vm = mb.addMenu("View")
        vm.addAction("Reset Camera", self.canvas.reset_view, QKeySequence("Ctrl+0"))
        vm.addAction("Top View", lambda: self._set_camera(0, 89))
        vm.addAction("Front View", lambda: self._set_camera(0, 0))
        vm.addAction("Side View", lambda: self._set_camera(90, 0))
        vm.addAction("Isometric View", lambda: self._set_camera(45, 35))
        tm = mb.addMenu("Tileset")
        tm.addAction("Edit Current...", self._edit_tileset)
        tm.addAction("New Tileset...", self._new_tileset)
        hm = mb.addMenu("Help")
        hm.addAction("About", lambda: QMessageBox.about(self, "About",
            "<h2>WFC 3D</h2><p>3D Wave Function Collapse with isometric voxel rendering.</p>"
            "<p><b>Controls:</b> Left-drag=rotate, Middle-drag=pan, Wheel=zoom, Double-click=reset</p>"))

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, self._step_once)
        QShortcut(QKeySequence("Return"), self, self._toggle_auto)
        QShortcut(QKeySequence("Escape"), self, self._stop)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)

    def _set_camera(self, az, el):
        self.canvas.azimuth = az; self.canvas.elevation = el
        self.canvas._zoom = 1.0; self.canvas._pan = QPoint(0, 0)
        self.canvas.update()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        ml = QHBoxLayout(central); ml.setContentsMargins(6, 6, 6, 6); ml.setSpacing(6)

        self.canvas = IsometricCanvas(); ml.addWidget(self.canvas, stretch=4)

        rs = QScrollArea(); rs.setWidgetResizable(True)
        rs.setMaximumWidth(340); rs.setMinimumWidth(270)
        rw = QWidget(); right = QVBoxLayout(rw); right.setSpacing(5)
        rs.setWidget(rw); ml.addWidget(rs, stretch=1)

        # Config
        cfg = QGroupBox("Configuration"); cl = QVBoxLayout(cfg); cl.setSpacing(3)
        cl.addWidget(QLabel("Tileset:"))
        self.ts_combo = QComboBox(); self.ts_combo.addItems(self._tilesets.keys())
        self.ts_combo.currentTextChanged.connect(self._on_tileset_changed)
        cl.addWidget(self.ts_combo)

        size_grid = QGroupBox("Grid Size"); sgl = QHBoxLayout(size_grid)
        sgl.addWidget(QLabel("X:"))
        self.sx_spin = QSpinBox(); self.sx_spin.setRange(2, 40); self.sx_spin.setValue(12)
        sgl.addWidget(self.sx_spin)
        sgl.addWidget(QLabel("Y:"))
        self.sy_spin = QSpinBox(); self.sy_spin.setRange(2, 40); self.sy_spin.setValue(12)
        sgl.addWidget(self.sy_spin)
        sgl.addWidget(QLabel("Z:"))
        self.sz_spin = QSpinBox(); self.sz_spin.setRange(2, 40); self.sz_spin.setValue(8)
        sgl.addWidget(self.sz_spin)
        cl.addWidget(size_grid)

        uniform_cb = QCheckBox("Uniform Size")
        uniform_cb.setChecked(True)
        uniform_cb.toggled.connect(lambda c: (
            self.sy_spin.setValue(self.sx_spin.value()),
            self.sz_spin.setValue(self.sx_spin.value())))
        self.sx_spin.valueChanged.connect(lambda v: (
            self.sy_spin.setValue(v), self.sz_spin.setValue(v)) if uniform_cb.isChecked() else None)
        cl.addWidget(uniform_cb)

        rs3 = QHBoxLayout(); rs3.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999); self.seed_spin.setValue(42)
        rs3.addWidget(self.seed_spin)
        br = QPushButton("🎲"); br.setFixedWidth(30); br.setToolTip("Random seed")
        br.clicked.connect(self._random_seed); rs3.addWidget(br)
        cl.addLayout(rs3)
        self.wrap_check = QCheckBox("Wrap Edges"); cl.addWidget(self.wrap_check)

        brs = QHBoxLayout()
        bs = QPushButton("Save Tileset"); bs.clicked.connect(self._save_tileset)
        bl = QPushButton("Load Tileset"); bl.clicked.connect(self._load_tileset)
        brs.addWidget(bs); brs.addWidget(bl); cl.addLayout(brs)
        right.addWidget(cfg)

        # Camera
        cam = QGroupBox("Camera"); cal = QVBoxLayout(cam)
        cal.addWidget(QLabel("Azimuth:"))
        self.az_slider = QSlider(Qt.Orientation.Horizontal)
        self.az_slider.setRange(0, 360); self.az_slider.setValue(45)
        self.az_slider.valueChanged.connect(lambda v: (setattr(self.canvas, 'azimuth', float(v)), self.canvas.update()))
        cal.addWidget(self.az_slider)
        cal.addWidget(QLabel("Elevation:"))
        self.el_slider = QSlider(Qt.Orientation.Horizontal)
        self.el_slider.setRange(-89, 89); self.el_slider.setValue(35)
        self.el_slider.valueChanged.connect(lambda v: (setattr(self.canvas, 'elevation', float(v)), self.canvas.update()))
        cal.addWidget(self.el_slider)
        cam_btns = QHBoxLayout()
        for label, az, el in [("Top", 0, 89), ("Front", 0, 0), ("Side", 90, 0), ("Iso", 45, 35)]:
            b = QPushButton(label); b.clicked.connect(lambda _, a=az, e=el: self._set_camera(a, e))
            cam_btns.addWidget(b)
        cal.addLayout(cam_btns)
        right.addWidget(cam)

        # Slicing
        slc = QGroupBox("Cross-Section"); sll = QVBoxLayout(slc)
        slice_row = QHBoxLayout()
        self.slice_none = QRadioButton("None"); self.slice_none.setChecked(True)
        self.slice_x = QRadioButton("X"); self.slice_y = QRadioButton("Y"); self.slice_z = QRadioButton("Z")
        for rb in [self.slice_none, self.slice_x, self.slice_y, self.slice_z]:
            rb.toggled.connect(self._on_slice_changed); slice_row.addWidget(rb)
        sll.addLayout(slice_row)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal); self.slice_slider.setRange(0, 39); self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self._on_slice_moved)
        self.slice_slider.setEnabled(False); sll.addWidget(self.slice_slider)
        self.slice_label = QLabel("Layer: 0")
        sll.addWidget(self.slice_label)
        right.addWidget(slc)

        # Visuals
        vis = QGroupBox("Visuals"); vl = QVBoxLayout(vis)
        self.trans_check = QCheckBox("Transparency (non-solid)")
        self.trans_check.setChecked(True)
        self.trans_check.toggled.connect(lambda c: (setattr(self.canvas, 'show_transparency', c), self.canvas.update()))
        vl.addWidget(self.trans_check)
        vl.addWidget(QLabel("Non-solid alpha:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(10, 255); self.alpha_slider.setValue(80)
        self.alpha_slider.valueChanged.connect(lambda v: (setattr(self.canvas, 'alpha_non_solid', v), self.canvas.update()))
        vl.addWidget(self.alpha_slider)
        self.grid_check = QCheckBox("Grid Lines"); self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(lambda c: (setattr(self.canvas, 'show_grid_lines', c), self.canvas.update()))
        vl.addWidget(self.grid_check)
        self.ent_check = QCheckBox("Entropy Heatmap")
        self.ent_check.toggled.connect(lambda c: (setattr(self.canvas, 'show_entropy', c), self.canvas.update()))
        vl.addWidget(self.ent_check)
        breset = QPushButton("Reset Camera"); breset.clicked.connect(self.canvas.reset_view)
        vl.addWidget(breset); right.addWidget(vis)

        # Layer Animation
        anim = QGroupBox("Layer Animation"); al = QVBoxLayout(anim)
        self.btn_anim = QPushButton("▶ Animate Layers")
        self.btn_anim.setCheckable(True)
        self.btn_anim.toggled.connect(self._toggle_layer_anim)
        al.addWidget(self.btn_anim)
        al.addWidget(QLabel("Anim Speed (ms):"))
        self.anim_speed = QSlider(Qt.Orientation.Horizontal)
        self.anim_speed.setRange(50, 2000); self.anim_speed.setValue(300)
        al.addWidget(self.anim_speed)
        right.addWidget(anim)

        # Paint Mode
        pnt = QGroupBox("Paint Mode"); pnl = QVBoxLayout(pnt)
        self.paint_check = QCheckBox("Enable Paint Mode")
        self.paint_check.toggled.connect(self._on_paint_toggled)
        pnl.addWidget(self.paint_check)
        pt_row = QHBoxLayout(); pt_row.addWidget(QLabel("Tile:"))
        self.paint_combo = QComboBox(); self.paint_combo.setMinimumWidth(100)
        pnl.addLayout(pt_row); pnl.addWidget(self.paint_combo)
        self._update_paint_combo()
        self.paint_combo.currentTextChanged.connect(self._on_paint_tile_changed)
        paint_info = QLabel("Left-click cells to place.\nRight-drag to still pan.")
        paint_info.setFont(QFont("Monospace", 8))
        paint_info.setStyleSheet("color: #a6adc8;")
        pnl.addWidget(paint_info)
        right.addWidget(pnt)

        # Simulation
        sim = QGroupBox("Simulation"); sl = QVBoxLayout(sim)
        sl.addWidget(QLabel("Speed (ms):"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500); self.speed_slider.setValue(50)
        self.speed_label = QLabel("50 ms")
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"{v} ms"))
        sl.addWidget(self.speed_slider); sl.addWidget(self.speed_label)
        spt_row = QHBoxLayout(); spt_row.addWidget(QLabel("Steps/tick:"))
        self.spt_spin = QSpinBox(); self.spt_spin.setRange(1, 100); self.spt_spin.setValue(1)
        spt_row.addWidget(self.spt_spin); sl.addLayout(spt_row)
        br2 = QHBoxLayout()
        self.btn_step = QPushButton("Step ▶"); self.btn_step.clicked.connect(self._step_once)
        self.btn_auto = QPushButton("Auto ▶▶"); self.btn_auto.clicked.connect(self._toggle_auto)
        br2.addWidget(self.btn_step); br2.addWidget(self.btn_auto); sl.addLayout(br2)
        self.btn_finish = QPushButton("⚡ Finish Instantly"); self.btn_finish.clicked.connect(self._finish_instantly)
        sl.addWidget(self.btn_finish)
        self.retry_check = QCheckBox("Auto-retry on contradiction"); self.retry_check.setChecked(True)
        sl.addWidget(self.retry_check)
        br3 = QHBoxLayout()
        bu = QPushButton("↩ Undo"); bu.clicked.connect(self._undo)
        bredo = QPushButton("Redo ↪"); bredo.clicked.connect(self._redo)
        br3.addWidget(bu); br3.addWidget(bredo); sl.addLayout(br3)
        self.btn_init = QPushButton("🔄 Reinitialize"); self.btn_init.clicked.connect(self._init_grid)
        sl.addWidget(self.btn_init); right.addWidget(sim)

        # Stats
        st = QGroupBox("Statistics"); stl = QVBoxLayout(st)
        self.prog_bar = QProgressBar(); self.prog_bar.setTextVisible(True)
        stl.addWidget(self.prog_bar)
        self.stats_lbl = QLabel("Ready"); self.stats_lbl.setWordWrap(True)
        stl.addWidget(self.stats_lbl)
        self.tile_counts_lbl = QLabel(""); self.tile_counts_lbl.setWordWrap(True)
        self.tile_counts_lbl.setFont(QFont("Monospace", 8))
        stl.addWidget(self.tile_counts_lbl)
        right.addWidget(st)
        right.addStretch()

        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Left-drag=rotate, Middle-drag=pan, Wheel=zoom")
        self._timer = QTimer(); self._timer.timeout.connect(self._auto_step)
        self.canvas.cell_hovered.connect(self._on_cell_hovered)
        self.canvas.cell_painted.connect(self._on_cell_painted)

    def _on_tileset_changed(self, name):
        if name in self._tilesets:
            self._init_grid()

    def _on_slice_changed(self):
        if self.slice_none.isChecked():
            self.canvas.slice_axis = "None"; self.slice_slider.setEnabled(False)
        elif self.slice_x.isChecked():
            self.canvas.slice_axis = "X"; self.slice_slider.setEnabled(True)
            self.slice_slider.setRange(0, self.sx_spin.value() - 1)
        elif self.slice_y.isChecked():
            self.canvas.slice_axis = "Y"; self.slice_slider.setEnabled(True)
            self.slice_slider.setRange(0, self.sy_spin.value() - 1)
        elif self.slice_z.isChecked():
            self.canvas.slice_axis = "Z"; self.slice_slider.setEnabled(True)
            self.slice_slider.setRange(0, self.sz_spin.value() - 1)
        self.canvas.slice_pos = self.slice_slider.value()
        self.canvas.update()

    def _on_slice_moved(self, v):
        self.canvas.slice_pos = v
        self.slice_label.setText(f"Layer: {v}")
        self.canvas.update()

    def _toggle_layer_anim(self, checked):
        if checked:
            self._layer_anim_dir = 1
            self.slice_z.setChecked(True)
            self.slice_slider.setValue(0)
            self._layer_anim_timer.start(self.anim_speed.value())
        else:
            self._layer_anim_timer.stop()

    def _animate_layer(self):
        self._layer_anim_timer.setInterval(self.anim_speed.value())
        cur = self.slice_slider.value()
        mx = self.slice_slider.maximum()
        nxt = cur + self._layer_anim_dir
        if nxt > mx:
            self._layer_anim_dir = -1; nxt = mx
        elif nxt < 0:
            self._layer_anim_dir = 1; nxt = 0
        self.slice_slider.setValue(nxt)

    def _save_tileset(self):
        ts = self._tilesets[self.ts_combo.currentText()]
        path, _ = QFileDialog.getSaveFileName(self, "Save", f"{ts.name}.json", "JSON (*.json)")
        if path:
            with open(path, 'w') as f: f.write(ts.to_json())
            self.status_bar.showMessage(f"Saved: {path}", 3000)

    def _load_tileset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load", "", "JSON (*.json)")
        if path:
            try:
                with open(path) as f: ts = TilesetConfig.from_json(f.read())
                self._tilesets[ts.name] = ts; self.ts_combo.addItem(ts.name)
                self.ts_combo.setCurrentText(ts.name); self._init_grid()
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def _edit_tileset(self):
        name = self.ts_combo.currentText()
        if name not in self._tilesets: return
        dlg = TilesetEditor3DDialog(self._tilesets[name], self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            nc = dlg.get_config(); self._tilesets[nc.name] = nc
            idx = self.ts_combo.findText(name)
            if idx >= 0: self.ts_combo.removeItem(idx)
            self.ts_combo.addItem(nc.name); self.ts_combo.setCurrentText(nc.name)
            self._init_grid()

    def _new_tileset(self):
        dlg = TilesetEditor3DDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            nc = dlg.get_config()
            if nc.tiles:
                self._tilesets[nc.name] = nc; self.ts_combo.addItem(nc.name)
                self.ts_combo.setCurrentText(nc.name); self._init_grid()

    def _init_grid(self):
        self._stop(); self._undo_stack.clear(); self._redo_stack.clear()
        self._layer_anim_timer.stop(); self.btn_anim.setChecked(False)
        name = self.ts_combo.currentText()
        if name not in self._tilesets:
            self.status_bar.showMessage("No tileset selected.")
            return
        config = self._tilesets[name]
        sx, sy, sz = self.sx_spin.value(), self.sy_spin.value(), self.sz_spin.value()
        try:
            self.worker = WFCWorker3D(sx, sy, sz, config, self.seed_spin.value(), self.wrap_check.isChecked())
        except ValueError as e:
            self.status_bar.showMessage(str(e))
            return
        self.canvas.set_worker(self.worker, config.colors, config.solid)
        # Update slice range
        self._on_slice_changed()
        self._update_paint_combo()
        self._t_start = 0; self._update_stats()
        self.status_bar.showMessage(f"Grid initialized: {sx}x{sy}x{sz}")

    def _random_seed(self):
        self.seed_spin.setValue(random.randint(0, 999999)); self._init_grid()

    def _push_undo(self):
        if not self.worker: return
        self._undo_stack.append(self.worker.get_snapshot())
        if len(self._undo_stack) > self._max_undo: self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack or not self.worker: return
        self._stop(); self._redo_stack.append(self.worker.get_snapshot())
        self.worker.restore_snapshot(self._undo_stack.pop())
        self.canvas.update(); self._update_stats()

    def _redo(self):
        if not self._redo_stack or not self.worker: return
        self._stop(); self._undo_stack.append(self.worker.get_snapshot())
        self.worker.restore_snapshot(self._redo_stack.pop())
        self.canvas.update(); self._update_stats()

    def _step_once(self):
        if not self.worker or self._is_running: return
        if self._t_start == 0: self._t_start = time.perf_counter()
        self._push_undo()
        if not self.worker.run_step():
            self._stop()
            if self.retry_check.isChecked() and not self.worker.is_complete():
                QTimer.singleShot(100, self._retry); return
            if self.worker.is_complete():
                self.status_bar.showMessage("Complete!")
        self.canvas.update(); self._update_stats()

    def _auto_step(self):
        if not self.worker: return
        self._push_undo()
        steps_this_tick = self.spt_spin.value()
        for _ in range(steps_this_tick):
            if not self.worker.run_step():
                self._timer.stop(); self.btn_auto.setText("Auto ▶▶"); self._is_running = False
                if self.retry_check.isChecked() and not self.worker.is_complete():
                    QTimer.singleShot(100, self._retry); return
                if self.worker.is_complete():
                    self.status_bar.showMessage("Complete!")
                break
        self.canvas.update(); self._update_stats()

    def _toggle_auto(self):
        if self._timer.isActive(): self._stop()
        else:
            if self._t_start == 0: self._t_start = time.perf_counter()
            self._timer.start(self.speed_slider.value())
            self.btn_auto.setText("⏹ Stop"); self._is_running = True

    def _stop(self):
        self._timer.stop(); self.btn_auto.setText("Auto ▶▶"); self._is_running = False
        if self.worker: self.worker.stop()
        if self._finish_thread and self._finish_thread.isRunning():
            if self.worker: self.worker.stop()
            self._finish_thread.quit(); self._finish_thread.wait(2000)

    def _finish_instantly(self):
        if not self.worker or self._is_running: return
        self._stop(); self._push_undo()
        if self._t_start == 0: self._t_start = time.perf_counter()
        self._finish_thread = QThread()
        iw = InstantFinishWorker(self.worker); iw.moveToThread(self._finish_thread)
        self._finish_thread.started.connect(iw.run)
        iw.progress.connect(self.prog_bar.setValue)
        iw.finished.connect(self._on_instant_done)
        self._finish_thread.start(); self._is_running = True; self.btn_finish.setEnabled(False)

    def _on_instant_done(self, success, elapsed):
        self._finish_thread.quit(); self._finish_thread.wait(3000)
        self._is_running = False; self.btn_finish.setEnabled(True)
        self.canvas.update(); self._update_stats()
        if success:
            self.status_bar.showMessage(f"Done in {elapsed:.3f}s!", 5000)
        elif self.retry_check.isChecked():
            QTimer.singleShot(100, self._retry)
        else:
            self.status_bar.showMessage("Contradiction.", 5000)

    def _retry(self):
        self._stop(); base = self.seed_spin.value()
        for attempt in range(1, 11):
            ns = (base + attempt * 7919) % 1000000
            self.seed_spin.setValue(ns)
            self._undo_stack.clear(); self._redo_stack.clear()
            config = self._tilesets[self.ts_combo.currentText()]
            sx, sy, sz = self.sx_spin.value(), self.sy_spin.value(), self.sz_spin.value()
            self.worker = WFCWorker3D(sx, sy, sz, config, ns, self.wrap_check.isChecked())
            self.canvas.set_worker(self.worker, config.colors, config.solid)
            self._t_start = time.perf_counter()
            limit = int(np.prod(self.worker.shape)) * 2
            for _ in range(limit):
                if not self.worker.run_step(): break
            if self.worker.is_complete():
                self.canvas.update(); self._update_stats()
                self.status_bar.showMessage(f"Success: attempt {attempt}, seed {ns}!", 5000)
                return
        self.status_bar.showMessage("Failed after 10 retries.", 5000)

    # ---- Paint Mode ----
    def _update_paint_combo(self):
        self.paint_combo.blockSignals(True)
        self.paint_combo.clear()
        if self.worker:
            self.paint_combo.addItems(self.worker.tiles)
        self.paint_combo.blockSignals(False)
        self._on_paint_tile_changed(self.paint_combo.currentText())

    def _on_paint_toggled(self, checked):
        self.canvas._paint_mode = checked
        self.canvas.setCursor(
            Qt.CursorShape.CrossCursor if checked else Qt.CursorShape.ArrowCursor)

    def _on_paint_tile_changed(self, text):
        self.canvas._paint_tile = text

    def _on_cell_painted(self, z, y, x, tname):
        if not self.worker: return
        self._push_undo()
        idx = self.worker._tile_to_idx.get(tname)
        if idx is None: return
        self.worker.wave[z, y, x] = np.uint64(1 << idx)
        self.worker.collapsed[z, y, x] = 1
        self.worker.step_count += 1
        if not self.worker._propagate((z, y, x)):
            self._undo()
            self.status_bar.showMessage(
                f"Contradiction painting {tname} at ({x},{y},{z})!", 3000)
            return
        self.canvas.update(); self._update_stats()
        self.status_bar.showMessage(f"Painted {tname} at ({x},{y},{z})", 2000)

    def _on_cell_hovered(self, z, y, x, tname):
        if z < 0:
            self.status_bar.showMessage("")
        else:
            ent = get_popcount(self.worker.wave[z, y, x]) if self.worker else 0
            extra = "" if ent <= 1 else f" [entropy={ent}]"
            self.status_bar.showMessage(f"Cell ({x}, {y}, {z}): {tname}{extra}")

    # ---- Screenshot Export ----
    def _export_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "wfc_screenshot.png", "PNG (*.png)")
        if path:
            pixmap = self.canvas.grab()
            pixmap.save(path)
            self.status_bar.showMessage(f"Screenshot saved: {path}", 3000)

    # ---- Save/Load Grid State ----
    def _save_state(self):
        if not self.worker: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save State", "wfc_state.json", "JSON (*.json)")
        if not path: return
        state = {
            "version": 2,
            "tileset_name": self.ts_combo.currentText(),
            "seed": self.seed_spin.value(),
            "wrap": self.wrap_check.isChecked(),
            "size": [self.sx_spin.value(), self.sy_spin.value(), self.sz_spin.value()],
            "step_count": self.worker.step_count,
            "wave": self.worker.wave.tolist(),
            "collapsed": self.worker.collapsed.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(state, f)
        self.status_bar.showMessage(f"State saved: {path}", 3000)

    def _load_state(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load State", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path) as f:
                state = json.load(f)
            ts_name = state.get("tileset_name", "")
            if ts_name not in self._tilesets:
                QMessageBox.warning(self, "Error",
                    f"Tileset '{ts_name}' not found. Load the tileset first.")
                return
            sx, sy, sz = state["size"]
            self.sx_spin.setValue(sx); self.sy_spin.setValue(sy); self.sz_spin.setValue(sz)
            self.seed_spin.setValue(state.get("seed", 0))
            self.wrap_check.setChecked(state.get("wrap", False))
            self.ts_combo.setCurrentText(ts_name)
            config = self._tilesets[ts_name]
            self._stop()
            self.worker = WFCWorker3D(sx, sy, sz, config, state.get("seed", 0),
                                       state.get("wrap", False))
            self.worker.wave = np.array(state["wave"], dtype=np.uint64)
            self.worker.collapsed = np.array(state["collapsed"], dtype=np.uint8)
            self.worker.step_count = state.get("step_count", 0)
            self.canvas.set_worker(self.worker, config.colors, config.solid)
            self._undo_stack.clear(); self._redo_stack.clear()
            self._t_start = 0
            self._on_slice_changed()
            self._update_stats()
            self.status_bar.showMessage(f"State loaded: {path}", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _update_stats(self):
        if not self.worker: return
        c = int(np.sum(self.worker.collapsed)); t = int(np.prod(self.worker.shape))
        self.prog_bar.setValue(int(c / t * 100) if t else 0)
        el = time.perf_counter() - self._t_start if self._t_start > 0 else 0
        cps = c / el if el > 0.01 else 0
        ok = "✅ Complete" if self.worker.is_complete() else ""
        self.stats_lbl.setText(
            f"Collapsed: {c}/{t} ({int(c/t*100) if t else 0}%)\n"
            f"Steps: {self.worker.step_count}\nTime: {el:.2f}s | {cps:.0f} cells/s\n{ok}")
        # Tile count statistics
        counts = {}
        for i, tname in enumerate(self.worker.tiles):
            mask = np.uint64(1 << i)
            cnt = int(np.sum((self.worker.wave == mask) & (self.worker.collapsed == 1)))
            if cnt > 0:
                counts[tname] = cnt
        if counts:
            lines = "\n".join(f"  {name}: {n}" for name, n in
                              sorted(counts.items(), key=lambda x: -x[1]))
            self.tile_counts_lbl.setText(f"Tile counts:\n{lines}")
        else:
            self.tile_counts_lbl.setText("")

    def _export_obj(self):
        if not self.worker: return
        path, _ = QFileDialog.getSaveFileName(self, "Export OBJ", "wfc_3d.obj", "OBJ (*.obj)")
        if path:
            self.canvas.export_obj(path)
            self.status_bar.showMessage(f"Exported: {path}", 5000)

    def _export_layer_pngs(self):
        if not self.worker: return
        dpath = QFileDialog.getExistingDirectory(self, "Select Folder for Layer PNGs")
        if dpath:
            self.canvas.export_layer_pngs(dpath)
            self.status_bar.showMessage(f"Exported {self.worker.sz} layers to {dpath}", 5000)

    def _export_json(self):
        if not self.worker: return
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "wfc_3d.json", "JSON (*.json)")
        if path:
            sz, sy, sx = self.worker.shape
            layers = []
            for z in range(sz):
                grid = []
                for y in range(sy):
                    row = []
                    for x in range(sx):
                        m = self.worker.wave[z, y, x]; e = get_popcount(m)
                        row.append(self.worker.tiles[int(m).bit_length() - 1] if e == 1 else f"?({e})")
                    grid.append(row)
                layers.append(grid)
            data = {"tileset": self.ts_combo.currentText(),
                    "size": [sx, sy, sz], "seed": self.seed_spin.value(),
                    "steps": self.worker.step_count, "layers": layers}
            with open(path, 'w') as f: json.dump(data, f, indent=2)
            self.status_bar.showMessage(f"Exported: {path}", 3000)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow3D(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()