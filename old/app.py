"""
Wave Function Collapse (WFC) — Enhanced High-Performance Edition
=================================================================
Features:
- Bitmasking: Superpositions stored as uint64 (up to 64 tiles).
- Popcount Table: O(1) entropy calculation.
- Min-Heap Priority Queue: O(log n) lowest-entropy observation.
- Direction-Specific Adjacency: Correct per-direction rule propagation.
- Multi-threading: Instant-finish runs in QThread to keep GUI responsive.
- Undo/Redo: Snapshot-based history.
- Zoom & Pan: Mouse wheel zoom, middle-click/drag pan.
- Export: PNG image export, JSON grid data export.
- Dark Theme: Modern dark UI with Fusion style.
- Tileset Editor: Create and edit tilesets in-app.
- Performance Stats: Cells/sec, elapsed time, step counter.
- Contradiction Auto-Retry: Automatically retries with new seed.
- Keyboard Shortcuts: Space=step, Enter=auto, Esc=stop, Ctrl+Z=undo.
- Grid Lines: Toggle grid overlay on canvas.
- Cell Hover Info: Tooltip showing cell coordinates and tile state.
- Multiple Built-in Tilesets: Terrain, Dungeon, City, Circuit, Cave.
"""

from __future__ import annotations

import copy
import heapq
import logging
import json
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PySide6.QtCore import (
    Qt, QTimer, QRectF, Signal, QObject, QThread, QPoint, QSize, QEvent
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QKeySequence, QShortcut,
    QPainterPath, QPixmap, QAction
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSpinBox, QLabel, QSlider, QGroupBox, QMessageBox,
    QComboBox, QCheckBox, QStatusBar, QProgressBar, QFileDialog,
    QLineEdit, QScrollArea, QColorDialog, QTableWidget, QTableWidgetItem,
    QDialog, QDialogButtonBox, QFormLayout, QAbstractItemView,
    QToolTip, QMenuBar, QMenu, QSizePolicy, QToolBar
)

# ---------------------------------------------------------------------------
# Performance Utilities
# ---------------------------------------------------------------------------

POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def get_popcount(n: int) -> int:
    """Fast popcount for 64-bit integers using the lookup table."""
    return (POPCOUNT_TABLE[n & 0xff] +
            POPCOUNT_TABLE[(n >> 8) & 0xff] +
            POPCOUNT_TABLE[(n >> 16) & 0xff] +
            POPCOUNT_TABLE[(n >> 24) & 0xff] +
            POPCOUNT_TABLE[(n >> 32) & 0xff] +
            POPCOUNT_TABLE[(n >> 40) & 0xff] +
            POPCOUNT_TABLE[(n >> 48) & 0xff] +
            POPCOUNT_TABLE[(n >> 56) & 0xff])

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

class Direction(IntEnum):
    UP = 0; DOWN = 1; LEFT = 2; RIGHT = 3; FORWARD = 4; BACKWARD = 5

    @property
    def delta(self) -> Tuple[int, ...]:
        return {0: (0, -1, 0), 1: (0, 1, 0), 2: (-1, 0, 0),
                3: (1, 0, 0), 4: (0, 0, 1), 5: (0, 0, -1)}[self]

    @property
    def opposite(self) -> Direction:
        """Return the opposite direction."""
        opp = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
        return Direction(opp[self])


@dataclass
class TilesetConfig:
    name: str
    tiles: List[str]
    colors: Dict[str, str]
    rules: List[Tuple[str, str, str]]  # (tile_a, direction, tile_b): "A can be to the LEFT of B"
    weights: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name, "tiles": self.tiles, "colors": self.colors,
            "rules": [list(r) for r in self.rules],
            "weights": self.weights
        }, indent=4)

    @staticmethod
    def from_json(data_str: str) -> TilesetConfig:
        d = json.loads(data_str)
        rules = [tuple(r) for r in d['rules']]
        return TilesetConfig(
            name=d['name'], tiles=d['tiles'], colors=d['colors'],
            rules=rules, weights=d.get('weights', {}))


# Direction name mapping for rule definitions
DIR_NAMES = {"UP": Direction.UP, "DOWN": Direction.DOWN,
             "LEFT": Direction.LEFT, "RIGHT": Direction.RIGHT,
             "FORWARD": Direction.FORWARD, "BACKWARD": Direction.BACKWARD}


def _make_sym_rules(bidirectional_rules: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """Helper: convert simple (A, B) pairs into direction-specific symmetric rules.
    Generates rules for all 4 directions (UP/DOWN/LEFT/RIGHT).
    For each pair (A, B), A and B can be neighbors in any horizontal/vertical direction."""
    result = []
    for a, b in bidirectional_rules:
        for dname in ["UP", "DOWN", "LEFT", "RIGHT"]:
            result.append((a, dname, b))
    return result


# ---------------------------------------------------------------------------
# Tileset Presets
# ---------------------------------------------------------------------------

def _get_terrain_tileset() -> TilesetConfig:
    tiles = ["WATER", "SAND", "GRASS", "FOREST", "MOUNTAIN"]
    colors = {
        "WATER": "#3498db", "SAND": "#f1c40f", "GRASS": "#2ecc71",
        "FOREST": "#196f3d", "MOUNTAIN": "#7f8c8d"
    }
    # Simple bidirectional adjacency (converted to per-direction rules)
    simple_rules = [
        ("WATER", "WATER"), ("SAND", "SAND"), ("GRASS", "GRASS"),
        ("FOREST", "FOREST"), ("MOUNTAIN", "MOUNTAIN"),
        ("WATER", "SAND"), ("SAND", "GRASS"), ("GRASS", "FOREST"),
        ("FOREST", "MOUNTAIN"),
    ]
    rules = _make_sym_rules(simple_rules)
    weights = {"WATER": 1.0, "SAND": 1.2, "GRASS": 2.0, "FOREST": 1.5, "MOUNTAIN": 0.5}
    return TilesetConfig("Terrain", tiles, colors, rules, weights)


def _get_dungeon_tileset() -> TilesetConfig:
    tiles = ["FLOOR", "WALL", "DOOR", "CORRIDOR", "ROOM"]
    colors = {
        "FLOOR": "#d4c5a9", "WALL": "#2c3e50", "DOOR": "#e67e22",
        "CORRIDOR": "#bdc3c7", "ROOM": "#ecf0f1"
    }
    simple_rules = [
        ("FLOOR", "FLOOR"), ("WALL", "WALL"), ("DOOR", "DOOR"),
        ("CORRIDOR", "CORRIDOR"), ("ROOM", "ROOM"),
        ("FLOOR", "DOOR"), ("DOOR", "FLOOR"), ("DOOR", "WALL"),
        ("FLOOR", "CORRIDOR"), ("CORRIDOR", "FLOOR"),
        ("FLOOR", "ROOM"), ("ROOM", "FLOOR"),
        ("ROOM", "WALL"), ("WALL", "ROOM"),
        ("CORRIDOR", "DOOR"), ("DOOR", "CORRIDOR"),
    ]
    rules = _make_sym_rules(simple_rules)
    weights = {"FLOOR": 2.0, "WALL": 3.0, "DOOR": 0.8, "CORRIDOR": 1.5, "ROOM": 1.0}
    return TilesetConfig("Dungeon", tiles, colors, rules, weights)


def _get_city_tileset() -> TilesetConfig:
    tiles = ["ROAD", "BUILDING", "PARK", "WATER", "PLAZA"]
    colors = {
        "ROAD": "#555555", "BUILDING": "#8e44ad", "PARK": "#27ae60",
        "WATER": "#2980b9", "PLAZA": "#f39c12"
    }
    simple_rules = [
        ("ROAD", "ROAD"), ("BUILDING", "BUILDING"), ("PARK", "PARK"),
        ("WATER", "WATER"), ("PLAZA", "PLAZA"),
        ("ROAD", "BUILDING"), ("ROAD", "PARK"), ("ROAD", "PLAZA"),
        ("PARK", "WATER"), ("WATER", "PARK"),
        ("PARK", "BUILDING"), ("BUILDING", "PLAZA"),
    ]
    rules = _make_sym_rules(simple_rules)
    weights = {"ROAD": 2.0, "BUILDING": 3.0, "PARK": 1.0, "WATER": 0.8, "PLAZA": 0.5}
    return TilesetConfig("City", tiles, colors, rules, weights)


def _get_circuit_tileset() -> TilesetConfig:
    tiles = ["WIRE_H", "WIRE_V", "WIRE_CROSS", "WIRE_TURN", "EMPTY", "NODE"]
    colors = {
        "WIRE_H": "#3498db", "WIRE_V": "#2980b9", "WIRE_CROSS": "#1abc9c",
        "WIRE_TURN": "#16a085", "EMPTY": "#1a1a2e", "NODE": "#e74c3c"
    }
    # Direction-specific rules for circuits
    rules = [
        # Horizontal wire connects left/right
        ("WIRE_H", "LEFT", "WIRE_H"), ("WIRE_H", "RIGHT", "WIRE_H"),
        ("WIRE_H", "LEFT", "WIRE_TURN"), ("WIRE_H", "RIGHT", "WIRE_TURN"),
        ("WIRE_H", "LEFT", "WIRE_CROSS"), ("WIRE_H", "RIGHT", "WIRE_CROSS"),
        ("WIRE_H", "LEFT", "NODE"), ("WIRE_H", "RIGHT", "NODE"),
        # Vertical wire connects up/down
        ("WIRE_V", "UP", "WIRE_V"), ("WIRE_V", "DOWN", "WIRE_V"),
        ("WIRE_V", "UP", "WIRE_TURN"), ("WIRE_V", "DOWN", "WIRE_TURN"),
        ("WIRE_V", "UP", "WIRE_CROSS"), ("WIRE_V", "DOWN", "WIRE_CROSS"),
        ("WIRE_V", "UP", "NODE"), ("WIRE_V", "DOWN", "NODE"),
        # Cross wire connects all directions
        ("WIRE_CROSS", "UP", "WIRE_V"), ("WIRE_CROSS", "DOWN", "WIRE_V"),
        ("WIRE_CROSS", "LEFT", "WIRE_H"), ("WIRE_CROSS", "RIGHT", "WIRE_H"),
        ("WIRE_CROSS", "UP", "WIRE_CROSS"), ("WIRE_CROSS", "DOWN", "WIRE_CROSS"),
        ("WIRE_CROSS", "LEFT", "WIRE_CROSS"), ("WIRE_CROSS", "RIGHT", "WIRE_CROSS"),
        # Turn wire
        ("WIRE_TURN", "UP", "WIRE_V"), ("WIRE_TURN", "DOWN", "WIRE_V"),
        ("WIRE_TURN", "LEFT", "WIRE_H"), ("WIRE_TURN", "RIGHT", "WIRE_H"),
        ("WIRE_TURN", "UP", "WIRE_TURN"), ("WIRE_TURN", "RIGHT", "WIRE_TURN"),
        ("WIRE_TURN", "DOWN", "WIRE_TURN"), ("WIRE_TURN", "LEFT", "WIRE_TURN"),
        # Node connects all
        ("NODE", "UP", "WIRE_V"), ("NODE", "DOWN", "WIRE_V"),
        ("NODE", "LEFT", "WIRE_H"), ("NODE", "RIGHT", "WIRE_H"),
        ("NODE", "UP", "WIRE_CROSS"), ("NODE", "DOWN", "WIRE_CROSS"),
        ("NODE", "LEFT", "WIRE_CROSS"), ("NODE", "RIGHT", "WIRE_CROSS"),
        ("NODE", "UP", "NODE"), ("NODE", "DOWN", "NODE"),
        ("NODE", "LEFT", "NODE"), ("NODE", "RIGHT", "NODE"),
        # Empty connects to empty
        ("EMPTY", "UP", "EMPTY"), ("EMPTY", "DOWN", "EMPTY"),
        ("EMPTY", "LEFT", "EMPTY"), ("EMPTY", "RIGHT", "EMPTY"),
    ]
    weights = {"WIRE_H": 1.5, "WIRE_V": 1.5, "WIRE_CROSS": 0.8,
               "WIRE_TURN": 1.2, "EMPTY": 2.0, "NODE": 0.5}
    return TilesetConfig("Circuit", tiles, colors, rules, weights)


def _get_cave_tileset() -> TilesetConfig:
    tiles = ["STONE", "EMPTY", "CRYSTAL", "WATER", "LAVA"]
    colors = {
        "STONE": "#616a6b", "EMPTY": "#1a1a2e", "CRYSTAL": "#9b59b6",
        "WATER": "#2471a3", "LAVA": "#e74c3c"
    }
    simple_rules = [
        ("STONE", "STONE"), ("EMPTY", "EMPTY"), ("CRYSTAL", "CRYSTAL"),
        ("WATER", "WATER"), ("LAVA", "LAVA"),
        ("STONE", "EMPTY"), ("EMPTY", "STONE"),
        ("EMPTY", "CRYSTAL"), ("CRYSTAL", "EMPTY"),
        ("EMPTY", "WATER"), ("WATER", "EMPTY"),
        ("STONE", "CRYSTAL"), ("CRYSTAL", "STONE"),
        ("STONE", "LAVA"), ("LAVA", "STONE"),
    ]
    rules = _make_sym_rules(simple_rules)
    weights = {"STONE": 3.0, "EMPTY": 2.0, "CRYSTAL": 0.5, "WATER": 1.0, "LAVA": 0.3}
    return TilesetConfig("Cave", tiles, colors, rules, weights)


# ---------------------------------------------------------------------------
# WFC Worker (Core Algorithm)
# ---------------------------------------------------------------------------

class WFCWorker(QObject):
    """Handles WFC computation. Can be used from any thread."""
    progress = Signal(float)
    cell_collapsed = Signal(int, int, int, int)  # z, y, x, tile_idx
    finished = Signal(bool)  # True if success, False if contradiction
    contradiction_pos = Signal(int, int, int)
    step_done = Signal()  # Emitted after each step

    def __init__(self, size: int, config: TilesetConfig, seed: int,
                 is_3d: bool, wrap: bool):
        super().__init__()
        self.size = size
        self.is_3d = is_3d
        self.wrap = wrap
        self.tiles = config.tiles
        self.n_tiles = len(self.tiles)
        self._tile_to_idx = {t: i for i, t in enumerate(self.tiles)}
        self.rng = np.random.default_rng(seed)
        self.weights = np.array(
            [config.weights.get(t, 1.0) for t in self.tiles], dtype=np.float64)

        # Direction-specific rules: rules[dir_idx][tile_idx] = bitmask of allowed neighbors
        self.rules = np.zeros((6, self.n_tiles), dtype=np.uint64)
        for tile_a, dir_name, tile_b in config.rules:
            if tile_a not in self._tile_to_idx or tile_b not in self._tile_to_idx:
                continue
            i_a = self._tile_to_idx[tile_a]
            i_b = self._tile_to_idx[tile_b]
            d_idx = int(DIR_NAMES[dir_name])
            # tile_a can have tile_b as its neighbor in direction dir_name
            self.rules[d_idx, i_a] |= (1 << i_b)
            # Also: tile_b can have tile_a as its neighbor in the opposite direction
            opp_idx = int(DIR_NAMES[dir_name].opposite)
            self.rules[opp_idx, i_b] |= (1 << i_a)

        self.shape = (size, size, size) if is_3d else (1, size, size)
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
        """Initialize the min-heap with all cells."""
        self._heap = []
        self._in_heap = np.zeros(self.shape, dtype=bool)
        for pos in np.ndindex(self.shape):
            ent = get_popcount(self.wave[pos])
            self._push(pos, ent)

    def _push(self, pos, entropy):
        """Push a cell onto the heap with its entropy."""
        # Add small noise to break ties deterministically by position
        tiebreak = pos[0] * 1e-10 + pos[1] * 1e-20 + pos[2] * 1e-30
        heapq.heappush(self._heap, (entropy, tiebreak, pos))
        self._in_heap[pos] = True

    def _remove_from_heap(self, target_pos):
        """Lazily remove by marking. Invalid entries are skipped during pop."""
        self._in_heap[target_pos] = False

    def _pop_lowest(self):
        """Pop the lowest-entropy uncollapsed cell from the heap."""
        while self._heap:
            ent, _, pos = heapq.heappop(self._heap)
            if not self._in_heap[pos]:
                continue  # Stale entry
            self._in_heap[pos] = False
            return pos, ent
        return None, -1

    def stop(self):
        self._stop_requested = True

    def get_snapshot(self) -> dict:
        """Capture current state for undo."""
        return {
            'wave': self.wave.copy(),
            'collapsed': self.collapsed.copy(),
            'step_count': self.step_count,
        }

    def restore_snapshot(self, snap: dict):
        """Restore state from a snapshot."""
        self.wave = snap['wave'].copy()
        self.collapsed = snap['collapsed'].copy()
        self.step_count = snap['step_count']
        self._init_heap()

    def run_step(self) -> bool:
        """Execute one iteration of WFC. Returns False if done or contradiction."""
        if self._stop_requested:
            return False

        # 1. Observe: Find lowest entropy uncollapsed cell via min-heap
        pos, ent = self._pop_lowest()
        if pos is None:
            return False  # All cells collapsed

        if ent == 0:
            self.contradiction_pos.emit(*pos)
            return False

        # 2. Collapse
        mask = self.wave[pos]
        poss = [i for i in range(self.n_tiles) if (mask >> i) & 1]
        w = self.weights[poss]
        w_sum = np.sum(w)
        if w_sum == 0:
            self.contradiction_pos.emit(*pos)
            return False
        chosen = self.rng.choice(poss, p=w / w_sum)

        self.wave[pos] = (1 << chosen)
        self.collapsed[pos] = 1
        self.step_count += 1

        # 3. Propagate
        if not self._propagate(pos):
            self.contradiction_pos.emit(*pos)
            return False

        # Re-push any affected neighbors back onto the heap with updated entropy
        # (Already handled during propagation — neighbors with changed masks get re-pushed)
        self.cell_collapsed.emit(*pos, chosen)
        self.step_done.emit()
        return True

    def _propagate(self, start_pos: Tuple[int, ...]) -> bool:
        """Propagate constraints using BFS with direction-specific bitmask rules."""
        stack = [start_pos]
        dirs = list(Direction) if self.is_3d else list(Direction)[:4]
        ndim = len(start_pos)

        while stack:
            curr = stack.pop()
            curr_mask = self.wave[curr]

            for d in dirs:
                d_idx = int(d)
                delta = d.delta
                neighbor = tuple(curr[i] + delta[i] for i in range(ndim))

                if self.wrap:
                    neighbor = tuple(neighbor[i] % self.shape[i] for i in range(ndim))
                elif not all(0 <= neighbor[i] < self.shape[i] for i in range(ndim)):
                    continue

                # Calculate what the neighbor is allowed to be based on
                # ALL possible tiles at the current cell in this direction
                allowed_for_dir = np.uint64(0)
                for i in range(self.n_tiles):
                    if (curr_mask >> i) & 1:
                        allowed_for_dir |= self.rules[d_idx, i]

                old_mask = self.wave[neighbor]
                new_mask = old_mask & allowed_for_dir

                if new_mask != old_mask:
                    if new_mask == 0:
                        return False
                    self.wave[neighbor] = new_mask
                    # Re-push neighbor with updated entropy
                    if self._in_heap[neighbor]:
                        self._remove_from_heap(neighbor)
                    self._push(neighbor, get_popcount(new_mask))
                    if not self.collapsed[neighbor]:
                        stack.append(neighbor)
        return True

    def is_complete(self) -> bool:
        return bool(np.all(self.collapsed))


# ---------------------------------------------------------------------------
# Threaded Instant-Finish Worker
# ---------------------------------------------------------------------------

class InstantFinishWorker(QObject):
    """Runs WFC to completion in a background thread."""
    progress = Signal(float)
    finished = Signal(bool, float)  # success, elapsed_seconds

    def __init__(self, worker: WFCWorker):
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
            collapsed = int(np.sum(self.worker.collapsed))
            self.progress.emit(collapsed / total)
        elapsed = time.perf_counter() - t0
        self.finished.emit(self.worker.is_complete(), elapsed)


# ---------------------------------------------------------------------------
# GUI Components
# ---------------------------------------------------------------------------

class WFCCanvas(QWidget):
    """Canvas widget for rendering the WFC grid with zoom/pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[WFCWorker] = None
        self._colors: Dict[str, str] = {}
        self.z_slice = 0
        self.show_entropy = False
        self.show_grid_lines = True
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        # Zoom/pan state
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._panning = False
        self._last_pan_pos = QPoint()

        # Hover state
        self._hover_cell = None  # (y, x) or None

    def set_worker(self, worker, colors):
        self.worker = worker
        self._colors = colors
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self.update()

    def _cell_size(self) -> float:
        if not self.worker:
            return 10
        base = min(self.width() / self.worker.size,
                    self.height() / self.worker.size)
        return base * self._zoom

    def _grid_origin(self) -> QPoint:
        if not self.worker:
            return QPoint(0, 0)
        cs = self._cell_size()
        grid_w = cs * self.worker.size
        grid_h = cs * self.worker.size
        ox = (self.width() - grid_w) / 2 + self._pan_offset.x()
        oy = (self.height() - grid_h) / 2 + self._pan_offset.y()
        return QPoint(int(ox), int(oy))

    def paintEvent(self, event):
        if not self.worker:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        z = self.z_slice
        cs = self._cell_size()
        origin = self._grid_origin()

        # Background
        p.fillRect(self.rect(), QColor("#1e1e2e"))

        for y in range(self.worker.size):
            for x in range(self.worker.size):
                rx = origin.x() + x * cs
                ry = origin.y() + y * cs
                rect = QRectF(rx, ry, cs, cs)

                # Skip if off-screen
                if rx + cs < 0 or ry + cs < 0 or rx > self.width() or ry > self.height():
                    continue

                mask = self.worker.wave[z, y, x]
                ent = get_popcount(mask)

                if self.show_entropy:
                    norm = (ent - 1) / (self.worker.n_tiles - 1) if self.worker.n_tiles > 1 else 0
                    color = QColor.fromHslF(0.6 * (1.0 - norm), 0.7, 0.5)
                elif ent == 1:
                    idx = mask.bit_length() - 1
                    color = QColor(self._colors[self.worker.tiles[idx]])
                else:
                    r, g, b, count = 0, 0, 0, 0
                    for i in range(self.worker.n_tiles):
                        if (mask >> i) & 1:
                            c = QColor(self._colors[self.worker.tiles[i]])
                            r += c.red(); g += c.green(); b += c.blue()
                            count += 1
                    color = QColor(r // count, g // count, b // count) if count > 0 else QColor(0, 0, 0)

                p.setBrush(QBrush(color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRect(rect)

                # Grid lines
                if self.show_grid_lines and cs > 4:
                    p.setPen(QPen(QColor(255, 255, 255, 30), 1))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(rect)

        # Hover highlight
        if self._hover_cell is not None and cs > 4:
            hy, hx = self._hover_cell
            if 0 <= hy < self.worker.size and 0 <= hx < self.worker.size:
                rx = origin.x() + hx * cs
                ry = origin.y() + hy * cs
                p.setPen(QPen(QColor(255, 255, 0, 180), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QRectF(rx, ry, cs, cs))

        p.end()

    def wheelEvent(self, event):
        """Zoom with mouse wheel."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom = max(0.1, min(20.0, self._zoom * factor))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._last_pan_pos
            self._pan_offset += delta
            self._last_pan_pos = event.pos()
            self.update()
        else:
            # Update hover cell
            self._update_hover(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _update_hover(self, pos: QPoint):
        if not self.worker:
            return
        cs = self._cell_size()
        origin = self._grid_origin()
        gx = (pos.x() - origin.x()) / cs
        gy = (pos.y() - origin.y()) / cs
        ix, iy = int(gx), int(gy)
        if 0 <= ix < self.worker.size and 0 <= iy < self.worker.size:
            self._hover_cell = (iy, ix)
            mask = self.worker.wave[self.z_slice, iy, ix]
            ent = get_popcount(mask)
            if ent == 1:
                idx = mask.bit_length() - 1
                tile_name = self.worker.tiles[idx]
                QToolTip.showText(
                    self.mapToGlobal(pos),
                    f"({iy}, {ix}) | Tile: {tile_name}")
            else:
                QToolTip.showText(
                    self.mapToGlobal(pos),
                    f"({iy}, {ix}) | Entropy: {ent} possibilities")
        else:
            self._hover_cell = None
            QToolTip.hideText()

    def export_png(self, path: str, scale: int = 1):
        """Export the current view as a PNG image."""
        if not self.worker:
            return
        s = self.worker.size
        img = QImage(s * scale, s * scale, QImage.Format.Format_RGB32)
        img.fill(QColor("#1e1e2e"))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cs = scale
        z = self.z_slice
        for y in range(s):
            for x in range(s):
                mask = self.worker.wave[z, y, x]
                ent = get_popcount(mask)
                if ent == 1:
                    idx = mask.bit_length() - 1
                    color = QColor(self._colors[self.worker.tiles[idx]])
                else:
                    r, g, b, count = 0, 0, 0, 0
                    for i in range(self.worker.n_tiles):
                        if (mask >> i) & 1:
                            c = QColor(self._colors[self.worker.tiles[i]])
                            r += c.red(); g += c.green(); b += c.blue()
                            count += 1
                    color = QColor(r // count, g // count, b // count) if count > 0 else QColor(0, 0, 0)
                p.setBrush(QBrush(color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRect(x * cs, y * cs, cs, cs)
        p.end()
        img.save(path)


# ---------------------------------------------------------------------------
# Tileset Editor Dialog
# ---------------------------------------------------------------------------

class TilesetEditorDialog(QDialog):
    """Dialog for creating and editing tilesets."""

    def __init__(self, config: Optional[TilesetConfig] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tileset Editor")
        self.setMinimumSize(600, 500)
        self._colors_map: Dict[str, str] = {}
        self._setup_ui(config)

    def _setup_ui(self, config: Optional[TilesetConfig]):
        layout = QVBoxLayout(self)

        # Name
        form = QFormLayout()
        self.name_edit = QLineEdit(config.name if config else "New Tileset")
        form.addRow("Name:", self.name_edit)
        layout.addLayout(form)

        # Tiles table
        tiles_grp = QGroupBox("Tiles (name, color, weight)")
        tiles_lay = QVBoxLayout(tiles_grp)
        self.tiles_table = QTableWidget(0, 3)
        self.tiles_table.setHorizontalHeaderLabels(["Name", "Color", "Weight"])
        self.tiles_table.horizontalHeader().setStretchLastSection(True)
        tiles_lay.addWidget(self.tiles_table)
        btn_row = QHBoxLayout()
        btn_add_tile = QPushButton("Add Tile")
        btn_add_tile.clicked.connect(self._add_tile_row)
        btn_del_tile = QPushButton("Remove Selected")
        btn_del_tile.clicked.connect(self._del_tile_row)
        btn_row.addWidget(btn_add_tile)
        btn_row.addWidget(btn_del_tile)
        tiles_lay.addLayout(btn_row)
        layout.addWidget(tiles_grp)

        # Rules table
        rules_grp = QGroupBox("Adjacency Rules (tile_a ↔ tile_b, bidirectional)")
        rules_lay = QVBoxLayout(rules_grp)
        self.rules_table = QTableWidget(0, 2)
        self.rules_table.setHorizontalHeaderLabels(["Tile A", "Tile B"])
        self.rules_table.horizontalHeader().setStretchLastSection(True)
        rules_lay.addWidget(self.rules_table)
        btn_row2 = QHBoxLayout()
        btn_add_rule = QPushButton("Add Rule")
        btn_add_rule.clicked.connect(self._add_rule_row)
        btn_del_rule = QPushButton("Remove Selected")
        btn_del_rule.clicked.connect(self._del_rule_row)
        btn_row2.addWidget(btn_add_rule)
        btn_row2.addWidget(btn_del_rule)
        rules_lay.addLayout(btn_row2)
        layout.addWidget(rules_grp)

        # Populate from existing config
        if config:
            for t in config.tiles:
                row = self.tiles_table.rowCount()
                self.tiles_table.insertRow(row)
                self.tiles_table.setItem(row, 0, QTableWidgetItem(t))
                color_item = QTableWidgetItem(config.colors.get(t, "#ffffff"))
                color_item.setBackground(QColor(config.colors.get(t, "#ffffff")))
                self.tiles_table.setItem(row, 1, color_item)
                self.tiles_table.setItem(row, 2, QTableWidgetItem(
                    str(config.weights.get(t, 1.0))))
                self._colors_map[t] = config.colors.get(t, "#ffffff")
            # Extract unique bidirectional rules
            seen = set()
            for a, d, b in config.rules:
                key = tuple(sorted([a, b]))
                if key not in seen:
                    seen.add(key)
                    row = self.rules_table.rowCount()
                    self.rules_table.insertRow(row)
                    self.rules_table.setItem(row, 0, QTableWidgetItem(a))
                    self.rules_table.setItem(row, 1, QTableWidgetItem(b))

        # Color picking on double-click
        self.tiles_table.cellDoubleClicked.connect(self._on_cell_double_click)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_tile_row(self):
        row = self.tiles_table.rowCount()
        self.tiles_table.insertRow(row)
        self.tiles_table.setItem(row, 0, QTableWidgetItem(f"TILE_{row}"))
        color_item = QTableWidgetItem("#ffffff")
        color_item.setBackground(QColor("#ffffff"))
        self.tiles_table.setItem(row, 1, color_item)
        self.tiles_table.setItem(row, 2, QTableWidgetItem("1.0"))

    def _del_tile_row(self):
        rows = set(i.row() for i in self.tiles_table.selectedItems())
        for r in sorted(rows, reverse=True):
            self.tiles_table.removeRow(r)

    def _add_rule_row(self):
        row = self.rules_table.rowCount()
        self.rules_table.insertRow(row)
        self.rules_table.setItem(row, 0, QTableWidgetItem(""))
        self.rules_table.setItem(row, 1, QTableWidgetItem(""))

    def _del_rule_row(self):
        rows = set(i.row() for i in self.rules_table.selectedItems())
        for r in sorted(rows, reverse=True):
            self.rules_table.removeRow(r)

    def _on_cell_double_click(self, row, col):
        if col == 1:  # Color column
            current = self._colors_map.get(
                self.tiles_table.item(row, 0).text() if self.tiles_table.item(row, 0) else "",
                "#ffffff")
            color = QColorDialog.getColor(QColor(current), self)
            if color.isValid():
                hex_color = color.name()
                item = self.tiles_table.item(row, 1)
                if item:
                    item.setText(hex_color)
                    item.setBackground(color)
                name_item = self.tiles_table.item(row, 0)
                if name_item:
                    self._colors_map[name_item.text()] = hex_color

    def get_config(self) -> TilesetConfig:
        tiles = []
        colors = {}
        weights = {}
        for r in range(self.tiles_table.rowCount()):
            name_item = self.tiles_table.item(r, 0)
            color_item = self.tiles_table.item(r, 1)
            weight_item = self.tiles_table.item(r, 2)
            if not name_item:
                continue
            tname = name_item.text().strip()
            if not tname:
                continue
            tiles.append(tname)
            colors[tname] = color_item.text().strip() if color_item else "#ffffff"
            try:
                weights[tname] = float(weight_item.text().strip()) if weight_item else 1.0
            except ValueError:
                weights[tname] = 1.0

        simple_rules = []
        for r in range(self.rules_table.rowCount()):
            a_item = self.rules_table.item(r, 0)
            b_item = self.rules_table.item(r, 1)
            if a_item and b_item:
                a, b = a_item.text().strip(), b_item.text().strip()
                if a and b:
                    simple_rules.append((a, b))

        rules = _make_sym_rules(simple_rules)
        return TilesetConfig(
            self.name_edit.text().strip() or "Untitled",
            tiles, colors, rules, weights)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WFC — Enhanced High-Performance Edition")
        self.resize(1280, 900)

        # Tileset library
        self._tilesets: Dict[str, TilesetConfig] = {
            "Terrain": _get_terrain_tileset(),
            "Dungeon": _get_dungeon_tileset(),
            "City": _get_city_tileset(),
            "Circuit": _get_circuit_tileset(),
            "Cave": _get_cave_tileset(),
        }

        self.worker: Optional[WFCWorker] = None
        self._undo_stack: List[dict] = []
        self._redo_stack: List[dict] = []
        self._max_undo = 50
        self._is_running = False
        self._finish_thread: Optional[QThread] = None
        self._t_start = 0.0

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self._apply_dark_theme()
        self._init_grid()

    # ---- Dark Theme ----
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; }
            QGroupBox { border: 1px solid #45475a; border-radius: 6px;
                        margin-top: 8px; padding-top: 14px; font-weight: bold; color: #cdd6f4; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background-color: #313244; border: 1px solid #45475a;
                          border-radius: 5px; padding: 6px 14px; color: #cdd6f4; }
            QPushButton:hover { background-color: #45475a; }
            QPushButton:pressed { background-color: #585b70; }
            QPushButton:checked { background-color: #585b70; border-color: #89b4fa; }
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
            QLabel { color: #cdd6f4; }
            QStatusBar { background-color: #181825; color: #a6adc8; }
            QMenuBar { background-color: #181825; color: #cdd6f4; }
            QMenuBar::item:selected { background-color: #313244; }
            QMenu { background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; }
            QMenu::item:selected { background-color: #313244; }
            QTableWidget { background-color: #1e1e2e; color: #cdd6f4;
                           gridline-color: #45475a; border: 1px solid #45475a; }
            QTableWidget::item { padding: 2px; }
            QHeaderView::section { background-color: #313244; color: #cdd6f4;
                                   border: 1px solid #45475a; padding: 4px; }
            QToolTip { background-color: #313244; color: #cdd6f4;
                       border: 1px solid #45475a; padding: 4px; }
            QScrollArea { border: none; }
            QLineEdit { background-color: #313244; border: 1px solid #45475a;
                        border-radius: 5px; padding: 4px; color: #cdd6f4; }
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
        """)

    # ---- Menu Bar ----
    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("Export PNG...", self._export_png, QKeySequence("Ctrl+E"))
        file_menu.addAction("Export Grid JSON...", self._export_json)
        file_menu.addSeparator()
        file_menu.addAction("Save Tileset...", self._save_tileset)
        file_menu.addAction("Load Tileset...", self._load_tileset)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, QKeySequence("Ctrl+Q"))

        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo", self._undo, QKeySequence.Undo)
        edit_menu.addAction("Redo", self._redo, QKeySequence.Redo)

        view_menu = menubar.addMenu("View")
        view_menu.addAction("Reset Zoom/Pan", self.canvas.reset_view, QKeySequence("Ctrl+0"))
        view_menu.addAction("Zoom In", lambda: self._zoom_by(1.3), QKeySequence.ZoomIn)
        view_menu.addAction("Zoom Out", lambda: self._zoom_by(1/1.3), QKeySequence.ZoomOut)

        tileset_menu = menubar.addMenu("Tileset")
        tileset_menu.addAction("Edit Current Tileset...", self._edit_tileset)
        tileset_menu.addAction("New Tileset...", self._new_tileset)

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self._show_about)
        help_menu.addAction("Keyboard Shortcuts", self._show_shortcuts)

    # ---- Keyboard Shortcuts ----
    def _build_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, self._step_once)
        QShortcut(QKeySequence("Return"), self, self._toggle_auto)
        QShortcut(QKeySequence("Escape"), self, self._stop)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)
        QShortcut(QKeySequence("R"), self, self._init_grid)

    def _show_shortcuts(self):
        QMessageBox.information(self, "Keyboard Shortcuts", """
<h3>Keyboard Shortcuts</h3>
<table>
<tr><td><b>Space</b></td><td>Single step</td></tr>
<tr><td><b>Enter</b></td><td>Toggle auto-step</td></tr>
<tr><td><b>Escape</b></td><td>Stop auto-step</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>
<tr><td><b>Ctrl+Y</b></td><td>Redo</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export PNG</td></tr>
<tr><td><b>Ctrl+Q</b></td><td>Quit</td></tr>
<tr><td><b>Ctrl+0</b></td><td>Reset zoom/pan</td></tr>
<tr><td><b>Ctrl++</b></td><td>Zoom in</td></tr>
<tr><td><b>Ctrl+-</b></td><td>Zoom out</td></tr>
<tr><td><b>R</b></td><td>Reinitialize grid</td></tr>
<tr><td><b>Mouse Wheel</b></td><td>Zoom in/out</td></tr>
<tr><td><b>Middle-click drag</b></td><td>Pan</td></tr>
</table>
""")

    # ---- UI Construction ----
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Canvas
        self.canvas = WFCCanvas()
        main_layout.addWidget(self.canvas, stretch=3)

        # Right panel in a scroll area
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(320)
        right_scroll.setMinimumWidth(260)
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setSpacing(6)
        right_scroll.setWidget(right_widget)
        main_layout.addWidget(right_scroll, stretch=1)

        # --- Configuration ---
        cfg_grp = QGroupBox("Configuration")
        cfg_lay = QVBoxLayout(cfg_grp)
        cfg_lay.setSpacing(4)

        cfg_lay.addWidget(QLabel("Tileset:"))
        self.ts_combo = QComboBox()
        self.ts_combo.addItems(self._tilesets.keys())
        self.ts_combo.currentTextChanged.connect(self._init_grid)
        cfg_lay.addWidget(self.ts_combo)

        cfg_lay.addWidget(QLabel("Dimension:"))
        self.dim_combo = QComboBox()
        self.dim_combo.addItems(["2D", "3D"])
        self.dim_combo.currentTextChanged.connect(self._init_grid)
        cfg_lay.addWidget(self.dim_combo)

        row_size = QHBoxLayout()
        row_size.addWidget(QLabel("Size:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(2, 200)
        self.size_spin.setValue(20)
        self.size_spin.valueChanged.connect(self._init_grid)
        row_size.addWidget(self.size_spin)
        cfg_lay.addLayout(row_size)

        row_seed = QHBoxLayout()
        row_seed.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        row_seed.addWidget(self.seed_spin)
        btn_random_seed = QPushButton("🎲")
        btn_random_seed.setFixedWidth(32)
        btn_random_seed.setToolTip("Random seed")
        btn_random_seed.clicked.connect(self._random_seed)
        row_seed.addWidget(btn_random_seed)
        cfg_lay.addLayout(row_seed)

        self.wrap_check = QCheckBox("Wrap Edges (toroidal)")
        cfg_lay.addWidget(self.wrap_check)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save Tileset")
        btn_save.clicked.connect(self._save_tileset)
        btn_load = QPushButton("Load Tileset")
        btn_load.clicked.connect(self._load_tileset)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_load)
        cfg_lay.addLayout(btn_row)

        right.addWidget(cfg_grp)

        # --- 3D View ---
        self.view_grp = QGroupBox("3D View")
        self.view_lay = QVBoxLayout(self.view_grp)
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.valueChanged.connect(
            lambda v: (setattr(self.canvas, 'z_slice', v), self.canvas.update()))
        self.view_lay.addWidget(QLabel("Z-Slice:"))
        self.view_lay.addWidget(self.z_slider)
        self.view_grp.setVisible(False)
        right.addWidget(self.view_grp)

        # --- Visuals ---
        vis_grp = QGroupBox("Visuals")
        vis_lay = QVBoxLayout(vis_grp)
        self.ent_check = QCheckBox("Entropy Heatmap")
        self.ent_check.toggled.connect(
            lambda c: (setattr(self.canvas, 'show_entropy', c), self.canvas.update()))
        vis_lay.addWidget(self.ent_check)
        self.grid_check = QCheckBox("Grid Lines")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(
            lambda c: (setattr(self.canvas, 'show_grid_lines', c), self.canvas.update()))
        vis_lay.addWidget(self.grid_check)
        btn_reset_view = QPushButton("Reset Zoom/Pan")
        btn_reset_view.clicked.connect(self.canvas.reset_view)
        vis_lay.addWidget(btn_reset_view)
        right.addWidget(vis_grp)

        # --- Simulation ---
        sim_grp = QGroupBox("Simulation")
        sim_lay = QVBoxLayout(sim_grp)

        sim_lay.addWidget(QLabel("Speed (ms between steps):"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500)
        self.speed_slider.setValue(50)
        self.speed_label = QLabel("50 ms")
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_label.setText(f"{v} ms"))
        sim_lay.addWidget(self.speed_slider)
        sim_lay.addWidget(self.speed_label)

        btn_row2 = QHBoxLayout()
        self.btn_step = QPushButton("Step ▶")
        self.btn_step.clicked.connect(self._step_once)
        btn_row2.addWidget(self.btn_step)
        self.btn_auto = QPushButton("Auto ▶▶")
        self.btn_auto.clicked.connect(self._toggle_auto)
        btn_row2.addWidget(self.btn_auto)
        sim_lay.addLayout(btn_row2)

        self.btn_finish = QPushButton("⚡ Finish Instantly")
        self.btn_finish.clicked.connect(self._finish_instantly)
        sim_lay.addWidget(self.btn_finish)

        self.retry_check = QCheckBox("Auto-retry on contradiction")
        self.retry_check.setChecked(True)
        sim_lay.addWidget(self.retry_check)

        btn_row3 = QHBoxLayout()
        self.btn_undo = QPushButton("↩ Undo")
        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo = QPushButton("Redo ↪")
        self.btn_redo.clicked.connect(self._redo)
        btn_row3.addWidget(self.btn_undo)
        btn_row3.addWidget(self.btn_redo)
        sim_lay.addLayout(btn_row3)

        self.btn_init = QPushButton("🔄 Reinitialize")
        self.btn_init.clicked.connect(self._init_grid)
        sim_lay.addWidget(self.btn_init)

        right.addWidget(sim_grp)

        # --- Stats ---
        stats_grp = QGroupBox("Statistics")
        stats_lay = QVBoxLayout(stats_grp)
        self.prog_bar = QProgressBar()
        self.prog_bar.setTextVisible(True)
        stats_lay.addWidget(self.prog_bar)
        self.stats_lbl = QLabel("Ready")
        self.stats_lbl.setWordWrap(True)
        stats_lay.addWidget(self.stats_lbl)
        right.addWidget(stats_grp)

        right.addStretch()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Press Space to step, Enter for auto-run")

        # Timer for auto-stepping
        self._timer = QTimer()
        self._timer.timeout.connect(self._auto_step)

    # ---- Tileset Management ----
    def _save_tileset(self):
        ts = self._tilesets[self.ts_combo.currentText()]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Tileset", f"{ts.name}.json", "JSON (*.json)")
        if path:
            with open(path, 'w') as f:
                f.write(ts.to_json())
            self.status_bar.showMessage(f"Tileset saved to {path}", 3000)

    def _load_tileset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Tileset", "", "JSON (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    ts = TilesetConfig.from_json(f.read())
                self._tilesets[ts.name] = ts
                self.ts_combo.addItem(ts.name)
                self.ts_combo.setCurrentText(ts.name)
                self._init_grid()
                self.status_bar.showMessage(f"Loaded tileset: {ts.name}", 3000)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load tileset:\n{e}")

    def _edit_tileset(self):
        name = self.ts_combo.currentText()
        if name not in self._tilesets:
            return
        dlg = TilesetEditorDialog(self._tilesets[name], self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_config = dlg.get_config()
            self._tilesets[new_config.name] = new_config
            # Update combo box
            idx = self.ts_combo.findText(name)
            if idx >= 0:
                self.ts_combo.removeItem(idx)
            self.ts_combo.addItem(new_config.name)
            self.ts_combo.setCurrentText(new_config.name)
            self._init_grid()
            self.status_bar.showMessage(f"Tileset '{new_config.name}' updated", 3000)

    def _new_tileset(self):
        dlg = TilesetEditorDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_config = dlg.get_config()
            if new_config.tiles:
                self._tilesets[new_config.name] = new_config
                self.ts_combo.addItem(new_config.name)
                self.ts_combo.setCurrentText(new_config.name)
                self._init_grid()
                self.status_bar.showMessage(f"New tileset '{new_config.name}' created", 3000)

    # ---- Grid Init ----
    def _init_grid(self):
        self._stop()
        self._undo_stack.clear()
        self._redo_stack.clear()

        config = self._tilesets[self.ts_combo.currentText()]
        is_3d = (self.dim_combo.currentText() == "3D")

        self.worker = WFCWorker(
            self.size_spin.value(), config, self.seed_spin.value(),
            is_3d, self.wrap_check.isChecked()
        )

        self.z_slider.setRange(0, self.worker.shape[0] - 1)
        self.z_slider.setValue(0)
        self.canvas.z_slice = 0
        self.view_grp.setVisible(is_3d)

        self.canvas.set_worker(self.worker, config.colors)
        self._t_start = 0
        self._update_stats()
        self.status_bar.showMessage("Grid initialized — Press Space to step")

    def _random_seed(self):
        import random
        self.seed_spin.setValue(random.randint(0, 999999))
        self._init_grid()

    # ---- Simulation Controls ----
    def _push_undo(self):
        if not self.worker:
            return
        snap = self.worker.get_snapshot()
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack or not self.worker:
            return
        self._stop()
        # Save current state to redo
        self._redo_stack.append(self.worker.get_snapshot())
        snap = self._undo_stack.pop()
        self.worker.restore_snapshot(snap)
        self.canvas.update()
        self._update_stats()
        self.status_bar.showMessage("Undo")

    def _redo(self):
        if not self._redo_stack or not self.worker:
            return
        self._stop()
        self._undo_stack.append(self.worker.get_snapshot())
        snap = self._redo_stack.pop()
        self.worker.restore_snapshot(snap)
        self.canvas.update()
        self._update_stats()
        self.status_bar.showMessage("Redo")

    def _step_once(self):
        if not self.worker or self._is_running:
            return
        if self._t_start == 0:
            self._t_start = time.perf_counter()
        self._push_undo()
        if not self.worker.run_step():
            self._stop()
            if self.retry_check.isChecked() and not self.worker.is_complete():
                self.status_bar.showMessage("Contradiction! Retrying with new seed...", 2000)
                QTimer.singleShot(100, self._retry_with_new_seed)
                return
            if self.worker.is_complete():
                self.status_bar.showMessage("Generation complete!")
        self.canvas.update()
        self._update_stats()

    def _auto_step(self):
        if not self.worker:
            return
        self._push_undo()
        if not self.worker.run_step():
            self._timer.stop()
            self.btn_auto.setText("Auto ▶▶")
            self._is_running = False
            if self.retry_check.isChecked() and not self.worker.is_complete():
                self.status_bar.showMessage("Contradiction! Retrying with new seed...", 2000)
                QTimer.singleShot(100, self._retry_with_new_seed)
                return
            if self.worker.is_complete():
                self.status_bar.showMessage("Generation complete!")
        self.canvas.update()
        self._update_stats()

    def _toggle_auto(self):
        if self._timer.isActive():
            self._stop()
        else:
            if self._t_start == 0:
                self._t_start = time.perf_counter()
            self._timer.start(self.speed_slider.value())
            self.btn_auto.setText("⏹ Stop")
            self._is_running = True
            self.status_bar.showMessage("Auto-running... Press Esc or Enter to stop")

    def _stop(self):
        self._timer.stop()
        self.btn_auto.setText("Auto ▶▶")
        self._is_running = False
        if self.worker:
            self.worker.stop()
        if self._finish_thread and self._finish_thread.isRunning():
            if self.worker:
                self.worker.stop()
            self._finish_thread.quit()
            self._finish_thread.wait(2000)

    def _finish_instantly(self):
        if not self.worker or self._is_running:
            return
        self._stop()
        self._push_undo()

        if self._t_start == 0:
            self._t_start = time.perf_counter()

        # Run in background thread to avoid GUI freeze
        self._finish_thread = QThread()
        self._instant_worker = InstantFinishWorker(self.worker)
        self._instant_worker.moveToThread(self._finish_thread)
        self._finish_thread.started.connect(self._instant_worker.run)
        self._instant_worker.progress.connect(self.prog_bar.setValue)
        self._instant_worker.finished.connect(self._on_instant_finished)
        self._finish_thread.start()
        self._is_running = True
        self.btn_finish.setEnabled(False)
        self.status_bar.showMessage("Generating...")

    def _on_instant_finished(self, success: bool, elapsed: float):
        self._finish_thread.quit()
        self._finish_thread.wait(3000)
        self._is_running = False
        self.btn_finish.setEnabled(True)
        self.canvas.update()
        self._update_stats()

        if success:
            self.status_bar.showMessage(
                f"Generation complete in {elapsed:.3f}s!", 5000)
        elif self.retry_check.isChecked():
            self.status_bar.showMessage("Contradiction! Retrying...", 2000)
            QTimer.singleShot(100, self._retry_with_new_seed)
        else:
            self.status_bar.showMessage("Contradiction encountered. Try a different seed or tileset.", 5000)

    def _retry_with_new_seed(self):
        self._stop()
        old_seed = self.seed_spin.value()
        for attempt in range(1, 11):
            new_seed = (old_seed + attempt * 7919) % 1000000  # Prime stride
            self.seed_spin.setValue(new_seed)
            self._undo_stack.clear()
            self._redo_stack.clear()

            config = self._tilesets[self.ts_combo.currentText()]
            is_3d = (self.dim_combo.currentText() == "3D")
            self.worker = WFCWorker(
                self.size_spin.value(), config, new_seed,
                is_3d, self.wrap_check.isChecked())
            self.z_slider.setRange(0, self.worker.shape[0] - 1)
            self.z_slider.setValue(0)
            self.canvas.z_slice = 0
            self.canvas.set_worker(self.worker, config.colors)
            self._t_start = time.perf_counter()

            # Try to finish
            total = int(np.prod(self.worker.shape))
            limit = total * 2
            for _ in range(limit):
                if not self.worker.run_step():
                    break

            if self.worker.is_complete():
                self.canvas.update()
                self._update_stats()
                self.status_bar.showMessage(
                    f"Success on attempt {attempt} with seed {new_seed}!", 5000)
                return

        self.status_bar.showMessage(
            "Failed after 10 retry attempts. Try different settings.", 5000)

    # ---- Stats ----
    def _update_stats(self):
        if not self.worker:
            return
        collapsed = int(np.sum(self.worker.collapsed))
        total = int(np.prod(self.worker.shape))
        pct = int((collapsed / total) * 100) if total > 0 else 0
        self.prog_bar.setValue(pct)

        elapsed = time.perf_counter() - self._t_start if self._t_start > 0 else 0
        cps = collapsed / elapsed if elapsed > 0.01 else 0
        complete = "✅ Complete" if self.worker.is_complete() else ""
        self.stats_lbl.setText(
            f"Collapsed: {collapsed}/{total} ({pct}%)\n"
            f"Steps: {self.worker.step_count}\n"
            f"Time: {elapsed:.2f}s\n"
            f"Cells/sec: {cps:.0f}\n"
            f"{complete}")

    # ---- Zoom ----
    def _zoom_by(self, factor: float):
        self.canvas._zoom = max(0.1, min(20.0, self.canvas._zoom * factor))
        self.canvas.update()

    # ---- Export ----
    def _export_png(self):
        if not self.worker:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", "wfc_output.png", "PNG (*.png)")
        if path:
            self.canvas.export_png(path, scale=max(1, 512 // self.worker.size))
            self.status_bar.showMessage(f"Exported PNG: {path}", 3000)

    def _export_json(self):
        if not self.worker:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Grid JSON", "wfc_grid.json", "JSON (*.json)")
        if path:
            z = self.z_slice
            grid = []
            for y in range(self.worker.size):
                row = []
                for x in range(self.worker.size):
                    mask = self.worker.wave[z, y, x]
                    ent = get_popcount(mask)
                    if ent == 1:
                        idx = mask.bit_length() - 1
                        row.append(self.worker.tiles[idx])
                    else:
                        row.append(f"?({ent})")
                grid.append(row)
            data = {
                "tileset": self.ts_combo.currentText(),
                "size": self.worker.size,
                "seed": self.seed_spin.value(),
                "steps": self.worker.step_count,
                "z_slice": z,
                "grid": grid
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.status_bar.showMessage(f"Exported JSON: {path}", 3000)

    # ---- About ----
    def _show_about(self):
        QMessageBox.about(self, "About WFC",
            "<h2>WFC — Enhanced Edition</h2>"
            "<p>Wave Function Collapse procedural generation tool.</p>"
            "<p>Features: bitmask optimization, min-heap observation, "
            "direction-specific rules, undo/redo, zoom/pan, "
            "tileset editor, PNG/JSON export, dark theme, "
            "and more.</p>"
            "<p>Built with Python, NumPy, and PySide6.</p>")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()