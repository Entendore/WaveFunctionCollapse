"""
WFC 2D — Wave Function Collapse (2D Edition)
==============================================
A standalone 2D procedural generation tool built with PySide6.
"""

from __future__ import annotations

import heapq
import json
import logging
import base64
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

LOG = logging.getLogger("wfc2d")
LOG.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
LOG.addHandler(_handler)

import numpy as np
from PySide6.QtCore import (
    Qt, QTimer, QRectF, QRect, Signal, QObject, QThread, QPoint, QPointF, QSize
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QKeySequence, QShortcut,
    QPainterPath, QPixmap, QImage
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSpinBox, QLabel, QSlider, QGroupBox, QMessageBox,
    QComboBox, QCheckBox, QStatusBar, QProgressBar, QFileDialog,
    QLineEdit, QScrollArea, QColorDialog, QTableWidget, QTableWidgetItem,
    QDialog, QDialogButtonBox, QFormLayout, QToolTip, QMenuBar, QMenu,
    QSizePolicy, QTextEdit, QInputDialog
)

# ---------------------------------------------------------------------------
# Performance Utilities
# ---------------------------------------------------------------------------

POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def get_popcount(n: int) -> int:
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
    UP = 0; DOWN = 1; LEFT = 2; RIGHT = 3

    @property
    def delta(self) -> Tuple[int, ...]:
        return {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[self]

    @property
    def opposite(self) -> Direction:
        return Direction({0: 1, 1: 0, 2: 3, 3: 2}[self])


@dataclass
class TilesetConfig:
    name: str
    tiles: List[str]
    colors: Dict[str, str]
    rules: List[Tuple[str, str, str]]  # (tile_a, direction_name, tile_b)
    weights: Dict[str, float] = field(default_factory=dict)
    images: Dict[str, str] = field(default_factory=dict)  # tile_name -> base64 PNG

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name, "tiles": self.tiles, "colors": self.colors,
            "rules": [list(r) for r in self.rules],
            "weights": self.weights,
            "images": self.images,
        }, indent=4)

    @staticmethod
    def from_json(data_str: str) -> TilesetConfig:
        d = json.loads(data_str)
        rules = [tuple(r) for r in d['rules']]
        return TilesetConfig(
            name=d['name'], tiles=d['tiles'], colors=d['colors'],
            rules=rules, weights=d.get('weights', {}),
            images=d.get('images', {}))

    @staticmethod
    def image_from_file(path: str) -> str:
        """Read an image file and return a base64-encoded PNG string."""
        with open(path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode('ascii')

    @staticmethod
    def image_to_pixmap(b64: str) -> Optional[QPixmap]:
        """Decode a base64 PNG string into a QPixmap (or None on failure)."""
        if not b64:
            return None
        try:
            data = base64.b64decode(b64)
            pix = QPixmap()
            if pix.loadFromData(data):
                return pix
        except Exception:
            pass
        return None


DIR_NAMES = {"UP": Direction.UP, "DOWN": Direction.DOWN,
             "LEFT": Direction.LEFT, "RIGHT": Direction.RIGHT}


def _make_sym_rules(bidirectional_rules: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """Convert (A, B) pairs into symmetric per-direction rules for all 4 2D directions."""
    result = []
    for a, b in bidirectional_rules:
        for dname in ["UP", "DOWN", "LEFT", "RIGHT"]:
            result.append((a, dname, b))
    return result

# ---------------------------------------------------------------------------
# WFC Worker (2D)
# ---------------------------------------------------------------------------


class WFCWorker(QObject):
    progress = Signal(float)
    cell_collapsed = Signal(int, int, int)  # y, x, tile_idx
    finished = Signal(bool)
    contradiction_pos = Signal(int, int)
    step_done = Signal()

    def __init__(self, size: int, config: TilesetConfig, seed: int, wrap: bool):
        super().__init__()
        self.size = size
        self.wrap = wrap
        self.tiles = config.tiles
        self.n_tiles = len(self.tiles)
        self._tile_to_idx = {t: i for i, t in enumerate(self.tiles)}
        self.rng = np.random.default_rng(seed)
        self.weights = np.array(
            [config.weights.get(t, 1.0) for t in self.tiles], dtype=np.float64)

        # Direction-specific rules
        self.rules = np.zeros((4, self.n_tiles), dtype=np.uint64)
        for tile_a, dir_name, tile_b in config.rules:
            if tile_a not in self._tile_to_idx or tile_b not in self._tile_to_idx:
                continue
            i_a, i_b = self._tile_to_idx[tile_a], self._tile_to_idx[tile_b]
            d_idx = int(DIR_NAMES[dir_name])
            self.rules[d_idx, i_a] |= (1 << i_b)
            self.rules[int(DIR_NAMES[dir_name].opposite), i_b] |= (1 << i_a)

        self.shape = (size, size)
        self.full_mask = (1 << self.n_tiles) - 1
        self._stop_requested = False
        self.step_count = 0
        self._seq = 0
        self.reset()

    def reset(self):
        self.wave = np.full(self.shape, self.full_mask, dtype=np.uint64)
        self.collapsed = np.zeros(self.shape, dtype=np.uint8)
        self.step_count = 0
        self._seq = 0
        self._heap = []
        self._in_heap = np.zeros(self.shape, dtype=bool)
        self._init_heap()

    def _init_heap(self):
        self._heap = []
        self._in_heap = np.zeros(self.shape, dtype=bool)
        for pos in np.ndindex(self.shape):
            self._push(pos, get_popcount(self.wave[pos]))

    def _push(self, pos, entropy):
        # Use a monotonically increasing counter as tiebreaker to avoid
        # float precision issues for large grids and to guarantee comparability.
        self._seq += 1
        heapq.heappush(self._heap, (entropy, self._seq, pos))
        self._in_heap[pos] = True

    def _pop_lowest(self):
        while self._heap:
            ent, _, pos = heapq.heappop(self._heap)
            if not self._in_heap[pos]:
                continue
            self._in_heap[pos] = False
            return pos, ent
        return None, -1

    def stop(self):
        self._stop_requested = True

    def get_snapshot(self) -> dict:
        return {'wave': self.wave.copy(), 'collapsed': self.collapsed.copy(),
                'step_count': self.step_count}

    def restore_snapshot(self, snap: dict):
        self.wave = snap['wave'].copy()
        self.collapsed = snap['collapsed'].copy()
        self.step_count = snap['step_count']
        self._seq = 0
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
        w_sum = np.sum(w)
        if w_sum == 0:
            self.contradiction_pos.emit(*pos)
            return False
        chosen = self.rng.choice(poss, p=w / w_sum)
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
        dirs = list(Direction)
        while stack:
            curr = stack.pop()
            curr_mask = self.wave[curr]
            for d in dirs:
                d_idx = int(d)
                delta = d.delta
                ny, nx = curr[0] + delta[0], curr[1] + delta[1]
                if self.wrap:
                    ny, nx = ny % self.size, nx % self.size
                elif not (0 <= ny < self.size and 0 <= nx < self.size):
                    continue
                neighbor = (ny, nx)

                allowed = np.uint64(0)
                for i in range(self.n_tiles):
                    if (curr_mask >> i) & 1:
                        allowed |= self.rules[d_idx, i]

                old_mask = self.wave[neighbor]
                new_mask = old_mask & allowed
                if new_mask != old_mask:
                    if new_mask == 0:
                        return False
                    self.wave[neighbor] = new_mask
                    if self._in_heap[neighbor]:
                        self._in_heap[neighbor] = False
                    self._push(neighbor, get_popcount(new_mask))
                    if not self.collapsed[neighbor]:
                        stack.append(neighbor)
        return True

    def is_complete(self) -> bool:
        return bool(np.all(self.collapsed))


class InstantFinishWorker(QObject):
    progress = Signal(float)
    finished = Signal(bool, float)

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
            self.progress.emit(int(np.sum(self.worker.collapsed)) / total)
        self.finished.emit(self.worker.is_complete(), time.perf_counter() - t0)

# ---------------------------------------------------------------------------
# 2D Canvas
# ---------------------------------------------------------------------------


class WFCCanvas2D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[WFCWorker] = None
        self._colors: Dict[str, str] = {}
        self._images: Dict[str, QPixmap] = {}  # tile_name -> cached QPixmap
        self.show_entropy = False
        self.show_grid_lines = True
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._panning = False
        self._last_pan_pos = QPoint()
        self._hover_cell = None
        self._contradiction_cells: Set[Tuple[int, int]] = set()

    def set_worker(self, worker, colors, images=None):
        self.worker = worker
        self._colors = colors
        self._images = {}
        if images:
            for tname, b64 in images.items():
                pix = TilesetConfig.image_to_pixmap(b64)
                if pix is not None:
                    self._images[tname] = pix
        if self._images:
            LOG.info("Loaded %d tile images for canvas", len(self._images))
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._contradiction_cells.clear()
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self.update()

    def _cell_size(self) -> float:
        if not self.worker:
            return 10
        base = min(self.width() / self.worker.size, self.height() / self.worker.size)
        return base * self._zoom

    def _grid_origin(self) -> QPoint:
        if not self.worker:
            return QPoint(0, 0)
        cs = self._cell_size()
        gw = cs * self.worker.size
        ox = (self.width() - gw) / 2 + self._pan_offset.x()
        oy = (self.height() - gw) / 2 + self._pan_offset.y()
        return QPoint(int(ox), int(oy))

    def paintEvent(self, event):
        if not self.worker:
            return
        p = QPainter(self)
        try:
            self._paint_grid(p)
        except Exception:
            LOG.exception("paintEvent error")
        finally:
            # Only call end() if the painter is actually active;
            # calling end() on an inactive painter triggers
            # "QPainter::end: Painter not active, aborted".
            if p.isActive():
                p.end()

    def _paint_grid(self, p: QPainter):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        cs = self._cell_size()
        origin = self._grid_origin()
        p.fillRect(self.rect(), QColor("#1e1e2e"))

        for y in range(self.worker.size):
            for x in range(self.worker.size):
                rx = origin.x() + x * cs
                ry = origin.y() + y * cs
                rect = QRectF(rx, ry, cs, cs)
                if rx + cs < 0 or ry + cs < 0 or rx > self.width() or ry > self.height():
                    continue
                mask_int = int(self.worker.wave[y, x])
                ent = get_popcount(mask_int)
                # Draw tile image if available and collapsed
                if not self.show_entropy and ent == 1 and cs > 2:
                    idx = mask_int.bit_length() - 1
                    tile_name = self.worker.tiles[idx]
                    if tile_name in self._images:
                        target = QRectF(rx, ry, cs, cs).toRect()
                        p.drawPixmap(target, self._images[tile_name])
                        if self.show_grid_lines and cs > 4:
                            p.setPen(QPen(QColor(255, 255, 255, 30), 1))
                            p.setBrush(Qt.BrushStyle.NoBrush)
                            p.drawRect(rect)
                        continue
                color = self._cell_color(mask_int)
                p.setBrush(QBrush(color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRect(rect)
                if self.show_grid_lines and cs > 4:
                    p.setPen(QPen(QColor(255, 255, 255, 30), 1))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(rect)

        # Draw contradiction highlights
        if self._contradiction_cells and cs > 2:
            p.setPen(QPen(QColor(255, 50, 50, 200), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for cy, cx in self._contradiction_cells:
                if 0 <= cy < self.worker.size and 0 <= cx < self.worker.size:
                    p.drawRect(QRectF(origin.x() + cx * cs, origin.y() + cy * cs, cs, cs))

        if self._hover_cell and cs > 4:
            hy, hx = self._hover_cell
            if 0 <= hy < self.worker.size and 0 <= hx < self.worker.size:
                p.setPen(QPen(QColor(255, 255, 0, 180), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QRectF(origin.x() + hx * cs, origin.y() + hy * cs, cs, cs))
        p.end()

    def wheelEvent(self, event):
        old_zoom = self._zoom
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom = max(0.1, min(20.0, self._zoom * factor))
        # Zoom toward mouse cursor position
        if self.worker:
            cs = self._cell_size()
            gw = cs * self.worker.size
            center_x = self.width() / 2 + self._pan_offset.x()
            center_y = self.height() / 2 + self._pan_offset.y()
            mouse = event.position()
            # Adjust pan so the point under the cursor stays fixed
            self._pan_offset.setX(
                int(mouse.x() - (mouse.x() - center_x) * (self._zoom / old_zoom) - self.width() / 2)
            )
            self._pan_offset.setY(
                int(mouse.y() - (mouse.y() - center_y) * (self._zoom / old_zoom) - self.height() / 2)
            )
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._panning:
            pos = event.position().toPoint()
            self._pan_offset += pos - self._last_pan_pos
            self._last_pan_pos = pos
            self.update()
        else:
            self._update_hover(event.position().toPoint())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _update_hover(self, pos: QPoint):
        if not self.worker:
            return
        cs = self._cell_size()
        o = self._grid_origin()
        ix, iy = int((pos.x() - o.x()) / cs), int((pos.y() - o.y()) / cs)
        if 0 <= ix < self.worker.size and 0 <= iy < self.worker.size:
            self._hover_cell = (iy, ix)
            mask = int(self.worker.wave[iy, ix])
            ent = get_popcount(mask)
            txt = (f"({iy}, {ix}) | Tile: {self.worker.tiles[mask.bit_length()-1]}"
                   if ent == 1 else f"({iy}, {ix}) | Entropy: {ent}")
            QToolTip.showText(self.mapToGlobal(pos), txt)
        else:
            self._hover_cell = None
            QToolTip.hideText()

    def _cell_color(self, mask) -> QColor:
        """Compute the display color for a cell given its superposition mask.

        Respects the current ``show_entropy`` flag so that
        ``export_png`` can mirror what the user sees on screen.
        Accepts both plain ``int`` and ``numpy.uint64`` values.
        """
        mask = int(mask)  # numpy scalars lack .bit_length()
        ent = get_popcount(mask)
        if not self.worker:
            return QColor(0, 0, 0)
        if self.show_entropy:
            norm = (ent - 1) / max(self.worker.n_tiles - 1, 1)
            return QColor.fromHslF(0.6 * (1.0 - norm), 0.7, 0.5)
        if ent == 1:
            idx = mask.bit_length() - 1
            return QColor(self._colors[self.worker.tiles[idx]])
        r, g, b, cnt = 0, 0, 0, 0
        for i in range(self.worker.n_tiles):
            if (mask >> i) & 1:
                c = QColor(self._colors[self.worker.tiles[i]])
                r += c.red(); g += c.green(); b += c.blue(); cnt += 1
        return QColor(r // cnt, g // cnt, b // cnt) if cnt else QColor(0, 0, 0)

    def export_png(self, path: str, scale: int = 1):
        if not self.worker:
            return
        s = self.worker.size
        img = QImage(s * scale, s * scale, QImage.Format.Format_ARGB32)
        img.fill(QColor("#1e1e2e"))
        p = QPainter(img)
        for y in range(s):
            for x in range(s):
                mask_int = int(self.worker.wave[y, x])
                ent = get_popcount(mask_int)
                if not self.show_entropy and ent == 1:
                    idx = mask_int.bit_length() - 1
                    tile_name = self.worker.tiles[idx]
                    if tile_name in self._images:
                        p.drawPixmap(x * scale, y * scale, scale, scale, self._images[tile_name])
                        continue
                color = self._cell_color(mask_int)
                p.setBrush(QBrush(color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRect(x * scale, y * scale, scale, scale)
        p.end()
        img.save(path)
        LOG.info("Exported PNG with %d tile images", len(self._images))

# ---------------------------------------------------------------------------
# Tileset Editor Dialog
# ---------------------------------------------------------------------------


class TilesetEditorDialog(QDialog):
    def __init__(self, config: Optional[TilesetConfig] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tileset Editor")
        self.setMinimumSize(750, 550)
        self._colors_map: Dict[str, str] = {}
        self._images_map: Dict[str, str] = {}  # tile_name -> base64
        self._setup_ui(config)

    def _setup_ui(self, config):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(config.name if config else "New Tileset")
        form.addRow("Name:", self.name_edit)
        layout.addLayout(form)

        tiles_grp = QGroupBox("Tiles (name, color, image, weight)")
        tl = QVBoxLayout(tiles_grp)
        self.tiles_table = QTableWidget(0, 5)
        self.tiles_table.setHorizontalHeaderLabels(["Name", "Color", "Image", "Weight", ""])
        hdr = self.tiles_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hdr.resizeSection(4, 70)
        self.tiles_table.verticalHeader().setDefaultSectionSize(36)
        tl.addWidget(self.tiles_table)
        br = QHBoxLayout()
        ba = QPushButton("Add Tile"); ba.clicked.connect(self._add_tile_row)
        bd = QPushButton("Remove Selected"); bd.clicked.connect(self._del_tile_row)
        br.addWidget(ba); br.addWidget(bd)
        tl.addLayout(br)
        layout.addWidget(tiles_grp)

        rules_grp = QGroupBox("Adjacency Rules (tile_a <-> tile_b)")
        rl = QVBoxLayout(rules_grp)
        self.rules_table = QTableWidget(0, 2)
        self.rules_table.setHorizontalHeaderLabels(["Tile A", "Tile B"])
        self.rules_table.horizontalHeader().setStretchLastSection(True)
        rl.addWidget(self.rules_table)
        br2 = QHBoxLayout()
        ba2 = QPushButton("Add Rule"); ba2.clicked.connect(self._add_rule_row)
        bd2 = QPushButton("Remove Selected"); bd2.clicked.connect(self._del_rule_row)
        br2.addWidget(ba2); br2.addWidget(bd2)
        rl.addLayout(br2)
        layout.addWidget(rules_grp)

        if config:
            self._images_map = dict(config.images) if config.images else {}
            for t in config.tiles:
                row = self.tiles_table.rowCount()
                self.tiles_table.insertRow(row)
                self.tiles_table.setItem(row, 0, QTableWidgetItem(t))
                ci = QTableWidgetItem(config.colors.get(t, "#ffffff"))
                ci.setBackground(QColor(config.colors.get(t, "#ffffff")))
                self.tiles_table.setItem(row, 1, ci)
                b64 = self._images_map.get(t, "")
                from os.path import basename as _bn
                img_label = ""
                if b64:
                    pix = TilesetConfig.image_to_pixmap(b64)
                    img_label = "(image set)" if pix else ""
                self.tiles_table.setItem(row, 2, QTableWidgetItem(img_label))
                self.tiles_table.setItem(row, 3, QTableWidgetItem(str(config.weights.get(t, 1.0))))
                self._colors_map[t] = config.colors.get(t, "#ffffff")
                img_btn = QPushButton("Clear" if b64 else "Load")
                img_btn.setFixedHeight(28)
                if b64:
                    img_btn.setStyleSheet("color: #a6e3a1;")
                img_btn.clicked.connect(lambda checked, r=row: self._load_image_for_row(r))
                self.tiles_table.setCellWidget(row, 4, img_btn)
            seen = set()
            for a, d, b in config.rules:
                key = tuple(sorted([a, b]))
                if key not in seen:
                    seen.add(key)
                    row = self.rules_table.rowCount()
                    self.rules_table.insertRow(row)
                    self.rules_table.setItem(row, 0, QTableWidgetItem(a))
                    self.rules_table.setItem(row, 1, QTableWidgetItem(b))
        self.tiles_table.cellDoubleClicked.connect(self._on_color_dblclick)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _add_tile_row(self):
        r = self.tiles_table.rowCount(); self.tiles_table.insertRow(r)
        self.tiles_table.setItem(r, 0, QTableWidgetItem(f"TILE_{r}"))
        ci = QTableWidgetItem("#ffffff"); ci.setBackground(QColor("#ffffff"))
        self.tiles_table.setItem(r, 1, ci)
        self.tiles_table.setItem(r, 2, QTableWidgetItem(""))
        self.tiles_table.setItem(r, 3, QTableWidgetItem("1.0"))
        img_btn = QPushButton("Load")
        img_btn.setFixedHeight(28)
        img_btn.clicked.connect(lambda checked, row=r: self._load_image_for_row(row))
        self.tiles_table.setCellWidget(r, 4, img_btn)

    def _del_tile_row(self):
        for r in sorted(set(i.row() for i in self.tiles_table.selectedItems()), reverse=True):
            name_item = self.tiles_table.item(r, 0)
            tname = name_item.text().strip() if name_item else ""
            self._images_map.pop(tname, None)
            self.tiles_table.removeRow(r)

    def _add_rule_row(self):
        r = self.rules_table.rowCount(); self.rules_table.insertRow(r)
        self.rules_table.setItem(r, 0, QTableWidgetItem(""))
        self.rules_table.setItem(r, 1, QTableWidgetItem(""))

    def _del_rule_row(self):
        for r in sorted(set(i.row() for i in self.rules_table.selectedItems()), reverse=True):
            self.rules_table.removeRow(r)

    def _on_color_dblclick(self, row, col):
        if col == 1:
            name = self.tiles_table.item(row, 0)
            cur = self._colors_map.get(name.text() if name else "", "#ffffff")
            c = QColorDialog.getColor(QColor(cur), self)
            if c.isValid():
                item = self.tiles_table.item(row, 1)
                if item: item.setText(c.name()); item.setBackground(c)
                if name: self._colors_map[name.text()] = c.name()

    def _load_image_for_row(self, row):
        name_item = self.tiles_table.item(row, 0)
        tname = name_item.text().strip() if name_item else ""
        if not tname:
            QMessageBox.warning(self, "Error", "Enter a tile name first.")
            return
        # If already has image, clear it
        if tname in self._images_map and self._images_map[tname]:
            self._images_map.pop(tname, None)
            self.tiles_table.setItem(row, 2, QTableWidgetItem(""))
            btn = self.tiles_table.cellWidget(row, 4)
            if btn:
                btn.setText("Load")
                btn.setStyleSheet("")
            LOG.info("Cleared image for tile %s", tname)
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Tile Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        try:
            b64 = TilesetConfig.image_from_file(path)
            self._images_map[tname] = b64
            from os.path import basename
            self.tiles_table.setItem(row, 2, QTableWidgetItem(basename(path)))
            btn = self.tiles_table.cellWidget(row, 4)
            if btn:
                btn.setText("Clear")
                btn.setStyleSheet("color: #a6e3a1;")
            LOG.info("Loaded image for tile %s (%s)", tname, basename(path))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image:\n{e}")

    def get_config(self) -> TilesetConfig:
        tiles, colors, weights = [], {}, {}
        for r in range(self.tiles_table.rowCount()):
            ni = self.tiles_table.item(r, 0)
            ci = self.tiles_table.item(r, 1)
            wi = self.tiles_table.item(r, 3)  # column 3 = weight
            if not ni: continue
            t = ni.text().strip()
            if not t: continue
            tiles.append(t)
            colors[t] = ci.text().strip() if ci else "#ffffff"
            try: weights[t] = float(wi.text().strip()) if wi else 1.0
            except ValueError: weights[t] = 1.0
        simple = []
        for r in range(self.rules_table.rowCount()):
            a, b = self.rules_table.item(r, 0), self.rules_table.item(r, 1)
            if a and b:
                aa, bb = a.text().strip(), b.text().strip()
                if aa and bb: simple.append((aa, bb))
        # Only include images for tiles that are still in the list
        images = {t: b64 for t, b64 in self._images_map.items() if t in tiles}
        return TilesetConfig(self.name_edit.text().strip() or "Untitled",
                             tiles, colors, _make_sym_rules(simple), weights, images)

# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

DARK_QSS = """
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
QHeaderView::section { background-color: #313244; color: #cdd6f4;
                       border: 1px solid #45475a; padding: 4px; }
QToolTip { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; padding: 4px; }
QScrollArea { border: none; }
QLineEdit { background-color: #313244; border: 1px solid #45475a;
            border-radius: 5px; padding: 4px; color: #cdd6f4; }
QDialog { background-color: #1e1e2e; color: #cdd6f4; }
QTextEdit { background-color: #181825; color: #cdd6f4; border: 1px solid #45475a;
           border-radius: 5px; padding: 4px; font-family: monospace; font-size: 11px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WFC 2D — Wave Function Collapse")
        self.resize(1280, 900)
        self._tilesets: Dict[str, TilesetConfig] = {}
        self._presets_dir = ""  # set after init
        self.worker: Optional[WFCWorker] = None
        self._undo_stack: List[dict] = []
        self._redo_stack: List[dict] = []
        self._max_undo = 50
        self._is_running = False
        self._finish_thread: Optional[QThread] = None
        self._finish_worker_ref: Optional[InstantFinishWorker] = None
        self._t_start = 0.0
        self._auto_undo_pushed = False  # Track whether undo was pushed for current auto-run
        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self.setStyleSheet(DARK_QSS)
        self._load_presets_dir()
        if not self._tilesets:
            self.status_bar.showMessage("No presets found — use Tileset > Load Preset File to add one.")

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction("Export PNG...", self._export_png, QKeySequence("Ctrl+E"))
        fm.addAction("Export Grid JSON...", self._export_json)
        fm.addSeparator()
        fm.addAction("Save Tileset...", self._save_tileset)
        fm.addAction("Load Tileset...", self._load_tileset)
        fm.addSeparator()
        fm.addAction("Exit", self.close, QKeySequence("Ctrl+Q"))
        em = mb.addMenu("Edit")
        em.addAction("Undo", self._undo, QKeySequence.Undo)
        em.addAction("Redo", self._redo, QKeySequence.Redo)
        vm = mb.addMenu("View")
        vm.addAction("Reset Zoom/Pan", self.canvas.reset_view, QKeySequence("Ctrl+0"))
        vm.addAction("Zoom In", lambda: self._zoom_by(1.3), QKeySequence.ZoomIn)
        vm.addAction("Zoom Out", lambda: self._zoom_by(1 / 1.3), QKeySequence.ZoomOut)
        tm = mb.addMenu("Tileset")
        tm.addAction("Edit Current Tileset...", self._edit_tileset)
        tm.addAction("New Tileset...", self._new_tileset)
        tm.addSeparator()
        tm.addAction("Load Preset Directory...", self._load_presets_dir_dialog)
        tm.addAction("Load Preset File...", self._load_preset_file)
        hm = mb.addMenu("Help")
        hm.addAction("About", lambda: QMessageBox.about(self, "About", "<h2>WFC 2D</h2><p>2D Wave Function Collapse tool.</p>"))

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, self._step_once)
        QShortcut(QKeySequence("Return"), self, self._toggle_auto)
        QShortcut(QKeySequence("Escape"), self, self._stop)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)
        QShortcut(QKeySequence("R"), self, self._init_grid)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        ml = QHBoxLayout(central); ml.setContentsMargins(6, 6, 6, 6); ml.setSpacing(6)
        self.canvas = WFCCanvas2D(); ml.addWidget(self.canvas, stretch=3)

        rs = QScrollArea(); rs.setWidgetResizable(True)
        rs.setMaximumWidth(320); rs.setMinimumWidth(260)
        rw = QWidget(); right = QVBoxLayout(rw); right.setSpacing(6)
        rs.setWidget(rw); ml.addWidget(rs, stretch=1)

        # Config
        cfg = QGroupBox("Configuration"); cl = QVBoxLayout(cfg); cl.setSpacing(4)
        cl.addWidget(QLabel("Tileset:"))
        self.ts_combo = QComboBox(); self.ts_combo.addItems(self._tilesets.keys())
        self.ts_combo.currentTextChanged.connect(self._on_tileset_changed); cl.addWidget(self.ts_combo)
        rs2 = QHBoxLayout(); rs2.addWidget(QLabel("Size:"))
        self.size_spin = QSpinBox(); self.size_spin.setRange(2, 200); self.size_spin.setValue(20)
        self.size_spin.valueChanged.connect(self._init_grid); rs2.addWidget(self.size_spin)
        cl.addLayout(rs2)
        rs3 = QHBoxLayout(); rs3.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999); self.seed_spin.setValue(42)
        rs3.addWidget(self.seed_spin)
        br = QPushButton("🎲"); br.setFixedWidth(32); br.setToolTip("Random seed")
        br.clicked.connect(self._random_seed); rs3.addWidget(br)
        cl.addLayout(rs3)
        self.wrap_check = QCheckBox("Wrap Edges (toroidal)"); cl.addWidget(self.wrap_check)
        brs = QHBoxLayout()
        bs = QPushButton("Save Tileset"); bs.clicked.connect(self._save_tileset)
        bl = QPushButton("Load Tileset"); bl.clicked.connect(self._load_tileset)
        brs.addWidget(bs); brs.addWidget(bl); cl.addLayout(brs)
        right.addWidget(cfg)

        # Visuals
        vis = QGroupBox("Visuals"); vl = QVBoxLayout(vis)
        self.ent_check = QCheckBox("Entropy Heatmap")
        self.ent_check.toggled.connect(lambda c: (setattr(self.canvas, 'show_entropy', c), self.canvas.update()))
        vl.addWidget(self.ent_check)
        self.grid_check = QCheckBox("Grid Lines"); self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(lambda c: (setattr(self.canvas, 'show_grid_lines', c), self.canvas.update()))
        vl.addWidget(self.grid_check)
        breset = QPushButton("Reset Zoom/Pan"); breset.clicked.connect(self.canvas.reset_view)
        vl.addWidget(breset); right.addWidget(vis)

        # Tileset Rules Viewer
        rules_grp = QGroupBox("Tileset Rules"); rl2 = QVBoxLayout(rules_grp)
        btn_row = QHBoxLayout()
        self.rules_toggle_btn = QPushButton("Show Rules")
        self.rules_toggle_btn.setCheckable(True)
        self.rules_toggle_btn.toggled.connect(self._toggle_rules_view)
        btn_row.addWidget(self.rules_toggle_btn)
        self.btn_set_image = QPushButton("🖼 Set Image")
        self.btn_set_image.setToolTip("Assign an image to a tile (shown in rules viewer and canvas)")
        self.btn_set_image.clicked.connect(self._assign_tile_image)
        self.btn_set_image.setEnabled(False)
        btn_row.addWidget(self.btn_set_image)
        self.btn_set_color = QPushButton("🎨 Set Color")
        self.btn_set_color.setToolTip("Change the color of a tile in the current tileset")
        self.btn_set_color.clicked.connect(self._assign_tile_color)
        self.btn_set_color.setEnabled(False)
        btn_row.addWidget(self.btn_set_color)
        rl2.addLayout(btn_row)
        self.rules_text = QTextEdit()
        self.rules_text.setReadOnly(True)
        self.rules_text.setMaximumHeight(260)
        self.rules_text.setVisible(False)
        rl2.addWidget(self.rules_text)
        right.addWidget(rules_grp)

        # Simulation
        sim = QGroupBox("Simulation"); sl = QVBoxLayout(sim)
        sl.addWidget(QLabel("Speed (ms between steps):"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500); self.speed_slider.setValue(50)
        self.speed_label = QLabel("50 ms")
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"{v} ms"))
        sl.addWidget(self.speed_slider); sl.addWidget(self.speed_label)
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
        stl.addWidget(self.stats_lbl); right.addWidget(st)
        right.addStretch()

        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Space=step, Enter=auto, Esc=stop")
        self._timer = QTimer(); self._timer.timeout.connect(self._auto_step)

    def _on_tileset_changed(self, name: str):
        self._init_grid()
        if self.rules_toggle_btn.isChecked():
            self._refresh_rules_view()

    def _toggle_rules_view(self, checked: bool):
        self.rules_text.setVisible(checked)
        self.rules_toggle_btn.setText("Hide Rules" if checked else "Show Rules")
        has_ts = checked and bool(self._tilesets)
        self.btn_set_image.setEnabled(has_ts)
        self.btn_set_color.setEnabled(has_ts)
        if checked:
            self._refresh_rules_view()

    def _refresh_rules_view(self):
        name = self.ts_combo.currentText()
        if name not in self._tilesets:
            self.rules_text.setPlainText("(no tileset selected)")
            return
        config = self._tilesets[name]
        lines = []
        # --- tiles + weights ---
        lines.append(f"<b>Tiles</b> ({len(config.tiles)}):<br>")
        for t in config.tiles:
            color = config.colors.get(t, "#ffffff")
            w = config.weights.get(t, 1.0)
            b64 = config.images.get(t, "")
            img_tag = ""
            if b64:
                img_tag = f' <img src="data:image/png;base64,{b64}" height="20" style="vertical-align:middle">'
            lines.append(
                f'&nbsp;&nbsp;<span style="color:{color}; font-size:13px">██</span> '
                f'{t} &nbsp;(weight {w}){img_tag}'
            )
        # --- adjacency rules ---
        lines.append(f"<br><b>Adjacency Rules</b> ({len(config.rules)}):<br>")
        dir_sym = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
        # Group by (a, b) pair, collect directions
        pairs: Dict[Tuple[str, str], List[str]] = {}
        for a, d, b in config.rules:
            pairs.setdefault((a, b), []).append(d)
        for (a, b), dirs in pairs.items():
            arrows = "&nbsp;".join(dir_sym.get(d, d) for d in sorted(dirs))
            ca = config.colors.get(a, "#ffffff")
            cb = config.colors.get(b, "#ffffff")
            img_a = config.images.get(a, "")
            img_b = config.images.get(b, "")
            tag_a = f' <img src="data:image/png;base64,{img_a}" height="16" style="vertical-align:middle">' if img_a else ""
            tag_b = f' <img src="data:image/png;base64,{img_b}" height="16" style="vertical-align:middle">' if img_b else ""
            lines.append(
                f'&nbsp;&nbsp;<span style="color:{ca}; font-size:13px">██</span> {a}{tag_a}'
                f'&nbsp;&nbsp;<span style="color:#89b4fa">{arrows}</span>'
                f'&nbsp;&nbsp;<span style="color:{cb}; font-size:13px">██</span> {b}{tag_b}'
            )
        self.rules_text.setHtml("<br>".join(lines))
        n_img = sum(1 for t in config.tiles if config.images.get(t))
        LOG.debug("Rules view: %d tiles (%d with images), %d rules", len(config.tiles), n_img, len(config.rules))

    def _pick_tile(self, title: str) -> Optional[str]:
        """Prompt user to pick a tile from the current tileset. Returns name or None."""
        name = self.ts_combo.currentText()
        if name not in self._tilesets:
            QMessageBox.warning(self, "No Tileset", "Select a tileset first.")
            return None
        config = self._tilesets[name]
        if not config.tiles:
            QMessageBox.warning(self, "Empty Tileset", "The current tileset has no tiles.")
            return None
        tile_name, ok = QInputDialog.getItem(
            self, "Select Tile", title,
            config.tiles, editable=False)
        if not ok or not tile_name:
            return None
        return tile_name

    def _assign_tile_color(self):
        """Open a color dialog to change a tile's color in the current tileset."""
        tile_name = self._pick_tile("Choose a tile to recolor:")
        if tile_name is None:
            return
        config = self._tilesets[self.ts_combo.currentText()]
        current = QColor(config.colors.get(tile_name, "#ffffff"))
        color = QColorDialog.getColor(current, self, f"Color for '{tile_name}'")
        if not color.isValid():
            return
        hex_color = color.name()
        config.colors[tile_name] = hex_color
        LOG.info("Tile '%s' color changed to %s", tile_name, hex_color)
        # Update canvas color mapping
        if self.worker:
            self.canvas._colors = config.colors
            self.canvas.update()
        if self.rules_toggle_btn.isChecked():
            self._refresh_rules_view()
        self.status_bar.showMessage(f"Color of '{tile_name}' set to {hex_color}", 3000)

    def _assign_tile_image(self):
        """Open a file dialog to assign an image to a tile in the current tileset."""
        tile_name = self._pick_tile("Choose a tile to assign an image to:")
        if tile_name is None:
            return
        config = self._tilesets[self.ts_combo.currentText()]
        # Open file dialog for image
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select Image for '{tile_name}'",
            self._presets_dir or "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tga *.tiff);;All Files (*)")
        if not path:
            return
        try:
            b64 = TilesetConfig.image_from_file(path)
            config.images[tile_name] = b64
            LOG.info("Assigned image to tile '%s' from %s (%d bytes base64)",
                     tile_name, path, len(b64))
            # Update canvas images cache
            if self.worker:
                pix = TilesetConfig.image_to_pixmap(b64)
                if pix is not None:
                    self.canvas._images[tile_name] = pix
                self.canvas.update()
            # Refresh rules view to show the new image
            if self.rules_toggle_btn.isChecked():
                self._refresh_rules_view()
            self.status_bar.showMessage(f"Image assigned to '{tile_name}'", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image:\n{e}")
            LOG.error("Failed to assign image for tile '%s': %s", tile_name, e)

    def _load_presets_dir(self, path: str = ""):
        """Scan a directory for .json preset files and load them."""
        if not path:
            # Default: presets/ next to the script
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
        self._presets_dir = path
        if not os.path.isdir(path):
            LOG.info("Presets directory not found: %s", path)
            return
        loaded = 0
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(path, fname)
            try:
                with open(fpath, "r") as f:
                    ts = TilesetConfig.from_json(f.read())
                self._tilesets[ts.name] = ts
                idx = self.ts_combo.findText(ts.name)
                if idx < 0:
                    self.ts_combo.addItem(ts.name)
                loaded += 1
                n_img = sum(1 for v in (ts.images or {}).values() if v)
                LOG.info("Loaded preset: %s (%d tiles, %d images) from %s",
                           ts.name, len(ts.tiles), n_img, fpath)
            except Exception as e:
                LOG.warning("Failed to load preset %s: %s", fname, e)
        if loaded:
            self.ts_combo.setCurrentIndex(0)
            self._init_grid()
            self.status_bar.showMessage(f"Loaded {loaded} preset(s) from {path}", 4000)

    def _load_presets_dir_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Presets Directory",
            self._presets_dir or "")
        if path:
            self._load_presets_dir(path)

    def _load_preset_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset File", self._presets_dir or "", "JSON (*.json)")
        if path:
            try:
                with open(path) as f:
                    ts = TilesetConfig.from_json(f.read())
                self._tilesets[ts.name] = ts
                idx = self.ts_combo.findText(ts.name)
                if idx >= 0:
                    self.ts_combo.removeItem(idx)
                self.ts_combo.addItem(ts.name)
                self.ts_combo.setCurrentText(ts.name)
                self._init_grid()
                n_img = sum(1 for v in (ts.images or {}).values() if v)
                self.status_bar.showMessage(
                    f"Loaded preset: {ts.name} ({len(ts.tiles)} tiles, {n_img} images) from {path}", 5000)
                LOG.info("Loaded preset file: %s", path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load preset:\n{e}")
                LOG.error("Failed to load preset file: %s — %s", path, e)

    def _save_tileset(self):
        name = self.ts_combo.currentText()
        if name not in self._tilesets:
            QMessageBox.warning(self, "Error", "No tileset selected.")
            return
        ts = self._tilesets[name]
        path, _ = QFileDialog.getSaveFileName(self, "Save", f"{ts.name}.json", "JSON (*.json)")
        if path:
            with open(path, 'w') as f: f.write(ts.to_json())
            self.status_bar.showMessage(f"Saved to {path}", 3000)

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
        dlg = TilesetEditorDialog(self._tilesets[name], self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            nc = dlg.get_config(); self._tilesets[nc.name] = nc
            idx = self.ts_combo.findText(name)
            if idx >= 0: self.ts_combo.removeItem(idx)
            self.ts_combo.addItem(nc.name); self.ts_combo.setCurrentText(nc.name)
            self._init_grid()

    def _new_tileset(self):
        dlg = TilesetEditorDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            nc = dlg.get_config()
            if nc.tiles:
                self._tilesets[nc.name] = nc; self.ts_combo.addItem(nc.name)
                self.ts_combo.setCurrentText(nc.name); self._init_grid()

    def _init_grid(self):
        # Guard: do not reinitialize while a finish thread is still running
        if self._is_running:
            LOG.warning("_init_grid: ignored — simulation still running")
            return
        self._stop()
        name = self.ts_combo.currentText()
        if name not in self._tilesets:
            LOG.warning("_init_grid: tileset %r not found", name)
            return
        self._undo_stack.clear(); self._redo_stack.clear()
        self.canvas._contradiction_cells.clear()
        config = self._tilesets[name]
        if not config.tiles:
            self.status_bar.showMessage("Tileset has no tiles — cannot initialize.")
            LOG.error("_init_grid: tileset %r has no tiles", name)
            return
        size = self.size_spin.value()
        seed = self.seed_spin.value()
        wrap = self.wrap_check.isChecked()
        LOG.info("Init grid: tileset=%s  size=%d  seed=%d  wrap=%s", name, size, seed, wrap)
        self.worker = WFCWorker(size, config, seed, wrap)
        self.worker.contradiction_pos.connect(self._on_contradiction)
        self.canvas.set_worker(self.worker, config.colors, config.images)
        self._t_start = 0.0; self._update_stats()
        self._auto_undo_pushed = False
        self.status_bar.showMessage("Grid initialized")
        LOG.debug("  tiles=%s  rules=%d  heap_size=%d",
                   config.tiles, len(config.rules), size * size)

    def _random_seed(self):
        import random as _rnd
        self.seed_spin.setValue(_rnd.randint(0, 999999)); self._init_grid()

    def _push_undo(self):
        if not self.worker: return
        self._undo_stack.append(self.worker.get_snapshot())
        if len(self._undo_stack) > self._max_undo: self._undo_stack.pop(0)
        self._redo_stack.clear()
        LOG.debug("Undo pushed (stack depth=%d)", len(self._undo_stack))

    def _undo(self):
        if not self._undo_stack or not self.worker: return
        self._stop(); self._redo_stack.append(self.worker.get_snapshot())
        self.worker.restore_snapshot(self._undo_stack.pop())
        self.canvas.update(); self._update_stats()
        LOG.info("Undo (undo=%d, redo=%d)", len(self._undo_stack), len(self._redo_stack))

    def _redo(self):
        if not self._redo_stack or not self.worker: return
        self._stop(); self._undo_stack.append(self.worker.get_snapshot())
        self.worker.restore_snapshot(self._redo_stack.pop())
        self.canvas.update(); self._update_stats()
        LOG.info("Redo (undo=%d, redo=%d)", len(self._undo_stack), len(self._redo_stack))

    def _on_contradiction(self, y, x):
        """Track contradiction cells so they can be highlighted on the canvas."""
        self.canvas._contradiction_cells.add((y, x))
        LOG.warning("Contradiction at (%d, %d)", y, x)

    def _step_once(self):
        if not self.worker or self._is_running: return
        if self._t_start == 0: self._t_start = time.perf_counter()
        self._push_undo()
        self.canvas._contradiction_cells.clear()
        ok = self.worker.run_step()
        if not ok:
            LOG.warning("Step failed (contradiction or stopped)")
            self._stop()
            if self.retry_check.isChecked() and not self.worker.is_complete():
                QTimer.singleShot(100, self._retry); return
            if self.worker.is_complete():
                self.status_bar.showMessage("Complete!")
                LOG.info("Grid complete after step")
        else:
            LOG.debug("Step ok  (step=%d)", self.worker.step_count)
        self.canvas.update(); self._update_stats()

    def _auto_step(self):
        # Guard: timer may fire once more after _stop() clears _is_running
        if not self.worker or not self._is_running:
            return
        # Push undo snapshot only once at the start of each auto-run sequence
        if not self._auto_undo_pushed:
            self._push_undo()
            self._auto_undo_pushed = True
        self.canvas._contradiction_cells.clear()
        ok = self.worker.run_step()
        if not ok:
            LOG.warning("Auto-step failed at step %d", self.worker.step_count)
            self._timer.stop(); self.btn_auto.setText("Auto ▶▶"); self._is_running = False
            self._auto_undo_pushed = False
            if self.retry_check.isChecked() and not self.worker.is_complete():
                QTimer.singleShot(100, self._retry); return
            if self.worker.is_complete():
                self.status_bar.showMessage("Complete!")
                LOG.info("Auto-run complete (%d steps)", self.worker.step_count)
        self.canvas.update(); self._update_stats()

    def _toggle_auto(self):
        if self._timer.isActive():
            LOG.info("Auto-run stopped (step=%d)", self.worker.step_count if self.worker else 0)
            self._stop()
            self._auto_undo_pushed = False
        else:
            if not self.worker or self._is_running:
                return
            if self._t_start == 0: self._t_start = time.perf_counter()
            self._timer.start(self.speed_slider.value())
            self.btn_auto.setText("⏹ Stop"); self._is_running = True
            self._auto_undo_pushed = False  # Will be set on first _auto_step
            LOG.info("Auto-run started (interval=%dms)", self.speed_slider.value())

    def _stop(self):
        self._timer.stop(); self.btn_auto.setText("Auto ▶▶"); self._is_running = False
        self._auto_undo_pushed = False
        if self.worker: self.worker.stop()
        if self._finish_thread and self._finish_thread.isRunning():
            if self.worker: self.worker.stop()
            LOG.debug("Waiting for finish thread to quit...")
            self._finish_thread.quit()
            self._finish_thread.wait(2000)
            # Clean up worker object to prevent memory leak
            if self._finish_worker_ref is not None:
                self._finish_worker_ref.deleteLater()
                self._finish_worker_ref = None
            self._finish_thread = None
            LOG.debug("Finish thread cleaned up")

    def _finish_instantly(self):
        if not self.worker or self._is_running: return
        self._stop(); self._push_undo()
        self.canvas._contradiction_cells.clear()
        if self._t_start == 0: self._t_start = time.perf_counter()
        self._finish_thread = QThread()
        iw = InstantFinishWorker(self.worker); iw.moveToThread(self._finish_thread)
        self._finish_worker_ref = iw  # Keep reference for cleanup
        self._finish_thread.started.connect(iw.run)
        iw.progress.connect(self.prog_bar.setValue)
        iw.finished.connect(self._on_instant_done)
        self._finish_thread.start(); self._is_running = True; self.btn_finish.setEnabled(False)
        LOG.info("Finish-instantly started on thread")

    def _on_instant_done(self, success, elapsed):
        # Clean up thread and worker object
        if self._finish_thread is not None:
            self._finish_thread.quit()
            self._finish_thread.wait(3000)
        if self._finish_worker_ref is not None:
            self._finish_worker_ref.deleteLater()
            self._finish_worker_ref = None
        self._finish_thread = None
        self._is_running = False; self.btn_finish.setEnabled(True)
        self.canvas.update(); self._update_stats()
        if success:
            self.status_bar.showMessage(f"Done in {elapsed:.3f}s!", 5000)
            LOG.info("Finish-instantly succeeded in %.3fs", elapsed)
        elif self.retry_check.isChecked():
            LOG.warning("Finish-instantly failed (contradiction), scheduling retry")
            QTimer.singleShot(100, self._retry)
        else:
            self.status_bar.showMessage("Contradiction. Try a different seed.", 5000)
            LOG.warning("Finish-instantly failed (contradiction), no retry")

    def _retry(self):
        self._stop()
        name = self.ts_combo.currentText()
        if name not in self._tilesets:
            self.status_bar.showMessage("No valid tileset for retry.", 5000)
            return
        # Save pre-retry state so we can restore if all attempts fail
        pre_retry_snap = None
        if self.worker:
            pre_retry_snap = self.worker.get_snapshot()
        base = self.seed_spin.value()
        LOG.info("Retry: starting up to 10 attempts from base seed %d", base)
        for attempt in range(1, 11):
            ns = (base + attempt * 7919) % 1000000
            self.seed_spin.setValue(ns)
            self._undo_stack.clear(); self._redo_stack.clear()
            config = self._tilesets[name]
            self.worker = WFCWorker(self.size_spin.value(), config, ns, self.wrap_check.isChecked())
            self.canvas.set_worker(self.worker, config.colors, config.images)
            self._t_start = time.perf_counter()
            limit = self.worker.size ** 2 * 2
            for _ in range(int(limit)):
                if not self.worker.run_step():
                    break
            if self.worker.is_complete():
                self.canvas.update(); self._update_stats()
                self.status_bar.showMessage(f"Success on attempt {attempt}, seed {ns}!", 5000)
                LOG.info("Retry: success on attempt %d, seed=%d", attempt, ns)
                return
            LOG.debug("Retry: attempt %d failed (seed=%d)", attempt, ns)
            # Briefly process events so the UI stays responsive between attempts
            QApplication.processEvents()
        # All retries failed — restore the previous state
        LOG.warning("Retry: all 10 attempts failed")
        self.status_bar.showMessage("Failed after 10 retries — restored previous state.", 5000)
        if pre_retry_snap is not None:
            self.worker = WFCWorker(self.size_spin.value(), self._tilesets[name],
                                    self.seed_spin.value(), self.wrap_check.isChecked())
            self.worker.restore_snapshot(pre_retry_snap)
            self.canvas.set_worker(self.worker, self._tilesets[name].colors)
            self.canvas.update(); self._update_stats()

    def _update_stats(self):
        if not self.worker: return
        c = int(np.sum(self.worker.collapsed)); t = self.worker.size ** 2
        self.prog_bar.setValue(int(c / t * 100))
        el = time.perf_counter() - self._t_start if self._t_start > 0 else 0
        cps = c / el if el > 0.01 else 0
        ok = "✅ Complete" if self.worker.is_complete() else ""
        self.stats_lbl.setText(f"Collapsed: {c}/{t} ({int(c/t*100)}%)\nSteps: {self.worker.step_count}\n"
                               f"Time: {el:.2f}s | Cells/sec: {cps:.0f}\n{ok}")

    def _zoom_by(self, f):
        self.canvas._zoom = max(0.1, min(20.0, self.canvas._zoom * f)); self.canvas.update()

    def _export_png(self):
        if not self.worker: return
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "wfc_2d.png", "PNG (*.png)")
        if path:
            self.canvas.export_png(path, max(1, 512 // self.worker.size))
            self.status_bar.showMessage(f"Exported: {path}", 3000)
            LOG.info("Exported PNG: %s", path)

    def _export_json(self):
        if not self.worker: return
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "wfc_2d.json", "JSON (*.json)")
        if path:
            grid = []
            for y in range(self.worker.size):
                row = []
                for x in range(self.worker.size):
                    m = int(self.worker.wave[y, x]); e = get_popcount(m)
                    row.append(self.worker.tiles[m.bit_length() - 1] if e == 1 else f"?({e})")
                grid.append(row)
            data = {"tileset": self.ts_combo.currentText(), "size": self.worker.size,
                    "seed": self.seed_spin.value(), "steps": self.worker.step_count, "grid": grid}
            with open(path, 'w') as f: json.dump(data, f, indent=2)
            self.status_bar.showMessage(f"Exported: {path}", 3000)
            LOG.info("Exported JSON: %s  (%d cells)", path, self.worker.size ** 2)


def main():
    LOG.info("WFC 2D starting...")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(); win.show()
    LOG.info("Window shown — ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()