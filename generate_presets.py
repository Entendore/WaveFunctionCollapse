"""
Preset Generator for WFC 2D
=============================
Generates three tileset preset JSON files with embedded base64 images.

Usage:
    python generate_presets.py [output_dir]

If no output_dir is given, defaults to ./presets/

Requires: Pillow (pip install Pillow)
"""

import sys
import os
import json
import base64
import math
import random as _rng_mod

from PIL import Image, ImageDraw

TILE = 32  # sprite size in pixels


def _img():
    return Image.new("RGBA", (TILE, TILE), (0, 0, 0, 255))


def _to_b64(img: Image.Image) -> str:
    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Terrain sprites
# ---------------------------------------------------------------------------

def _terrain_water():
    img = _img()
    d = ImageDraw.Draw(img)
    # Gradient-like blue
    for y in range(TILE):
        t = y / TILE
        r = int(41 + (26 - 41) * t)
        g = int(128 + (82 - 128) * t)
        b = int(185 + (118 - 185) * t)
        d.line([(0, y), (TILE, y)], fill=(r, g, b))
    # Wave lines
    for row in [8, 16, 24]:
        for x in range(0, TILE, 2):
            y = row + int(2 * math.sin(x * 0.5))
            d.point((x, y), fill=(255, 255, 255, 80))
            if x + 1 < TILE:
                d.point((x + 1, y), fill=(255, 255, 255, 80))
    return img


def _terrain_sand():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(240, 215, 140))
    rng = _rng_mod.Random(42)
    for _ in range(30):
        x, y = rng.randint(0, TILE - 2), rng.randint(0, TILE - 2)
        c = (200 + rng.randint(-20, 20), 180 + rng.randint(-20, 20), 100 + rng.randint(-20, 20), 255)
        d.ellipse([x, y, x + 2, y + 2], fill=c)
    return img


def _terrain_grass():
    img = _img()
    d = ImageDraw.Draw(img)
    for y in range(TILE):
        t = y / TILE
        r = int(39 + (30 - 39) * t)
        g = int(174 + (132 - 174) * t)
        b = int(96 + (73 - 96) * t)
        d.line([(0, y), (TILE, y)], fill=(r, g, b))
    rng = _rng_mod.Random(7)
    for _ in range(20):
        x = rng.randint(1, TILE - 2)
        y = rng.randint(TILE // 3, TILE - 2)
        dx, dy = rng.randint(-3, 3), rng.randint(3, 7)
        d.line([(x, y), (x + dx, y - dy)], fill=(20, 90, 50, 200))
    return img


def _terrain_forest():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(26, 94, 42))
    # Tree canopies
    for tx, ty, r in [(8, 10, 10), (20, 8, 11), (16, 22, 9), (6, 24, 8)]:
        d.ellipse([tx - r, ty - r, tx + r, ty + r], fill=(25, 111, 61))
        d.ellipse([tx - r + 2, ty - r + 2, tx + r - 2, ty + r - 2], fill=(14, 74, 31))
    # Trunks
    for tx, ty, _ in [(8, 10, 10), (20, 8, 11), (16, 22, 9), (6, 24, 8)]:
        d.line([(tx, ty + 4), (tx, TILE - 2)], fill=(93, 58, 26), width=2)
    return img


def _terrain_mountain():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(127, 140, 141))
    # Main peak
    d.polygon([(4, TILE - 2), (16, 4), (28, TILE - 2)], fill=(149, 165, 166))
    # Snow cap
    d.polygon([(10, 14), (16, 4), (22, 14)], fill=(236, 240, 241))
    # Smaller peak
    d.polygon([(18, TILE - 2), (28, 10), (TILE - 2, TILE - 2)], fill=(170, 183, 184))
    # Peak edge
    d.polygon([(18, TILE - 2), (28, 10), (TILE - 2, TILE - 2)], outline=(93, 109, 126))
    return img


# ---------------------------------------------------------------------------
# Dungeon sprites
# ---------------------------------------------------------------------------

def _dungeon_floor():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(196, 167, 125))
    d.line([(16, 0), (16, TILE)], fill=(160, 140, 110), width=1)
    d.line([(0, 16), (TILE, 16)], fill=(160, 140, 110), width=1)
    rng = _rng_mod.Random(12)
    for _ in range(6):
        x, y = rng.randint(0, TILE - 4), rng.randint(0, TILE - 4)
        d.line([(x, y), (x + rng.randint(2, 6), y)], fill=(180, 160, 130, 100))
    return img


def _dungeon_wall():
    img = _img()
    d = ImageDraw.Draw(img)
    for row_i, y in enumerate(range(0, TILE, 8)):
        offset = 8 if row_i % 2 else 0
        for x in range(-8 + offset, TILE, 16):
            d.rectangle([x, y, x + 16, y + 8], fill=(52, 73, 94), outline=(44, 62, 80))
            d.rectangle([x + 1, y + 1, x + 15, y + 7], fill=(44, 62, 80))
    return img


def _dungeon_door():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(44, 62, 80))
    d.rounded_rectangle([6, 2, 26, 30], radius=2, fill=(181, 101, 29))
    d.line([(16, 2), (16, 30)], fill=(139, 69, 19))
    d.line([(6, 16), (26, 16)], fill=(139, 69, 19))
    d.ellipse([20, 14, 24, 18], fill=(212, 167, 106))
    return img


def _dungeon_corridor():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(168, 160, 142))
    d.rectangle([6, 0, 26, TILE], fill=(142, 135, 122))
    d.rectangle([0, 0, 6, TILE], fill=(107, 99, 87))
    d.rectangle([26, 0, 32, TILE], fill=(107, 99, 87))
    return img


def _dungeon_room():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(221, 214, 198))
    for x in [0, 10, 20, 30]:
        d.line([(x, 0), (x, TILE)], fill=(190, 180, 165))
    for y in [0, 10, 20, 30]:
        d.line([(0, y), (TILE, y)], fill=(190, 180, 165))
    d.rectangle([1, 1, TILE - 2, TILE - 2], outline=(184, 176, 160), width=2)
    return img


# ---------------------------------------------------------------------------
# Circuit sprites
# ---------------------------------------------------------------------------

def _circuit_wire_h():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(26, 26, 46))
    cy = TILE // 2
    d.line([(2, cy), (TILE - 2, cy)], fill=(52, 152, 219), width=3)
    d.ellipse([0, cy - 3, 6, cy + 3], fill=(52, 152, 219))
    d.ellipse([TILE - 6, cy - 3, TILE, cy + 3], fill=(52, 152, 219))
    return img


def _circuit_wire_v():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(26, 26, 46))
    cx = TILE // 2
    d.line([(cx, 2), (cx, TILE - 2)], fill=(41, 128, 185), width=3)
    d.ellipse([cx - 3, 0, cx + 3, 6], fill=(41, 128, 185))
    d.ellipse([cx - 3, TILE - 6, cx + 3, TILE], fill=(41, 128, 185))
    return img


def _circuit_wire_cross():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(26, 26, 46))
    cx, cy = TILE // 2, TILE // 2
    d.line([(2, cy), (TILE - 2, cy)], fill=(26, 188, 156), width=3)
    d.line([(cx, 2), (cx, TILE - 2)], fill=(26, 188, 156), width=3)
    for dx, dy in [(0, 0), (0, -6), (-6, 0), (-6, -6)]:
        d.ellipse([cx - 3 + dx, cy - 3 + dy, cx + 3 + dx, cy + 3 + dy], fill=(26, 188, 156))
    for dx, dy in [(0, TILE - 6), (TILE - 6, 0), (TILE - 6, TILE - 6)]:
        d.ellipse([cx - 3 + dx - (1 if dx else 0), cy - 3 + dy - (1 if dy else 0),
                    cx + 3 + dx + (1 if dx else 0), cy + 3 + dy + (1 if dy else 0)], fill=(26, 188, 156))
    return img


def _circuit_wire_turn():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(26, 26, 46))
    cx, cy = TILE // 2, TILE // 2
    d.line([(cx, 2), (cx, cy)], fill=(22, 160, 133), width=3)
    d.line([(cx, cy), (TILE - 2, cy)], fill=(22, 160, 133), width=3)
    d.ellipse([cx - 3, 0, cx + 3, 6], fill=(22, 160, 133))
    d.ellipse([TILE - 6, cy - 3, TILE, cy + 3], fill=(22, 160, 133))
    return img


def _circuit_empty():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(15, 15, 35))
    for x in [8, 16, 24]:
        for y in [8, 16, 24]:
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255, 20))
    return img


def _circuit_node():
    img = _img()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TILE, TILE], fill=(26, 26, 46))
    cx, cy = TILE // 2, TILE // 2
    d.line([(2, cy), (TILE - 2, cy)], fill=(231, 76, 60), width=2)
    d.line([(cx, 2), (cx, TILE - 2)], fill=(231, 76, 60), width=2)
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(231, 76, 60), outline=(192, 57, 43))
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(241, 148, 138))
    return img


# ---------------------------------------------------------------------------
# Tileset builders
# ---------------------------------------------------------------------------

def _make_sym_rules(pairs):
    result = []
    for a, b in pairs:
        for dname in ["UP", "DOWN", "LEFT", "RIGHT"]:
            result.append([a, dname, b])
    return result


def build_terrain():
    sprites = {
        "WATER": _terrain_water(), "SAND": _terrain_sand(),
        "GRASS": _terrain_grass(), "FOREST": _terrain_forest(),
        "MOUNTAIN": _terrain_mountain(),
    }
    colors = {t: "#000000" for t in sprites}
    tiles = list(sprites.keys())
    simple = [
        ("WATER", "WATER"), ("SAND", "SAND"), ("GRASS", "GRASS"),
        ("FOREST", "FOREST"), ("MOUNTAIN", "MOUNTAIN"),
        ("WATER", "SAND"), ("SAND", "GRASS"), ("GRASS", "FOREST"),
        ("FOREST", "MOUNTAIN"),
    ]
    images = {t: _to_b64(pix) for t, pix in sprites.items()}
    return {
        "name": "Terrain", "tiles": tiles, "colors": colors,
        "rules": _make_sym_rules(simple),
        "weights": {"WATER": 1.0, "SAND": 1.2, "GRASS": 2.0,
                     "FOREST": 1.5, "MOUNTAIN": 0.5},
        "images": images,
    }


def build_dungeon():
    sprites = {
        "FLOOR": _dungeon_floor(), "WALL": _dungeon_wall(),
        "DOOR": _dungeon_door(), "CORRIDOR": _dungeon_corridor(),
        "ROOM": _dungeon_room(),
    }
    colors = {t: "#000000" for t in sprites}
    tiles = list(sprites.keys())
    simple = [
        ("FLOOR", "FLOOR"), ("WALL", "WALL"), ("DOOR", "DOOR"),
        ("CORRIDOR", "CORRIDOR"), ("ROOM", "ROOM"),
        ("FLOOR", "DOOR"), ("DOOR", "FLOOR"), ("DOOR", "WALL"),
        ("FLOOR", "CORRIDOR"), ("CORRIDOR", "FLOOR"),
        ("FLOOR", "ROOM"), ("ROOM", "FLOOR"),
        ("ROOM", "WALL"), ("WALL", "ROOM"),
        ("CORRIDOR", "DOOR"), ("DOOR", "CORRIDOR"),
    ]
    images = {t: _to_b64(pix) for t, pix in sprites.items()}
    return {
        "name": "Dungeon", "tiles": tiles, "colors": colors,
        "rules": _make_sym_rules(simple),
        "weights": {"FLOOR": 2.0, "WALL": 3.0, "DOOR": 0.8,
                     "CORRIDOR": 1.5, "ROOM": 1.0},
        "images": images,
    }


def build_circuit():
    sprites = {
        "WIRE_H": _circuit_wire_h(), "WIRE_V": _circuit_wire_v(),
        "WIRE_CROSS": _circuit_wire_cross(), "WIRE_TURN": _circuit_wire_turn(),
        "EMPTY": _circuit_empty(), "NODE": _circuit_node(),
    }
    colors = {t: "#000000" for t in sprites}
    tiles = list(sprites.keys())
    rules = [
        ["WIRE_H", "LEFT", "WIRE_H"], ["WIRE_H", "RIGHT", "WIRE_H"],
        ["WIRE_H", "LEFT", "WIRE_TURN"], ["WIRE_H", "RIGHT", "WIRE_TURN"],
        ["WIRE_H", "LEFT", "WIRE_CROSS"], ["WIRE_H", "RIGHT", "WIRE_CROSS"],
        ["WIRE_H", "LEFT", "NODE"], ["WIRE_H", "RIGHT", "NODE"],
        ["WIRE_V", "UP", "WIRE_V"], ["WIRE_V", "DOWN", "WIRE_V"],
        ["WIRE_V", "UP", "WIRE_TURN"], ["WIRE_V", "DOWN", "WIRE_TURN"],
        ["WIRE_V", "UP", "WIRE_CROSS"], ["WIRE_V", "DOWN", "WIRE_CROSS"],
        ["WIRE_V", "UP", "NODE"], ["WIRE_V", "DOWN", "NODE"],
        ["WIRE_CROSS", "UP", "WIRE_V"], ["WIRE_CROSS", "DOWN", "WIRE_V"],
        ["WIRE_CROSS", "LEFT", "WIRE_H"], ["WIRE_CROSS", "RIGHT", "WIRE_H"],
        ["WIRE_CROSS", "UP", "WIRE_CROSS"], ["WIRE_CROSS", "DOWN", "WIRE_CROSS"],
        ["WIRE_CROSS", "LEFT", "WIRE_CROSS"], ["WIRE_CROSS", "RIGHT", "WIRE_CROSS"],
        ["WIRE_TURN", "UP", "WIRE_V"], ["WIRE_TURN", "DOWN", "WIRE_V"],
        ["WIRE_TURN", "LEFT", "WIRE_H"], ["WIRE_TURN", "RIGHT", "WIRE_H"],
        ["WIRE_TURN", "UP", "WIRE_TURN"], ["WIRE_TURN", "RIGHT", "WIRE_TURN"],
        ["WIRE_TURN", "DOWN", "WIRE_TURN"], ["WIRE_TURN", "LEFT", "WIRE_TURN"],
        ["NODE", "UP", "WIRE_V"], ["NODE", "DOWN", "WIRE_V"],
        ["NODE", "LEFT", "WIRE_H"], ["NODE", "RIGHT", "WIRE_H"],
        ["NODE", "UP", "WIRE_CROSS"], ["NODE", "DOWN", "WIRE_CROSS"],
        ["NODE", "LEFT", "WIRE_CROSS"], ["NODE", "RIGHT", "WIRE_CROSS"],
        ["NODE", "UP", "NODE"], ["NODE", "DOWN", "NODE"],
        ["NODE", "LEFT", "NODE"], ["NODE", "RIGHT", "NODE"],
        ["EMPTY", "UP", "EMPTY"], ["EMPTY", "DOWN", "EMPTY"],
        ["EMPTY", "LEFT", "EMPTY"], ["EMPTY", "RIGHT", "EMPTY"],
    ]
    images = {t: _to_b64(pix) for t, pix in sprites.items()}
    return {
        "name": "Circuit", "tiles": tiles, "colors": colors,
        "rules": rules,
        "weights": {"WIRE_H": 1.5, "WIRE_V": 1.5, "WIRE_CROSS": 0.8,
                     "WIRE_TURN": 1.2, "EMPTY": 2.0, "NODE": 0.5},
        "images": images,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "presets")
    os.makedirs(out_dir, exist_ok=True)

    presets = [build_terrain, build_dungeon, build_circuit]
    for builder in presets:
        data = builder()
        path = os.path.join(out_dir, f"{data['name'].lower()}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        n_img = len(data["images"])
        size_kb = os.path.getsize(path) / 1024
        print(f"  {data['name']:10s}  {len(data['tiles'])} tiles, "
              f"{n_img} images, {size_kb:.0f} KB  -> {path}")

    print(f"\nDone — {len(presets)} presets written to {out_dir}/")


if __name__ == "__main__":
    main()