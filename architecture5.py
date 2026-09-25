# ============================================================
# png_to_editable_diagram.py
#
# Converts a raster architecture figure (PNG/JPG) into an
# EDITABLE vector reconstruction using computer vision +
# OCR + shape detection.
#
# Usage:
#   python png_to_editable_diagram.py path\to\figure.png
# ============================================================

import os
import sys
import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Rect:
    id: str
    x: int
    y: int
    w: int
    h: int
    fill_color: Tuple[int, int, int] = (255, 255, 255)
    stroke_color: Tuple[int, int, int] = (200, 200, 200)
    stroke_width: int = 1
    corner_radius: int = 6
    layer: str = "box"


@dataclass
class TextItem:
    id: str
    x: int
    y: int
    w: int
    h: int
    text: str
    font_size_px: int = 12
    color: Tuple[int, int, int] = (30, 30, 30)
    bold: bool = False
    italic: bool = False


@dataclass
class ArrowItem:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int] = (30, 30, 30)
    stroke_width: int = 2


@dataclass
class IconItem:
    id: str
    x: int
    y: int
    w: int
    h: int
    crop_path: str = ""
    kind: str = "crop"


@dataclass
class DiagramAnalysis:
    image_path: str
    image_w: int
    image_h: int
    panels: List[Rect] = field(default_factory=list)
    boxes: List[Rect] = field(default_factory=list)
    texts: List[TextItem] = field(default_factory=list)
    arrows: List[ArrowItem] = field(default_factory=list)
    icons: List[IconItem] = field(default_factory=list)


# ============================================================
# STAGE 1 — LOAD + PREPROCESS
# ============================================================

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_background(img: np.ndarray) -> Tuple[int, int, int]:
    flat = img.reshape(-1, 3)
    small = flat[::max(1, len(flat) // 5000)]
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        small.astype(np.float32), k, None, criteria,
        3, cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten())
    bg_bgr = centers[np.argmax(counts)].astype(int)
    # return RGB for convenience
    return (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))


# ============================================================
# STAGE 2 — OCR
# ============================================================

def detect_text_easyocr(img: np.ndarray) -> List[TextItem]:
    try:
        import easyocr
    except ImportError:
        print("  [OCR] easyocr not installed; falling back to pytesseract.")
        return detect_text_tesseract(img)

    print("  [OCR] running easyocr (this may take a minute) …")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(img)

    items = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs)) - x, int(max(ys)) - y
        if w < 4 or h < 4:
            continue
        font_px = max(10, int(h * 0.75))
        items.append(TextItem(
            id=str(uuid.uuid4())[:8],
            x=x, y=y, w=w, h=h,
            text=text.strip(),
            font_size_px=font_px,
        ))
    print(f"  [OCR] detected {len(items)} text items.")
    return items


def detect_text_tesseract(img: np.ndarray) -> List[TextItem]:
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        print("  [OCR] pytesseract not installed either. Skipping OCR.")
        return []

    gray = to_gray(img)
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    items = []
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        if not t:
            continue
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
        if conf < 40 or w < 4 or h < 4:
            continue
        items.append(TextItem(
            id=str(uuid.uuid4())[:8],
            x=x, y=y, w=w, h=h,
            text=t,
            font_size_px=max(10, int(h * 0.75)),
        ))
    return items


# ============================================================
# STAGE 3 — BOXES
# ============================================================

def detect_boxes(img: np.ndarray,
                 bg_color: Tuple[int, int, int]) -> List[Rect]:
    gray = to_gray(img)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[Rect] = []
    H, W = img.shape[:2]
    min_area = (W * H) * 0.0005
    max_area = (W * H) * 0.5

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(c)
        if w < 20 or h < 20:
            continue
        if len(approx) < 4:
            continue

        aspect = w / max(h, 1)
        if aspect > 30 or aspect < 0.03:
            continue

        # center pixel fill (BGR -> RGB)
        b, g, r = img[y + h // 2, x + w // 2]
        fill = (int(r), int(g), int(b))
        b, g, r = img[y, x]
        stroke = (int(r), int(g), int(b))

        boxes.append(Rect(
            id=str(uuid.uuid4())[:8],
            x=x, y=y, w=w, h=h,
            fill_color=fill,
            stroke_color=stroke,
            stroke_width=1,
            corner_radius=8,
            layer="box",
        ))

    boxes = _dedupe_boxes(boxes)
    print(f"  [BOX] detected {len(boxes)} candidate boxes.")
    return boxes


def _dedupe_boxes(boxes: List[Rect]) -> List[Rect]:
    boxes = sorted(boxes, key=lambda b: b.w * b.h, reverse=True)
    kept: List[Rect] = []
    for b in boxes:
        if any(_iom(b, k) > 0.85 for k in kept):
            continue
        kept.append(b)
    return kept


def _iom(a: Rect, b: Rect) -> float:
    xa1, ya1, xa2, ya2 = a.x, a.y, a.x + a.w, a.y + a.h
    xb1, yb1, xb2, yb2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix = max(0, min(xa2, xb2) - max(xa1, xb1))
    iy = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = ix * iy
    smaller = min(a.w * a.h, b.w * b.h)
    return inter / max(smaller, 1)


# ============================================================
# STAGE 4 — ARROWS  (fixed: robust to all HoughLinesP shapes)
# ============================================================

def detect_arrows(img: np.ndarray,
                  boxes: List[Rect]) -> List[ArrowItem]:
    """
    Hough line transform; each entry may come back as a scalar,
    a (1,4) array, a (4,) array, or a (1,1,4) array depending on
    OpenCV version. We flatten every entry to a flat tuple of 4.
    """
    gray = to_gray(img)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=30, maxLineGap=6,
    )
    arrows: List[ArrowItem] = []
    if lines is None:
        print("  [ARR] no lines detected.")
        return arrows

    for raw in lines:
        # flatten to 4 ints no matter the shape
        arr = np.array(raw).ravel()
        if arr.size < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in arr[:4]]

        # length filter
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 40:
            continue

        # color sample from mid-point (BGR -> RGB)
        mx = int((x1 + x2) / 2)
        my = int((y1 + y2) / 2)
        if 0 <= my < img.shape[0] and 0 <= mx < img.shape[1]:
            b, g, r = img[my, mx]
            col = (int(r), int(g), int(b))
        else:
            col = (30, 30, 30)

        arrows.append(ArrowItem(
            id=str(uuid.uuid4())[:8],
            x1=x1, y1=y1, x2=x2, y2=y2,
            color=col,
            stroke_width=2,
        ))

    arrows = _dedupe_arrows(arrows)
    print(f"  [ARR] detected {len(arrows)} candidate arrows.")
    return arrows


def _dedupe_arrows(arrows: List[ArrowItem], tol: int = 6) -> List[ArrowItem]:
    kept: List[ArrowItem] = []
    for a in arrows:
        dup = False
        for k in kept:
            if (abs(a.x1 - k.x1) < tol and abs(a.y1 - k.y1) < tol
                    and abs(a.x2 - k.x2) < tol and abs(a.y2 - k.y2) < tol):
                dup = True
                break
            if (abs(a.x1 - k.x2) < tol and abs(a.y1 - k.y2) < tol
                    and abs(a.x2 - k.x1) < tol and abs(a.y2 - k.y1) < tol):
                dup = True
                break
        if not dup:
            kept.append(a)
    return kept


# ============================================================
# STAGE 5 — ICONS
# ============================================================

def detect_icons(img: np.ndarray,
                 boxes: List[Rect],
                 texts: List[TextItem],
                 out_dir: str,
                 bg_color: Tuple[int, int, int]) -> List[IconItem]:
    bg_bgr = np.array([bg_color[2], bg_color[1], bg_color[0]], dtype=np.int16)
    diff = np.abs(img.astype(np.int16) - bg_bgr).sum(axis=2)
    mask = (diff > 30).astype(np.uint8) * 255

    for t in texts:
        cv2.rectangle(mask, (t.x - 2, t.y - 2),
                      (t.x + t.w + 2, t.y + t.h + 2), 0, -1)
    for b in boxes:
        cv2.rectangle(mask, (b.x, b.y),
                      (b.x + b.w, b.y + b.h), 0, 4)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    icons: List[IconItem] = []
    H, W = img.shape[:2]
    min_side = max(12, int(min(H, W) * 0.012))

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < min_side or h < min_side:
            continue
        if w > W * 0.25 or h > H * 0.25:
            continue
        if area < min_side * min_side * 0.3:
            continue
        if w / h > 6 or h / w > 6:
            continue

        crop = img[y:y + h, x:x + w]
        crop_path = os.path.join(out_dir, f"icon_{len(icons):03d}.png")
        cv2.imwrite(crop_path, crop)

        icons.append(IconItem(
            id=str(uuid.uuid4())[:8],
            x=int(x), y=int(y), w=int(w), h=int(h),
            crop_path=crop_path,
            kind="crop",
        ))

    print(f"  [ICO] extracted {len(icons)} icon crops.")
    return icons


# ============================================================
# STAGE 6 — PANELS
# ============================================================

def detect_panels(img: np.ndarray,
                  bg_color: Tuple[int, int, int]) -> List[Rect]:
    gray = to_gray(img)
    _, th = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((15, 15), np.uint8)
    merged = cv2.dilate(th, kernel, iterations=2)

    contours, _ = cv2.findContours(
        merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = img.shape[:2]
    panels: List[Rect] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < (W * H) * 0.03:
            continue
        if area > (W * H) * 0.9:
            continue
        if w < W * 0.3 or h < H * 0.08:
            continue

        b, g, r = img[y + h // 2, x + w // 2]
        fill = (int(r), int(g), int(b))

        panels.append(Rect(
            id=str(uuid.uuid4())[:8],
            x=x, y=y, w=w, h=h,
            fill_color=fill,
            stroke_color=fill,
            stroke_width=2,
            corner_radius=12,
            layer="panel",
        ))

    print(f"  [PNL] detected {len(panels)} panels.")
    return panels


# ============================================================
# STAGE 7 — RENDER RECONSTRUCTION
# ============================================================

def _hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def render_reconstruction(analysis: DiagramAnalysis,
                          out_png: str,
                          out_pdf: str,
                          out_svg: str) -> None:
    W_px, H_px = analysis.image_w, analysis.image_h
    W = W_px / 100
    H = H_px / 100

    fig, ax = plt.subplots(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, W_px)
    ax.set_ylim(H_px, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for p in analysis.panels:
        ax.add_patch(FancyBboxPatch(
            (p.x, p.y), p.w, p.h,
            boxstyle=f"round,pad=0,rounding_size={p.corner_radius}",
            facecolor=_hex(p.fill_color),
            edgecolor=_hex(p.stroke_color),
            linewidth=0.9, zorder=1,
        ))

    for b in analysis.boxes:
        ax.add_patch(FancyBboxPatch(
            (b.x, b.y), b.w, b.h,
            boxstyle=f"round,pad=0,rounding_size={b.corner_radius}",
            facecolor=_hex(b.fill_color),
            edgecolor=_hex(b.stroke_color),
            linewidth=0.8, zorder=3,
        ))

    for a in analysis.arrows:
        ax.add_patch(FancyArrowPatch(
            (a.x1, a.y1), (a.x2, a.y2),
            arrowstyle="-|>", mutation_scale=9,
            linewidth=1.0, color=_hex(a.color),
            shrinkA=0, shrinkB=0, zorder=8,
        ))

    for ic in analysis.icons:
        try:
            im = Image.open(ic.crop_path).convert("RGBA")
            ax.imshow(
                np.array(im),
                extent=[ic.x, ic.x + ic.w, ic.y + ic.h, ic.y],
                zorder=9,
            )
        except Exception:
            pass

    for t in analysis.texts:
        cx = t.x + t.w / 2
        cy = t.y + t.h / 2
        ax.text(
            cx, cy, t.text,
            ha="center", va="center",
            fontsize=max(6, t.font_size_px * 0.55),
            color=_hex(t.color), zorder=20,
        )

    fig.savefig(out_png, dpi=200, facecolor="#FFFFFF")
    fig.savefig(out_pdf, facecolor="#FFFFFF")
    fig.savefig(out_svg, facecolor="#FFFFFF")
    plt.close(fig)


# ============================================================
# STAGE 8 — EDITABLE PYTHON SOURCE
# ============================================================

def write_editable_python(analysis: DiagramAnalysis, out_py: str) -> None:
    lines = [
        "# ============================================================",
        "# Auto-generated editable reconstruction",
        f"# Source: {os.path.basename(analysis.image_path)}",
        f"# Canvas: {analysis.image_w} x {analysis.image_h} px",
        "# ============================================================",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from matplotlib.patches import FancyBboxPatch, FancyArrowPatch",
        "from PIL import Image",
        "import numpy as np",
        "",
        f"FIG_W = {analysis.image_w / 100:.2f}",
        f"FIG_H = {analysis.image_h / 100:.2f}",
        "",
        "fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=200)",
        "fig.patch.set_facecolor('#FFFFFF')",
        "ax.set_facecolor('#FFFFFF')",
        f"ax.set_xlim(0, {analysis.image_w})",
        f"ax.set_ylim({analysis.image_h}, 0)",
        "ax.set_aspect('equal', adjustable='box')",
        "ax.axis('off')",
        "plt.subplots_adjust(left=0, right=1, bottom=0, top=1)",
        "",
    ]

    def hexc(c):
        return "#{:02X}{:02X}{:02X}".format(*c)

    lines.append("# ---------- panels ----------")
    for p in analysis.panels:
        lines.append(
            f"ax.add_patch(FancyBboxPatch(({p.x}, {p.y}), {p.w}, {p.h}, "
            f"boxstyle='round,pad=0,rounding_size={p.corner_radius}', "
            f"facecolor='{hexc(p.fill_color)}', "
            f"edgecolor='{hexc(p.stroke_color)}', "
            f"linewidth=1.0, zorder=1))"
        )

    lines.append("")
    lines.append("# ---------- boxes ----------")
    for b in analysis.boxes:
        lines.append(
            f"ax.add_patch(FancyBboxPatch(({b.x}, {b.y}), {b.w}, {b.h}, "
            f"boxstyle='round,pad=0,rounding_size={b.corner_radius}', "
            f"facecolor='{hexc(b.fill_color)}', "
            f"edgecolor='{hexc(b.stroke_color)}', "
            f"linewidth=0.9, zorder=3))"
        )

    lines.append("")
    lines.append("# ---------- arrows ----------")
    for a in analysis.arrows:
        lines.append(
            f"ax.add_patch(FancyArrowPatch(({a.x1}, {a.y1}), "
            f"({a.x2}, {a.y2}), arrowstyle='-|>', mutation_scale=9, "
            f"linewidth=1.0, color='{hexc(a.color)}', "
            f"shrinkA=0, shrinkB=0, zorder=8))"
        )

    lines.append("")
    lines.append("# ---------- icons (embedded crops) ----------")
    for ic in analysis.icons:
        rel = os.path.basename(ic.crop_path)
        lines.append(f"_im = Image.open('{rel}').convert('RGBA')")
        lines.append(
            f"ax.imshow(np.array(_im), "
            f"extent=[{ic.x}, {ic.x + ic.w}, {ic.y + ic.h}, {ic.y}], "
            f"zorder=9)"
        )

    lines.append("")
    lines.append("# ---------- text ----------")
    for t in analysis.texts:
        cx = t.x + t.w / 2
        cy = t.y + t.h / 2
        safe = t.text.replace("\\", "\\\\").replace("'", "\\'")
        lines.append(
            f"ax.text({cx:.1f}, {cy:.1f}, '{safe}', "
            f"ha='center', va='center', "
            f"fontsize={max(6, t.font_size_px * 0.55):.2f}, "
            f"color='{hexc(t.color)}', zorder=20)"
        )

    lines.append("")
    lines.append("fig.savefig('reconstruction.png', dpi=200, "
                 "facecolor='#FFFFFF')")
    lines.append("fig.savefig('reconstruction.pdf', facecolor='#FFFFFF')")
    lines.append("fig.savefig('reconstruction.svg', facecolor='#FFFFFF')")
    lines.append("plt.close(fig)")

    with open(out_py, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OUT] wrote editable Python -> {out_py}")


# ============================================================
# STAGE 9 — DEBUG OVERLAY
# ============================================================

def draw_debug_overlay(img: np.ndarray,
                       analysis: DiagramAnalysis,
                       out_path: str) -> None:
    dbg = img.copy()
    for p in analysis.panels:
        cv2.rectangle(dbg, (p.x, p.y),
                      (p.x + p.w, p.y + p.h), (255, 0, 255), 2)
    for b in analysis.boxes:
        cv2.rectangle(dbg, (b.x, b.y),
                      (b.x + b.w, b.y + b.h), (0, 200, 0), 2)
    for a in analysis.arrows:
        cv2.arrowedLine(dbg, (a.x1, a.y1), (a.x2, a.y2),
                        (0, 140, 255), 2, tipLength=0.05)
    for t in analysis.texts:
        cv2.rectangle(dbg, (t.x, t.y),
                      (t.x + t.w, t.y + t.h), (255, 100, 0), 1)
    for ic in analysis.icons:
        cv2.rectangle(dbg, (ic.x, ic.y),
                      (ic.x + ic.w, ic.y + ic.h), (0, 0, 255), 2)
    cv2.imwrite(out_path, dbg)
    print(f"  [DBG] wrote overlay -> {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python png_to_editable_diagram.py <figure.png>")
        sys.exit(1)

    src = sys.argv[1]
    if not os.path.isfile(src):
        print(f"File not found: {src}")
        sys.exit(1)

    out_dir = os.path.splitext(src)[0] + "_recon"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/8] Loading {src}")
    img = load_image(src)
    H, W = img.shape[:2]
    print(f"  size = {W} x {H}")

    print("[2/8] Estimating background color")
    bg = estimate_background(img)
    print(f"  background = RGB{bg}")

    print("[3/8] Detecting panels")
    panels = detect_panels(img, bg)

    print("[4/8] Detecting boxes")
    boxes = detect_boxes(img, bg)

    print("[5/8] Detecting text (OCR)")
    texts = detect_text_easyocr(img)

    print("[6/8] Detecting arrows")
    arrows = detect_arrows(img, boxes)

    print("[7/8] Extracting icons")
    icons = detect_icons(img, boxes, texts, out_dir, bg)

    analysis = DiagramAnalysis(
        image_path=src, image_w=W, image_h=H,
        panels=panels, boxes=boxes, texts=texts,
        arrows=arrows, icons=icons,
    )

    json_path = os.path.join(out_dir, "analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(analysis), f, indent=2, default=list)
    print(f"  [OUT] analysis -> {json_path}")

    print("[8/8] Reconstructing and writing outputs")
    render_reconstruction(
        analysis,
        out_png=os.path.join(out_dir, "reconstruction.png"),
        out_pdf=os.path.join(out_dir, "reconstruction.pdf"),
        out_svg=os.path.join(out_dir, "reconstruction.svg"),
    )

    write_editable_python(
        analysis,
        out_py=os.path.join(out_dir, "editable_reconstruction.py"),
    )

    draw_debug_overlay(
        img, analysis,
        out_path=os.path.join(out_dir, "debug_overlay.png"),
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Output folder : {out_dir}")
    print("  - reconstruction.png")
    print("  - reconstruction.pdf")
    print("  - reconstruction.svg")
    print("  - editable_reconstruction.py    <- tweak this")
    print("  - analysis.json                 <- all detected items")
    print("  - debug_overlay.png             <- sanity check")
    print("  - icon_*.png                    <- extracted icon crops")


if __name__ == "__main__":
    main()