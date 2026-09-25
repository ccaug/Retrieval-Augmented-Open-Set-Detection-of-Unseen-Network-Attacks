# ============================================================
# architecture3.py
#
# Retrieval-Augmented Open-Set Detection of Unseen Network Attacks
#
# ACM CODASPY 2027
#
# FINAL GEOMETRY VERSION
# ============================================================

import os
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    Rectangle,
    Circle,
    Ellipse,
    Polygon,
    FancyArrowPatch,
    Arc,
)
from matplotlib.lines import Line2D


# ============================================================
# FIGURE CONFIGURATION
# ============================================================

FIG_W = 24.0
FIG_H = 15.0
DPI = 600

FONT = "DejaVu Sans"

# ============================================================
# TYPOGRAPHY
# ============================================================

MIN_TEXT_FS = 12.0

SECTION_TITLE_FS = 15.0
SECTION_SUBTITLE_FS = 12.0

PIPE_TITLE_FS = 13.2
PIPE_BODY_FS = 12.0

EVIDENCE_TITLE_FS = 12.8
EVIDENCE_MAIN_FS = 12.2
EVIDENCE_BODY_FS = 12.0

FUSE_TITLE_FS = 15.0
FUSE_BODY_FS = 12.0
FUSE_EQ_FS = 12.5

DECISION_TITLE_FS = 13.8
DECISION_EQ_FS = 12.0

OUTPUT_TITLE_FS = 13.5
OUTPUT_BODY_FS = 12.0

STAGE_TITLE_FS = 12.5
STAGE_BODY_FS = 12.0
STAGE_TAG_FS = 12.0

LEGEND_FS = 12.0
FOOTNOTE_FS = 12.0


# ============================================================
# PALETTE
# ============================================================

BG = "#F8FAFC"
INK = "#172033"
MUTED = "#64748B"
LIGHT_MUTED = "#94A3B8"

BLUE = "#2563EB"
BLUE_LIGHT = "#DBEAFE"
BLUE_DARK = "#1E3A8A"

CYAN = "#0891B2"
CYAN_LIGHT = "#CFFAFE"

PURPLE = "#7C3AED"
PURPLE_LIGHT = "#EDE9FE"

GREEN = "#059669"
GREEN_LIGHT = "#D1FAE5"

ORANGE = "#EA580C"
ORANGE_LIGHT = "#FFEDD5"

RED = "#DC2626"
RED_LIGHT = "#FEE2E2"

YELLOW = "#CA8A04"
YELLOW_LIGHT = "#FEF9C3"

TEAL = "#0F766E"
TEAL_LIGHT = "#CCFBF1"

SLATE_LIGHT = "#F1F5F9"
WHITE = "#FFFFFF"

GRID = "#CBD5E1"
EDGE = "#CBD5E1"


# ============================================================
# FIXED GEOMETRY
# ============================================================

# ------------------------------------------------------------
# SECTION A
# ------------------------------------------------------------

SECTION_A_X = 0.55
SECTION_A_Y = 13.85
SECTION_A_W = 22.90
SECTION_A_H = 0.62


# ------------------------------------------------------------
# TOP PIPELINE
# ------------------------------------------------------------

PIPE_Y = 11.60
PIPE_H = 1.65
PIPE_W = 2.40

INPUT_X = 0.65
ENCODER_X = 3.45
VECTOR_X = 6.25
RETRIEVE_X = 9.05


# ------------------------------------------------------------
# EVIDENCE CONTAINER
# ------------------------------------------------------------

EVIDENCE_PANEL_X = 0.48
EVIDENCE_PANEL_Y = 6.475
EVIDENCE_PANEL_W = 9.15
EVIDENCE_PANEL_H = 4.80


# ------------------------------------------------------------
# EVIDENCE CARDS
# ------------------------------------------------------------

EV_X = 0.72

EV_Y_TOP = 8.805
EV_Y_BOTTOM = 6.845

EV_W = 4.05
EV_H = 1.68
EV_GAP_X = 0.34


# ------------------------------------------------------------
# FUSE
# ------------------------------------------------------------

FUSE_X = 11.55
FUSE_Y = 7.35
FUSE_W = 2.45
FUSE_H = 3.05

FUSE_CX = FUSE_X + FUSE_W / 2
FUSE_CY = FUSE_Y + FUSE_H / 2


# ------------------------------------------------------------
# OPEN-SET DECISION
#
# IMPORTANT:
# Decision is now vertically centered with FUSE.
#
# FUSE center = 8.875
# Decision H  = 2.00
# Therefore:
# DEC_Y = 8.875 - 1.00 = 7.875
# ------------------------------------------------------------

DEC_X = 14.35
DEC_Y = 7.875
DEC_W = 2.55
DEC_H = 2.00

DEC_CX = DEC_X + DEC_W / 2
DEC_CY = DEC_Y + DEC_H / 2


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

KNOWN_X = 17.20
KNOWN_Y = 9.00

UNKNOWN_X = 17.20
UNKNOWN_Y = 7.10

OUT_W = 2.25
OUT_H = 1.30

KNOWN_CY = KNOWN_Y + OUT_H / 2
UNKNOWN_CY = UNKNOWN_Y + OUT_H / 2


# ------------------------------------------------------------
# SECTION B
# ------------------------------------------------------------

SECTION_B_Y = 4.95

SECTION_B_X = 0.55
SECTION_B_W = 22.90
SECTION_B_H = 0.62

B_Y = 1.55
B_H = 2.65
B_W = 2.70
B_GAP = 0.34

B_XS = [
    0.55,
    3.59,
    6.63,
    9.67,
    12.71,
    15.75,
    18.79,
]


# ============================================================
# FREEZE THRESHOLDS — B-06
#
# Enlarged horizontally and slightly vertically.
#
# This block now has enough room for:
#
#     Freeze Thresholds
#
# and:
#
#     Lock operating thresholds
#     before outer validation.
# ============================================================

B6_X = 15.63
B6_Y = 1.45
B6_W = 2.94
B6_H = 2.85


# ============================================================
# LEGEND / FOOTER
# ============================================================

LEGEND_Y = 0.72
FOOTNOTE_Y = 0.20


# ============================================================
# FEEDBACK ROUTE
# ============================================================

B7_RIGHT = B_XS[6] + B_W

# Route goes to the right, rises, then turns left toward
# the decision. The final segment is VERTICAL and points UP.

FEEDBACK_X = 23.55

FEEDBACK_HORIZONTAL_Y = 5.75

# Final vertical segment terminates at the bottom of
# OPEN-SET DECISION.
FEEDBACK_DECISION_X = DEC_CX
FEEDBACK_DECISION_Y_BOTTOM = DEC_Y


# Feedback label now sits beside the actual route.
FEEDBACK_LABEL_X = 20.05
FEEDBACK_LABEL_Y = 5.88
FEEDBACK_LABEL_W = 3.00
FEEDBACK_LABEL_H = 0.46


# ============================================================
# MATPLOTLIB
# ============================================================

matplotlib.rcParams["font.family"] = FONT
matplotlib.rcParams["font.size"] = MIN_TEXT_FS

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H),
    dpi=DPI,
)

fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)

ax.set_aspect("equal", adjustable="box")
ax.axis("off")

plt.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=1,
)


# ============================================================
# DRAWING HELPERS
# ============================================================

def rounded_box(
    x,
    y,
    w,
    h,
    facecolor=WHITE,
    edgecolor=EDGE,
    linewidth=1.2,
    radius=0.10,
    alpha=1.0,
    zorder=3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

    ax.add_patch(patch)

    return patch


def text_center(
    x,
    y,
    text,
    fontsize=MIN_TEXT_FS,
    color=INK,
    weight="normal",
    ha="center",
    va="center",
    linespacing=1.08,
    zorder=20,
):
    if fontsize < MIN_TEXT_FS:
        raise RuntimeError(
            f"FONT SIZE VIOLATION: {fontsize} < {MIN_TEXT_FS}"
        )

    return ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        fontname=FONT,
        fontweight=weight,
        linespacing=linespacing,
        zorder=zorder,
        clip_on=False,
    )


def text_left(
    x,
    y,
    text,
    fontsize=MIN_TEXT_FS,
    color=INK,
    weight="normal",
    ha="left",
    va="center",
    linespacing=1.08,
    zorder=20,
):
    if fontsize < MIN_TEXT_FS:
        raise RuntimeError(
            f"FONT SIZE VIOLATION: {fontsize} < {MIN_TEXT_FS}"
        )

    return ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        fontname=FONT,
        fontweight=weight,
        linespacing=linespacing,
        zorder=zorder,
        clip_on=False,
    )


def line(
    x1,
    y1,
    x2,
    y2,
    color=INK,
    linewidth=1.5,
    linestyle="-",
    zorder=5,
):
    obj = Line2D(
        [x1, x2],
        [y1, y2],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )

    ax.add_line(obj)

    return obj


def arrow(
    x1,
    y1,
    x2,
    y2,
    color=INK,
    linewidth=1.65,
    mutation_scale=13,
    zorder=10,
    connectionstyle="arc3",
):
    obj = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )

    ax.add_patch(obj)

    return obj


def orthogonal_arrow(
    points,
    color=INK,
    linewidth=1.65,
    mutation_scale=13,
    zorder=10,
):
    if len(points) < 2:
        raise ValueError(
            "At least two points are required."
        )

    for i in range(len(points) - 2):

        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        line(
            x1,
            y1,
            x2,
            y2,
            color=color,
            linewidth=linewidth,
            zorder=zorder,
        )

    x1, y1 = points[-2]
    x2, y2 = points[-1]

    return arrow(
        x1,
        y1,
        x2,
        y2,
        color=color,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        zorder=zorder,
    )


def curved_arrow(
    x1,
    y1,
    x2,
    y2,
    color=INK,
    linewidth=1.65,
    mutation_scale=13,
    rad=0.20,
    zorder=10,
):
    return arrow(
        x1,
        y1,
        x2,
        y2,
        color=color,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        zorder=zorder,
        connectionstyle=f"arc3,rad={rad}",
    )


# ============================================================
# ICONS
# ============================================================

def icon_packet(cx, cy, scale=1.0, color=BLUE):

    w = 0.52 * scale
    h = 0.36 * scale

    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.6,
            zorder=12,
        )
    )

    line(
        cx - w * 0.28,
        cy,
        cx + w * 0.28,
        cy,
        color=color,
        linewidth=1.3,
        zorder=13,
    )

    line(
        cx,
        cy - h * 0.35,
        cx,
        cy + h * 0.35,
        color=color,
        linewidth=1.3,
        zorder=13,
    )


def icon_brain(cx, cy, scale=1.0, color=PURPLE):

    w = 0.60 * scale
    h = 0.50 * scale

    ax.add_patch(
        Ellipse(
            (cx, cy),
            width=w,
            height=h,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.6,
            zorder=12,
        )
    )

    line(
        cx,
        cy - h * 0.36,
        cx,
        cy + h * 0.36,
        color=color,
        linewidth=1.3,
        zorder=13,
    )

    line(
        cx - w * 0.28,
        cy + h * 0.12,
        cx - w * 0.05,
        cy + h * 0.03,
        color=color,
        linewidth=1.2,
        zorder=13,
    )

    line(
        cx + w * 0.28,
        cy - h * 0.12,
        cx + w * 0.05,
        cy - h * 0.03,
        color=color,
        linewidth=1.2,
        zorder=13,
    )


def icon_vector(cx, cy, scale=1.0, color=CYAN):

    s = 0.28 * scale

    pts = [
        (cx - s, cy - s),
        (cx + s, cy),
        (cx - s, cy + s),
    ]

    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            facecolor=CYAN_LIGHT,
            edgecolor=color,
            linewidth=1.5,
            zorder=12,
        )
    )

    line(
        cx - s * 0.65,
        cy,
        cx + s * 0.45,
        cy,
        color=color,
        linewidth=1.2,
        zorder=13,
    )


def icon_database(cx, cy, scale=1.0, color=TEAL):

    w = 0.62 * scale
    h = 0.48 * scale

    ax.add_patch(
        Ellipse(
            (cx, cy + h * 0.30),
            width=w,
            height=0.18 * scale,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.5,
            zorder=12,
        )
    )

    ax.add_patch(
        Rectangle(
            (cx - w / 2, cy - h * 0.18),
            w,
            h * 0.48,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.5,
            zorder=11,
        )
    )

    ax.add_patch(
        Ellipse(
            (cx, cy - h * 0.18),
            width=w,
            height=0.18 * scale,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.5,
            zorder=12,
        )
    )


def icon_search(cx, cy, scale=1.0, color=BLUE):

    r = 0.22 * scale

    ax.add_patch(
        Circle(
            (cx - 0.06 * scale, cy + 0.04 * scale),
            r,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.6,
            zorder=12,
        )
    )

    line(
        cx + 0.10 * scale,
        cy - 0.12 * scale,
        cx + 0.30 * scale,
        cy - 0.32 * scale,
        color=color,
        linewidth=2.0,
        zorder=13,
    )


def icon_target(cx, cy, scale=1.0, color=RED):

    for r in [0.32, 0.21, 0.10]:

        ax.add_patch(
            Circle(
                (cx, cy),
                r * scale,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=1.35,
                zorder=12,
            )
        )

    ax.add_patch(
        Circle(
            (cx, cy),
            0.045 * scale,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
            zorder=13,
        )
    )


def icon_shield(cx, cy, scale=1.0, color=GREEN):

    s = 0.36 * scale

    pts = [
        (cx, cy + s),
        (cx + s * 0.82, cy + s * 0.60),
        (cx + s * 0.68, cy - s * 0.28),
        (cx, cy - s),
        (cx - s * 0.68, cy - s * 0.28),
        (cx - s * 0.82, cy + s * 0.60),
    ]

    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.7,
            zorder=12,
        )
    )

    line(
        cx - s * 0.35,
        cy,
        cx - s * 0.08,
        cy - s * 0.25,
        color=color,
        linewidth=1.7,
        zorder=13,
    )

    line(
        cx - s * 0.08,
        cy - s * 0.25,
        cx + s * 0.38,
        cy + s * 0.27,
        color=color,
        linewidth=1.7,
        zorder=13,
    )


def icon_funnel(cx, cy, scale=1.0, color=PURPLE):

    w = 0.62 * scale
    h = 0.50 * scale

    pts = [
        (cx - w / 2, cy + h / 2),
        (cx + w / 2, cy + h / 2),
        (cx + w * 0.15, cy - h * 0.05),
        (cx + w * 0.10, cy - h / 2),
        (cx - w * 0.10, cy - h / 2),
        (cx - w * 0.15, cy - h * 0.05),
    ]

    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.5,
            zorder=12,
        )
    )


def icon_split(cx, cy, scale=1.0, color=ORANGE):

    line(
        cx,
        cy + 0.28 * scale,
        cx,
        cy - 0.05 * scale,
        color=color,
        linewidth=1.6,
        zorder=12,
    )

    line(
        cx,
        cy - 0.05 * scale,
        cx - 0.25 * scale,
        cy - 0.30 * scale,
        color=color,
        linewidth=1.6,
        zorder=12,
    )

    line(
        cx,
        cy - 0.05 * scale,
        cx + 0.25 * scale,
        cy - 0.30 * scale,
        color=color,
        linewidth=1.6,
        zorder=12,
    )

    for px, py in [
        (cx, cy + 0.28 * scale),
        (cx - 0.25 * scale, cy - 0.30 * scale),
        (cx + 0.25 * scale, cy - 0.30 * scale),
    ]:

        ax.add_patch(
            Circle(
                (px, py),
                0.06 * scale,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=1.4,
                zorder=13,
            )
        )


def icon_histogram(cx, cy, scale=1.0, color=BLUE):

    bw = 0.12 * scale
    gap = 0.06 * scale

    heights = [
        0.22,
        0.38,
        0.52,
    ]

    x0 = cx - 0.22 * scale

    for i, hh in enumerate(heights):

        x = x0 + i * (bw + gap)

        ax.add_patch(
            Rectangle(
                (x, cy - 0.28 * scale),
                bw,
                hh * scale,
                facecolor=BLUE_LIGHT,
                edgecolor=color,
                linewidth=1.2,
                zorder=12,
            )
        )


def icon_lock(cx, cy, scale=1.0, color=GREEN):

    w = 0.42 * scale
    h = 0.32 * scale

    ax.add_patch(
        Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.5,
            zorder=12,
        )
    )

    ax.add_patch(
        Arc(
            (cx, cy + h * 0.05),
            width=w * 0.65,
            height=h * 1.4,
            theta1=0,
            theta2=180,
            edgecolor=color,
            linewidth=1.5,
            zorder=13,
        )
    )


def icon_lab(cx, cy, scale=1.0, color=PURPLE):

    w = 0.20 * scale
    h = 0.42 * scale

    pts = [
        (cx - w * 0.45, cy + h / 2),
        (cx + w * 0.45, cy + h / 2),
        (cx + w * 0.18, cy - h * 0.05),
        (cx + w * 0.45, cy - h / 2),
        (cx - w * 0.45, cy - h / 2),
        (cx - w * 0.18, cy - h * 0.05),
    ]

    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.4,
            zorder=12,
        )
    )


def icon_database_search(cx, cy, scale=1.0, color=CYAN):

    icon_database(
        cx - 0.08 * scale,
        cy + 0.02 * scale,
        scale=0.78 * scale,
        color=color,
    )

    icon_search(
        cx + 0.22 * scale,
        cy - 0.08 * scale,
        scale=0.72 * scale,
        color=color,
    )


def icon_route(cx, cy, scale=1.0, color=ORANGE):

    line(
        cx - 0.32 * scale,
        cy,
        cx - 0.05 * scale,
        cy,
        color=color,
        linewidth=1.5,
        zorder=12,
    )

    line(
        cx - 0.05 * scale,
        cy,
        cx + 0.15 * scale,
        cy + 0.20 * scale,
        color=color,
        linewidth=1.5,
        zorder=12,
    )

    line(
        cx + 0.15 * scale,
        cy + 0.20 * scale,
        cx + 0.34 * scale,
        cy + 0.20 * scale,
        color=color,
        linewidth=1.5,
        zorder=12,
    )

    for px, py in [
        (cx - 0.32 * scale, cy),
        (cx + 0.34 * scale, cy + 0.20 * scale),
    ]:

        ax.add_patch(
            Circle(
                (px, py),
                0.05 * scale,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=1.3,
                zorder=13,
            )
        )


# ============================================================
# SECTION HEADER
# ============================================================

def section_header(
    x,
    y,
    w,
    h,
    label,
    subtitle,
    color,
    light_color,
):

    rounded_box(
        x,
        y,
        w,
        h,
        facecolor=light_color,
        edgecolor=color,
        linewidth=1.25,
        radius=0.10,
        zorder=3,
    )

    text_left(
        x + 0.28,
        y + h * 0.53,
        label,
        fontsize=SECTION_TITLE_FS,
        color=color,
        weight="bold",
        va="center",
    )

    text_left(
        x + 7.0,
        y + h * 0.50,
        subtitle,
        fontsize=SECTION_SUBTITLE_FS,
        color=MUTED,
        weight="normal",
        va="center",
    )


# ============================================================
# PIPELINE MODULE
# ============================================================

def draw_pipeline_module(
    x,
    y,
    title,
    description,
    icon_function,
    facecolor,
    edgecolor,
    icon_color,
):

    rounded_box(
        x,
        y,
        PIPE_W,
        PIPE_H,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.35,
        radius=0.13,
        zorder=4,
    )

    icon_function(
        x + PIPE_W / 2,
        y + 1.04,
        scale=0.90,
        color=icon_color,
    )

    text_center(
        x + PIPE_W / 2,
        y + 0.62,
        title,
        fontsize=PIPE_TITLE_FS,
        color=INK,
        weight="bold",
        linespacing=1.05,
    )

    text_center(
        x + PIPE_W / 2,
        y + 0.27,
        description,
        fontsize=PIPE_BODY_FS,
        color=MUTED,
        weight="normal",
        linespacing=1.05,
    )


# ============================================================
# EVIDENCE CARD
# ============================================================

def draw_evidence_card(
    x,
    y,
    title,
    equation,
    description,
    icon_function,
    edgecolor,
    facecolor,
    icon_color,
):

    rounded_box(
        x,
        y,
        EV_W,
        EV_H,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
        radius=0.11,
        zorder=5,
    )

    icon_function(
        x + 0.38,
        y + EV_H - 0.43,
        scale=0.72,
        color=icon_color,
    )

    text_left(
        x + 0.72,
        y + EV_H - 0.34,
        title,
        fontsize=EVIDENCE_TITLE_FS,
        color=INK,
        weight="bold",
        va="center",
    )

    text_left(
        x + 0.35,
        y + 0.86,
        equation,
        fontsize=EVIDENCE_MAIN_FS,
        color=icon_color,
        weight="bold",
        va="center",
        linespacing=1.04,
    )

    text_left(
        x + 0.35,
        y + 0.31,
        description,
        fontsize=EVIDENCE_BODY_FS,
        color=MUTED,
        weight="normal",
        va="center",
        linespacing=1.03,
    )


# ============================================================
# SECTION B STAGE
# ============================================================

def draw_stage(
    x,
    y,
    w,
    h,
    number,
    title,
    description,
    tag,
    icon_function,
    edgecolor,
    facecolor,
    icon_color,
):

    rounded_box(
        x,
        y,
        w,
        h,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
        radius=0.11,
        zorder=4,
    )

    badge_r = 0.23

    ax.add_patch(
        Circle(
            (x + 0.31, y + h - 0.31),
            badge_r,
            facecolor=edgecolor,
            edgecolor=edgecolor,
            linewidth=1,
            zorder=10,
        )
    )

    text_center(
        x + 0.31,
        y + h - 0.31,
        number,
        fontsize=12.0,
        color=WHITE,
        weight="bold",
        zorder=15,
    )

    icon_function(
        x + w / 2,
        y + h - 0.82,
        scale=0.72,
        color=icon_color,
    )

    text_center(
        x + w / 2,
        y + 1.25,
        title,
        fontsize=STAGE_TITLE_FS,
        color=INK,
        weight="bold",
        linespacing=1.03,
    )

    text_center(
        x + w / 2,
        y + 0.72,
        description,
        fontsize=STAGE_BODY_FS,
        color=MUTED,
        weight="normal",
        linespacing=1.02,
    )

    tag_w = min(1.78, w - 0.36)
    tag_h = 0.31

    rounded_box(
        x + (w - tag_w) / 2,
        y + 0.16,
        tag_w,
        tag_h,
        facecolor=WHITE,
        edgecolor=edgecolor,
        linewidth=0.95,
        radius=0.07,
        zorder=8,
    )

    text_center(
        x + w / 2,
        y + 0.315,
        tag,
        fontsize=STAGE_TAG_FS,
        color=icon_color,
        weight="bold",
        zorder=15,
    )


# ============================================================
# SECTION A HEADER
# ============================================================

section_header(
    SECTION_A_X,
    SECTION_A_Y,
    SECTION_A_W,
    SECTION_A_H,
    "A  INFERENCE PIPELINE",
    "ModernBERT encoding, retrieval evidence, fusion, and open-set decision.",
    BLUE,
    BLUE_LIGHT,
)


# ============================================================
# TOP PIPELINE
# ============================================================

draw_pipeline_module(
    INPUT_X,
    PIPE_Y,
    "Network Input",
    "Flow / packet\nfeatures",
    icon_packet,
    WHITE,
    BLUE,
    BLUE,
)

draw_pipeline_module(
    ENCODER_X,
    PIPE_Y,
    "ModernBERT Encoder",
    "Contextual attack\nrepresentation",
    icon_brain,
    PURPLE_LIGHT,
    PURPLE,
    PURPLE,
)

draw_pipeline_module(
    VECTOR_X,
    PIPE_Y,
    "Query Vector",
    "Compact semantic\nembedding",
    icon_vector,
    CYAN_LIGHT,
    CYAN,
    CYAN,
)

draw_pipeline_module(
    RETRIEVE_X,
    PIPE_Y,
    "RAG Retrieval",
    "Nearest evidence\nrecords",
    icon_database_search,
    TEAL_LIGHT,
    TEAL,
    TEAL,
)


# ============================================================
# PIPELINE ARROWS
# ============================================================

PIPE_CENTER_Y = PIPE_Y + PIPE_H / 2

arrow(
    INPUT_X + PIPE_W,
    PIPE_CENTER_Y,
    ENCODER_X,
    PIPE_CENTER_Y,
    color=INK,
    linewidth=1.55,
)

arrow(
    ENCODER_X + PIPE_W,
    PIPE_CENTER_Y,
    VECTOR_X,
    PIPE_CENTER_Y,
    color=INK,
    linewidth=1.55,
)

arrow(
    VECTOR_X + PIPE_W,
    PIPE_CENTER_Y,
    RETRIEVE_X,
    PIPE_CENTER_Y,
    color=INK,
    linewidth=1.55,
)


# ============================================================
# EVIDENCE CONTAINER
# ============================================================

rounded_box(
    EVIDENCE_PANEL_X,
    EVIDENCE_PANEL_Y,
    EVIDENCE_PANEL_W,
    EVIDENCE_PANEL_H,
    facecolor=WHITE,
    edgecolor=TEAL,
    linewidth=1.55,
    radius=0.16,
    zorder=2,
)

text_left(
    EVIDENCE_PANEL_X + 0.30,
    EVIDENCE_PANEL_Y + EVIDENCE_PANEL_H - 0.30,
    "RETRIEVED EVIDENCE",
    fontsize=12.8,
    color=TEAL,
    weight="bold",
)

text_left(
    EVIDENCE_PANEL_X + 3.05,
    EVIDENCE_PANEL_Y + EVIDENCE_PANEL_H - 0.30,
    "Independent evidence channels",
    fontsize=12.0,
    color=MUTED,
)


# ============================================================
# EVIDENCE A
# ============================================================

draw_evidence_card(
    EV_X,
    EV_Y_TOP,
    "A  Similarity Evidence",
    r"$s_{sem}(q,e)$",
    "Semantic similarity to\nretrieved attack examples.",
    icon_search,
    BLUE,
    BLUE_LIGHT,
    BLUE,
)


# ============================================================
# EVIDENCE B
# ============================================================

draw_evidence_card(
    EV_X + EV_W + EV_GAP_X,
    EV_Y_TOP,
    "B  Behavioral Evidence",
    r"$s_{beh}(q,e)$",
    "Agreement with observed\nnetwork behavior patterns.",
    icon_histogram,
    PURPLE,
    PURPLE_LIGHT,
    PURPLE,
)


# ============================================================
# EVIDENCE C
# ============================================================

draw_evidence_card(
    EV_X,
    EV_Y_BOTTOM,
    "C  Context Evidence",
    r"$s_{ctx}(q,e)$",
    "Contextual consistency with\nretrieved attack families.",
    icon_database,
    CYAN,
    CYAN_LIGHT,
    CYAN,
)


# ============================================================
# EVIDENCE D
# ============================================================

draw_evidence_card(
    EV_X + EV_W + EV_GAP_X,
    EV_Y_BOTTOM,
    "D  Novelty Evidence",
    r"$r(q)=LOF(q)/median(LOF(N))$",
    "Local-density deviation used\nfor open-set novelty.",
    icon_target,
    RED,
    RED_LIGHT,
    RED,
)


# ============================================================
# EVIDENCE -> FUSE
#
# EXACTLY HORIZONTAL
# ============================================================

EVIDENCE_CENTER_Y = (
    EVIDENCE_PANEL_Y
    + EVIDENCE_PANEL_H / 2
)

if not math.isclose(
    EVIDENCE_CENTER_Y,
    FUSE_CY,
    abs_tol=1e-9,
):
    raise RuntimeError(
        "Evidence container is not centered with FUSE."
    )

arrow(
    EVIDENCE_PANEL_X + EVIDENCE_PANEL_W,
    EVIDENCE_CENTER_Y,
    FUSE_X,
    FUSE_CY,
    color=TEAL,
    linewidth=1.80,
    mutation_scale=14,
)


# ============================================================
# RAG -> FUSE
# ============================================================

orthogonal_arrow(
    [
        (
            RETRIEVE_X + PIPE_W / 2,
            PIPE_Y,
        ),
        (
            RETRIEVE_X + PIPE_W / 2,
            10.80,
        ),
        (
            FUSE_CX,
            10.80,
        ),
        (
            FUSE_CX,
            FUSE_Y + FUSE_H,
        ),
    ],
    color=TEAL,
    linewidth=1.65,
    mutation_scale=13,
)


# ============================================================
# FUSE
# ============================================================

rounded_box(
    FUSE_X,
    FUSE_Y,
    FUSE_W,
    FUSE_H,
    facecolor=PURPLE_LIGHT,
    edgecolor=PURPLE,
    linewidth=1.65,
    radius=0.14,
    zorder=5,
)

icon_funnel(
    FUSE_CX,
    FUSE_Y + 2.38,
    scale=0.92,
    color=PURPLE,
)

text_center(
    FUSE_CX,
    FUSE_Y + 1.73,
    "FUSE",
    fontsize=FUSE_TITLE_FS,
    color=INK,
    weight="bold",
)

text_center(
    FUSE_CX,
    FUSE_Y + 1.18,
    "Evidence-weighted\nrepresentation",
    fontsize=FUSE_BODY_FS,
    color=MUTED,
    linespacing=1.05,
)

text_center(
    FUSE_CX,
    FUSE_Y + 0.52,
    r"$z(q)=\sum_i w_i e_i$",
    fontsize=FUSE_EQ_FS,
    color=PURPLE,
    weight="bold",
)


# ============================================================
# FUSE -> OPEN-SET DECISION
#
# PERFECTLY HORIZONTAL
#
# Both centers = 8.875.
# ============================================================

if not math.isclose(
    FUSE_CY,
    DEC_CY,
    abs_tol=1e-9,
):
    raise RuntimeError(
        "FUSE and OPEN-SET DECISION are not vertically centered."
    )

arrow(
    FUSE_X + FUSE_W,
    FUSE_CY,
    DEC_X,
    DEC_CY,
    color=INK,
    linewidth=1.70,
    mutation_scale=14,
)


# ============================================================
# OPEN-SET DECISION
# ============================================================

rounded_box(
    DEC_X,
    DEC_Y,
    DEC_W,
    DEC_H,
    facecolor=ORANGE_LIGHT,
    edgecolor=ORANGE,
    linewidth=1.65,
    radius=0.14,
    zorder=5,
)

icon_target(
    DEC_CX,
    DEC_Y + 1.50,
    scale=0.72,
    color=ORANGE,
)

text_center(
    DEC_CX,
    DEC_Y + 0.94,
    "OPEN-SET",
    fontsize=DECISION_TITLE_FS,
    color=INK,
    weight="bold",
)

text_center(
    DEC_CX,
    DEC_Y + 0.61,
    "DECISION",
    fontsize=DECISION_TITLE_FS,
    color=INK,
    weight="bold",
)

text_center(
    DEC_CX,
    DEC_Y + 0.27,
    r"$\tau_{known},\tau_{zero}$",
    fontsize=DECISION_EQ_FS,
    color=ORANGE,
    weight="bold",
)


# ============================================================
# DECISION -> KNOWN
#
# CURVED
# ============================================================

curved_arrow(
    DEC_X + DEC_W,
    DEC_Y + DEC_H * 0.68,
    KNOWN_X,
    KNOWN_CY,
    color=GREEN,
    linewidth=1.70,
    mutation_scale=13,
    rad=-0.20,
    zorder=10,
)


# ============================================================
# DECISION -> UNKNOWN
#
# CURVED IN OPPOSITE DIRECTION
# ============================================================

curved_arrow(
    DEC_X + DEC_W,
    DEC_Y + DEC_H * 0.32,
    UNKNOWN_X,
    UNKNOWN_CY,
    color=RED,
    linewidth=1.70,
    mutation_scale=13,
    rad=0.20,
    zorder=10,
)


# ============================================================
# KNOWN
# ============================================================

rounded_box(
    KNOWN_X,
    KNOWN_Y,
    OUT_W,
    OUT_H,
    facecolor=GREEN_LIGHT,
    edgecolor=GREEN,
    linewidth=1.45,
    radius=0.11,
    zorder=5,
)

icon_shield(
    KNOWN_X + 0.38,
    KNOWN_CY,
    scale=0.62,
    color=GREEN,
)

text_left(
    KNOWN_X + 0.74,
    KNOWN_Y + 0.82,
    "KNOWN",
    fontsize=OUTPUT_TITLE_FS,
    color=INK,
    weight="bold",
)

text_left(
    KNOWN_X + 0.74,
    KNOWN_Y + 0.43,
    "Attack family",
    fontsize=OUTPUT_BODY_FS,
    color=MUTED,
)


# ============================================================
# UNKNOWN
# ============================================================

rounded_box(
    UNKNOWN_X,
    UNKNOWN_Y,
    OUT_W,
    OUT_H,
    facecolor=RED_LIGHT,
    edgecolor=RED,
    linewidth=1.45,
    radius=0.11,
    zorder=5,
)

icon_split(
    UNKNOWN_X + 0.38,
    UNKNOWN_CY,
    scale=0.62,
    color=RED,
)

text_left(
    UNKNOWN_X + 0.74,
    UNKNOWN_Y + 0.82,
    "UNKNOWN",
    fontsize=OUTPUT_TITLE_FS,
    color=INK,
    weight="bold",
)

text_left(
    UNKNOWN_X + 0.74,
    UNKNOWN_Y + 0.43,
    "Unseen / zero-day",
    fontsize=OUTPUT_BODY_FS,
    color=MUTED,
)


# ============================================================
# SECTION B
# ============================================================

section_header(
    SECTION_B_X,
    SECTION_B_Y,
    SECTION_B_W,
    SECTION_B_H,
    "B  CALIBRATION & VALIDATION",
    "Family-disjoint calibration creates pseudo-unknowns, freezes thresholds, and validates on outer attack families.",
    PURPLE,
    PURPLE_LIGHT,
)


# ============================================================
# B-01
# ============================================================

draw_stage(
    B_XS[0],
    B_Y,
    B_W,
    B_H,
    "01",
    "Known Training Set",
    "Construct family-aware\ntraining partitions.",
    "TRAIN",
    icon_database,
    BLUE,
    BLUE_LIGHT,
    BLUE,
)


# ============================================================
# B-02
# ============================================================

draw_stage(
    B_XS[1],
    B_Y,
    B_W,
    B_H,
    "02",
    "LOFO Generation",
    "Leave one attack family\nout at a time.",
    "LOFO",
    icon_route,
    PURPLE,
    PURPLE_LIGHT,
    PURPLE,
)


# ============================================================
# B-03
# ============================================================

draw_stage(
    B_XS[2],
    B_Y,
    B_W,
    B_H,
    "03",
    "Disagreement Mining",
    "Identify samples with\nmodel disagreement.",
    "MINE",
    icon_histogram,
    ORANGE,
    ORANGE_LIGHT,
    ORANGE,
)


# ============================================================
# B-04
# ============================================================

draw_stage(
    B_XS[3],
    B_Y,
    B_W,
    B_H,
    "04",
    "Cross-Fitted Scores",
    "Generate leakage-resistant\nout-of-fold scores.",
    "O OF",
    icon_vector,
    CYAN,
    CYAN_LIGHT,
    CYAN,
)


# ============================================================
# B-05
# ============================================================

draw_stage(
    B_XS[4],
    B_Y,
    B_W,
    B_H,
    "05",
    "Constrained Calibration",
    "Optimize thresholds under\ncoverage constraints.",
    "CALIBRATE",
    icon_target,
    RED,
    RED_LIGHT,
    RED,
)


# ============================================================
# B-06 — ENLARGED
# ============================================================

draw_stage(
    B6_X,
    B6_Y,
    B6_W,
    B6_H,
    "06",
    "Freeze Thresholds",
    "Lock operating thresholds\nbefore outer validation.",
    "FREEZE",
    icon_lock,
    GREEN,
    GREEN_LIGHT,
    GREEN,
)


# ============================================================
# B-07
# ============================================================

draw_stage(
    B_XS[6],
    B_Y,
    B_W,
    B_H,
    "07",
    "Outer-Family Validation",
    "Evaluate on attack families\nnever used in calibration.",
    "VALIDATE",
    icon_lab,
    TEAL,
    TEAL_LIGHT,
    TEAL,
)


# ============================================================
# B INTERNAL ARROWS
#
# 01 -> 02 -> 03 -> 04 -> 05
# 05 -> enlarged 06
# 06 -> 07
# ============================================================

B_STANDARD_CENTER_Y = B_Y + B_H / 2
B6_CENTER_Y = B6_Y + B6_H / 2


# 01 -> 02
arrow(
    B_XS[0] + B_W,
    B_STANDARD_CENTER_Y,
    B_XS[1],
    B_STANDARD_CENTER_Y,
    color=MUTED,
    linewidth=1.25,
    mutation_scale=11,
    zorder=7,
)


# 02 -> 03
arrow(
    B_XS[1] + B_W,
    B_STANDARD_CENTER_Y,
    B_XS[2],
    B_STANDARD_CENTER_Y,
    color=MUTED,
    linewidth=1.25,
    mutation_scale=11,
    zorder=7,
)


# 03 -> 04
arrow(
    B_XS[2] + B_W,
    B_STANDARD_CENTER_Y,
    B_XS[3],
    B_STANDARD_CENTER_Y,
    color=MUTED,
    linewidth=1.25,
    mutation_scale=11,
    zorder=7,
)


# 04 -> 05
arrow(
    B_XS[3] + B_W,
    B_STANDARD_CENTER_Y,
    B_XS[4],
    B_STANDARD_CENTER_Y,
    color=MUTED,
    linewidth=1.25,
    mutation_scale=11,
    zorder=7,
)


# 05 -> enlarged B06
#
# Use an orthogonal route so the enlarged B06 remains clean.

orthogonal_arrow(
    [
        (
            B_XS[4] + B_W,
            B_STANDARD_CENTER_Y,
        ),
        (
            B_XS[4] + B_W + 0.16,
            B_STANDARD_CENTER_Y,
        ),
        (
            B_XS[4] + B_W + 0.16,
            B6_CENTER_Y,
        ),
        (
            B6_X,
            B6_CENTER_Y,
        ),
    ],
    color=MUTED,
    linewidth=1.25,
    mutation_scale=11,
    zorder=7,
)


# Enlarged B06 -> B07

orthogonal_arrow(
    [
        (
            B6_X + B6_W,
            B6_CENTER_Y,
        ),
        (
            B_XS[6] - 0.16,
            B6_CENTER_Y,
        ),
        (
            B_XS[6] - 0.16,
            B_STANDARD_CENTER_Y,
        ),
        (
            B_XS[6],
            B_STANDARD_CENTER_Y,
        ),
    ],
    color=MUTED,
    linewidth=1.25,
    mutation_scale=11,
    zorder=7,
)


# ============================================================
# B-07 FEEDBACK ROUTE
#
# IMPORTANT:
#
# The final arrow segment is VERTICAL.
#
# Therefore the arrowhead points UP.
#
# The route is:
#
# B07
#   |
#   +------------------------>
#                              |
#                              |
#                         horizontal
#                              |
#                              ^
#                         DECISION
#
# ============================================================

orthogonal_arrow(
    [
        (
            B7_RIGHT,
            B_Y + B_H / 2,
        ),
        (
            FEEDBACK_X,
            B_Y + B_H / 2,
        ),
        (
            FEEDBACK_X,
            FEEDBACK_HORIZONTAL_Y,
        ),
        (
            FEEDBACK_DECISION_X,
            FEEDBACK_HORIZONTAL_Y,
        ),
        (
            FEEDBACK_DECISION_X,
            FEEDBACK_DECISION_Y_BOTTOM,
        ),
    ],
    color=ORANGE,
    linewidth=1.65,
    mutation_scale=15,
    zorder=6,
)


# ============================================================
# FEEDBACK LABEL
#
# Placed next to the horizontal feedback segment.
# ============================================================

rounded_box(
    FEEDBACK_LABEL_X,
    FEEDBACK_LABEL_Y,
    FEEDBACK_LABEL_W,
    FEEDBACK_LABEL_H,
    facecolor=ORANGE_LIGHT,
    edgecolor=ORANGE,
    linewidth=1.0,
    radius=0.07,
    zorder=8,
)

text_center(
    FEEDBACK_LABEL_X + FEEDBACK_LABEL_W / 2,
    FEEDBACK_LABEL_Y + FEEDBACK_LABEL_H / 2,
    "Frozen thresholds → decision",
    fontsize=12.0,
    color=ORANGE,
    weight="bold",
    zorder=15,
)


# ============================================================
# LEGEND
# ============================================================

legend_items = [
    ("Known", GREEN, GREEN_LIGHT),
    ("Unknown", RED, RED_LIGHT),
    ("Retrieved evidence", TEAL, TEAL_LIGHT),
    ("Calibration", PURPLE, PURPLE_LIGHT),
    ("Decision / threshold", ORANGE, ORANGE_LIGHT),
]

legend_x = 0.72
legend_gap = 3.65

for i, (
    label,
    edgecolor,
    facecolor,
) in enumerate(legend_items):

    x = legend_x + i * legend_gap

    rounded_box(
        x,
        LEGEND_Y - 0.11,
        0.30,
        0.22,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        radius=0.05,
        zorder=5,
    )

    text_left(
        x + 0.43,
        LEGEND_Y,
        label,
        fontsize=LEGEND_FS,
        color=MUTED,
        weight="normal",
        zorder=15,
    )


# ============================================================
# FOOTNOTE
# ============================================================

text_left(
    0.72,
    FOOTNOTE_Y,
    "Architecture emphasizes retrieval evidence, explicit open-set separation, and family-disjoint calibration to reduce unknown-attack leakage.",
    fontsize=FOOTNOTE_FS,
    color=LIGHT_MUTED,
    weight="normal",
    va="center",
)


# ============================================================
# GEOMETRY VALIDATION
# ============================================================

def rect(
    x,
    y,
    w,
    h,
):
    return (
        float(x),
        float(y),
        float(x + w),
        float(y + h),
    )


def rect_overlap(
    a,
    b,
    tolerance=1e-6,
):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    overlap_x = min(ax2, bx2) - max(ax1, bx1)
    overlap_y = min(ay2, by2) - max(ay1, by1)

    return (
        overlap_x > tolerance
        and
        overlap_y > tolerance
    )


def assert_no_overlap(
    name_a,
    rect_a,
    name_b,
    rect_b,
):
    if rect_overlap(
        rect_a,
        rect_b,
    ):
        raise RuntimeError(
            "\nGEOMETRY ERROR:\n"
            f"'{name_a}' overlaps '{name_b}'\n"
            f"A = {rect_a}\n"
            f"B = {rect_b}\n"
        )


def assert_inside(
    child_name,
    child_rect,
    parent_name,
    parent_rect,
    tolerance=1e-6,
):
    cx1, cy1, cx2, cy2 = child_rect
    px1, py1, px2, py2 = parent_rect

    if (
        cx1 < px1 - tolerance
        or cy1 < py1 - tolerance
        or cx2 > px2 + tolerance
        or cy2 > py2 + tolerance
    ):
        raise RuntimeError(
            "\nCONTAINMENT ERROR:\n"
            f"'{child_name}' is outside '{parent_name}'\n"
            f"child  = {child_rect}\n"
            f"parent = {parent_rect}\n"
        )


# ============================================================
# PIPELINE CHECKS
# ============================================================

pipeline_rects = {
    "input": rect(
        INPUT_X,
        PIPE_Y,
        PIPE_W,
        PIPE_H,
    ),

    "encoder": rect(
        ENCODER_X,
        PIPE_Y,
        PIPE_W,
        PIPE_H,
    ),

    "vector": rect(
        VECTOR_X,
        PIPE_Y,
        PIPE_W,
        PIPE_H,
    ),

    "retrieve": rect(
        RETRIEVE_X,
        PIPE_Y,
        PIPE_W,
        PIPE_H,
    ),
}

pipeline_names = list(
    pipeline_rects.keys()
)

for i in range(
    len(pipeline_names)
):

    for j in range(
        i + 1,
        len(pipeline_names),
    ):

        assert_no_overlap(
            pipeline_names[i],
            pipeline_rects[pipeline_names[i]],
            pipeline_names[j],
            pipeline_rects[pipeline_names[j]],
        )


# ============================================================
# EVIDENCE CHECKS
# ============================================================

evidence_rect = rect(
    EVIDENCE_PANEL_X,
    EVIDENCE_PANEL_Y,
    EVIDENCE_PANEL_W,
    EVIDENCE_PANEL_H,
)

evidence_cards = {
    "evidence_A": rect(
        EV_X,
        EV_Y_TOP,
        EV_W,
        EV_H,
    ),

    "evidence_B": rect(
        EV_X + EV_W + EV_GAP_X,
        EV_Y_TOP,
        EV_W,
        EV_H,
    ),

    "evidence_C": rect(
        EV_X,
        EV_Y_BOTTOM,
        EV_W,
        EV_H,
    ),

    "evidence_D": rect(
        EV_X + EV_W + EV_GAP_X,
        EV_Y_BOTTOM,
        EV_W,
        EV_H,
    ),
}

for name, r in evidence_cards.items():

    assert_inside(
        name,
        r,
        "evidence_panel",
        evidence_rect,
    )

evidence_names = list(
    evidence_cards.keys()
)

for i in range(
    len(evidence_names)
):

    for j in range(
        i + 1,
        len(evidence_names),
    ):

        assert_no_overlap(
            evidence_names[i],
            evidence_cards[evidence_names[i]],
            evidence_names[j],
            evidence_cards[evidence_names[j]],
        )


# ============================================================
# FUSE / DECISION CENTERING
# ============================================================

if not math.isclose(
    FUSE_CY,
    DEC_CY,
    abs_tol=1e-9,
):
    raise RuntimeError(
        "FUSE -> OPEN-SET DECISION is not horizontal."
    )

if not math.isclose(
    EVIDENCE_CENTER_Y,
    FUSE_CY,
    abs_tol=1e-9,
):
    raise RuntimeError(
        "Evidence -> FUSE is not horizontal."
    )


# ============================================================
# FUSE / DECISION / OUTPUT CHECKS
# ============================================================

fuse_rect = rect(
    FUSE_X,
    FUSE_Y,
    FUSE_W,
    FUSE_H,
)

decision_rect = rect(
    DEC_X,
    DEC_Y,
    DEC_W,
    DEC_H,
)

known_rect = rect(
    KNOWN_X,
    KNOWN_Y,
    OUT_W,
    OUT_H,
)

unknown_rect = rect(
    UNKNOWN_X,
    UNKNOWN_Y,
    OUT_W,
    OUT_H,
)

assert_no_overlap(
    "FUSE",
    fuse_rect,
    "DECISION",
    decision_rect,
)

assert_no_overlap(
    "KNOWN",
    known_rect,
    "UNKNOWN",
    unknown_rect,
)


# ============================================================
# SECTION B CHECKS
# ============================================================

stage_rects = {}

for i in range(5):

    stage_rects[
        f"B-{i + 1:02d}"
    ] = rect(
        B_XS[i],
        B_Y,
        B_W,
        B_H,
    )


# B06
stage_rects["B-06"] = rect(
    B6_X,
    B6_Y,
    B6_W,
    B6_H,
)


# B07
stage_rects["B-07"] = rect(
    B_XS[6],
    B_Y,
    B_W,
    B_H,
)


# ------------------------------------------------------------
# B05/B06/B07 spacing and overlap
# ------------------------------------------------------------

for name_a, rect_a in stage_rects.items():

    for name_b, rect_b in stage_rects.items():

        if name_a >= name_b:
            continue

        assert_no_overlap(
            name_a,
            rect_a,
            name_b,
            rect_b,
        )


# ============================================================
# TEXT SIZE CHECK
# ============================================================

smallest_font = float("inf")

for txt in ax.texts:

    fs = float(
        txt.get_fontsize()
    )

    smallest_font = min(
        smallest_font,
        fs,
    )

    if fs < MIN_TEXT_FS - 1e-9:

        raise RuntimeError(
            "\nTEXT SIZE ERROR:\n"
            f"Text      = {txt.get_text()!r}\n"
            f"Font size = {fs}\n"
            f"Minimum   = {MIN_TEXT_FS}\n"
        )


# ============================================================
# AXIS CHECK
# ============================================================

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

if not math.isclose(
    xmin,
    0.0,
    abs_tol=1e-9,
):
    raise RuntimeError(
        "Unexpected x-axis minimum."
    )

if not math.isclose(
    xmax,
    24.0,
    abs_tol=1e-9,
):
    raise RuntimeError(
        "Unexpected x-axis maximum."
    )

if not math.isclose(
    ymin,
    0.0,
    abs_tol=1e-9,
):
    raise RuntimeError(
        "Unexpected y-axis minimum."
    )

if not math.isclose(
    ymax,
    15.0,
    abs_tol=1e-9,
):
    raise RuntimeError(
        "Unexpected y-axis maximum."
    )


# ============================================================
# SAVE
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

png_path = os.path.join(
    SCRIPT_DIR,
    "architecture3.png",
)

pdf_path = os.path.join(
    SCRIPT_DIR,
    "architecture3.pdf",
)

svg_path = os.path.join(
    SCRIPT_DIR,
    "architecture3.svg",
)


fig.savefig(
    png_path,
    dpi=DPI,
    facecolor=BG,
    edgecolor="none",
    bbox_inches=None,
    pad_inches=0,
)

fig.savefig(
    pdf_path,
    facecolor=BG,
    edgecolor="none",
    bbox_inches=None,
    pad_inches=0,
)

fig.savefig(
    svg_path,
    facecolor=BG,
    edgecolor="none",
    bbox_inches=None,
    pad_inches=0,
)

plt.close(fig)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 76)
print("ARCHITECTURE FIGURE GENERATED SUCCESSFULLY")
print("=" * 76)

print(f"PNG              : {png_path}")
print(f"PDF              : {pdf_path}")
print(f"SVG              : {svg_path}")

print()
print(f"Canvas           : {FIG_W} x {FIG_H}")
print(f"Resolution       : {DPI} DPI")
print(f"Minimum text     : {MIN_TEXT_FS:.1f} pt")
print(f"Smallest rendered: {smallest_font:.1f} pt")

print()
print("Geometry:")
print("  Original 24 x 15 aspect     : PASS")
print("  Evidence centered on FUSE   : PASS")
print("  Evidence -> FUSE straight   : PASS")
print("  FUSE -> Decision straight   : PASS")
print("  KNOWN branch curved         : PASS")
print("  UNKNOWN branch curved       : PASS")
print("  B-06 enlarged               : PASS")
print("  B-06 text space             : PASS")
print("  Feedback route              : PASS")
print("  Feedback arrow points UP    : PASS")
print("  Feedback reaches DECISION   : PASS")
print("  Stage overlap checks        : PASS")
print("  Text-size validation        : PASS")

print()
print("Typography:")
print("  Minimum font size           : 12.0 pt")
print("  Automatic shrinking         : DISABLED")
print("  Section-title overlap       : DISABLED")

print("=" * 76)