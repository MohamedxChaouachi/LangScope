"""
theme.py — Central design tokens for the Live Translator UI.

Named constants for the color palette, spacing scale, and border-radius
values used to build the QSS stylesheet in ui.py. Extracted from the
previously inline hex/pixel values in ControlPanel._apply_styles() —
values are unchanged, only named.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────────────

COLOR_BG = "#1A1A2E"              # window / base background
COLOR_TEXT = "#E0E0E0"            # primary text
COLOR_WHITE = "#FFF"              # text on solid accent buttons

COLOR_ACCENT = "#4FC3F7"          # title, fps counter, combo hover border
COLOR_ACCENT_SOFT = "#90CAF9"     # section labels, info text, checkbox text
COLOR_SURFACE = "#16213E"         # info pill / combo box / dropdown backgrounds
COLOR_BADGE = "#80CBC4"           # badge label text
COLOR_STATUS = "#B0BEC5"          # status label text
COLOR_SEPARATOR = "#2A2A4A"       # QFrame separator line
COLOR_BORDER = "#2A2A6A"          # combo box / dropdown border

COLOR_PRIMARY = "#4CAF50"
COLOR_PRIMARY_HOVER = "#66BB6A"
COLOR_PRIMARY_PRESSED = "#388E3C"

COLOR_SECONDARY = "#1565C0"       # also used as dropdown selection background
COLOR_SECONDARY_HOVER = "#1976D2"
COLOR_SECONDARY_PRESSED = "#0D47A1"

COLOR_DANGER = "#E53935"
COLOR_DANGER_HOVER = "#EF5350"
COLOR_DANGER_PRESSED = "#B71C1C"

COLOR_DISABLED_BG = "#37474F"
COLOR_DISABLED_TEXT = "#78909C"

# ─────────────────────────────────────────────────────────────────────────────
# Spacing scale (px)
# ─────────────────────────────────────────────────────────────────────────────

SPACE_XS = 5    # info label vertical padding
SPACE_SM = 6    # combo padding-vertical / checkbox spacing
SPACE_MD = 8    # info label horizontal padding / fps padding-left
SPACE_LG = 9    # button padding-vertical
SPACE_XL = 10   # combo padding-horizontal
SPACE_XXL = 14  # button padding-horizontal / checkbox indicator size

# ─────────────────────────────────────────────────────────────────────────────
# Border radius (px)
# ─────────────────────────────────────────────────────────────────────────────

RADIUS_SM = 6   # info label / combo box
RADIUS_MD = 8   # buttons
