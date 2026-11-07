"""
theme.py — Central design tokens for the LangScope / Live Translator UI.

Light, modern desktop theme inspired by consumer apps like NordVPN:
a clean white/soft-gray canvas, one strong brand-blue call-to-action,
rounded pill-style buttons, soft neutral borders, and a friendly UI sans.

Named constants for the color palette, typography, spacing scale, and
border-radius values consumed by the QSS stylesheet in ui.py.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Typography
# ─────────────────────────────────────────────────────────────────────────────

# Prefer the modern Windows variable UI font, then graceful fallbacks.
FONT_FAMILY = (
    "'Segoe UI Variable Display', 'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif"
)
FONT_FAMILY_MONO = "'Cascadia Code', 'Consolas', 'SF Mono', monospace"

FONT_SIZE_BASE = 13  # body / default
FONT_SIZE_SM = 12  # secondary text, checkboxes
FONT_SIZE_XS = 11  # badges, stats, captions
FONT_SIZE_LG = 14  # buttons
FONT_SIZE_TITLE = 20  # brand title

# ─────────────────────────────────────────────────────────────────────────────
# Color palette  (LIGHT MODE)
# ─────────────────────────────────────────────────────────────────────────────

COLOR_BG = "#F5F7FB"  # window / base canvas (soft cool gray)
COLOR_TEXT = "#1B1F2A"  # primary text (near-black slate)
COLOR_WHITE = "#FFFFFF"  # text on solid accent buttons / card surfaces

COLOR_ACCENT = "#4687FF"  # brand blue — title, globe, active accents
COLOR_ACCENT_SOFT = "#5B6478"  # muted slate — section labels, secondary text
COLOR_SURFACE = "#FFFFFF"  # cards / combo / info pill backgrounds
COLOR_SURFACE_ALT = "#EEF2F9"  # subtle raised fills (hover, stats strip)
COLOR_BADGE = "#2E8B77"  # badge label text (teal-green)
COLOR_STATUS = "#6B7385"  # status label text (muted slate)
COLOR_SEPARATOR = "#E4E8F0"  # separator line
COLOR_BORDER = "#DDE3EE"  # combo box / card border

# Primary call-to-action = brand blue (NordVPN-style prominent CTA)
COLOR_PRIMARY = "#4687FF"
COLOR_PRIMARY_HOVER = "#5E97FF"
COLOR_PRIMARY_PRESSED = "#2E5BFF"

# Secondary = neutral outline button (white fill, blue ink)
COLOR_SECONDARY = "#FFFFFF"
COLOR_SECONDARY_HOVER = "#EEF2F9"
COLOR_SECONDARY_PRESSED = "#E0E7F5"
COLOR_SECONDARY_TEXT = "#3A6FE0"  # ink/border for the outline button
COLOR_SELECTION = "#E7EFFF"  # dropdown selection highlight (light blue)

# Danger = modern red for Stop
COLOR_DANGER = "#F04770"
COLOR_DANGER_HOVER = "#F45C82"
COLOR_DANGER_PRESSED = "#D5305C"

COLOR_DISABLED_BG = "#E7EAF1"
COLOR_DISABLED_TEXT = "#A6ADBC"

# Live-feed canvas (the video preview area) — kept slightly darker than the
# app canvas so letterboxing reads as a distinct surface, but light overall.
COLOR_FEED_BG = "#E9EDF5"
COLOR_FEED_PLACEHOLDER = "#9AA3B7"

# ─────────────────────────────────────────────────────────────────────────────
# Spacing scale (px)
# ─────────────────────────────────────────────────────────────────────────────

SPACE_XS = 6  # info label vertical padding
SPACE_SM = 8  # combo padding-vertical / checkbox spacing
SPACE_MD = 12  # info label horizontal padding
SPACE_LG = 11  # button padding-vertical
SPACE_XL = 14  # combo padding-horizontal
SPACE_XXL = 20  # button padding-horizontal

# ─────────────────────────────────────────────────────────────────────────────
# Border radius (px)
# ─────────────────────────────────────────────────────────────────────────────

# Kept intentionally subtle — only slight rounding, no pill shapes.
RADIUS_SM = 6  # combo box / inputs
RADIUS_MD = 8  # cards / toolbar / buttons
