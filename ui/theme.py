"""
App-wide look and UI-speed plumbing (Gradio 6.31).

- build_theme(): the light/dark design tokens as a Gradio theme.
- DESIGN_CSS: layout, tab bar, header, accordion and readability rules. Gradio copies
  every custom rule with a `.gradio-container.gradio-container-6-31-0 .contain` prefix
  and only treats the FIRST selector of a `.dark` rule list as dark, so colours come
  from tokens (:root = light, .dark = dark) and dark-only rules use single selectors.
- trim_progress_tracking(): Gradio re-walks the loading status of every component that
  was ever a progress target on each event start/finish (O(components^2) per event).
  Events that show no progress are untracked, the rest track only their media outputs.
- live_poller(): one hidden gr.HTML that polls a server function (no Gradio event, so no
  per-tick page work) and fires a refresh event only when the watched data changed.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Callable

import gradio as gr


def build_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Cascadia Code", "Consolas", "monospace"],
    ).set(
        body_background_fill="#eef2f8",
        body_background_fill_dark="#0a0f1c",
        body_text_color="#0f172a",
        body_text_color_dark="#e6ebf4",
        body_text_color_subdued="#4b5a73",
        body_text_color_subdued_dark="#9eabc2",
        body_text_size="16px",
        body_text_weight="400",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#111a2d",
        background_fill_secondary="#f5f7fb",
        background_fill_secondary_dark="#0d1526",
        border_color_primary="#d8e0ed",
        border_color_primary_dark="#24314d",
        color_accent="#6366f1",
        color_accent_soft="#eef2ff",
        color_accent_soft_dark="#1c2644",
        link_text_color="#4f46e5",
        link_text_color_dark="#a5b4fc",
        link_text_color_hover="#4338ca",
        link_text_color_hover_dark="#c7d2fe",
        link_text_color_visited="#6d28d9",
        link_text_color_visited_dark="#c4b5fd",
        code_background_fill="#eef2f8",
        code_background_fill_dark="#0b1322",
        block_background_fill="#ffffff",
        block_background_fill_dark="#111a2d",
        block_border_color="#d8e0ed",
        block_border_color_dark="#24314d",
        block_border_width="1px",
        block_radius="14px",
        block_shadow="0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 14px -6px rgba(15, 23, 42, 0.10)",
        block_shadow_dark="0 1px 2px rgba(0, 0, 0, 0.35), 0 8px 22px -12px rgba(0, 0, 0, 0.60)",
        # Field labels as plain readable text (Soft paints them as filled chips).
        block_title_background_fill="transparent",
        block_title_background_fill_dark="transparent",
        block_title_text_color="#1e293b",
        block_title_text_color_dark="#e2e8f0",
        block_title_text_size="15px",
        block_title_text_weight="650",
        block_title_padding="0",
        block_title_radius="0",
        block_title_border_width="0",
        block_label_background_fill="#eef2ff",
        block_label_background_fill_dark="#1c2644",
        block_label_border_color="#c7d2fe",
        block_label_border_color_dark="#33417060",
        block_label_border_width="1px",
        block_label_text_color="#4338ca",
        block_label_text_color_dark="#c7d2fe",
        block_label_text_size="14px",
        block_label_text_weight="650",
        block_info_text_color="#4b5a73",
        block_info_text_color_dark="#9eabc2",
        block_info_text_size="13px",
        panel_background_fill="#f5f7fb",
        panel_background_fill_dark="#0d1526",
        panel_border_color="#d8e0ed",
        panel_border_color_dark="#24314d",
        accordion_text_color="#1e293b",
        accordion_text_color_dark="#e2e8f0",
        input_background_fill="#f8fafd",
        input_background_fill_dark="#0b1322",
        input_background_fill_focus="#ffffff",
        input_background_fill_focus_dark="#0e1729",
        input_border_color="#cdd6e5",
        input_border_color_dark="#2b3b5c",
        input_border_color_hover="#a5b4fc",
        input_border_color_hover_dark="#4b5d8a",
        input_border_color_focus="#6366f1",
        input_border_color_focus_dark="#818cf8",
        input_border_width="1px",
        input_shadow="none",
        input_shadow_dark="none",
        input_shadow_focus="0 0 0 3px rgba(99, 102, 241, 0.18)",
        input_shadow_focus_dark="0 0 0 3px rgba(129, 140, 248, 0.28)",
        input_radius="10px",
        input_text_size="15px",
        input_placeholder_color="#94a3b8",
        input_placeholder_color_dark="#62728f",
        slider_color="#6366f1",
        slider_color_dark="#818cf8",
        checkbox_border_color="#b8c4d8",
        checkbox_border_color_dark="#3a4b70",
        checkbox_background_color_selected="#6366f1",
        checkbox_background_color_selected_dark="#6366f1",
        checkbox_border_color_selected="#6366f1",
        checkbox_border_color_selected_dark="#818cf8",
        checkbox_label_text_size="15px",
        table_even_background_fill="#f8fafd",
        table_even_background_fill_dark="#0f1829",
        table_odd_background_fill="#ffffff",
        table_odd_background_fill_dark="#111a2d",
        button_border_width="1px",
        button_large_padding="12px 22px",
        button_large_radius="12px",
        button_large_text_size="17px",
        button_large_text_weight="650",
        button_medium_radius="11px",
        button_medium_text_weight="600",
        button_small_radius="10px",
        button_primary_background_fill="linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #4338ca 0%, #6d28d9 100%)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, #5b54f0 0%, #8b5cf6 100%)",
        button_primary_border_color="transparent",
        button_primary_border_color_dark="transparent",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_shadow="0 6px 16px -6px rgba(79, 70, 229, 0.55)",
        button_primary_shadow_dark="0 6px 18px -6px rgba(124, 58, 237, 0.65)",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#18223a",
        button_secondary_background_fill_hover="#f1f4fb",
        button_secondary_background_fill_hover_dark="#1f2b47",
        button_secondary_border_color="#cdd6e5",
        button_secondary_border_color_dark="#2d3c5e",
        button_secondary_text_color="#1e293b",
        button_secondary_text_color_dark="#e2e8f0",
        button_cancel_background_fill="linear-gradient(135deg, #b91c1c 0%, #e11d48 100%)",
        button_cancel_background_fill_dark="linear-gradient(135deg, #b91c1c 0%, #e11d48 100%)",
        button_cancel_text_color="#ffffff",
        button_cancel_text_color_dark="#ffffff",
    )


# (deep, mid, bright) per hue. The gradient runs deep -> mid -> bright, so white text stays
# readable on every stop in both themes; light mode stretches the mid stop and uses a
# solid border with a softer shadow.
BUTTON_HUES = {
    "red":     ("#991b1b", "#dc2626", "#f87171"),
    "crimson": ("#4c0519", "#be123c", "#fb7185"),
    "rose":    ("#9f1239", "#e11d48", "#fda4af"),
    "pink":    ("#9d174d", "#db2777", "#f9a8d4"),
    "fuchsia": ("#86198f", "#c026d3", "#e879f9"),
    "purple":  ("#6b21a8", "#9333ea", "#c084fc"),
    "violet":  ("#5b21b6", "#7c3aed", "#a78bfa"),
    "indigo":  ("#3730a3", "#4f46e5", "#818cf8"),
    "blue":    ("#1e40af", "#2563eb", "#60a5fa"),
    "sky":     ("#075985", "#0284c7", "#38bdf8"),
    "cyan":    ("#155e75", "#0891b2", "#22d3ee"),
    "teal":    ("#115e59", "#0d9488", "#2dd4bf"),
    "emerald": ("#065f46", "#059669", "#34d399"),
    "green":   ("#166534", "#16a34a", "#4ade80"),
    "lime":    ("#3f6212", "#65a30d", "#a3e635"),
    "yellow":  ("#854d0e", "#ca8a04", "#facc15"),
    "amber":   ("#92400e", "#d97706", "#fbbf24"),
    "orange":  ("#9a3412", "#ea580c", "#fb923c"),
    "bronze":  ("#78350f", "#b45309", "#d6a05a"),
    "slate":   ("#334155", "#475569", "#94a3b8"),
}


def _rgb(hex_color: str) -> str:
    value = hex_color.lstrip("#")
    return ", ".join(str(int(value[i:i + 2], 16)) for i in (0, 2, 4))


def button_hue_css() -> str:
    rules = []
    for name, (deep, mid, bright) in BUTTON_HUES.items():
        m, b = _rgb(mid), _rgb(bright)
        rules.append(f"""
button.sec-btn-{name}, .sec-btn-{name} button {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 66%, {bright} 100%) !important;
  border-color: {mid} !important;
  color: #f8fafc !important;
  text-shadow: 0 1px 2px rgba(2, 6, 23, 0.45);
  box-shadow: 0 7px 17px -4px rgba({m}, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.26) !important;
}}
button.sec-btn-{name}:hover:not(:disabled), .sec-btn-{name} button:hover:not(:disabled) {{
  box-shadow: 0 11px 24px -6px rgba({m}, 0.50), inset 0 1px 0 rgba(255, 255, 255, 0.32) !important;
}}
.dark button.sec-btn-{name} {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 55%, {bright} 100%) !important;
  border-color: rgba({b}, 0.72) !important;
  box-shadow: 0 8px 20px -4px rgba({m}, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.20) !important;
}}
.dark button.sec-btn-{name}:hover:not(:disabled) {{
  border-color: rgba({b}, 0.98) !important;
  box-shadow: 0 12px 26px -6px rgba({m}, 0.48), inset 0 1px 0 rgba(255, 255, 255, 0.28) !important;
}}""")
    return "\n".join(rules)


DESIGN_CSS = r"""
:root {
  --sec-bg-glow-1: rgba(99, 102, 241, 0.10);
  --sec-bg-glow-2: rgba(20, 184, 166, 0.08);
  --sec-surface: #ffffff;
  --sec-surface-2: #f5f7fb;
  --sec-border: #d8e0ed;
  --sec-text: #0f172a;
  --sec-text-2: #1e293b;
  --sec-muted: #4b5a73;
  --sec-accent: #6366f1;
  --sec-ok: #15803d;
  --sec-warn: #b45309;
  --sec-bad: #be123c;
  --sec-card-tint: linear-gradient(135deg, rgba(99, 102, 241, 0.06), rgba(14, 165, 233, 0.05));
  --sec-card-inner: #f8fafd;
  --sec-card-title: #4338ca;
  --sec-tabbar-bg: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(217, 70, 239, 0.05) 55%, rgba(20, 184, 166, 0.06));
  --sec-tabbar-border: rgba(99, 102, 241, 0.22);
  --sec-tab-bg: #ffffff;
  --sec-tab-border: #d8e0ed;
  --sec-tab-text: #1e293b;
  --sec-tab-hover: #eef2ff;
  --sec-tab-hover-border: #a5b4fc;
  --sec-shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
  --sec-hero-sub: #4b5a73;
}
.dark {
  --sec-bg-glow-1: rgba(99, 102, 241, 0.16);
  --sec-bg-glow-2: rgba(20, 184, 166, 0.07);
  --sec-surface: #111a2d;
  --sec-surface-2: #0d1526;
  --sec-border: #24314d;
  --sec-text: #e6ebf4;
  --sec-text-2: #dbe3f0;
  --sec-muted: #9eabc2;
  --sec-accent: #818cf8;
  --sec-ok: #4ade80;
  --sec-warn: #fbbf24;
  --sec-bad: #fb7185;
  --sec-card-tint: linear-gradient(135deg, rgba(30, 41, 79, 0.55), rgba(14, 116, 144, 0.14));
  --sec-card-inner: rgba(10, 16, 30, 0.55);
  --sec-card-title: #c7d2fe;
  --sec-tabbar-bg: linear-gradient(135deg, rgba(49, 46, 129, 0.34), rgba(112, 26, 117, 0.16) 55%, rgba(17, 94, 89, 0.20));
  --sec-tabbar-border: rgba(129, 140, 248, 0.26);
  --sec-tab-bg: #131c31;
  --sec-tab-border: #26344f;
  --sec-tab-text: #d3dbea;
  --sec-tab-hover: #1b2643;
  --sec-tab-hover-border: #4b5d8a;
  --sec-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.35);
  --sec-hero-sub: #9eabc2;
}

/* Page: Gradio paints the backdrop from the OS colour scheme, not the chosen theme. */
body {
  background: var(--body-background-fill) !important;
}
.gradio-container {
  background:
    radial-gradient(1200px 520px at 12% -8%, var(--sec-bg-glow-1), transparent 70%),
    radial-gradient(900px 480px at 92% -12%, var(--sec-bg-glow-2), transparent 70%),
    var(--body-background-fill) !important;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

/* Header */
#sec-hero .html-container {
  padding: 0 !important;
}
.sec-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  padding: 14px 18px;
  border-radius: 18px;
  border: 1px solid var(--sec-tabbar-border);
  background: var(--sec-surface);
  box-shadow: var(--sec-shadow-sm), 0 10px 30px -18px rgba(79, 70, 229, 0.45);
}
.sec-hero-main {
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
}
.sec-hero-logo {
  width: 52px;
  height: 52px;
  flex: 0 0 auto;
  border-radius: 14px;
  box-shadow: 0 8px 20px -8px rgba(49, 46, 129, 0.7);
}
.sec-hero-title {
  font-size: 24px;
  font-weight: 800;
  letter-spacing: -0.01em;
  line-height: 1.2;
  background-image: linear-gradient(90deg, #4f46e5, #7c3aed 45%, #0d9488);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.dark .sec-hero-title {
  background-image: linear-gradient(90deg, #a5b4fc, #d8b4fe 45%, #5eead4);
}
.sec-hero-version {
  display: inline-block;
  vertical-align: 3px;
  margin-left: 8px;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 750;
  letter-spacing: 0.02em;
  color: #ffffff;
  background: linear-gradient(135deg, #4f46e5, #c026d3);
  -webkit-text-fill-color: #ffffff;
}
.sec-hero-sub {
  margin-top: 3px;
  font-size: 13.5px;
  color: var(--sec-hero-sub);
  line-height: 1.4;
}
.sec-hero-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.sec-hero-actions a,
.sec-hero-actions button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border-radius: 11px;
  font-size: 14px;
  font-weight: 650;
  line-height: 1.2;
  cursor: pointer;
  text-decoration: none !important;
  border: 1px solid var(--sec-tab-border);
  background: var(--sec-tab-bg);
  color: var(--sec-tab-text) !important;
  transition: background-color 0.12s ease, border-color 0.12s ease;
}
.sec-hero-actions a:hover,
.sec-hero-actions button:hover {
  background: var(--sec-tab-hover);
  border-color: var(--sec-tab-hover-border);
}
.sec-hero-actions a.sec-hero-primary {
  border-color: transparent;
  color: #ffffff !important;
  background: linear-gradient(135deg, #e11d48, #f97316);
  box-shadow: 0 6px 16px -6px rgba(225, 29, 72, 0.6);
}
.sec-hero-actions a.sec-hero-primary:hover {
  filter: brightness(1.06);
}

/* Top status row */
.top-status-row .health-banner {
  margin: 0;
  padding: 9px 14px;
  border-radius: 12px;
  border: 1px solid rgba(16, 185, 129, 0.35);
  background: linear-gradient(120deg, rgba(16, 185, 129, 0.10), rgba(14, 165, 233, 0.08));
  color: var(--sec-text) !important;
  font-weight: 600;
  font-size: 14.5px;
}

/* Main tab bar: every tab visible, wrapped into rows (Tabs overflow_behavior="wrap"). */
#secourses-main-tabs > .tab-wrapper {
  height: auto !important;
  padding: 8px !important;
  margin-bottom: 14px;
  border-radius: 16px !important;
  border: 1px solid var(--sec-tabbar-border) !important;
  background: var(--sec-tabbar-bg) !important;
  box-shadow: var(--sec-shadow-sm) !important;
}
#secourses-main-tabs > .tab-wrapper .tab-container {
  height: auto !important;
  display: grid !important;
  grid-template-columns: repeat(auto-fill, minmax(214px, 1fr));
  gap: 6px !important;
  background: none !important;
  overflow: visible !important;
}
#secourses-main-tabs > .tab-wrapper .tab-container::after {
  display: none !important;
}
#secourses-main-tabs > .tab-wrapper .tab-container button {
  width: auto !important;
  max-width: none !important;
  min-width: 0 !important;
  height: auto !important;
  min-height: 42px !important;
  margin: 0 !important;
  padding: 8px 10px !important;
  justify-content: center !important;
  text-align: center !important;
  white-space: normal !important;
  line-height: 1.2 !important;
  border-radius: 11px !important;
  border: 1px solid var(--sec-tab-border) !important;
  background: var(--sec-tab-bg) !important;
  color: var(--sec-tab-text) !important;
  font-size: 14.5px !important;
  font-weight: 650 !important;
  box-shadow: none !important;
  transform: none !important;
  transition: background-color 0.1s ease, border-color 0.1s ease, color 0.1s ease !important;
}
#secourses-main-tabs > .tab-wrapper .tab-container button:hover:not(.selected) {
  background: var(--sec-tab-hover) !important;
  border-color: var(--sec-tab-hover-border) !important;
}
#secourses-main-tabs > .tab-wrapper .tab-container button.selected {
  color: #ffffff !important;
  border-color: transparent !important;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #c026d3 100%) !important;
  box-shadow: 0 6px 16px -6px rgba(124, 58, 237, 0.65) !important;
}
#secourses-main-tabs > .tab-wrapper .tab-container button.selected::after {
  display: none !important;
}

/* Nested tab sets inside a page */
.tabitem .tabs > .tab-wrapper .tab-container {
  gap: 4px;
  background: none !important;
}
.tabitem .tabs > .tab-wrapper .tab-container button {
  border-radius: 10px !important;
  padding: 6px 12px !important;
  font-weight: 600 !important;
}
.tabitem .tabs > .tab-wrapper .tab-container button.selected {
  color: var(--sec-card-title) !important;
  background: var(--color-accent-soft) !important;
}

/* Section headings written as markdown ### inside pages */
.tabitem .prose h3,
.tabitem .md h3 {
  font-size: 18px;
  font-weight: 750;
  letter-spacing: -0.005em;
  color: var(--sec-text-2);
}
.block-info,
[data-testid="block-info"] + span {
  line-height: 1.45;
}

/* Accordions: a clear header row */
.gr-accordion > button.label-wrap {
  padding: 4px 2px !important;
  font-size: 15.5px !important;
  font-weight: 700 !important;
  color: var(--sec-text-2) !important;
}
.gr-accordion > button.label-wrap > span:first-child {
  font-size: 15.5px !important;
  font-weight: 700 !important;
}
.gr-accordion > button.label-wrap:hover {
  color: var(--sec-accent) !important;
}
.gr-accordion > button.label-wrap .icon {
  color: var(--sec-accent);
}

/* Buttons: shared shape; colours come from the sec-btn-<hue> / action-btn-* classes. */
.gradio-container button.lg,
.gradio-container button.md {
  transition: transform 0.12s ease, filter 0.12s ease, box-shadow 0.12s ease, background-color 0.12s ease;
}
.action-btn button,
button.action-btn {
  border-radius: 12px !important;
  font-weight: 700 !important;
  min-height: 44px;
  letter-spacing: 0.01em;
}
.action-btn button:hover:not(:disabled),
button.action-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  filter: brightness(1.05) saturate(1.05);
}
.action-btn button:active:not(:disabled),
button.action-btn:active:not(:disabled) {
  transform: translateY(1px);
  filter: brightness(0.96);
}
.action-btn button:focus-visible,
button.action-btn:focus-visible {
  outline: 2px solid #a5b4fc !important;
  outline-offset: 2px;
}
/* The always-running sweep and pulse repainted the page ~60 times a second while idle;
   the sweep now plays once on hover (transform only, compositor-driven). */
.action-btn-upscale button::after,
button.action-btn-upscale::after,
.action-btn-optimize button::after,
button.action-btn-optimize::after {
  left: 0 !important;
  opacity: 0;
  animation: none !important;
  transform: translateX(-140%) rotate(18deg);
}
.action-btn-upscale button:hover:not(:disabled)::after,
button.action-btn-upscale:hover:not(:disabled)::after,
.action-btn-optimize button:hover:not(:disabled)::after,
button.action-btn-optimize:hover:not(:disabled)::after {
  animation: secSweepOnce 0.85s ease-out 1 !important;
}
.action-btn-optimize button,
button.action-btn-optimize {
  animation: none !important;
}
@keyframes secSweepOnce {
  0% { transform: translateX(-140%) rotate(18deg); opacity: 1; }
  100% { transform: translateX(480%) rotate(18deg); opacity: 0; }
}

/* Theme-aware custom cards (they were dark-only and unreadable in light mode) */
.resolution-info .resolution-stats-shell {
  background: var(--sec-card-tint) !important;
  border-color: color-mix(in srgb, var(--sec-accent) 28%, transparent) !important;
}
.resolution-info .resolution-stat-card {
  background: var(--sec-card-inner) !important;
  border-color: var(--sec-border) !important;
}
.resolution-info .resolution-stat-card-title {
  color: var(--sec-card-title) !important;
}
.resolution-info .resolution-stat-row {
  border-top-color: color-mix(in srgb, var(--sec-muted) 22%, transparent) !important;
}
.resolution-info .resolution-stat-val.is-up {
  color: var(--sec-ok) !important;
}
.resolution-info .resolution-stat-val.is-down,
.resolution-info .resolution-stat-val.is-override {
  color: var(--sec-bad) !important;
}
.resolution-info .resolution-stat-val.is-neutral {
  color: var(--sec-text) !important;
}
.health-report-summary,
.health-report-card {
  background: var(--sec-card-tint) !important;
}
.health-report-detail {
  background: #172033 !important;
  color: #e2e8f0 !important;
}
.health-report-metrics span,
.health-report-badge {
  background: var(--sec-card-inner) !important;
}
.health-report-badge.is-ok {
  color: var(--sec-ok) !important;
}
.health-report-badge.is-warning {
  color: var(--sec-warn) !important;
}
.health-report-badge.is-error {
  color: var(--sec-bad) !important;
}
.health-report-badge.is-skipped {
  color: var(--sec-muted) !important;
}
.autotune-modal-card {
  background: var(--sec-surface) !important;
  border-color: color-mix(in srgb, var(--sec-accent) 45%, transparent) !important;
  color: var(--sec-text) !important;
}
.vram-oom-card {
  background: var(--sec-card-inner) !important;
}

/* Groups: Gradio fills them with the border colour, which shows as a pale patch wherever
   one column is shorter than its neighbour. */
.gr-group,
.gr-group > .styler {
  background: var(--block-background-fill) !important;
}

/* Gradio 6 resets min-width of fields wrapped in a row's .form box to 0 (cut labels). */
.row > .form > .block {
  min-width: min(160px, 100%);
}

/* Scrollbars */
.gradio-container ::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}
.gradio-container ::-webkit-scrollbar-thumb {
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
  background-color: color-mix(in srgb, var(--sec-muted) 45%, transparent);
}
.gradio-container ::-webkit-scrollbar-thumb:hover {
  background-color: color-mix(in srgb, var(--sec-accent) 70%, transparent);
}
.gradio-container ::-webkit-scrollbar-track {
  background: transparent;
}

#sec-live-poller {
  display: none !important;
}
"""


# Removes run-history keys left by launches before run_history=False (it clears only
# the current app id's key) and the obsolete per-launch entries.
RUN_HISTORY_CLEANUP_HEAD = """
<script>
try {
  Object.keys(localStorage).forEach(function (k) {
    if (k.indexOf("gradio:run-history:") === 0) { localStorage.removeItem(k); }
  });
} catch (e) {}
</script>
"""


# Chrome allows a limited number of media players per page and frees the player of a removed
# <video>/<audio> only when that element is garbage collected. Every run replaces preview,
# output and comparison videos, so after many runs new players were refused ("too many
# WebMediaPlayers") and previews stayed on "Loading videos..." until a page refresh. Release the
# player of every media element that has left the page (checked a moment later, so an element
# that is only moved keeps playing).
# DLSS 5 and Chains tabs (V9.7). Built on the --sec-* tokens, so light and dark follow the app theme.
FEATURE_CSS = r"""
.dlss5-hero, .chain-hero {
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 12px;
  border: 1px solid var(--sec-tabbar-border);
  background: linear-gradient(120deg, rgba(16, 185, 129, 0.14), rgba(99, 102, 241, 0.12) 55%, rgba(14, 165, 233, 0.12));
  box-shadow: var(--sec-shadow-sm);
}
.chain-hero {
  background: linear-gradient(120deg, rgba(217, 70, 239, 0.13), rgba(99, 102, 241, 0.12) 50%, rgba(20, 184, 166, 0.13));
}
.dlss5-hero-title, .chain-hero-title {
  font-size: 1.45rem;
  font-weight: 800;
  letter-spacing: 0.2px;
  color: var(--sec-text);
}
.dlss5-hero-sub, .chain-hero-sub {
  margin-top: 6px;
  color: var(--sec-hero-sub, var(--sec-muted));
  line-height: 1.5;
  font-size: 0.95rem;
}
.dlss5-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.dlss5-chip {
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--sec-text-2);
  background: var(--sec-surface);
  border: 1px solid var(--sec-border);
}
.dlss5-card-title p, .chain-card-title p, .dlss5-card-title h4, .chain-card-title h4 {
  margin: 2px 0 6px 0 !important;
  color: var(--sec-card-title) !important;
}
.dlss5-post-card {
  border: 1px solid rgba(16, 185, 129, 0.45) !important;
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.07), rgba(99, 102, 241, 0.05)) !important;
}
.chain-palette { flex-wrap: wrap !important; gap: 8px !important; }
.chain-palette > button, .chain-palette .chain-add-btn {
  flex: 0 1 auto !important;
  border-radius: 999px !important;
  font-weight: 650 !important;
}
.chain-flow {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 12px;
  border-radius: 14px;
  background: var(--sec-card-inner);
  border: 1px dashed var(--sec-border);
}
.chain-node {
  min-width: 110px;
  max-width: 240px;
  padding: 8px 12px;
  border-radius: 12px;
  background: var(--sec-surface);
  border: 1px solid var(--sec-tab-hover-border, var(--sec-border));
  box-shadow: 0 4px 14px -8px rgba(79, 70, 229, 0.55);
}
.chain-node-io {
  min-width: 0;
  font-weight: 700;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.16), rgba(20, 184, 166, 0.14));
}
.chain-node-off { opacity: 0.45; filter: grayscale(0.7); }
.chain-node-top { font-weight: 700; color: var(--sec-text); white-space: nowrap; }
.chain-node-sub { font-size: 0.75rem; color: var(--sec-muted); margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chain-node-size { font-size: 0.74rem; font-weight: 700; color: var(--sec-accent); margin-top: 2px; white-space: nowrap; }
.chain-input-info p { margin: 4px 2px 2px !important; color: var(--sec-text-2); }
.chain-preview { margin-top: 4px; }
/* Path fields with native folder / file picker buttons (ui/path_picker.py, V10) */
.path-row { gap: 6px !important; align-items: flex-end !important; flex-wrap: nowrap !important; }
.path-row > .form { flex: 1 1 auto !important; min-width: 0 !important; }
.path-row:not(:has(.block)) { display: none !important; }
button.path-pick-btn {
  flex: 0 0 auto !important;
  min-width: 46px !important;
  width: 46px !important;
  height: 41px !important;
  margin-bottom: 11px !important;
  padding: 0 !important;
  font-size: 1.25rem !important;
  line-height: 1 !important;
  border-radius: 10px !important;
  border: none !important;
  color: #fff !important;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
}
button.path-pick-folder {
  background: linear-gradient(135deg, #1e40af 0%, #2563eb 60%, #60a5fa 100%) !important;
  box-shadow: 0 6px 14px -4px rgba(37, 99, 235, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
}
button.path-pick-file {
  background: linear-gradient(135deg, #115e59 0%, #0d9488 60%, #2dd4bf 100%) !important;
  box-shadow: 0 6px 14px -4px rgba(13, 148, 136, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
}
button.path-pick-btn:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.08) saturate(1.05); }
button.path-pick-btn:active:not(:disabled) { transform: scale(0.97); filter: brightness(0.95); }
.chain-batch-card .chain-batch-info p { margin: 2px 2px 0 !important; color: var(--sec-text-2); }
.chain-node-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 20px; height: 20px; border-radius: 50%;
  font-size: 0.72rem; font-weight: 800; color: #fff;
  background: linear-gradient(135deg, #4f46e5, #0891b2);
}
.chain-badge {
  margin-left: 4px; padding: 1px 7px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 800; color: #065f46;
  background: rgba(52, 211, 153, 0.25); border: 1px solid rgba(16, 185, 129, 0.5);
}
.dark .chain-badge { color: #a7f3d0 !important; }
.chain-arrow { font-size: 1.1rem; color: var(--sec-accent); font-weight: 800; }
.chain-flow-summary { margin: 6px 4px 2px; font-size: 0.85rem; color: var(--sec-muted); }
.chain-step {
  align-items: center !important;
  padding: 6px 10px !important;
  margin: 6px 0 !important;
  border-radius: 12px !important;
  background: var(--sec-surface) !important;
  border: 1px solid var(--sec-border) !important;
}
.chain-step-title { font-weight: 700; color: var(--sec-text); padding: 6px 2px; white-space: nowrap; }
.chain-timeline { display: flex; flex-direction: column; gap: 6px; margin: 6px 0 10px; }
.chain-tl-head { font-weight: 700; color: var(--sec-text); margin-bottom: 2px; }
.chain-tl-row {
  display: grid; grid-template-columns: 26px minmax(110px, 170px) 1fr; align-items: center; gap: 8px;
  padding: 8px 10px; border-radius: 10px; border: 1px solid var(--sec-border); background: var(--sec-surface);
}
.chain-tl-name { font-weight: 650; color: var(--sec-text); white-space: nowrap; }
.chain-tl-detail { font-size: 0.82rem; color: var(--sec-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chain-tl-running { border-color: rgba(99, 102, 241, 0.75); box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.18); }
.chain-tl-running .chain-tl-icon { animation: chain-pulse 1.2s ease-in-out infinite; }
.chain-tl-done { border-color: rgba(16, 185, 129, 0.6); }
.chain-tl-failed { border-color: rgba(225, 29, 72, 0.7); background: rgba(225, 29, 72, 0.06); }
.chain-tl-skipped { opacity: 0.6; }
@keyframes chain-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
@media (max-width: 720px) {
  .chain-node { max-width: 100%; }
  .chain-tl-row { grid-template-columns: 26px 1fr; }
  .chain-tl-detail { grid-column: 1 / -1; }
}
"""


MEDIA_PLAYER_RELEASE_HEAD = """
<script>
(() => {
  if (window.__secoursesMediaReleaseInstalled) return;
  window.__secoursesMediaReleaseInstalled = true;
  const removed = new Set();
  let timer = null;
  const release = (el) => {
    try { el.pause(); } catch (e) {}
    el.removeAttribute("src");
    el.querySelectorAll("source").forEach((s) => s.remove());
    try { el.load(); } catch (e) {}
  };
  const flush = () => {
    timer = null;
    removed.forEach((el) => { if (!el.isConnected) release(el); });
    removed.clear();
  };
  const collect = (node) => {
    if (node.nodeType !== 1) return;
    if (node.matches("video, audio")) removed.add(node);
    node.querySelectorAll("video, audio").forEach((el) => removed.add(el));
  };
  const start = () => {
    new MutationObserver((records) => {
      records.forEach((r) => r.removedNodes.forEach(collect));
      if (removed.size && !timer) timer = setTimeout(flush, 1000);
    }).observe(document.body, { childList: true, subtree: true });
  };
  if (document.body) start(); else document.addEventListener("DOMContentLoaded", start, { once: true });
})();
</script>
"""


def _logo_data_uri(favicon_path: Path) -> str:
    try:
        return "data:image/svg+xml;base64," + base64.b64encode(Path(favicon_path).read_bytes()).decode("ascii")
    except Exception:
        return ""


HERO_JS = """
const toggle = element.querySelector('.sec-theme-toggle');
const paintLabel = () => {
  const dark = document.body.classList.contains('dark');
  if (toggle) toggle.textContent = dark ? '☀️ Light theme' : '🌙 Dark theme';
};
paintLabel();
if (toggle) toggle.addEventListener('click', () => {
  const mode = document.body.classList.contains('dark') ? 'light' : 'dark';
  document.body.classList.toggle('dark', mode === 'dark');
  try { localStorage.setItem('secourses_theme_mode', mode); } catch (e) {}
  const url = new URL(window.location.href);
  url.searchParams.set('__theme', mode);
  window.history.replaceState(null, '', url.toString());
  paintLabel();
  // A frontend-only event copies the mode into the Global Settings Theme radio
  // (saved with presets), even before that tab was ever opened.
  trigger('click');
});
new MutationObserver(paintLabel).observe(document.body, { attributes: true, attributeFilter: ['class'] });
"""


def hero_header(version: str, favicon_path: Path, patreon_url: str) -> gr.HTML:
    logo = _logo_data_uri(favicon_path)
    logo_html = f'<img class="sec-hero-logo" src="{logo}" alt="">' if logo else ""
    html = f"""
<div class="sec-hero">
  <div class="sec-hero-main">
    {logo_html}
    <div>
      <div class="sec-hero-title">SECourses Ultimate Video &amp; Image Upscaler Pro<span class="sec-hero-version">V{version}</span></div>
      <div class="sec-hero-sub">SeedVR2 · FlashVSR · FlashVSR+ · Open-H3 Regenerate · LTX 2.5 · SparkVSR · RTX Super Resolution · DLSS 5 · Chains · GAN · RIFE · Face Restoration</div>
    </div>
  </div>
  <div class="sec-hero-actions">
    <button type="button" class="sec-theme-toggle">🌓 Theme</button>
    <a class="sec-hero-primary" href="{patreon_url}" target="_blank" rel="noopener">Patreon post ↗</a>
  </div>
</div>
"""
    return gr.HTML(html, js_on_load=HERO_JS, apply_default_css=False, padding=False, elem_id="sec-hero")


POLLER_JS = """
let last = null;
let busy = false;
const tick = async () => {
  if (busy || document.visibilityState !== 'visible') return;
  busy = true;
  try {
    const snap = await server.sec_live_snapshot(0);
    const key = JSON.stringify(snap);
    if (key !== last) {
      const first = last === null;
      last = key;
      if (!first || (snap && (snap.waiting || snap.active || snap.oom))) trigger('click');
    }
  } catch (e) {
  } finally {
    busy = false;
  }
};
setInterval(tick, 2000);
document.addEventListener('visibilitychange', tick);
setTimeout(tick, 400);
"""


def live_poller(snapshot_fn: Callable) -> gr.HTML:
    """Hidden poller; its click event fires only when snapshot_fn's result changes."""
    if snapshot_fn.__name__ != "sec_live_snapshot":
        raise ValueError("live_poller expects a function named sec_live_snapshot")
    return gr.HTML(
        "",
        visible="hidden",
        elem_id="sec-live-poller",
        server_functions=[snapshot_fn],
        js_on_load=POLLER_JS,
        apply_default_css=False,
        padding=False,
    )


def front_state_components(demo: gr.Blocks) -> int:
    """Move every gr.State to the start of the page layout; returns how many moved.

    Gradio's browser finds a component by walking the layout depth-first, for each
    state input of every event and for each trigger at page load. States created deep
    inside the tabs made those lookups walk most of the ~1750 components. A State
    renders nothing, so the page looks the same."""
    from gradio.blocks import BlockContext

    moved = []

    def walk(context) -> None:
        kept = []
        for child in context.children:
            if isinstance(child, gr.State) and getattr(child, "rendered_in", None) is None:
                moved.append(child)
                continue
            kept.append(child)
            if isinstance(child, BlockContext):
                walk(child)
        context.children[:] = kept

    walk(demo)
    for child in moved:
        child.parent = demo
    demo.children[:0] = moved
    return len(moved)


_PROGRESS_MEDIA = (gr.Video, gr.Image, gr.ImageSlider, gr.Gallery, gr.File, gr.Audio)


def trim_progress_tracking(demo: gr.Blocks) -> int:
    """Shrink every event's progress targets; returns how many events changed.

    Gradio refreshes the loading status of every component that was ever a progress
    target on each status change of every later event (a full tree walk per component),
    so the tracked set must stay tiny. Only button clicks keep an overlay: a processing
    run shows it on its first media output, a small click (up to 3 outputs) keeps Gradio's
    default; change/select/load/then events and bulk updates track nothing. The tabs
    show their own status and progress text for long runs.

    Every UI event also becomes an undocumented endpoint (still callable): the page
    otherwise embeds 2.5 MB of API docs for ~400 UI events, parsed on every page load."""
    changed = 0
    for fn in demo.fns.values():
        if fn.api_visibility == "public":
            fn.api_visibility = "undocumented"
        if fn.show_progress_on is not None or fn.fn is None:
            continue
        clicked = any(event == "click" for _block_id, event in fn.targets)
        media = [block for block in fn.outputs if isinstance(block, _PROGRESS_MEDIA)]
        if fn.show_progress == "hidden" or not clicked:
            fn.show_progress_on = []
        elif media:
            shown = [block for block in media if getattr(block, "visible", True) is True]
            fn.show_progress_on = (shown or media)[:1]
        elif len(fn.outputs) > 3:
            fn.show_progress_on = []
        else:
            continue
        changed += 1
    return changed
