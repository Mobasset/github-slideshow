"""
Plotly time-series timeline panel.

Renders RSRP/SINR line chart with event markers as vertical colored spans,
supporting brush-zoom linked to the map.
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Event type → (color, symbol, display name)
EVENT_STYLES: Dict[str, tuple] = {
    "INTERNAL_HANDOVER_ATTEMPT": ("#3498db", "triangle-up", "HO Attempt"),
    "INTERNAL_HANDOVER_SUCCESS": ("#2ecc71", "circle", "HO Success"),
    "INTERNAL_HANDOVER_FAILURE": ("#e74c3c", "x", "HO Failure"),
    "RRC_RLF": ("#8e44ad", "diamond", "RLF"),
    "NR_SCG_FAILURE": ("#f39c12", "star", "SCG Failure"),
    "A3_TRIGGER": ("#1abc9c", "triangle-right", "A3 Trigger"),
    "A5_TRIGGER": ("#e67e22", "triangle-left", "A5 Trigger"),
    "INTERNAL_ERAB_SETUP": ("#27ae60", "circle-open", "ERAB Setup"),
    "INTERNAL_UE_CONTEXT_RELEASE": ("#c0392b", "cross", "UE Release"),
}

DARK_THEME = {
    "bg": "#1a1a2e",
    "panel": "#16213e",
    "grid": "#2a2a4a",
    "text": "#e0e0e0",
    "axis": "#888888",
    "rsrp_color": "#00d4ff",
    "sinr_color": "#ff6b6b",
    "rsrq_color": "#a8d08d",
}


def build_timeline_figure(
    df: pd.DataFrame,
    time_range: Optional[tuple] = None,
    show_events: bool = True,
    show_sinr: bool = True,
    show_rsrq: bool = False,
    height: int = 280,
) -> go.Figure:
    """
    Build the main RSRP/SINR timeline Plotly figure.

    Args:
        df: Unified GeoDataFrame
        time_range: Optional (ts_min_ms, ts_max_ms) filter
        show_events: Overlay event markers
        show_sinr: Show SINR secondary axis
        show_rsrq: Show RSRQ secondary axis
        height: Chart height in pixels
    """
    T = DARK_THEME

    if df.empty or "timestamp_ms" not in df.columns:
        return _empty_figure(height)

    df_plot = df.copy()
    df_plot = df_plot.sort_values("timestamp_ms")

    if time_range:
        ts_min, ts_max = time_range
        mask = (df_plot["timestamp_ms"] >= ts_min) & (df_plot["timestamp_ms"] <= ts_max)
        df_plot = df_plot[mask]

    if df_plot.empty:
        return _empty_figure(height)

    # Convert timestamp to datetime for x-axis
    x_times = pd.to_datetime(df_plot["timestamp_ms"], unit="ms", utc=True)

    # Downsample for rendering performance (max 5000 points)
    if len(df_plot) > 5000:
        step = len(df_plot) // 5000
        df_plot = df_plot.iloc[::step]
        x_times = pd.to_datetime(df_plot["timestamp_ms"], unit="ms", utc=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── RSRP line (colored segments by bin) ──────────────────────────────────
    if "rsrp_dbm" in df_plot.columns:
        rsrp_vals = df_plot["rsrp_dbm"].values

        # Color RSRP line by quality bin
        if "rsrp_color" in df_plot.columns:
            # Use a single trace with marker colors for gradient effect
            fig.add_trace(
                go.Scatter(
                    x=x_times,
                    y=rsrp_vals,
                    mode="lines+markers",
                    name="RSRP (dBm)",
                    line={"color": T["rsrp_color"], "width": 1.5},
                    marker={
                        "size": 3,
                        "color": df_plot["rsrp_color"].tolist()
                        if "rsrp_color" in df_plot.columns
                        else T["rsrp_color"],
                        "opacity": 0.8,
                    },
                    hovertemplate=(
                        "<b>%{x}</b><br>RSRP: %{y:.1f} dBm<extra></extra>"
                    ),
                    yaxis="y1",
                ),
                secondary_y=False,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_times,
                    y=rsrp_vals,
                    mode="lines",
                    name="RSRP (dBm)",
                    line={"color": T["rsrp_color"], "width": 1.5},
                    hovertemplate="<b>%{x}</b><br>RSRP: %{y:.1f} dBm<extra></extra>",
                ),
                secondary_y=False,
            )

    # ── SINR line (secondary y-axis) ─────────────────────────────────────────
    if show_sinr and "sinr_db" in df_plot.columns:
        sinr_vals = df_plot["sinr_db"].values
        fig.add_trace(
            go.Scatter(
                x=x_times,
                y=sinr_vals,
                mode="lines",
                name="SINR (dB)",
                line={"color": T["sinr_color"], "width": 1.5, "dash": "dash"},
                hovertemplate="<b>%{x}</b><br>SINR: %{y:.1f} dB<extra></extra>",
                opacity=0.85,
            ),
            secondary_y=True,
        )

    # ── RSRQ line (secondary y-axis) ─────────────────────────────────────────
    if show_rsrq and "rsrq_db" in df_plot.columns:
        rsrq_vals = df_plot["rsrq_db"].values
        fig.add_trace(
            go.Scatter(
                x=x_times,
                y=rsrq_vals,
                mode="lines",
                name="RSRQ (dB)",
                line={"color": T["rsrq_color"], "width": 1.0, "dash": "dot"},
                hovertemplate="<b>%{x}</b><br>RSRQ: %{y:.1f} dB<extra></extra>",
                opacity=0.8,
            ),
            secondary_y=True,
        )

    # ── Event markers ─────────────────────────────────────────────────────────
    if show_events and "event_type" in df_plot.columns:
        event_mask = df_plot["event_type"].notna()
        for etype, (color, symbol, display) in EVENT_STYLES.items():
            e_mask = event_mask & (df_plot["event_type"] == etype)
            if not e_mask.any():
                continue

            e_times = pd.to_datetime(df_plot.loc[e_mask, "timestamp_ms"], unit="ms", utc=True)
            # Place event markers at a fixed y position (10% from top of RSRP range)
            e_rsrp = df_plot.loc[e_mask, "rsrp_dbm"] if "rsrp_dbm" in df_plot.columns else pd.Series([-80] * e_mask.sum())

            fig.add_trace(
                go.Scatter(
                    x=e_times,
                    y=e_rsrp.values,
                    mode="markers",
                    name=display,
                    marker={
                        "symbol": symbol,
                        "size": 10,
                        "color": color,
                        "line": {"width": 1.5, "color": "#ffffff"},
                    },
                    hovertemplate=(
                        f"<b>{display}</b><br>%{{x}}<br>"
                        f"RSRP: %{{y:.1f}} dBm<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

    # ── RSRP quality threshold bands ─────────────────────────────────────────
    rsrp_thresholds = [
        (-80, "Excellent", "#2ecc71"),
        (-90, "Good", "#a8d08d"),
        (-100, "Fair", "#f0a500"),
        (-110, "Poor", "#e06000"),
    ]
    for y_val, label, color in rsrp_thresholds:
        fig.add_hline(
            y=y_val,
            line={"color": color, "width": 0.8, "dash": "dot"},
            annotation_text=label,
            annotation_position="right",
            annotation={"font": {"size": 9, "color": color}},
            secondary_y=False,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=height,
        paper_bgcolor=T["bg"],
        plot_bgcolor=T["panel"],
        font={"color": T["text"], "family": "monospace"},
        margin={"l": 50, "r": 60, "t": 10, "b": 40},
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.12,
            "font": {"size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis={
            "gridcolor": T["grid"],
            "tickfont": {"size": 10},
            "rangeslider": {"visible": True, "thickness": 0.05},
            "type": "date",
        },
        hovermode="x unified",
        dragmode="zoom",
        selectdirection="h",
    )

    fig.update_yaxes(
        title_text="RSRP (dBm)",
        gridcolor=T["grid"],
        tickfont={"size": 10, "family": "monospace"},
        secondary_y=False,
        range=[-145, -30],
    )

    if show_sinr or show_rsrq:
        fig.update_yaxes(
            title_text="SINR / RSRQ (dB)",
            gridcolor=T["grid"],
            tickfont={"size": 10, "family": "monospace"},
            secondary_y=True,
            range=[-30, 45],
            showgrid=False,
        )

    return fig


def build_cdf_figure(rsrp_x: list, rsrp_y: list, height: int = 220) -> go.Figure:
    """Build RSRP CDF curve chart."""
    T = DARK_THEME
    fig = go.Figure()

    if rsrp_x and rsrp_y:
        # Color segments by RSRP bin
        fig.add_trace(
            go.Scatter(
                x=rsrp_x,
                y=[y * 100 for y in rsrp_y],
                mode="lines",
                name="RSRP CDF",
                line={"color": T["rsrp_color"], "width": 2},
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.08)",
                hovertemplate="RSRP: %{x:.1f} dBm<br>CDF: %{y:.1f}%<extra></extra>",
            )
        )

        # P5/P50/P95 annotations
        if len(rsrp_x) > 0:
            for pct_idx, pct_label in [(5, "P5"), (50, "P50"), (95, "P95")]:
                idx = min(
                    int(pct_idx / 100 * (len(rsrp_x) - 1)), len(rsrp_x) - 1
                )
                fig.add_vline(
                    x=rsrp_x[idx],
                    line={"color": "#e94560", "dash": "dot", "width": 1},
                    annotation_text=f"{pct_label}={rsrp_x[idx]:.0f}",
                    annotation={"font": {"size": 9, "color": "#e94560"}},
                )

    fig.update_layout(
        height=height,
        paper_bgcolor=T["bg"],
        plot_bgcolor=T["panel"],
        font={"color": T["text"], "family": "monospace"},
        margin={"l": 50, "r": 20, "t": 20, "b": 40},
        xaxis={
            "title": "RSRP (dBm)",
            "gridcolor": T["grid"],
            "tickfont": {"size": 10},
        },
        yaxis={
            "title": "Cumulative %",
            "gridcolor": T["grid"],
            "tickfont": {"size": 10},
            "range": [0, 100],
        },
        showlegend=False,
    )
    return fig


def _empty_figure(height: int = 280) -> go.Figure:
    T = DARK_THEME
    fig = go.Figure()
    fig.add_annotation(
        text="No data loaded — upload files or use Demo Mode",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": T["text"]},
    )
    fig.update_layout(
        height=height,
        paper_bgcolor=T["bg"],
        plot_bgcolor=T["panel"],
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return fig
