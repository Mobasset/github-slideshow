"""
Virtual Drive Test (VDT) Tool — Main Dash Application

Entry point: python app.py
Open: http://localhost:8050
"""

import base64
import io
import json
import logging
import os
import sys

# Ensure parsers/metrics/visualization are importable from this directory
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, no_update
from dash.exceptions import PreventUpdate

# ── Internal imports ──────────────────────────────────────────────────────────
from parsers.l3_parser import parse_l3_bytes
from parsers.events_parser import parse_events_bytes
from parsers.gps_correlator import (
    parse_gps_bytes, correlate_gps_to_samples, apply_cell_centroid_positions
)
from metrics.kpi_engine import compute_summary_kpis, compute_per_cell_table
from metrics.classifier import add_all_classes, BIN_COLOR, RSRP_BINS
from visualization.dot_map import build_plotly_scatter, decimate_samples
from visualization.heatmap import build_plotly_density, get_map_center
from visualization.idw_grid import build_idw_raster, build_colorbar_trace
from visualization.hexbin import build_hexbin_layer, H3_AVAILABLE
from visualization.sector_overlay import build_sector_traces
from visualization.timeline import build_timeline_figure, build_cdf_figure
from export.kmz_exporter import export_kmz
from export.geojson_exporter import export_geojson, export_csv
from export.pdf_reporter import export_pdf_report
from demo_data import generate_demo_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vdt_app")

# ── Theme constants ────────────────────────────────────────────────────────────
BG = "#1a1a2e"
PANEL = "#16213e"
ACCENT = "#0f3460"
HIGHLIGHT = "#e94560"
TEXT = "#e0e0e0"
MUTED = "#888888"
BORDER = "#2a2a4a"

MAPBOX_STYLES = {
    "carto-positron": "CartoDB Positron",
    "carto-darkmatter": "CartoDB Dark Matter",
    "open-street-map": "OpenStreetMap",
    "white-bg": "Satellite (Esri)",
}

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="VDT Tool — Virtual Drive Test",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server


# ─── Helper: style utilities ──────────────────────────────────────────────────

def card(children, style=None):
    s = {
        "background": PANEL,
        "border": f"1px solid {BORDER}",
        "borderRadius": "6px",
        "padding": "10px 12px",
        "marginBottom": "8px",
    }
    if style:
        s.update(style)
    return html.Div(children, style=s)


def kpi_row(label, value, color=TEXT, mono=True):
    val_style = {
        "fontFamily": "monospace" if mono else "sans-serif",
        "fontSize": "15px",
        "fontWeight": "bold",
        "color": color,
    }
    return html.Div(
        [
            html.Span(label, style={"color": MUTED, "fontSize": "10px", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div(str(value), style=val_style),
        ],
        style={"marginBottom": "4px"},
    )


def section_header(text):
    return html.Div(
        text,
        style={
            "color": HIGHLIGHT,
            "fontSize": "11px",
            "fontWeight": "bold",
            "textTransform": "uppercase",
            "letterSpacing": "1px",
            "marginBottom": "6px",
            "marginTop": "4px",
            "borderBottom": f"1px solid {BORDER}",
            "paddingBottom": "3px",
        },
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def build_sidebar():
    return html.Div(
        [
            # Logo / title
            html.Div(
                [
                    html.Span("VDT", style={"color": HIGHLIGHT, "fontWeight": "900", "fontSize": "20px", "fontFamily": "monospace"}),
                    html.Span(" TOOL", style={"color": TEXT, "fontSize": "14px", "fontFamily": "monospace"}),
                    html.Div("Virtual Drive Test", style={"color": MUTED, "fontSize": "9px", "letterSpacing": "2px", "textTransform": "uppercase"}),
                ],
                style={"marginBottom": "12px", "paddingBottom": "8px", "borderBottom": f"1px solid {BORDER}"},
            ),

            # ── File Upload ──────────────────────────────────────────────────
            section_header("Input Files"),

            dcc.Upload(
                id="upload-l3",
                children=html.Div(["4G/5G RRC Log / .pcap"], style={"fontSize": "11px", "color": MUTED}),
                style={
                    "border": f"1px dashed {ACCENT}", "borderRadius": "4px",
                    "padding": "6px 8px", "marginBottom": "5px",
                    "textAlign": "center", "cursor": "pointer",
                    "background": "rgba(15,52,96,0.3)",
                },
                multiple=True,
            ),
            dcc.Upload(
                id="upload-events",
                children=html.Div(["Events CSV / Log (LTE · NR)"], style={"fontSize": "11px", "color": MUTED}),
                style={
                    "border": f"1px dashed {ACCENT}", "borderRadius": "4px",
                    "padding": "6px 8px", "marginBottom": "5px",
                    "textAlign": "center", "cursor": "pointer",
                    "background": "rgba(15,52,96,0.3)",
                },
                multiple=True,
            ),
            dcc.Upload(
                id="upload-gps",
                children=html.Div(["GPS CSV / KML / GPX"], style={"fontSize": "11px", "color": MUTED}),
                style={
                    "border": f"1px dashed {ACCENT}", "borderRadius": "4px",
                    "padding": "6px 8px", "marginBottom": "5px",
                    "textAlign": "center", "cursor": "pointer",
                    "background": "rgba(15,52,96,0.3)",
                },
                multiple=False,
            ),
            dcc.Upload(
                id="upload-cellmeta",
                children=html.Div(["Cell Metadata CSV"], style={"fontSize": "11px", "color": MUTED}),
                style={
                    "border": f"1px dashed {ACCENT}", "borderRadius": "4px",
                    "padding": "6px 8px", "marginBottom": "5px",
                    "textAlign": "center", "cursor": "pointer",
                    "background": "rgba(15,52,96,0.3)",
                },
                multiple=False,
            ),

            html.Div(
                [
                    html.Button(
                        "Load Demo Data",
                        id="btn-demo",
                        style={
                            "width": "100%", "padding": "6px",
                            "background": HIGHLIGHT, "color": "white",
                            "border": "none", "borderRadius": "4px",
                            "cursor": "pointer", "fontSize": "12px",
                            "fontWeight": "bold", "marginBottom": "4px",
                        },
                    ),
                    html.Button(
                        "Clear All",
                        id="btn-clear",
                        style={
                            "width": "100%", "padding": "5px",
                            "background": "transparent", "color": MUTED,
                            "border": f"1px solid {BORDER}", "borderRadius": "4px",
                            "cursor": "pointer", "fontSize": "11px",
                        },
                    ),
                ],
                style={"marginBottom": "10px"},
            ),

            # ── KPI Cards ──────────────────────────────────────────────────
            section_header("KPIs"),
            html.Div(id="kpi-panel"),

            # ── Layer Controls ─────────────────────────────────────────────
            section_header("Visualization Mode"),
            dcc.RadioItems(
                id="viz-mode",
                options=[
                    {"label": " Colored Dots", "value": "dots"},
                    {"label": " Heatmap", "value": "heatmap"},
                    {"label": " IDW Grid", "value": "idw"},
                    {"label": " HexBin", "value": "hexbin"},
                    {"label": " Cluster Events", "value": "cluster"},
                ],
                value="dots",
                labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "3px",
                            "fontSize": "12px", "color": TEXT, "cursor": "pointer"},
                style={"marginBottom": "8px"},
            ),

            section_header("Metric"),
            dcc.Dropdown(
                id="metric-select",
                options=[
                    {"label": "RSRP (dBm)", "value": "rsrp_dbm"},
                    {"label": "RSRQ (dB)", "value": "rsrq_db"},
                    {"label": "SINR (dB)", "value": "sinr_db"},
                    {"label": "Throughput DL (Mbps)", "value": "throughput_dl_mbps"},
                    {"label": "CQI", "value": "cqi"},
                ],
                value="rsrp_dbm",
                clearable=False,
                style={"fontSize": "11px", "background": PANEL, "marginBottom": "8px"},
            ),

            section_header("Overlays"),
            dcc.Checklist(
                id="overlay-checks",
                options=[
                    {"label": " GPS Track", "value": "track"},
                    {"label": " eNB Sites", "value": "sites"},
                    {"label": " Sector Wedges", "value": "sectors"},
                    {"label": " Event Markers", "value": "events"},
                ],
                value=["events"],
                labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "3px",
                            "fontSize": "12px", "color": TEXT, "cursor": "pointer"},
                style={"marginBottom": "8px"},
            ),

            section_header("Map Style"),
            dcc.Dropdown(
                id="map-style",
                options=[{"label": v, "value": k} for k, v in MAPBOX_STYLES.items()],
                value="carto-darkmatter",
                clearable=False,
                style={"fontSize": "11px", "marginBottom": "8px"},
            ),

            section_header("Filters"),
            html.Div(
                [
                    html.Label("RSRP Threshold (dBm)", style={"fontSize": "10px", "color": MUTED}),
                    dcc.Slider(
                        id="rsrp-threshold",
                        min=-140, max=-40, step=5,
                        value=-140,
                        marks={-140: {"label": "-140", "style": {"color": MUTED, "fontSize": "9px"}},
                               -80: {"label": "-80", "style": {"color": MUTED, "fontSize": "9px"}},
                               -40: {"label": "-40", "style": {"color": MUTED, "fontSize": "9px"}}},
                        tooltip={"always_visible": False, "placement": "bottom"},
                    ),
                ],
                style={"marginBottom": "8px"},
            ),

            dcc.Dropdown(
                id="cell-filter",
                options=[],
                value=[],
                multi=True,
                placeholder="Filter by Cell ID…",
                style={"fontSize": "11px", "marginBottom": "8px"},
            ),

            dcc.Dropdown(
                id="tech-filter",
                options=[
                    {"label": "LTE", "value": "LTE"},
                    {"label": "NR", "value": "NR"},
                    {"label": "2G", "value": "2G"},
                ],
                value=[],
                multi=True,
                placeholder="Filter by Technology…",
                style={"fontSize": "11px", "marginBottom": "8px"},
            ),

            # ── HexBin Config ──────────────────────────────────────────────
            html.Div(
                [
                    html.Label("H3 Resolution", style={"fontSize": "10px", "color": MUTED}),
                    dcc.Slider(
                        id="hex-resolution",
                        min=7, max=10, step=1,
                        value=8,
                        marks={7: {"label": "7", "style": {"color": MUTED, "fontSize": "9px"}},
                               8: {"label": "8", "style": {"color": MUTED, "fontSize": "9px"}},
                               9: {"label": "9", "style": {"color": MUTED, "fontSize": "9px"}},
                               10: {"label": "10", "style": {"color": MUTED, "fontSize": "9px"}}},
                    ),
                ],
                id="hexbin-config",
                style={"display": "none", "marginBottom": "8px"},
            ),

            # ── IDW Config ─────────────────────────────────────────────────
            html.Div(
                [
                    html.Label("Grid Resolution", style={"fontSize": "10px", "color": MUTED}),
                    dcc.Slider(
                        id="idw-resolution",
                        min=50, max=200, step=25,
                        value=100,
                        marks={50: {"label": "50", "style": {"color": MUTED, "fontSize": "9px"}},
                               100: {"label": "100", "style": {"color": MUTED, "fontSize": "9px"}},
                               200: {"label": "200", "style": {"color": MUTED, "fontSize": "9px"}}},
                    ),
                ],
                id="idw-config",
                style={"display": "none", "marginBottom": "8px"},
            ),

            # ── Export Buttons ─────────────────────────────────────────────
            section_header("Export"),
            html.Div(
                [
                    html.Button("CSV",       id="btn-export-csv",     style=_export_btn_style()),
                    html.Button("GeoJSON",   id="btn-export-geojson", style=_export_btn_style()),
                    html.Button("KMZ",       id="btn-export-kmz",     style=_export_btn_style()),
                    html.Button("HTML Map",  id="btn-export-html",    style=_export_btn_style()),
                    html.Button("PDF Report",id="btn-export-pdf",     style=_export_btn_style("#e94560")),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "4px"},
            ),
            dcc.Download(id="download-csv"),
            dcc.Download(id="download-geojson"),
            dcc.Download(id="download-kmz"),
            dcc.Download(id="download-html-map"),
            dcc.Download(id="download-pdf"),

            # Status bar
            html.Div(id="status-bar", style={"marginTop": "8px", "fontSize": "10px", "color": MUTED}),
        ],
        style={
            "width": "280px",
            "minWidth": "280px",
            "height": "100vh",
            "overflowY": "auto",
            "padding": "12px 10px",
            "background": BG,
            "borderRight": f"1px solid {BORDER}",
            "boxSizing": "border-box",
        },
    )


def _export_btn_style(bg=ACCENT):
    return {
        "background": bg, "color": "white", "border": "none",
        "borderRadius": "3px", "padding": "5px 8px",
        "cursor": "pointer", "fontSize": "11px", "width": "100%",
        "fontWeight": "bold",
    }


def _empty_map_figure():
    fig = go.Figure(go.Scattermapbox())
    fig.update_layout(
        mapbox={
            "style": "carto-darkmatter",
            "center": {"lat": 23.5880, "lon": 58.3829},
            "zoom": 12,
        },
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        annotations=[
            {
                "text": "Upload files or click <b>Load Demo Data</b>",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5,
                "showarrow": False,
                "font": {"size": 16, "color": "#888888"},
                "bgcolor": "rgba(26,26,46,0.7)",
            }
        ],
    )
    return fig


# ─── Main content ─────────────────────────────────────────────────────────────

def build_main_content():
    return html.Div(
        [
            # ── Map panel ─────────────────────────────────────────────────
            html.Div(
                [
                    dcc.Graph(
                        id="main-map",
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "scrollZoom": True,
                            "toImageButtonOptions": {"format": "png", "scale": 2},
                        },
                        style={"height": "100%", "width": "100%"},
                        figure=_empty_map_figure(),
                    ),
                ],
                style={"flex": "1", "minHeight": "0", "position": "relative"},
            ),

            # ── Bottom panel (collapsible) ────────────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Timeline", style={"fontSize": "11px", "color": HIGHLIGHT, "fontWeight": "bold"}),
                            html.Span(" — RSRP / SINR / Events", style={"fontSize": "10px", "color": MUTED}),
                            html.Button(
                                "▼",
                                id="btn-collapse-timeline",
                                style={
                                    "float": "right", "background": "transparent",
                                    "border": "none", "color": MUTED,
                                    "cursor": "pointer", "fontSize": "12px",
                                    "padding": "0 4px",
                                },
                            ),
                        ],
                        style={
                            "padding": "4px 10px", "background": ACCENT,
                            "borderBottom": f"1px solid {BORDER}",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="timeline-chart",
                                config={"displayModeBar": False},
                                style={"height": "220px"},
                                figure=build_timeline_figure(pd.DataFrame()),
                            ),
                        ],
                        id="timeline-container",
                    ),
                ],
                style={
                    "height": "260px",
                    "minHeight": "260px",
                    "background": PANEL,
                    "borderTop": f"1px solid {BORDER}",
                    "flexShrink": "0",
                },
            ),
        ],
        style={
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "overflow": "hidden",
            "minWidth": "0",
        },
    )


def build_detail_panel():
    """Right-side detail drawer (appears on map point click)."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Sample Detail", style={"fontWeight": "bold", "color": HIGHLIGHT, "fontSize": "12px"}),
                    html.Button(
                        "✕",
                        id="btn-close-detail",
                        style={
                            "float": "right", "background": "transparent",
                            "border": "none", "color": MUTED,
                            "cursor": "pointer", "fontSize": "14px",
                        },
                    ),
                ],
                style={"marginBottom": "8px", "borderBottom": f"1px solid {BORDER}", "paddingBottom": "6px"},
            ),
            html.Div(id="detail-content"),

            html.Hr(style={"borderColor": BORDER, "margin": "8px 0"}),
            section_header("Per-Cell Table"),
            html.Div(
                id="per-cell-table",
                style={"overflowX": "auto", "fontSize": "10px"},
            ),
        ],
        id="detail-panel",
        style={
            "width": "300px",
            "minWidth": "300px",
            "height": "100vh",
            "overflowY": "auto",
            "padding": "12px 10px",
            "background": BG,
            "borderLeft": f"1px solid {BORDER}",
            "boxSizing": "border-box",
            "display": "none",
        },
    )


# ─── App layout ───────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        # Data stores
        dcc.Store(id="store-samples", storage_type="memory"),
        dcc.Store(id="store-cellmeta", storage_type="memory"),
        dcc.Store(id="store-kpis", storage_type="memory"),
        dcc.Store(id="store-time-filter", storage_type="memory"),
        dcc.Store(id="store-cell-filter", storage_type="memory"),

        # Main layout row
        html.Div(
            [
                build_sidebar(),
                build_main_content(),
                build_detail_panel(),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "height": "100vh",
                "overflow": "hidden",
                "background": BG,
                "color": TEXT,
                "fontFamily": "Inter, Roboto, sans-serif",
            },
        ),
    ]
)


# ─── Utility functions ────────────────────────────────────────────────────────

def _df_from_store(data) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    try:
        return pd.read_json(io.StringIO(data), orient="split")
    except Exception:
        return pd.DataFrame()


def _df_to_store(df: pd.DataFrame) -> str:
    if df.empty:
        return "{}"
    # Convert timestamps to strings for JSON serialization
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["datetime64[ns, UTC]", "datetimetz"]).columns:
        df_copy[col] = df_copy[col].astype(str)
    return df_copy.to_json(orient="split", date_format="iso")


def _decode_upload(contents: str, filename: str) -> bytes:
    """Decode base64 upload content to raw bytes."""
    content_type, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)



def _apply_filters(
    df: pd.DataFrame,
    rsrp_thresh: float,
    cell_ids: list,
    techs: list,
    time_range: tuple = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series([True] * len(df), index=df.index)

    if "rsrp_dbm" in df.columns and rsrp_thresh > -140:
        mask &= df["rsrp_dbm"].fillna(-200) >= rsrp_thresh

    if cell_ids:
        mask &= df["cell_id"].astype(str).isin([str(c) for c in cell_ids])

    if techs and "earfcn" in df.columns:
        tech_mask = pd.Series([False] * len(df), index=df.index)
        earfcn = df["earfcn"].fillna(0)
        if "LTE" in techs:
            # LTE EARFCN bands: 1–262143 (B1–B86, excluding NR ranges)
            tech_mask |= (earfcn > 0) & (earfcn < 600000)
        if "NR" in techs:
            # NR ARFCN: 600000–3279165
            tech_mask |= earfcn >= 600000
        if "2G" in techs:
            # GSM ARFCN: 0–1023
            tech_mask |= (earfcn >= 0) & (earfcn <= 1023)
        mask &= tech_mask

    if time_range and len(time_range) == 2 and "timestamp_ms" in df.columns:
        ts_min, ts_max = time_range
        if ts_min is not None and ts_max is not None:
            mask &= (df["timestamp_ms"] >= ts_min) & (df["timestamp_ms"] <= ts_max)

    return df[mask]


def _build_map_figure(
    df: pd.DataFrame,
    cell_meta_df: pd.DataFrame,
    viz_mode: str,
    metric: str,
    overlays: list,
    map_style: str,
    hex_res: int,
    idw_res: int,
) -> go.Figure:
    """Build the main map Plotly figure from current state."""

    traces = []
    layout_images = []

    center = get_map_center(df) if not df.empty else (23.5880, 58.3829)

    # ── GPS Track polyline ─────────────────────────────────────────────────
    if "track" in overlays and not df.empty and "latitude" in df.columns:
        mask = df["latitude"].notna() & df["longitude"].notna()
        track = df[mask].sort_values("timestamp_ms")
        if len(track) >= 2:
            track_dec = decimate_samples(track, threshold=2000)
            traces.append(
                {
                    "type": "scattermapbox",
                    "lat": track_dec["latitude"].tolist(),
                    "lon": track_dec["longitude"].tolist(),
                    "mode": "lines",
                    "line": {"color": "#ffffaa", "width": 1.5},
                    "opacity": 0.5,
                    "name": "GPS Track",
                    "hoverinfo": "skip",
                }
            )

    # ── Main visualization layer ───────────────────────────────────────────
    if not df.empty:

        if viz_mode == "dots":
            scatter = build_plotly_scatter(df)
            if scatter:
                traces.append(scatter)

        elif viz_mode == "heatmap":
            density = build_plotly_density(df, metric=metric, radius=15)
            if density:
                traces.append(density)

        elif viz_mode == "idw":
            idw_result = build_idw_raster(df, metric=metric, grid_resolution=idw_res)
            if idw_result:
                lat_range = idw_result["lat_range"]
                lon_range = idw_result["lon_range"]
                layout_images.append(
                    {
                        "source": f"data:image/png;base64,{idw_result['png_b64']}",
                        "xref": "x", "yref": "y",
                        "x": lon_range[0], "y": lat_range[1],
                        "sizex": lon_range[1] - lon_range[0],
                        "sizey": lat_range[1] - lat_range[0],
                        "sizing": "stretch",
                        "opacity": 0.75,
                        "layer": "above",
                    }
                )
                cb_trace = build_colorbar_trace(
                    idw_result["vmin"], idw_result["vmax"],
                    idw_result["colorscale"], metric
                )
                traces.append(cb_trace)

        elif viz_mode == "hexbin" and H3_AVAILABLE:
            hex_result = build_hexbin_layer(df, metric=metric, resolution=hex_res)
            if hex_result:
                traces.append(hex_result["trace"])

        elif viz_mode == "cluster":
            # Event markers per type
            event_styles = {
                "INTERNAL_HANDOVER_FAILURE": ("#e74c3c", "✕", "HO Fail"),
                "RRC_RLF":                   ("#8e44ad", "☠", "RLF"),
                "NR_SCG_FAILURE":            ("#f39c12", "⚠", "SCG Fail"),
                "INTERNAL_HANDOVER_SUCCESS": ("#2ecc71", "✓", "HO OK"),
                "A3_TRIGGER":               ("#3498db", "▶", "A3"),
                "A5_TRIGGER":               ("#e67e22", "◀", "A5"),
            }
            if "event_type" in df.columns:
                mask_gps = df["latitude"].notna() & df["longitude"].notna()
                for etype, (color, sym, label) in event_styles.items():
                    emask = mask_gps & (df["event_type"] == etype)
                    if not emask.any():
                        continue
                    edf = df[emask]
                    traces.append(
                        {
                            "type": "scattermapbox",
                            "lat": edf["latitude"].tolist(),
                            "lon": edf["longitude"].tolist(),
                            "mode": "markers+text",
                            "marker": {"size": 14, "color": color, "opacity": 0.9},
                            "text": [sym] * len(edf),
                            "textfont": {"size": 10, "color": "white"},
                            "name": label,
                            "hovertemplate": f"<b>{label}</b><br>%{{lat:.5f}}, %{{lon:.5f}}<extra></extra>",
                        }
                    )
        # Fall back to dots if no valid mode
        if not traces and viz_mode not in ("idw",):
            scatter = build_plotly_scatter(df)
            if scatter:
                traces.append(scatter)

    # ── Sector wedge overlays ──────────────────────────────────────────────
    if "sectors" in overlays and not cell_meta_df.empty:
        sector_traces = build_sector_traces(cell_meta_df, df, color_by=metric)
        traces.extend(sector_traces)

    # ── Event overlay ──────────────────────────────────────────────────────
    if "events" in overlays and not df.empty and "event_type" in df.columns and viz_mode != "cluster":
        event_mask = (
            df["latitude"].notna()
            & df["longitude"].notna()
            & df["event_type"].isin(["INTERNAL_HANDOVER_FAILURE", "RRC_RLF", "NR_SCG_FAILURE"])
        )
        edf = df[event_mask]
        if not edf.empty:
            colors_map = {
                "INTERNAL_HANDOVER_FAILURE": "#e74c3c",
                "RRC_RLF": "#8e44ad",
                "NR_SCG_FAILURE": "#f39c12",
            }
            ec = edf["event_type"].map(colors_map).fillna("#888888").tolist()
            traces.append(
                {
                    "type": "scattermapbox",
                    "lat": edf["latitude"].tolist(),
                    "lon": edf["longitude"].tolist(),
                    "mode": "markers",
                    "marker": {"size": 10, "color": ec, "opacity": 0.95, "symbol": "circle"},
                    "name": "Critical Events",
                    "hovertemplate": "<b>%{customdata}</b><extra></extra>",
                    "customdata": edf["event_type"].tolist(),
                }
            )

    # ── eNB site markers ───────────────────────────────────────────────────
    if "sites" in overlays and not cell_meta_df.empty and "sectors" not in overlays:
        col_map = {}
        for c in cell_meta_df.columns:
            cl = c.lower().strip()
            if cl in ("lat", "latitude"):
                col_map[c] = "lat"
            elif cl in ("lon", "longitude"):
                col_map[c] = "lon"
            elif cl in ("cellid", "cell_id"):
                col_map[c] = "cell_id"
            elif cl in ("site_name", "sitename"):
                col_map[c] = "site_name"
        meta = cell_meta_df.rename(columns=col_map)
        if "lat" in meta.columns and "lon" in meta.columns:
            site_groups = meta.drop_duplicates(subset=["lat", "lon"])
            traces.append(
                {
                    "type": "scattermapbox",
                    "lat": site_groups["lat"].tolist(),
                    "lon": site_groups["lon"].tolist(),
                    "mode": "markers",
                    "marker": {"size": 14, "color": HIGHLIGHT, "symbol": "triangle"},
                    "text": site_groups.get("site_name", site_groups.get("cell_id", pd.Series())).astype(str).tolist(),
                    "hovertemplate": "<b>%{text}</b><extra></extra>",
                    "name": "eNB/gNB Sites",
                }
            )

    # ── Assemble figure ────────────────────────────────────────────────────
    go_traces = []
    for t in traces:
        ttype = t.pop("type", "scattermapbox")
        if ttype == "scattermapbox":
            go_traces.append(go.Scattermapbox(**t))
        elif ttype == "densitymapbox":
            go_traces.append(go.Densitymapbox(**t))
        elif ttype == "choroplethmapbox":
            go_traces.append(go.Choroplethmapbox(**t))
        else:
            go_traces.append(go.Scattermapbox(**t))

    if not go_traces:
        go_traces = [go.Scattermapbox(lat=[], lon=[])]

    fig = go.Figure(data=go_traces)

    zoom = 12 if df.empty else _estimate_zoom(df)

    fig.update_layout(
        mapbox={
            "style": map_style,
            "center": {"lat": center[0], "lon": center[1]},
            "zoom": zoom,
        },
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={
            "bgcolor": "rgba(22,33,62,0.85)",
            "bordercolor": BORDER,
            "borderwidth": 1,
            "font": {"color": TEXT, "size": 10},
            "x": 0.01, "y": 0.99,
            "xanchor": "left", "yanchor": "top",
        },
        uirevision="map",
    )

    if layout_images:
        fig.update_layout(images=layout_images)

    return fig


def _estimate_zoom(df: pd.DataFrame) -> float:
    """Estimate appropriate Mapbox zoom level from GPS extent."""
    mask = df["latitude"].notna() & df["longitude"].notna()
    if not mask.any():
        return 12
    lat_span = df.loc[mask, "latitude"].max() - df.loc[mask, "latitude"].min()
    lon_span = df.loc[mask, "longitude"].max() - df.loc[mask, "longitude"].min()
    span = max(lat_span, lon_span)
    if span < 0.005:
        return 14
    if span < 0.02:
        return 13
    if span < 0.1:
        return 12
    if span < 0.5:
        return 10
    return 8


def _build_kpi_panel(kpis: dict) -> list:
    """Build KPI card children from kpis dict."""
    if not kpis:
        return [html.Div("No data loaded.", style={"color": MUTED, "fontSize": "11px"})]

    def _rsrp_color(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return MUTED
        if v >= -80:
            return "#2ecc71"
        if v >= -90:
            return "#a8d08d"
        if v >= -100:
            return "#f0a500"
        if v >= -110:
            return "#e06000"
        return "#c0392b"

    def _fmt(v, fmt=".1f"):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "N/A"
        return format(v, fmt)

    rsrp_med = kpis.get("rsrp_median")

    items = [
        kpi_row("Samples", f"{kpis.get('n_samples', 0):,}"),
        kpi_row("GPS Match", f"{_fmt(kpis.get('gps_match_pct'))}%"),
        kpi_row("Duration", f"{_fmt(kpis.get('duration_min'))} min"),
        kpi_row("RSRP P50 / P5 / P95",
                f"{_fmt(rsrp_med)} / {_fmt(kpis.get('rsrp_p5'))} / {_fmt(kpis.get('rsrp_p95'))} dBm",
                color=_rsrp_color(rsrp_med)),
        kpi_row("RSRQ Median", f"{_fmt(kpis.get('rsrq_median'))} dB"),
        kpi_row("SINR Median", f"{_fmt(kpis.get('sinr_median'))} dB"),
        kpi_row("HO Success Rate",
                f"{_fmt(kpis.get('ho_sr'))}% ({kpis.get('ho_success', 0)}/{kpis.get('ho_attempt', 0)})",
                color="#2ecc71" if (kpis.get("ho_sr") or 0) >= 90 else "#f0a500"),
        kpi_row("RLF Count", f"{kpis.get('rlf_count', 0)} ({_fmt(kpis.get('rlf_rate_per_10k'))}/10k)",
                color="#e74c3c" if kpis.get("rlf_count", 0) > 0 else TEXT),
        kpi_row("SCG Failures", f"{kpis.get('scg_failure_count', 0)}",
                color="#f39c12" if kpis.get("scg_failure_count", 0) > 0 else TEXT),
        kpi_row("DL Throughput", f"{_fmt(kpis.get('dl_throughput_median_mbps'), '.2f')} Mbps"),
    ]

    # Coverage donut chart
    cov = kpis.get("coverage_bins", {})
    if cov and sum(cov.values()) > 0:
        labels = list(cov.keys())
        values = list(cov.values())
        pie_colors = [BIN_COLOR.get(l, "#888") for l in labels]
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                marker={"colors": pie_colors},
                hole=0.55,
                textinfo="percent",
                textfont={"size": 9, "color": "white"},
                hovertemplate="%{label}: %{value}<br>%{percent}<extra></extra>",
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend={"font": {"color": TEXT, "size": 8}, "bgcolor": "rgba(0,0,0,0)"},
            margin={"l": 0, "r": 0, "t": 5, "b": 0},
            height=130,
            annotations=[{
                "text": "RSRP",
                "x": 0.5, "y": 0.5,
                "font": {"size": 10, "color": TEXT},
                "showarrow": False,
            }],
        )
        items.append(
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False},
                style={"height": "130px", "marginTop": "4px"},
            )
        )

    return items


# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-samples", "data"),
    Output("store-cellmeta", "data"),
    Output("store-kpis", "data"),
    Output("cell-filter", "options"),
    Output("status-bar", "children"),
    Input("btn-demo", "n_clicks"),
    Input("btn-clear", "n_clicks"),
    Input("upload-l3", "contents"),
    Input("upload-events", "contents"),
    Input("upload-gps", "contents"),
    Input("upload-cellmeta", "contents"),
    State("upload-l3", "filename"),
    State("upload-events", "filename"),
    State("upload-gps", "filename"),
    State("upload-cellmeta", "filename"),
    State("store-samples", "data"),
    State("store-cellmeta", "data"),
    prevent_initial_call=True,
)
def handle_data_load(
    demo_clicks, clear_clicks,
    l3_contents, ev_contents, gps_contents, meta_contents,
    l3_fnames, ev_fnames, gps_fname, meta_fname,
    existing_samples, existing_meta,
):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "btn-clear":
        return "{}", "{}", "{}", [], "All data cleared."

    if trigger == "btn-demo":
        df, cell_meta = generate_demo_data()
        kpis = compute_summary_kpis(df)
        cell_options = [{"label": str(c), "value": c} for c in sorted(df["cell_id"].unique())]
        status = f"Demo data loaded: {len(df):,} samples, {len(cell_meta)} cells."
        return _df_to_store(df), _df_to_store(cell_meta), json.dumps(kpis), cell_options, status

    # File uploads — accumulate into existing data
    frames = []
    existing_df = _df_from_store(existing_samples)
    existing_meta_df = _df_from_store(existing_meta)

    gps_df = pd.DataFrame()
    cell_meta_df = existing_meta_df.copy()
    status_parts = []

    # Cell metadata
    if trigger == "upload-cellmeta" and meta_contents:
        try:
            raw = _decode_upload(meta_contents, meta_fname or "meta.csv")
            cell_meta_df = pd.read_csv(io.BytesIO(raw))
            status_parts.append(f"Metadata: {len(cell_meta_df)} cells")
        except Exception as e:
            logger.warning("Meta CSV error: %s", e)

    # GPS
    if trigger == "upload-gps" and gps_contents:
        try:
            raw = _decode_upload(gps_contents, gps_fname or "gps.csv")
            gps_df = parse_gps_bytes(raw, filename=gps_fname or "gps.csv")
            status_parts.append(f"GPS: {len(gps_df)} points")
        except Exception as e:
            logger.warning("GPS parse error: %s", e)

    # L3 logs (4G/5G RRC)
    if trigger == "upload-l3" and l3_contents:
        for content, fname in zip(l3_contents, l3_fnames or [""]):
            try:
                raw = _decode_upload(content, fname)
                f = parse_l3_bytes(raw, filename=fname)
                if not f.empty:
                    frames.append(f)
                    status_parts.append(f"{fname}: {len(f)} records")
            except Exception as e:
                logger.warning("L3 error %s: %s", fname, e)

    # Events
    if trigger == "upload-events" and ev_contents:
        for content, fname in zip(ev_contents, ev_fnames or [""]):
            try:
                raw = _decode_upload(content, fname)
                f = parse_events_bytes(raw, filename=fname)
                if not f.empty:
                    frames.append(f)
                    status_parts.append(f"{fname}: {len(f)} records")
            except Exception as e:
                logger.warning("Events error %s: %s", fname, e)

    # Merge new frames with existing
    if frames:
        new_df = pd.concat(frames, ignore_index=True)
        for col in ["rsrp_dbm", "rsrq_db", "sinr_db", "ho_attempt", "ho_success",
                    "ho_failure", "rlf_flag", "rab_setup", "scg_failure", "a3_trigger", "a5_trigger"]:
            if col not in new_df.columns:
                new_df[col] = 0 if col not in ("rsrp_dbm", "rsrq_db", "sinr_db") else np.nan
        if "timestamp_ms" not in new_df.columns:
            new_df["timestamp_ms"] = 0
        new_df = add_all_classes(new_df)

        # Correlate GPS if available
        if not gps_df.empty:
            new_df = correlate_gps_to_samples(new_df, gps_df)
        elif not cell_meta_df.empty and ("latitude" not in new_df.columns or new_df["latitude"].isna().all()):
            new_df = apply_cell_centroid_positions(new_df, cell_meta_df)

        if not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
    else:
        combined = existing_df

    if combined.empty:
        return "{}", _df_to_store(cell_meta_df), "{}", [], " | ".join(status_parts) or "No data parsed."

    kpis = compute_summary_kpis(combined)
    cell_options = [{"label": str(c), "value": c} for c in sorted(combined["cell_id"].dropna().unique())]
    status = f"Total: {len(combined):,} samples | " + " | ".join(status_parts)

    return _df_to_store(combined), _df_to_store(cell_meta_df), json.dumps(kpis), cell_options, status


@app.callback(
    Output("kpi-panel", "children"),
    Input("store-kpis", "data"),
)
def update_kpi_panel(kpis_json):
    if not kpis_json or kpis_json == "{}":
        return [html.Div("No data loaded.", style={"color": MUTED, "fontSize": "11px"})]
    try:
        kpis = json.loads(kpis_json)
    except Exception:
        return [html.Div("KPI error.", style={"color": MUTED})]
    return _build_kpi_panel(kpis)


@app.callback(
    Output("main-map", "figure"),
    Input("store-samples", "data"),
    Input("store-cellmeta", "data"),
    Input("viz-mode", "value"),
    Input("metric-select", "value"),
    Input("overlay-checks", "value"),
    Input("map-style", "value"),
    Input("rsrp-threshold", "value"),
    Input("cell-filter", "value"),
    Input("tech-filter", "value"),
    Input("hex-resolution", "value"),
    Input("idw-resolution", "value"),
    Input("store-time-filter", "data"),
)
def update_map(
    samples_data, meta_data, viz_mode, metric, overlays,
    map_style, rsrp_thresh, cell_ids, techs, hex_res, idw_res, time_filter,
):
    df = _df_from_store(samples_data)
    cell_meta = _df_from_store(meta_data)

    if df.empty:
        return _empty_map_figure()

    time_range = None
    if time_filter:
        try:
            tr = json.loads(time_filter)
            if tr.get("min") and tr.get("max"):
                time_range = (tr["min"], tr["max"])
        except Exception:
            pass

    df = _apply_filters(df, rsrp_thresh or -140, cell_ids or [], techs or [], time_range)

    return _build_map_figure(
        df, cell_meta, viz_mode or "dots", metric or "rsrp_dbm",
        overlays or [], map_style or "carto-darkmatter",
        hex_res or 8, idw_res or 100,
    )


@app.callback(
    Output("timeline-chart", "figure"),
    Input("store-samples", "data"),
    Input("store-time-filter", "data"),
)
def update_timeline(samples_data, time_filter):
    df = _df_from_store(samples_data)
    if df.empty:
        return build_timeline_figure(pd.DataFrame())

    time_range = None
    if time_filter:
        try:
            tr = json.loads(time_filter)
            time_range = (tr.get("min"), tr.get("max"))
        except Exception:
            pass

    return build_timeline_figure(df, time_range=time_range)


@app.callback(
    Output("hexbin-config", "style"),
    Output("idw-config", "style"),
    Input("viz-mode", "value"),
)
def toggle_config_panels(viz_mode):
    hidden = {"display": "none", "marginBottom": "8px"}
    visible = {"display": "block", "marginBottom": "8px"}
    return (
        visible if viz_mode == "hexbin" else hidden,
        visible if viz_mode == "idw" else hidden,
    )


@app.callback(
    Output("detail-panel", "style"),
    Output("detail-content", "children"),
    Output("per-cell-table", "children"),
    Input("main-map", "clickData"),
    Input("btn-close-detail", "n_clicks"),
    State("store-samples", "data"),
    State("store-kpis", "data"),
    prevent_initial_call=True,
)
def handle_map_click(click_data, close_clicks, samples_data, kpis_json):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    base_style = {
        "width": "300px", "minWidth": "300px", "height": "100vh",
        "overflowY": "auto", "padding": "12px 10px",
        "background": BG, "borderLeft": f"1px solid {BORDER}",
        "boxSizing": "border-box",
    }

    if trigger == "btn-close-detail":
        return {**base_style, "display": "none"}, [], []

    if not click_data:
        raise PreventUpdate

    # Show detail panel
    df = _df_from_store(samples_data)
    detail_items = []

    if click_data and "points" in click_data:
        pt = click_data["points"][0]
        lat = pt.get("lat")
        lon = pt.get("lon")
        custom = pt.get("customdata", [])

        detail_items = [
            kpi_row("Latitude", f"{lat:.5f}" if lat else "N/A"),
            kpi_row("Longitude", f"{lon:.5f}" if lon else "N/A"),
        ]

        if isinstance(custom, list) and len(custom) >= 5:
            rsrp, rsrq, sinr, cell_id, event_type = custom[:5]
            detail_items += [
                kpi_row("RSRP", f"{rsrp:.1f} dBm" if rsrp else "N/A",
                        color="#2ecc71" if rsrp and rsrp >= -80 else "#e06000"),
                kpi_row("RSRQ", f"{rsrq:.1f} dB" if rsrq else "N/A"),
                kpi_row("SINR", f"{sinr:.1f} dB" if sinr else "N/A"),
                kpi_row("Cell ID", str(cell_id) if cell_id else "N/A"),
                kpi_row("Event", str(event_type) if event_type else "—"),
            ]

    # Per-cell table
    cell_table = []
    if not df.empty:
        per_cell = compute_per_cell_table(df)
        if not per_cell.empty:
            display_cols = [c for c in ["Cell_ID", "pci", "sample_count", "rsrp_median",
                                          "ho_sr", "rlf_rate"] if c in per_cell.columns]
            cell_table = [
                dash_table.DataTable(
                    data=per_cell[display_cols].round(1).to_dict("records"),
                    columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display_cols],
                    sort_action="native",
                    filter_action="native",
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": ACCENT, "color": TEXT,
                        "fontWeight": "bold", "fontSize": "10px",
                    },
                    style_cell={
                        "backgroundColor": PANEL, "color": TEXT,
                        "fontSize": "10px", "padding": "4px 6px",
                        "border": f"1px solid {BORDER}",
                        "fontFamily": "monospace",
                    },
                    style_data_conditional=[
                        {
                            "if": {"filter_query": "{rsrp_median} < -100"},
                            "color": "#e06000",
                        },
                        {
                            "if": {"filter_query": "{rsrp_median} < -110"},
                            "color": "#c0392b",
                        },
                    ],
                )
            ]

    return {**base_style, "display": "block"}, detail_items, cell_table


@app.callback(
    Output("download-csv", "data"),
    Input("btn-export-csv", "n_clicks"),
    State("store-samples", "data"),
    prevent_initial_call=True,
)
def export_csv_download(n, samples_data):
    df = _df_from_store(samples_data)
    if df.empty:
        raise PreventUpdate
    csv_bytes = export_csv(df)
    return dcc.send_bytes(csv_bytes, "vdt_export.csv")


@app.callback(
    Output("download-geojson", "data"),
    Input("btn-export-geojson", "n_clicks"),
    State("store-samples", "data"),
    prevent_initial_call=True,
)
def export_geojson_download(n, samples_data):
    df = _df_from_store(samples_data)
    if df.empty:
        raise PreventUpdate
    gj_bytes = export_geojson(df)
    return dcc.send_bytes(gj_bytes, "vdt_export.geojson")


@app.callback(
    Output("download-kmz", "data"),
    Input("btn-export-kmz", "n_clicks"),
    State("store-samples", "data"),
    State("store-cellmeta", "data"),
    prevent_initial_call=True,
)
def export_kmz_download(n, samples_data, meta_data):
    df = _df_from_store(samples_data)
    if df.empty:
        raise PreventUpdate
    cell_meta = _df_from_store(meta_data)
    kmz_bytes = export_kmz(df, cell_meta_df=cell_meta if not cell_meta.empty else None)
    return dcc.send_bytes(kmz_bytes, "vdt_export.kmz")


@app.callback(
    Output("download-pdf", "data"),
    Input("btn-export-pdf", "n_clicks"),
    State("store-samples", "data"),
    State("store-kpis", "data"),
    prevent_initial_call=True,
)
def export_pdf_download(n, samples_data, kpis_json):
    df = _df_from_store(samples_data)
    if df.empty:
        raise PreventUpdate
    kpis = json.loads(kpis_json) if kpis_json and kpis_json != "{}" else {}
    per_cell = compute_per_cell_table(df) if not df.empty else None
    pdf_bytes = export_pdf_report(kpis, per_cell_df=per_cell)
    if not pdf_bytes:
        raise PreventUpdate
    return dcc.send_bytes(pdf_bytes, "vdt_report.pdf")


@app.callback(
    Output("store-time-filter", "data"),
    Input("timeline-chart", "relayoutData"),
    State("store-samples", "data"),
    prevent_initial_call=True,
)
def sync_timeline_brush(relayout_data, samples_data):
    """Translate Plotly rangeslider / zoom gestures into a time filter stored in ms."""
    if not relayout_data:
        raise PreventUpdate

    # Plotly emits xaxis.range[0]/[1] when the user zooms/brushes the chart
    x0 = relayout_data.get("xaxis.range[0]") or relayout_data.get("xaxis.range", [None, None])[0]
    x1 = relayout_data.get("xaxis.range[1]") or (relayout_data.get("xaxis.range", [None, None]) + [None])[1]

    # If user double-clicked to auto-range, clear the filter
    if relayout_data.get("xaxis.autorange") or relayout_data.get("autosize"):
        return json.dumps({})

    if x0 is None or x1 is None:
        raise PreventUpdate

    try:
        ts_min = int(pd.Timestamp(x0).timestamp() * 1000)
        ts_max = int(pd.Timestamp(x1).timestamp() * 1000)
    except Exception:
        raise PreventUpdate

    return json.dumps({"min": ts_min, "max": ts_max})


@app.callback(
    Output("timeline-container", "style"),
    Output("btn-collapse-timeline", "children"),
    Input("btn-collapse-timeline", "n_clicks"),
    State("timeline-container", "style"),
    prevent_initial_call=True,
)
def toggle_timeline(n_clicks, current_style):
    if current_style and current_style.get("display") == "none":
        return {"display": "block"}, "▼"
    return {"display": "none"}, "▶"


@app.callback(
    Output("download-html-map", "data"),
    Input("btn-export-html", "n_clicks"),
    State("store-samples", "data"),
    State("store-cellmeta", "data"),
    State("metric-select", "value"),
    prevent_initial_call=True,
)
def export_html_map(n, samples_data, meta_data, metric):
    """Build a self-contained Folium HTML map and send as download."""
    df = _df_from_store(samples_data)
    if df.empty:
        raise PreventUpdate

    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster
    except ImportError:
        raise PreventUpdate

    cell_meta = _df_from_store(meta_data)
    center = get_map_center(df)

    fmap = folium.Map(
        location=[center[0], center[1]],
        zoom_start=13,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # ── Colored dot layer ──────────────────────────────────────────────────
    dot_group = folium.FeatureGroup(name="RSRP Samples", show=True)
    mask = df["latitude"].notna() & df["longitude"].notna()
    df_valid = df[mask]
    if len(df_valid) > 5000:
        df_valid = df_valid.sample(n=5000, random_state=42)

    for _, row in df_valid.iterrows():
        color = row.get("rsrp_color", "#888888")
        rsrp = row.get("rsrp_dbm", float("nan"))
        cell_id = row.get("cell_id", "")
        tooltip = (
            f"RSRP: {rsrp:.1f} dBm | Cell: {cell_id}"
            if not (isinstance(rsrp, float) and np.isnan(rsrp))
            else f"Cell: {cell_id}"
        )
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=0,
            tooltip=tooltip,
        ).add_to(dot_group)
    dot_group.add_to(fmap)

    # ── Heatmap layer ──────────────────────────────────────────────────────
    from visualization.heatmap import build_heatmap_data
    heat_data = build_heatmap_data(df, metric=metric or "rsrp_dbm")
    if heat_data:
        heat_group = folium.FeatureGroup(name=f"Heatmap ({metric})", show=False)
        HeatMap(heat_data, radius=15, blur=20, min_opacity=0.3).add_to(heat_group)
        heat_group.add_to(fmap)

    # ── GPS track ─────────────────────────────────────────────────────────
    if len(df_valid) >= 2:
        track = df_valid.sort_values("timestamp_ms")
        coords = list(zip(track["latitude"].tolist(), track["longitude"].tolist()))
        if len(coords) > 2000:
            step = len(coords) // 2000
            coords = coords[::step]
        track_group = folium.FeatureGroup(name="GPS Track", show=True)
        folium.PolyLine(coords, color="#ffffaa", weight=2, opacity=0.5).add_to(track_group)
        track_group.add_to(fmap)

    # ── Event markers ─────────────────────────────────────────────────────
    event_colors = {
        "INTERNAL_HANDOVER_FAILURE": "red",
        "RRC_RLF": "darkred",
        "NR_SCG_FAILURE": "orange",
        "INTERNAL_HANDOVER_SUCCESS": "green",
        "A3_TRIGGER": "blue",
        "A5_TRIGGER": "cadetblue",
    }
    if "event_type" in df.columns:
        evt_group = folium.FeatureGroup(name="Events", show=True)
        evt_cluster = MarkerCluster().add_to(evt_group)
        emask = (
            df["latitude"].notna() & df["longitude"].notna()
            & df["event_type"].isin(event_colors.keys())
        )
        for _, row in df[emask].iterrows():
            etype = row["event_type"]
            folium.Marker(
                location=[float(row["latitude"]), float(row["longitude"])],
                icon=folium.Icon(color=event_colors.get(etype, "gray"), icon="info-sign"),
                tooltip=etype.replace("INTERNAL_", ""),
            ).add_to(evt_cluster)
        evt_group.add_to(fmap)

    # ── Sector wedges ─────────────────────────────────────────────────────
    if not cell_meta.empty:
        from visualization.sector_overlay import _build_wedge_polygon, DEFAULT_HPBW_DEG, DEFAULT_ISD_M
        col_map = {}
        for c in cell_meta.columns:
            cl = c.lower().strip()
            if cl in ("cellid", "cell_id"):
                col_map[c] = "cell_id"
            elif cl in ("lat", "latitude"):
                col_map[c] = "lat"
            elif cl in ("lon", "longitude"):
                col_map[c] = "lon"
            elif cl in ("azimuth", "az"):
                col_map[c] = "azimuth"
            elif cl in ("site_name", "sitename"):
                col_map[c] = "site_name"
        meta = cell_meta.rename(columns=col_map)
        if all(c in meta.columns for c in ["lat", "lon", "azimuth"]):
            sector_group = folium.FeatureGroup(name="Sector Wedges", show=True)
            for _, row in meta.iterrows():
                polygon = _build_wedge_polygon(
                    float(row["lat"]), float(row["lon"]),
                    float(row["azimuth"]), DEFAULT_HPBW_DEG, DEFAULT_ISD_M,
                )
                folium.Polygon(
                    locations=polygon,
                    color="#e94560",
                    fill=True,
                    fill_color="#0f3460",
                    fill_opacity=0.25,
                    weight=1.5,
                    tooltip=str(row.get("site_name", row.get("cell_id", ""))),
                ).add_to(sector_group)
            sector_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    html_str = fmap._repr_html_()
    return dcc.send_string(html_str, "vdt_map.html")


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("VDT_DEBUG", "0") == "1"
    logger.info("Starting VDT Tool on http://localhost:%d", port)
    app.run(debug=debug, host="0.0.0.0", port=port)
