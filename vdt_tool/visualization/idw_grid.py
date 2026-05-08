"""
IDW (Inverse Distance Weighting) interpolated grid layer.

Generates a raster PNG overlay from scatter signal samples using
scipy.interpolate.griddata, then encodes it as a base64 PNG for
Plotly layout image overlay or Folium ImageOverlay.
"""

import base64
import io
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLORSCALE_MAP = {
    "rsrp_dbm": "RdYlGn",
    "rsrq_db": "RdYlGn",
    "sinr_db": "RdBu",
    "throughput_dl_mbps": "plasma",
    "cqi": "viridis",
}

VRANGE_MAP = {
    "rsrp_dbm": (-140, -40),
    "rsrq_db": (-19.5, 0),
    "sinr_db": (-23, 40),
    "throughput_dl_mbps": (0, 150),
    "cqi": (0, 15),
}


def _get_bounds(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) with 10% padding."""
    lat_min = df["latitude"].min()
    lat_max = df["latitude"].max()
    lon_min = df["longitude"].min()
    lon_max = df["longitude"].max()
    lat_pad = (lat_max - lat_min) * 0.10 + 0.001
    lon_pad = (lon_max - lon_min) * 0.10 + 0.001
    return (
        lat_min - lat_pad,
        lat_max + lat_pad,
        lon_min - lon_pad,
        lon_max + lon_pad,
    )


def build_idw_raster(
    df: pd.DataFrame,
    metric: str = "rsrp_dbm",
    grid_resolution: int = 100,
    max_points: int = 5000,
) -> Optional[dict]:
    """
    Interpolate signal metric to a regular grid using scipy griddata.

    Returns dict with:
      - png_b64: base64-encoded PNG image string
      - bounds: [[lat_sw, lon_sw], [lat_ne, lon_ne]] for Folium overlay
      - lat_range: [lat_min, lat_max]
      - lon_range: [lon_min, lon_max]
      - colorscale: plotly colorscale name
      - vmin, vmax: value range used for color mapping
    """
    try:
        from scipy.interpolate import griddata
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.cm import get_cmap
    except ImportError as exc:
        logger.warning("IDW raster requires scipy + matplotlib: %s", exc)
        return None

    if df.empty or metric not in df.columns:
        return None

    mask = df["latitude"].notna() & df["longitude"].notna() & df[metric].notna()
    df_valid = df[mask].copy()

    if len(df_valid) < 10:
        logger.debug("Too few valid points for IDW interpolation: %d", len(df_valid))
        return None

    # Downsample if needed
    if len(df_valid) > max_points:
        df_valid = df_valid.sample(n=max_points, random_state=42)

    lat_min, lat_max, lon_min, lon_max = _get_bounds(df_valid)

    # Create grid
    grid_lat = np.linspace(lat_min, lat_max, grid_resolution)
    grid_lon = np.linspace(lon_min, lon_max, grid_resolution)
    glon, glat = np.meshgrid(grid_lon, grid_lat)

    points = df_valid[["longitude", "latitude"]].values
    values = df_valid[metric].values

    # Linear interpolation first, then nearest for fill
    grid_z = griddata(points, values, (glon, glat), method="linear")
    grid_z_nearest = griddata(points, values, (glon, glat), method="nearest")
    # Fill NaN from linear with nearest
    nan_mask = np.isnan(grid_z)
    grid_z[nan_mask] = grid_z_nearest[nan_mask]

    if np.all(np.isnan(grid_z)):
        return None

    vmin, vmax = VRANGE_MAP.get(metric, (values.min(), values.max()))
    cmap_name = COLORSCALE_MAP.get(metric, "RdYlGn")

    # Map matplotlib cmap names to actual colormaps
    cmap_aliases = {
        "RdYlGn": "RdYlGn",
        "RdBu": "RdBu",
        "plasma": "plasma",
        "viridis": "viridis",
    }
    cmap = plt.get_cmap(cmap_aliases.get(cmap_name, "RdYlGn"))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(grid_z))
    # Set NaN pixels transparent
    if np.any(np.isnan(grid_z)):
        rgba[np.isnan(grid_z), 3] = 0.0

    # Convert to PNG bytes
    fig, ax = plt.subplots(figsize=(grid_resolution / 50, grid_resolution / 50), dpi=50)
    ax.imshow(
        rgba,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        aspect="auto",
        interpolation="bilinear",
    )
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "png_b64": png_b64,
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "lat_range": [lat_min, lat_max],
        "lon_range": [lon_min, lon_max],
        "colorscale": cmap_name,
        "vmin": vmin,
        "vmax": vmax,
        "metric": metric,
    }


def build_colorbar_trace(vmin: float, vmax: float, colorscale: str, label: str) -> dict:
    """Build a dummy Plotly trace that shows only the colorbar."""
    return {
        "type": "scattermapbox",
        "lat": [],
        "lon": [],
        "mode": "markers",
        "marker": {
            "color": [],
            "colorscale": colorscale,
            "cmin": vmin,
            "cmax": vmax,
            "showscale": True,
            "colorbar": {
                "title": label,
                "bgcolor": "#16213e",
                "tickfont": {"color": "#ffffff"},
                "titlefont": {"color": "#ffffff"},
            },
        },
        "showlegend": False,
        "hoverinfo": "skip",
        "name": f"IDW: {label}",
    }
