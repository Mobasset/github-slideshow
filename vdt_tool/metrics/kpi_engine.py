"""
KPI derivation engine.

Computes all network KPIs from the unified GeoDataFrame and produces:
  - Summary KPI dict (for sidebar cards)
  - Per-cell statistics table
  - Time-aggregated rolling metrics
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from metrics.classifier import (
    add_all_classes,
    coverage_bin_distribution,
    cdf_series,
    BIN_COLOR,
)

logger = logging.getLogger(__name__)

BLER_THRESHOLD_PERCENT = 10.0  # Above this → poor BLER


def safe_pct(numerator: float, denominator: float, scale: float = 100.0) -> float:
    if denominator == 0 or np.isnan(denominator):
        return float("nan")
    return round((numerator / denominator) * scale, 2)


def percentile(series: pd.Series, pct: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean, pct))


def compute_summary_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary KPI dict from unified DataFrame.

    Returns dict with keys matching sidebar card labels.
    """
    n_total = len(df)
    if n_total == 0:
        return _empty_kpis()

    # GPS match rate
    gps_matched = df.get("gps_matched", pd.Series([False] * n_total))
    if "latitude" in df.columns:
        gps_matched = df["latitude"].notna() & df["longitude"].notna()
    n_gps = int(gps_matched.sum())
    gps_pct = safe_pct(n_gps, n_total)

    # Time range
    ts = df["timestamp_ms"] if "timestamp_ms" in df.columns else pd.Series(dtype=float)
    ts_valid = ts[ts > 0].dropna()
    if ts_valid.empty:
        time_range = "N/A"
        duration_min = 0.0
    else:
        t_min = pd.Timestamp(ts_valid.min(), unit="ms", tz="UTC")
        t_max = pd.Timestamp(ts_valid.max(), unit="ms", tz="UTC")
        time_range = f"{t_min.strftime('%H:%M:%S')} – {t_max.strftime('%H:%M:%S')} UTC"
        duration_min = round((ts_valid.max() - ts_valid.min()) / 60_000, 1)

    # RSRP
    rsrp = df["rsrp_dbm"].dropna() if "rsrp_dbm" in df.columns else pd.Series(dtype=float)
    rsrp_median = round(float(rsrp.median()), 1) if not rsrp.empty else float("nan")
    rsrp_p5 = round(percentile(rsrp, 5), 1)
    rsrp_p95 = round(percentile(rsrp, 95), 1)

    # RSRQ
    rsrq = df["rsrq_db"].dropna() if "rsrq_db" in df.columns else pd.Series(dtype=float)
    rsrq_median = round(float(rsrq.median()), 1) if not rsrq.empty else float("nan")
    rsrq_p5 = round(percentile(rsrq, 5), 1)
    rsrq_p95 = round(percentile(rsrq, 95), 1)

    # SINR
    sinr = df["sinr_db"].dropna() if "sinr_db" in df.columns else pd.Series(dtype=float)
    sinr_median = round(float(sinr.median()), 1) if not sinr.empty else float("nan")
    sinr_p5 = round(percentile(sinr, 5), 1)
    sinr_p95 = round(percentile(sinr, 95), 1)

    # HO KPIs
    ho_att = int(df["ho_attempt"].sum()) if "ho_attempt" in df.columns else 0
    ho_succ = int(df["ho_success"].sum()) if "ho_success" in df.columns else 0
    ho_fail = int(df["ho_failure"].sum()) if "ho_failure" in df.columns else 0
    ho_sr = safe_pct(ho_succ, ho_att)

    # RLF
    rlf_count = int(df["rlf_flag"].sum()) if "rlf_flag" in df.columns else 0
    rlf_rate = safe_pct(rlf_count, n_total, scale=10_000)  # per 10k samples

    # RAB
    rab_att = int(df["rab_setup"].sum()) if "rab_setup" in df.columns else 0

    # SCG failures (EN-DC)
    scg_count = int(df["scg_failure"].sum()) if "scg_failure" in df.columns else 0

    # A3/A5 triggers
    a3_count = int(df["a3_trigger"].sum()) if "a3_trigger" in df.columns else 0
    a5_count = int(df["a5_trigger"].sum()) if "a5_trigger" in df.columns else 0

    # Throughput
    dl = df["throughput_dl_mbps"].dropna() if "throughput_dl_mbps" in df.columns else pd.Series(dtype=float)
    ul = df["throughput_ul_mbps"].dropna() if "throughput_ul_mbps" in df.columns else pd.Series(dtype=float)
    dl_median = round(float(dl.median()), 2) if not dl.empty else float("nan")
    ul_median = round(float(ul.median()), 2) if not ul.empty else float("nan")

    # Coverage distribution
    df_cls = add_all_classes(df.copy())
    coverage_bins = coverage_bin_distribution(df_cls)

    # RSRP CDF
    rsrp_cdf_x, rsrp_cdf_y = cdf_series(rsrp)

    return {
        "n_samples": n_total,
        "n_gps_matched": n_gps,
        "gps_match_pct": gps_pct,
        "time_range": time_range,
        "duration_min": duration_min,
        "rsrp_median": rsrp_median,
        "rsrp_p5": rsrp_p5,
        "rsrp_p95": rsrp_p95,
        "rsrq_median": rsrq_median,
        "rsrq_p5": rsrq_p5,
        "rsrq_p95": rsrq_p95,
        "sinr_median": sinr_median,
        "sinr_p5": sinr_p5,
        "sinr_p95": sinr_p95,
        "ho_attempt": ho_att,
        "ho_success": ho_succ,
        "ho_failure": ho_fail,
        "ho_sr": ho_sr,
        "rlf_count": rlf_count,
        "rlf_rate_per_10k": rlf_rate,
        "rab_setup_count": rab_att,
        "scg_failure_count": scg_count,
        "a3_trigger_count": a3_count,
        "a5_trigger_count": a5_count,
        "dl_throughput_median_mbps": dl_median,
        "ul_throughput_median_mbps": ul_median,
        "coverage_bins": coverage_bins,
        "rsrp_cdf_x": rsrp_cdf_x.tolist() if len(rsrp_cdf_x) > 0 else [],
        "rsrp_cdf_y": rsrp_cdf_y.tolist() if len(rsrp_cdf_y) > 0 else [],
    }


def compute_per_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cell statistics table.

    Returns DataFrame with columns:
    Cell_ID, PCI, Samples, Median_RSRP, P5_RSRP, HO_SR, RLF_Rate,
    Avg_TA, Median_CQI, Median_Throughput_DL
    """
    if df.empty or "cell_id" not in df.columns:
        return pd.DataFrame()

    agg_funcs: Dict[str, Any] = {}

    if "rsrp_dbm" in df.columns:
        agg_funcs["rsrp_median"] = ("rsrp_dbm", "median")
        agg_funcs["rsrp_p5"] = ("rsrp_dbm", lambda x: np.percentile(x.dropna(), 5) if x.notna().any() else np.nan)

    if "pci" in df.columns:
        agg_funcs["pci"] = ("pci", lambda x: int(x.mode().iloc[0]) if not x.empty else 0)

    if "earfcn" in df.columns:
        agg_funcs["earfcn"] = ("earfcn", lambda x: int(x.mode().iloc[0]) if not x.empty else 0)

    if "ho_attempt" in df.columns:
        agg_funcs["ho_attempt"] = ("ho_attempt", "sum")
    if "ho_success" in df.columns:
        agg_funcs["ho_success"] = ("ho_success", "sum")
    if "rlf_flag" in df.columns:
        agg_funcs["rlf_count"] = ("rlf_flag", "sum")
    if "ta_us" in df.columns:
        agg_funcs["avg_ta"] = ("ta_us", "mean")
    if "cqi" in df.columns:
        agg_funcs["cqi_median"] = ("cqi", "median")
    if "throughput_dl_mbps" in df.columns:
        agg_funcs["dl_mbps_median"] = ("throughput_dl_mbps", "median")
    if "sinr_db" in df.columns:
        agg_funcs["sinr_median"] = ("sinr_db", "median")

    # Build aggregation dict
    named_agg = {k: v for k, v in agg_funcs.items()}
    cell_grouped = df.groupby("cell_id", as_index=True)

    try:
        result = cell_grouped.agg(**{k: pd.NamedAgg(*v) for k, v in named_agg.items()})
        result["sample_count"] = cell_grouped.size()
    except Exception as exc:
        logger.warning("Per-cell aggregation error: %s — using fallback", exc)
        result = _fallback_per_cell(df)

    result = result.reset_index()
    result = result.rename(columns={"cell_id": "Cell_ID"})

    # Derived KPIs
    if "ho_attempt" in result.columns and "ho_success" in result.columns:
        result["ho_sr"] = result.apply(
            lambda r: safe_pct(r.get("ho_success", 0), r.get("ho_attempt", 0)), axis=1
        )
    else:
        result["ho_sr"] = float("nan")

    if "rlf_count" in result.columns and "sample_count" in result.columns:
        result["rlf_rate"] = result.apply(
            lambda r: safe_pct(r.get("rlf_count", 0), r.get("sample_count", 1), scale=10_000),
            axis=1,
        )
    else:
        result["rlf_rate"] = float("nan")

    # Round floats
    for col in ["rsrp_median", "rsrp_p5", "sinr_median", "avg_ta", "cqi_median", "dl_mbps_median"]:
        if col in result.columns:
            result[col] = result[col].round(1)

    return result


def _fallback_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Simple fallback aggregation when named agg fails."""
    groups = df.groupby("cell_id")
    rows = []
    for cell_id, grp in groups:
        rows.append(
            {
                "cell_id": cell_id,
                "rsrp_median": grp["rsrp_dbm"].median() if "rsrp_dbm" in grp else np.nan,
                "rsrp_p5": np.percentile(grp["rsrp_dbm"].dropna(), 5)
                if "rsrp_dbm" in grp and grp["rsrp_dbm"].notna().any()
                else np.nan,
                "pci": int(grp["pci"].mode().iloc[0]) if "pci" in grp and not grp["pci"].empty else 0,
                "ho_attempt": grp["ho_attempt"].sum() if "ho_attempt" in grp else 0,
                "ho_success": grp["ho_success"].sum() if "ho_success" in grp else 0,
                "rlf_count": grp["rlf_flag"].sum() if "rlf_flag" in grp else 0,
                "avg_ta": grp["ta_us"].mean() if "ta_us" in grp else np.nan,
                "cqi_median": grp["cqi"].median() if "cqi" in grp else np.nan,
                "dl_mbps_median": grp["throughput_dl_mbps"].median()
                if "throughput_dl_mbps" in grp
                else np.nan,
                "sinr_median": grp["sinr_db"].median() if "sinr_db" in grp else np.nan,
                "sample_count": len(grp),
            }
        )
    return pd.DataFrame(rows).set_index("cell_id")


def compute_rolling_metrics(
    df: pd.DataFrame, window_ms: int = 60_000
) -> pd.DataFrame:
    """Compute rolling window statistics (1-minute default) sorted by time."""
    if df.empty or "timestamp_ms" not in df.columns:
        return df

    df_sorted = df.sort_values("timestamp_ms").copy()
    df_sorted = df_sorted.set_index(
        pd.to_datetime(df_sorted["timestamp_ms"], unit="ms", utc=True)
    )

    window = f"{window_ms}ms"

    for col in ["rsrp_dbm", "rsrq_db", "sinr_db", "throughput_dl_mbps"]:
        if col in df_sorted.columns:
            df_sorted[f"{col}_rolling_median"] = (
                df_sorted[col].rolling(window, min_periods=1).median()
            )

    return df_sorted.reset_index(drop=True)


def _empty_kpis() -> Dict[str, Any]:
    return {
        "n_samples": 0,
        "n_gps_matched": 0,
        "gps_match_pct": 0.0,
        "time_range": "N/A",
        "duration_min": 0.0,
        "rsrp_median": float("nan"),
        "rsrp_p5": float("nan"),
        "rsrp_p95": float("nan"),
        "rsrq_median": float("nan"),
        "rsrq_p5": float("nan"),
        "rsrq_p95": float("nan"),
        "sinr_median": float("nan"),
        "sinr_p5": float("nan"),
        "sinr_p95": float("nan"),
        "ho_attempt": 0,
        "ho_success": 0,
        "ho_failure": 0,
        "ho_sr": float("nan"),
        "rlf_count": 0,
        "rlf_rate_per_10k": float("nan"),
        "rab_setup_count": 0,
        "scg_failure_count": 0,
        "a3_trigger_count": 0,
        "a5_trigger_count": 0,
        "dl_throughput_median_mbps": float("nan"),
        "ul_throughput_median_mbps": float("nan"),
        "coverage_bins": {},
        "rsrp_cdf_x": [],
        "rsrp_cdf_y": [],
    }
