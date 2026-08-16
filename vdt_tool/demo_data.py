"""
Demo mode data generator.

Produces a 2000-sample synthetic drive test dataset:
  - Random walk trajectory around Muscat, Oman (23.5880°N, 58.3829°E)
  - RSRP: log-distance path loss from 3 synthetic eNBs + Gaussian shadow fading
  - 12 random HO events, 3 RLF events, 2 SCG failures
  - All metrics populated for full visualization demo
"""

import math
import numpy as np
import pandas as pd

# ── Synthetic eNB sites around Muscat CBD ────────────────────────────────────
DEMO_ENB_SITES = [
    {"cell_id": 10101, "site_name": "MUSCAT_CBD_1", "lat": 23.5880, "lon": 58.3829,
     "azimuth": 0,   "pci": 42, "earfcn": 1300, "band": "B3",  "tech": "LTE"},
    {"cell_id": 10102, "site_name": "MUSCAT_CBD_1", "lat": 23.5880, "lon": 58.3829,
     "azimuth": 120, "pci": 43, "earfcn": 1300, "band": "B3",  "tech": "LTE"},
    {"cell_id": 10103, "site_name": "MUSCAT_CBD_1", "lat": 23.5880, "lon": 58.3829,
     "azimuth": 240, "pci": 44, "earfcn": 1300, "band": "B3",  "tech": "LTE"},
    {"cell_id": 10201, "site_name": "MUSCAT_RDC_2", "lat": 23.5950, "lon": 58.3920,
     "azimuth": 10,  "pci": 51, "earfcn": 3050, "band": "B7",  "tech": "LTE"},
    {"cell_id": 10202, "site_name": "MUSCAT_RDC_2", "lat": 23.5950, "lon": 58.3920,
     "azimuth": 130, "pci": 52, "earfcn": 3050, "band": "B7",  "tech": "LTE"},
    {"cell_id": 10203, "site_name": "MUSCAT_RDC_2", "lat": 23.5950, "lon": 58.3920,
     "azimuth": 250, "pci": 53, "earfcn": 3050, "band": "B7",  "tech": "LTE"},
    {"cell_id": 10301, "site_name": "MUSCAT_GRN_3", "lat": 23.5810, "lon": 58.3760,
     "azimuth": 30,  "pci": 61, "earfcn": 627980, "band": "n78", "tech": "NR"},
    {"cell_id": 10302, "site_name": "MUSCAT_GRN_3", "lat": 23.5810, "lon": 58.3760,
     "azimuth": 150, "pci": 62, "earfcn": 627980, "band": "n78", "tech": "NR"},
    {"cell_id": 10303, "site_name": "MUSCAT_GRN_3", "lat": 23.5810, "lon": 58.3760,
     "azimuth": 270, "pci": 63, "earfcn": 627980, "band": "n78", "tech": "NR"},
]

EARTH_R = 6_371_000.0
N_SAMPLES = 2000
CENTER_LAT = 23.5880
CENTER_LON = 58.3829
ROUTE_RADIUS_M = 800.0
SHADOW_FADING_DB = 8.0
PATH_LOSS_EXPONENT = 3.5
TX_POWER_DBM = 46.0
ANTENNA_GAIN_DBI = 18.0

# 3 primary serving sites (indices into DEMO_ENB_SITES list)
PRIMARY_SITES = [
    {"cell_id": 10101, "lat": 23.5880, "lon": 58.3829},
    {"cell_id": 10201, "lat": 23.5950, "lon": 58.3920},
    {"cell_id": 10301, "lat": 23.5810, "lon": 58.3760},
]


def _haversine_m(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _log_distance_rsrp(dist_m: float, shadow_fading: float = 0.0) -> float:
    """Log-distance path loss model → RSRP in dBm."""
    dist_m = max(dist_m, 10.0)
    fspl_db = 20 * math.log10(2600e6) + 20 * math.log10(dist_m) - 147.55
    path_loss = fspl_db + (PATH_LOSS_EXPONENT - 2) * 10 * math.log10(dist_m / 100.0)
    rsrp = TX_POWER_DBM + ANTENNA_GAIN_DBI - path_loss + shadow_fading
    return max(-140.0, min(-40.0, rsrp))


def _generate_route() -> tuple:
    """Generate random walk GPS trajectory around center point."""
    rng = np.random.default_rng(42)
    # Route: figure-8 / loop around the center, then random walk
    t = np.linspace(0, 2 * np.pi, N_SAMPLES)
    # Lissajous-like route for realistic drive test path
    route_lat = CENTER_LAT + (ROUTE_RADIUS_M / EARTH_R) * np.sin(t) * (180 / np.pi)
    route_lon = CENTER_LON + (ROUTE_RADIUS_M / EARTH_R) * np.cos(2 * t) * (180 / np.pi) / math.cos(math.radians(CENTER_LAT))

    # Add small random perturbation for realism
    route_lat += rng.normal(0, 0.0002, N_SAMPLES)
    route_lon += rng.normal(0, 0.0002, N_SAMPLES)

    return route_lat, route_lon


def generate_demo_data() -> tuple:
    """
    Generate synthetic demo dataset.

    Returns:
        (samples_df, cell_meta_df)
    """
    rng = np.random.default_rng(42)

    route_lat, route_lon = _generate_route()

    # Base timestamp: 2024-01-15 08:00:00 UTC
    import datetime
    base_ts = int(
        datetime.datetime(2024, 1, 15, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000
    )
    # Sample interval: ~1.5 seconds
    timestamps_ms = base_ts + np.arange(N_SAMPLES) * 1500

    # ── Signal metrics from path loss model ───────────────────────────────────
    rsrp_vals = np.zeros(N_SAMPLES)
    rsrq_vals = np.zeros(N_SAMPLES)
    sinr_vals = np.zeros(N_SAMPLES)
    cell_ids = np.zeros(N_SAMPLES, dtype=int)
    pcis = np.zeros(N_SAMPLES, dtype=int)

    # Shadow fading: spatially correlated via low-pass filtered noise
    raw_shadow = rng.normal(0, SHADOW_FADING_DB, N_SAMPLES)
    # Simple 20-sample moving average for spatial correlation
    shadow_fading = np.convolve(raw_shadow, np.ones(20) / 20, mode="same")

    for i in range(N_SAMPLES):
        lat, lon = route_lat[i], route_lon[i]
        # Compute RSRP from all 3 primary sites, serve from strongest
        site_rsrps = []
        for site in PRIMARY_SITES:
            dist = _haversine_m(lat, lon, site["lat"], site["lon"])
            sf = shadow_fading[i] * 0.5  # reduced per-site
            rsrp = _log_distance_rsrp(dist, sf)
            site_rsrps.append((rsrp, site["cell_id"]))

        site_rsrps.sort(reverse=True, key=lambda x: x[0])
        best_rsrp, best_cell = site_rsrps[0]

        rsrp_vals[i] = best_rsrp
        cell_ids[i] = best_cell

        # RSRQ: function of RSRP and interference
        n_rb = 100
        n_interferers = max(0.5, (best_rsrp + 120) / 20)
        rsrq_vals[i] = 10 * math.log10(n_rb / (1 + n_interferers)) - (120 + best_rsrp) * 0.15
        rsrq_vals[i] = max(-19.5, min(-3.0, rsrq_vals[i]))

        # SINR: higher when RSRP is better
        sinr_vals[i] = (best_rsrp + 120) * 0.6 - 5 + rng.normal(0, 2)
        sinr_vals[i] = max(-10.0, min(40.0, sinr_vals[i]))

    # Assign PCI from cell_id lookup
    cell_pci_map = {s["cell_id"]: s["pci"] for s in DEMO_ENB_SITES}
    pcis = np.array([cell_pci_map.get(c, 0) for c in cell_ids])

    # ── Derived metrics ───────────────────────────────────────────────────────
    # CQI: mapped from SINR (roughly)
    cqi_vals = np.clip(np.round((sinr_vals + 10) / 3.5).astype(int), 0, 15)

    # Timing advance: proportional to distance from serving site
    ta_vals = np.zeros(N_SAMPLES)
    for i in range(N_SAMPLES):
        serving = next((s for s in PRIMARY_SITES if s["cell_id"] == cell_ids[i]), PRIMARY_SITES[0])
        dist = _haversine_m(route_lat[i], route_lon[i], serving["lat"], serving["lon"])
        ta_vals[i] = (dist / 150000.0) * 1000  # rough: propagation delay in µs

    # Throughput: function of CQI (Shannon approximation)
    dl_tput = np.clip(cqi_vals * 4.5 + rng.normal(0, 3, N_SAMPLES), 0, 150).astype(float)
    ul_tput = np.clip(cqi_vals * 1.2 + rng.normal(0, 1.5, N_SAMPLES), 0, 50).astype(float)

    # RSRP IE values (reverse from dBm)
    rsrp_ie = np.clip(rsrp_vals + 140, 0, 127).astype(int)
    rsrq_ie = np.clip((rsrq_vals + 19.5) * 2, 0, 63).astype(int)
    sinr_ie = np.clip((sinr_vals + 23) / 0.5, 0, 127).astype(int)

    # ── Event injection ───────────────────────────────────────────────────────
    n_samples = N_SAMPLES
    ho_attempt = np.zeros(n_samples, dtype=int)
    ho_success = np.zeros(n_samples, dtype=int)
    ho_failure = np.zeros(n_samples, dtype=int)
    rlf_flag = np.zeros(n_samples, dtype=int)
    scg_failure = np.zeros(n_samples, dtype=int)
    a3_trigger = np.zeros(n_samples, dtype=int)
    a5_trigger = np.zeros(n_samples, dtype=int)
    event_type = np.array(["" for _ in range(n_samples)], dtype=object)
    event_cause = np.array(["" for _ in range(n_samples)], dtype=object)

    # 12 HO events: inject at cell boundary crossings (low RSRP zones)
    ho_indices = rng.choice(n_samples, size=12, replace=False)
    for idx in ho_indices:
        ho_attempt[idx] = 1
        event_type[idx] = "INTERNAL_HANDOVER_ATTEMPT"
        event_cause[idx] = "A3"
        if rng.random() > 0.2:  # 80% success rate
            ho_success[idx] = 1
            event_type[idx] = "INTERNAL_HANDOVER_SUCCESS"
        else:
            ho_failure[idx] = 1
            event_type[idx] = "INTERNAL_HANDOVER_FAILURE"
            event_cause[idx] = "FAILED_IN_TARGET_RAN"

    # 3 RLF events: inject at worst RSRP positions
    worst_rsrp_idx = np.argsort(rsrp_vals)[:10]
    rlf_indices = rng.choice(worst_rsrp_idx, size=3, replace=False)
    for idx in rlf_indices:
        rlf_flag[idx] = 1
        event_type[idx] = "RRC_RLF"
        event_cause[idx] = "T310_EXPIRY"
        rsrp_vals[idx] = rng.uniform(-115, -125)

    # 2 SCG failures: random positions in NR coverage area
    scg_indices = rng.choice(n_samples, size=2, replace=False)
    for idx in scg_indices:
        scg_failure[idx] = 1
        event_type[idx] = "NR_SCG_FAILURE"
        event_cause[idx] = "SCG_CHANGE"

    # A3/A5 triggers (events before HOs)
    a3_indices = rng.choice(n_samples, size=8, replace=False)
    for idx in a3_indices:
        if event_type[idx] == "":
            a3_trigger[idx] = 1
            event_type[idx] = "A3_TRIGGER"

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "timestamp_ms": timestamps_ms.astype("int64"),
            "timestamp_utc": pd.to_datetime(timestamps_ms, unit="ms", utc=True),
            "latitude": route_lat,
            "longitude": route_lon,
            "cell_id": cell_ids,
            "pci": pcis,
            "earfcn": np.array([
                1300 if c in (10101, 10102, 10103) else
                (3050 if c in (10201, 10202, 10203) else 627980)
                for c in cell_ids
            ]),
            "rsrp_dbm": rsrp_vals,
            "rsrq_db": rsrq_vals,
            "sinr_db": sinr_vals,
            "cqi": cqi_vals.astype(float),
            "ta_us": ta_vals,
            "throughput_dl_mbps": dl_tput,
            "throughput_ul_mbps": ul_tput,
            "bler_dl": np.clip(rng.exponential(2, n_samples), 0, 30),
            "bler_ul": np.clip(rng.exponential(1.5, n_samples), 0, 20),
            "ho_attempt": ho_attempt,
            "ho_success": ho_success,
            "ho_failure": ho_failure,
            "rlf_flag": rlf_flag,
            "rab_setup": np.zeros(n_samples, dtype=int),
            "scg_failure": scg_failure,
            "a3_trigger": a3_trigger,
            "a5_trigger": a5_trigger,
            "event_type": event_type,
            "event_cause": event_cause,
            "source_file": "demo_synthetic",
            "gps_matched": True,
        }
    )

    # ── Add classification columns ────────────────────────────────────────────
    from metrics.classifier import add_all_classes
    df = add_all_classes(df)

    # ── Cell metadata DataFrame ───────────────────────────────────────────────
    cell_meta_df = pd.DataFrame(DEMO_ENB_SITES)
    cell_meta_df = cell_meta_df.rename(columns={
        "cell_id": "CellID",
        "site_name": "Site_Name",
        "lat": "Lat",
        "lon": "Lon",
        "azimuth": "Azimuth",
        "pci": "PCI",
        "earfcn": "EARFCN",
        "band": "Band",
        "tech": "Tech",
    })
    cell_meta_df["eNB_ID"] = cell_meta_df["CellID"] // 10
    cell_meta_df["Sector"] = cell_meta_df["CellID"] % 10
    cell_meta_df["Tilt"] = 6
    cell_meta_df["Height"] = 30

    return df, cell_meta_df
