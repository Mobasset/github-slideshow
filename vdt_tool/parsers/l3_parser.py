"""
L3 message parser: handles both .pcap (via pyshark) and Ericsson OSS
RRC log text files (.txt / .log).

Extracts from RRC messages:
  - RRCMeasurementReport: RSRP, RSRQ, SINR per PCI/EARFCN
  - RRCConnectionSetupComplete / RRCReconfiguration
  - SCGFailureInformation (EN-DC NR failures)
  - UECapabilityInformation (overheatingInd-r14/r16)

Log file format (Ericsson OSS export):
  2024-01-15 10:32:44.123 UTC [RRC][CELL:12345][RNTI:0xABCD] RRCMeasurementReport
    measResults: {
      measId: 1, pci: 42, rsrpResult: 85, rsrqResult: 48, ...
    }
"""

import re
import io
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Regex patterns for Ericsson OSS RRC log text format ──────────────────────

_TS_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*(?:UTC)?"
)
_CELL_PATTERN = re.compile(r"\[CELL[:\s](\d+)\]", re.IGNORECASE)
_RNTI_PATTERN = re.compile(r"\[RNTI[:\s]0x([0-9A-Fa-f]+)\]", re.IGNORECASE)
_PCI_PATTERN = re.compile(r"\bpci[:\s]+(\d+)", re.IGNORECASE)
_EARFCN_PATTERN = re.compile(r"\b(?:earfcn|arfcn|dl-CarrierFreq)[:\s]+(\d+)", re.IGNORECASE)
_RSRP_PATTERN = re.compile(r"\brsrp(?:Result)?[:\s]+(\d+)", re.IGNORECASE)
_RSRQ_PATTERN = re.compile(r"\brsrq(?:Result)?[:\s]+(\d+)", re.IGNORECASE)
_SINR_PATTERN = re.compile(r"\bsinr(?:Result)?[:\s]+(\d+)", re.IGNORECASE)
_CQI_PATTERN = re.compile(r"\bcqi[:\s]+(\d+)", re.IGNORECASE)
_TA_PATTERN = re.compile(r"\b(?:timingAdvance|ta)[:\s]+(\d+)", re.IGNORECASE)
_MSG_TYPE_PATTERN = re.compile(
    r"\b(RRCMeasurementReport|RRCConnectionSetupComplete|RRCReconfiguration"
    r"|RRCConnectionSetup|SCGFailureInformation|UECapabilityInformation"
    r"|RRCConnectionRelease|RRCConnectionReestablishment)\b",
    re.IGNORECASE,
)
_CAUSE_PATTERN = re.compile(r"\bcause[:\s]+(\w+)", re.IGNORECASE)
_NEIGHBOR_BLOCK = re.compile(
    r"measResultNeighCells.*?(?=measResultNeighCells|\Z)", re.DOTALL | re.IGNORECASE
)
_NEIGH_ENTRY = re.compile(
    r"pci[:\s]+(\d+).*?rsrp(?:Result)?[:\s]+(\d+)(?:.*?rsrq(?:Result)?[:\s]+(\d+))?",
    re.DOTALL | re.IGNORECASE,
)


def _parse_iso_ts(ts_str: str) -> int:
    """Parse timestamp string to epoch ms."""
    import datetime
    ts_str = ts_str.strip()
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(ts_str, fmt)
            dt = dt.replace(tzinfo=datetime.timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return 0


def _extract_int(pattern: re.Pattern, text: str, default: int = 0) -> int:
    m = pattern.search(text)
    return int(m.group(1)) if m else default


def _extract_float(pattern: re.Pattern, text: str, default: float = float("nan")) -> float:
    m = pattern.search(text)
    return float(m.group(1)) if m else default


def _parse_log_block(block: str, source_file: str) -> Optional[dict]:
    """Parse a single RRC log message block into a record dict."""
    ts_m = _TS_PATTERN.search(block)
    if not ts_m:
        return None
    ts_ms = _parse_iso_ts(ts_m.group(1))
    if ts_ms == 0:
        return None

    msg_m = _MSG_TYPE_PATTERN.search(block)
    msg_type = msg_m.group(1) if msg_m else "UNKNOWN"

    cell_id = _extract_int(_CELL_PATTERN, block)
    rnti_m = _RNTI_PATTERN.search(block)
    rnti = int(rnti_m.group(1), 16) if rnti_m else 0

    pci = _extract_int(_PCI_PATTERN, block)
    earfcn = _extract_int(_EARFCN_PATTERN, block)
    rsrp_ie = _extract_int(_RSRP_PATTERN, block)
    rsrq_ie = _extract_int(_RSRQ_PATTERN, block)
    sinr_ie = _extract_int(_SINR_PATTERN, block)
    cqi = _extract_int(_CQI_PATTERN, block)
    ta = _extract_int(_TA_PATTERN, block)

    rsrp_dbm = float(rsrp_ie - 140) if rsrp_ie > 0 else float("nan")
    rsrq_db = float(rsrq_ie) / 2.0 - 19.5 if rsrq_ie > 0 else float("nan")
    sinr_db = float(sinr_ie) * 0.5 - 23.0 if sinr_ie > 0 else float("nan")
    ta_us = ta * 16.0 / 1000.0 if ta > 0 else float("nan")

    cause_m = _CAUSE_PATTERN.search(block)
    cause = cause_m.group(1) if cause_m else ""

    scg_failure = 1 if "SCGFailureInformation" in msg_type else 0

    # Parse neighbor cells
    neighbors: List[Dict[str, Any]] = []
    for neigh_block_m in _NEIGHBOR_BLOCK.finditer(block):
        neigh_text = neigh_block_m.group(0)
        for entry_m in _NEIGH_ENTRY.finditer(neigh_text):
            n_pci = int(entry_m.group(1))
            n_rsrp_ie = int(entry_m.group(2))
            n_rsrq_ie = int(entry_m.group(3)) if entry_m.group(3) else 0
            neighbors.append(
                {
                    "pci": n_pci,
                    "rsrp_dbm": float(n_rsrp_ie - 140),
                    "rsrq_db": float(n_rsrq_ie) / 2.0 - 19.5 if n_rsrq_ie else float("nan"),
                }
            )

    return {
        "timestamp_utc": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
        "timestamp_ms": ts_ms,
        "cell_id": cell_id,
        "rnti": rnti,
        "pci": pci,
        "earfcn": earfcn,
        "rsrp_dbm": rsrp_dbm,
        "rsrq_db": rsrq_db,
        "sinr_db": sinr_db,
        "cqi": float(cqi) if cqi > 0 else float("nan"),
        "ta_us": ta_us,
        "throughput_dl_mbps": float("nan"),
        "throughput_ul_mbps": float("nan"),
        "ho_attempt": 0,
        "ho_success": 0,
        "ho_failure": 0,
        "rlf_flag": 0,
        "rab_setup": 0,
        "scg_failure": scg_failure,
        "a3_trigger": 0,
        "a5_trigger": 0,
        "event_type": msg_type,
        "event_cause": cause,
        "source_file": source_file,
        "neighbors": neighbors,
    }


def parse_rrc_log(data: str, source_file: str = "rrc_log") -> pd.DataFrame:
    """Parse Ericsson OSS RRC text log into a DataFrame."""
    # Split into message blocks on timestamp boundaries
    lines = data.splitlines()
    blocks: List[str] = []
    current: List[str] = []

    for line in lines:
        if _TS_PATTERN.match(line) and _MSG_TYPE_PATTERN.search(line):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current))

    records = []
    for block in blocks:
        rec = _parse_log_block(block, source_file)
        if rec:
            records.append(rec)

    if not records:
        return _empty_df()

    df = pd.DataFrame(records)
    # Store neighbors as JSON string for optional detail panel
    if "neighbors" in df.columns:
        import json
        df["neighbors_json"] = df["neighbors"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else "[]"
        )
        df = df.drop(columns=["neighbors"])
    return df


def parse_pcap(file_path: str, source_file: str = "") -> pd.DataFrame:
    """
    Parse .pcap file using pyshark, extracting LTE/NR RRC messages.
    Falls back gracefully if pyshark is unavailable.
    """
    sf = source_file or file_path
    try:
        import pyshark  # type: ignore
    except ImportError:
        logger.warning("pyshark not installed — cannot parse .pcap files")
        return _empty_df()

    records = []
    try:
        cap = pyshark.FileCapture(
            file_path,
            display_filter="lte_rrc or nr_rrc",
            keep_packets=False,
            use_json=True,
            include_raw=False,
        )
        for pkt in cap:
            try:
                ts_ms = int(float(pkt.sniff_timestamp) * 1000)
                # Try to extract LTE RRC measurement report fields
                rrc = getattr(pkt, "lte_rrc", None) or getattr(pkt, "nr_rrc", None)
                if rrc is None:
                    continue

                msg_type = getattr(rrc, "message_type", "UNKNOWN")

                rsrp_ie = _safe_int(getattr(rrc, "rsrpResult", 0))
                rsrq_ie = _safe_int(getattr(rrc, "rsrqResult", 0))
                sinr_ie = _safe_int(getattr(rrc, "sinr_result", 0))
                pci = _safe_int(getattr(rrc, "physCellId", 0))
                earfcn = _safe_int(getattr(rrc, "dl_CarrierFreq", 0)) or _safe_int(
                    getattr(rrc, "arfcn_ValueEUTRA", 0)
                )

                rsrp_dbm = float(rsrp_ie - 140) if rsrp_ie > 0 else float("nan")
                rsrq_db = float(rsrq_ie) / 2.0 - 19.5 if rsrq_ie > 0 else float("nan")
                sinr_db = float(sinr_ie) * 0.5 - 23.0 if sinr_ie > 0 else float("nan")

                records.append(
                    {
                        "timestamp_utc": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                        "timestamp_ms": ts_ms,
                        "cell_id": 0,
                        "rnti": 0,
                        "pci": pci,
                        "earfcn": earfcn,
                        "rsrp_dbm": rsrp_dbm,
                        "rsrq_db": rsrq_db,
                        "sinr_db": sinr_db,
                        "cqi": float("nan"),
                        "ta_us": float("nan"),
                        "throughput_dl_mbps": float("nan"),
                        "throughput_ul_mbps": float("nan"),
                        "ho_attempt": 0,
                        "ho_success": 0,
                        "ho_failure": 0,
                        "rlf_flag": 0,
                        "rab_setup": 0,
                        "scg_failure": 0,
                        "a3_trigger": 0,
                        "a5_trigger": 0,
                        "event_type": str(msg_type),
                        "event_cause": "",
                        "source_file": sf,
                    }
                )
            except Exception as inner_exc:
                logger.debug("Skipping packet: %s", inner_exc)
                continue

        cap.close()
    except Exception as exc:
        logger.warning("pyshark parse error for %s: %s", file_path, exc)
        return _empty_df()

    if not records:
        return _empty_df()
    return pd.DataFrame(records)


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(str(val))
    except (ValueError, TypeError):
        return default


def parse_l3_file(file_path: str) -> pd.DataFrame:
    """Auto-detect and parse an L3 message file."""
    if file_path.lower().endswith(".pcap") or file_path.lower().endswith(".pcapng"):
        return parse_pcap(file_path, source_file=file_path)

    with open(file_path, "r", errors="replace") as f:
        data = f.read()
    return parse_rrc_log(data, source_file=file_path)


def parse_l3_bytes(data: bytes, filename: str = "l3_log") -> pd.DataFrame:
    """Parse L3 data from raw bytes (used for in-memory upload)."""
    if filename.lower().endswith((".pcap", ".pcapng")):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return parse_pcap(tmp_path, source_file=filename)
        finally:
            os.unlink(tmp_path)

    text = data.decode("utf-8", errors="replace")
    return parse_rrc_log(text, source_file=filename)


def _empty_df() -> pd.DataFrame:
    cols = [
        "timestamp_utc", "timestamp_ms", "cell_id", "rnti", "pci", "earfcn",
        "rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "ta_us",
        "throughput_dl_mbps", "throughput_ul_mbps",
        "ho_attempt", "ho_success", "ho_failure", "rlf_flag", "rab_setup",
        "scg_failure", "a3_trigger", "a5_trigger",
        "event_type", "event_cause", "source_file",
    ]
    return pd.DataFrame(columns=cols)
