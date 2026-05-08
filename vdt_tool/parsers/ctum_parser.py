"""
CTUM (Cell Trace UE Measurement) gzip XML parser for Ericsson OSS exports.

CTUM files are gzipped XML with the following structure:
<measCollecFile>
  <fileHeader>
    <fileSender localDn="..."/>
    <measCollec beginTime="2024-01-01T00:00:00Z"/>
  </fileHeader>
  <measData>
    <managedElement localDn="SubNetwork=X,MeContext=eNB123,..."/>
    <measInfo measInfoId="PM_...">
      <granPeriod duration="PT900S" endTime="..."/>
      <measType p="1">pmRrcConnEstabAtt</measType>
      ...
      <measValue measObjLdn="...">
        <r p="1">42</r>
        ...
      </measValue>
    </measInfo>
  </measData>
</measCollecFile>

For cell trace (CTUM UE-level), events are encoded as:
<event eventType="INTERNAL_HANDOVER_ATTEMPT" timestamp="..." rnti="..." .../>
"""

import gzip
import io
import logging
import re
from typing import List, Dict, Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Namespace used in Ericsson CTUM files
NS_CTUM = {
    "c": "http://www.3gpp.org/ftp/specs/archive/32_series/32.435#measCollec",
    "e": "urn:ericsson:ctum:events:v1",
}

EVENT_TYPE_MAP = {
    "INTERNAL_HANDOVER_ATTEMPT": {"ho_attempt": 1},
    "INTERNAL_HANDOVER_SUCCESS": {"ho_success": 1},
    "INTERNAL_HANDOVER_FAILURE": {"ho_failure": 1},
    "RRC_RLF": {"rlf_flag": 1},
    "INTERNAL_ERAB_SETUP": {"rab_setup": 1},
    "INTERNAL_UE_CONTEXT_RELEASE": {},
    "NR_SCG_FAILURE": {"scg_failure": 1},
    "A3_TRIGGER": {"a3_trigger": 1},
    "A5_TRIGGER": {"a5_trigger": 1},
    "INTERNAL_RAB_ESTABLISHMENT": {"rab_setup": 1},
    "INTERNAL_SYSTEM_RELEASE": {},
}

_TS_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?Z?"
)


def _parse_iso_ts(ts_str: str) -> int:
    """Parse ISO8601 timestamp to epoch milliseconds."""
    m = _TS_RE.match(ts_str.strip())
    if not m:
        return 0
    y, mo, d, h, mi, s = [int(x) for x in m.groups()[:6]]
    frac_str = m.group(7) or "0"
    frac_ms = int(frac_str[:3].ljust(3, "0"))
    import calendar
    import datetime
    dt = datetime.datetime(y, mo, d, h, mi, s, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000) + frac_ms


def _attr_int(elem: ET.Element, name: str, default: int = 0) -> int:
    v = elem.get(name, "")
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _attr_float(elem: ET.Element, name: str, default: float = float("nan")) -> float:
    v = elem.get(name, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _extract_cell_id(ldn: str) -> int:
    """Extract numeric cell ID from Ericsson LDN string."""
    m = re.search(r"EUtranCellFDD=(\d+)|EUtranCellTDD=(\d+)|NRCellDU=(\d+)", ldn)
    if m:
        return int(next(v for v in m.groups() if v is not None))
    m2 = re.search(r"=(\d+)$", ldn)
    if m2:
        return int(m2.group(1))
    return 0


def _parse_event_element(elem: ET.Element, cell_id: int, source_file: str) -> dict:
    """Parse a single CTUM event XML element into a record dict."""
    etype = elem.get("eventType", elem.tag.split("}")[-1])
    ts_str = elem.get("timestamp", elem.get("t", ""))
    ts_ms = _parse_iso_ts(ts_str) if ts_str else 0

    rsrp_ie = _attr_int(elem, "rsrpResult", 0)
    rsrq_ie = _attr_int(elem, "rsrqResult", 0)
    sinr_ie = _attr_int(elem, "sinrResult", 0)

    rsrp_dbm = float(rsrp_ie - 140) if rsrp_ie > 0 else float("nan")
    rsrq_db = float(rsrq_ie) / 2.0 - 19.5 if rsrq_ie > 0 else float("nan")
    sinr_db = float(sinr_ie) * 0.5 - 23.0 if sinr_ie > 0 else float("nan")

    flags = EVENT_TYPE_MAP.get(etype, {})

    cause_raw = elem.get("causeCode", elem.get("cause", "0"))
    try:
        cause_int = int(cause_raw)
    except ValueError:
        cause_int = 0

    return {
        "timestamp_utc": pd.Timestamp(ts_ms, unit="ms", tz="UTC") if ts_ms else pd.NaT,
        "timestamp_ms": ts_ms,
        "cell_id": _attr_int(elem, "cellId", cell_id),
        "rnti": _attr_int(elem, "rnti"),
        "pci": _attr_int(elem, "pci"),
        "earfcn": _attr_int(elem, "earfcn"),
        "rsrp_dbm": rsrp_dbm,
        "rsrq_db": rsrq_db,
        "sinr_db": sinr_db,
        "cqi": _attr_float(elem, "cqi"),
        "ta_us": _attr_float(elem, "ta"),
        "throughput_dl_mbps": _attr_float(elem, "dlThroughput"),
        "throughput_ul_mbps": _attr_float(elem, "ulThroughput"),
        "ho_attempt": flags.get("ho_attempt", 0),
        "ho_success": flags.get("ho_success", 0),
        "ho_failure": flags.get("ho_failure", 0),
        "rlf_flag": flags.get("rlf_flag", 0),
        "rab_setup": flags.get("rab_setup", 0),
        "scg_failure": flags.get("scg_failure", 0),
        "a3_trigger": flags.get("a3_trigger", 0),
        "a5_trigger": flags.get("a5_trigger", 0),
        "event_type": etype,
        "event_cause": str(cause_int),
        "source_file": source_file,
    }


def _parse_xml_tree(root: ET.Element, source_file: str) -> List[dict]:
    records: List[dict] = []

    # Try to strip namespace prefix for robust matching
    def _strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    # Walk all elements looking for event-like nodes
    for elem in root.iter():
        tag = _strip_ns(elem.tag)

        if tag in (
            "event",
            "ctumEvent",
            "traceRecord",
            "INTERNAL_HANDOVER_ATTEMPT",
            "INTERNAL_HANDOVER_SUCCESS",
            "INTERNAL_HANDOVER_FAILURE",
            "RRC_RLF",
            "NR_SCG_FAILURE",
            "A3_TRIGGER",
            "A5_TRIGGER",
            "INTERNAL_ERAB_SETUP",
            "INTERNAL_UE_CONTEXT_RELEASE",
            "INTERNAL_RAB_ESTABLISHMENT",
            "INTERNAL_SYSTEM_RELEASE",
        ):
            # Determine cell_id from parent context
            cell_id = 0
            ldn = elem.get("localDn", elem.get("measObjLdn", ""))
            if ldn:
                cell_id = _extract_cell_id(ldn)

            if tag not in ("event", "ctumEvent", "traceRecord"):
                # tag IS the event type
                elem.set("eventType", tag)

            rec = _parse_event_element(elem, cell_id, source_file)
            records.append(rec)

        elif tag == "managedElement":
            ldn = elem.get("localDn", "")
            cell_id = _extract_cell_id(ldn)
            # Check children for measurements
            for child in elem:
                ctag = _strip_ns(child.tag)
                if ctag == "measInfo":
                    _parse_meas_info(child, cell_id, source_file, records)

    return records


def _parse_meas_info(
    meas_info: ET.Element, cell_id: int, source_file: str, records: List[dict]
) -> None:
    """Parse a 3GPP PM measInfo block for counter-based metrics."""
    def _strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    type_map: Dict[int, str] = {}
    end_time_str = ""

    for child in meas_info:
        ctag = _strip_ns(child.tag)
        if ctag == "granPeriod":
            end_time_str = child.get("endTime", "")
        elif ctag == "measType":
            p = int(child.get("p", 0))
            type_map[p] = child.text or ""

    ts_ms = _parse_iso_ts(end_time_str) if end_time_str else 0

    for child in meas_info:
        ctag = _strip_ns(child.tag)
        if ctag == "measValue":
            obj_ldn = child.get("measObjLdn", "")
            cid = _extract_cell_id(obj_ldn) or cell_id
            values: Dict[str, float] = {}
            for r_elem in child:
                rtag = _strip_ns(r_elem.tag)
                if rtag == "r":
                    p = int(r_elem.get("p", 0))
                    name = type_map.get(p, f"pm_{p}")
                    try:
                        values[name] = float(r_elem.text or "nan")
                    except ValueError:
                        values[name] = float("nan")

            if values:
                records.append(
                    {
                        "timestamp_utc": pd.Timestamp(ts_ms, unit="ms", tz="UTC")
                        if ts_ms
                        else pd.NaT,
                        "timestamp_ms": ts_ms,
                        "cell_id": cid,
                        "rnti": 0,
                        "pci": 0,
                        "earfcn": 0,
                        "rsrp_dbm": float("nan"),
                        "rsrq_db": float("nan"),
                        "sinr_db": float("nan"),
                        "cqi": values.get("pmCqiPuschRank1Distr", float("nan")),
                        "ta_us": float("nan"),
                        "throughput_dl_mbps": values.get("pmPdcpVolDlDrb", float("nan")),
                        "throughput_ul_mbps": values.get("pmPdcpVolUlDrb", float("nan")),
                        "ho_attempt": int(
                            values.get("pmErabHoInterFreqAttLteAdm", 0) or 0
                        ),
                        "ho_success": int(
                            values.get("pmErabHoInterFreqSuccLteAdm", 0) or 0
                        ),
                        "ho_failure": 0,
                        "rlf_flag": 0,
                        "rab_setup": int(values.get("pmRrcConnEstabSucc", 0) or 0),
                        "scg_failure": 0,
                        "a3_trigger": 0,
                        "a5_trigger": 0,
                        "event_type": "PM_COUNTER",
                        "event_cause": "",
                        "source_file": source_file,
                    }
                )


def parse_ctum_bytes(data: bytes, source_file: str = "ctum") -> pd.DataFrame:
    """Parse CTUM gzip XML bytes into a DataFrame."""
    # Decompress if gzipped
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except Exception as exc:
            logger.warning("Failed to decompress CTUM gzip: %s", exc)

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        logger.warning("XML parse error in CTUM: %s", exc)
        # Try to recover by stripping BOM
        try:
            root = ET.fromstring(data.lstrip(b"\xef\xbb\xbf"))
        except ET.ParseError:
            return _empty_df()

    records = _parse_xml_tree(root, source_file)
    if not records:
        return _empty_df()

    df = pd.DataFrame(records)
    # Ensure all expected columns exist
    for col in _empty_df().columns:
        if col not in df.columns:
            df[col] = np.nan if col not in {"event_type", "event_cause", "source_file"} else ""
    return df


def parse_ctum_file(file_path: str) -> pd.DataFrame:
    with open(file_path, "rb") as f:
        data = f.read()
    return parse_ctum_bytes(data, source_file=file_path)


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
