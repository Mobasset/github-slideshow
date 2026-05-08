"""
GPEH (.gpb) binary decoder for Ericsson OSS cell trace exports.

GPEH uses a TLV-style binary format with a file header followed by
records. Each record starts with a 2-byte record type, 2-byte length,
then payload bytes.

Record types relevant to VDT analysis:
  0x04  GPEH_CELL_TRACE_HEADER
  0x08  RRC_MEASUREMENT_REPORT
  0x09  INTERNAL_HANDOVER_ATTEMPT
  0x0A  INTERNAL_HANDOVER_SUCCESS
  0x0B  INTERNAL_HANDOVER_FAILURE
  0x0C  RRC_RLF
  0x10  INTERNAL_ERAB_SETUP
  0x11  INTERNAL_UE_CONTEXT_RELEASE
  0x14  NR_SCG_FAILURE
  0x15  A3_TRIGGER
  0x16  A5_TRIGGER
  0x20  INTERNAL_RAB_ESTABLISHMENT
  0x21  INTERNAL_SYSTEM_RELEASE
"""

import struct
import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

RECORD_TYPE_NAMES = {
    0x04: "CELL_TRACE_HEADER",
    0x08: "RRC_MEASUREMENT_REPORT",
    0x09: "INTERNAL_HANDOVER_ATTEMPT",
    0x0A: "INTERNAL_HANDOVER_SUCCESS",
    0x0B: "INTERNAL_HANDOVER_FAILURE",
    0x0C: "RRC_RLF",
    0x10: "INTERNAL_ERAB_SETUP",
    0x11: "INTERNAL_UE_CONTEXT_RELEASE",
    0x14: "NR_SCG_FAILURE",
    0x15: "A3_TRIGGER",
    0x16: "A5_TRIGGER",
    0x20: "INTERNAL_RAB_ESTABLISHMENT",
    0x21: "INTERNAL_SYSTEM_RELEASE",
}

HO_CAUSE_CODES = {
    0: "UNSPECIFIED",
    1: "HANDOVER_TRIGGERED_BY_UE",
    2: "RADIO_LINK_FAILURE",
    3: "LOAD_BALANCING",
    4: "COVERAGE_OPTIMIZATION",
    5: "SIGNAL_STRENGTH",
    6: "VELOCITY_CRITERIA",
    7: "TIMER_EXPIRY",
    8: "UE_CONTEXT_RELEASE_COMMAND",
    9: "FAILED_IN_TARGET_EPC",
    10: "FAILED_IN_TARGET_RAN",
}

RLF_CAUSE_CODES = {
    0: "UNSPECIFIED",
    1: "T310_EXPIRY",
    2: "RANDOM_ACCESS_PROBLEM",
    3: "RLC_MAX_RETRANSMISSION",
    4: "SYNC_RECONFIGURATION_FAILURE",
    5: "SCG_CHANGE",
    6: "SRB3_INTEGRITY_FAILURE",
}


@dataclass
class GPEHRecord:
    record_type: int
    timestamp_ms: int
    cell_id: int = 0
    rnti: int = 0
    pci: int = 0
    earfcn: int = 0
    rsrp_ie: int = 0
    rsrq_ie: int = 0
    sinr_ie: int = 0
    cqi: int = 0
    ta: int = 0
    cause_code: int = 0
    ho_type: int = 0
    target_cell_id: int = 0
    rlf_cause: int = 0
    event_type: str = ""
    raw_bytes: bytes = field(default_factory=bytes, repr=False)

    @property
    def rsrp_dbm(self) -> float:
        if self.rsrp_ie > 0:
            return float(self.rsrp_ie - 140)
        return float("nan")

    @property
    def rsrq_db(self) -> float:
        if self.rsrq_ie > 0:
            return float(self.rsrq_ie) / 2.0 - 19.5
        return float("nan")

    @property
    def sinr_db(self) -> float:
        if self.sinr_ie > 0:
            return float(self.sinr_ie) * 0.5 - 23.0
        return float("nan")


GPEH_FILE_MAGIC = b"\x00\x00\x00\x04"


def _read_be16(buf: bytes, offset: int) -> int:
    return struct.unpack_from(">H", buf, offset)[0]


def _read_be32(buf: bytes, offset: int) -> int:
    return struct.unpack_from(">I", buf, offset)[0]


def _read_be64(buf: bytes, offset: int) -> int:
    return struct.unpack_from(">Q", buf, offset)[0]


def _parse_meas_report(payload: bytes, ts: int, cell_id: int) -> GPEHRecord:
    rec = GPEHRecord(
        record_type=0x08,
        timestamp_ms=ts,
        cell_id=cell_id,
        event_type="RRC_MEASUREMENT_REPORT",
    )
    if len(payload) >= 4:
        rec.rnti = _read_be16(payload, 0)
    if len(payload) >= 6:
        rec.pci = _read_be16(payload, 2)
    if len(payload) >= 8:
        rec.earfcn = _read_be16(payload, 4)
    if len(payload) >= 9:
        rec.rsrp_ie = payload[6] & 0x7F
    if len(payload) >= 10:
        rec.rsrq_ie = payload[7] & 0x3F
    if len(payload) >= 11:
        rec.sinr_ie = payload[8] & 0x7F
    if len(payload) >= 12:
        rec.cqi = payload[9] & 0x0F
    if len(payload) >= 14:
        rec.ta = _read_be16(payload, 10)
    return rec


def _parse_ho_attempt(payload: bytes, ts: int, cell_id: int) -> GPEHRecord:
    rec = GPEHRecord(
        record_type=0x09,
        timestamp_ms=ts,
        cell_id=cell_id,
        event_type="INTERNAL_HANDOVER_ATTEMPT",
    )
    if len(payload) >= 2:
        rec.rnti = _read_be16(payload, 0)
    if len(payload) >= 4:
        rec.target_cell_id = _read_be16(payload, 2)
    if len(payload) >= 5:
        rec.cause_code = payload[4]
        rec.ho_type = (payload[4] >> 4) & 0x0F
    return rec


def _parse_ho_success(payload: bytes, ts: int, cell_id: int) -> GPEHRecord:
    rec = GPEHRecord(
        record_type=0x0A,
        timestamp_ms=ts,
        cell_id=cell_id,
        event_type="INTERNAL_HANDOVER_SUCCESS",
    )
    if len(payload) >= 2:
        rec.rnti = _read_be16(payload, 0)
    if len(payload) >= 4:
        rec.target_cell_id = _read_be16(payload, 2)
    return rec


def _parse_ho_failure(payload: bytes, ts: int, cell_id: int) -> GPEHRecord:
    rec = GPEHRecord(
        record_type=0x0B,
        timestamp_ms=ts,
        cell_id=cell_id,
        event_type="INTERNAL_HANDOVER_FAILURE",
    )
    if len(payload) >= 2:
        rec.rnti = _read_be16(payload, 0)
    if len(payload) >= 4:
        rec.target_cell_id = _read_be16(payload, 2)
    if len(payload) >= 5:
        rec.cause_code = payload[4]
    return rec


def _parse_rlf(payload: bytes, ts: int, cell_id: int) -> GPEHRecord:
    rec = GPEHRecord(
        record_type=0x0C,
        timestamp_ms=ts,
        cell_id=cell_id,
        event_type="RRC_RLF",
    )
    if len(payload) >= 2:
        rec.rnti = _read_be16(payload, 0)
    if len(payload) >= 3:
        rec.rlf_cause = payload[2] & 0x07
    return rec


def _parse_scg_failure(payload: bytes, ts: int, cell_id: int) -> GPEHRecord:
    rec = GPEHRecord(
        record_type=0x14,
        timestamp_ms=ts,
        cell_id=cell_id,
        event_type="NR_SCG_FAILURE",
    )
    if len(payload) >= 2:
        rec.rnti = _read_be16(payload, 0)
    if len(payload) >= 3:
        rec.cause_code = payload[2]
    return rec


def _parse_generic_event(
    rtype: int, payload: bytes, ts: int, cell_id: int
) -> GPEHRecord:
    name = RECORD_TYPE_NAMES.get(rtype, f"UNKNOWN_0x{rtype:02X}")
    rec = GPEHRecord(
        record_type=rtype,
        timestamp_ms=ts,
        cell_id=cell_id,
        event_type=name,
    )
    if len(payload) >= 2:
        rec.rnti = _read_be16(payload, 0)
    return rec


_PARSERS = {
    0x08: _parse_meas_report,
    0x09: _parse_ho_attempt,
    0x0A: _parse_ho_success,
    0x0B: _parse_ho_failure,
    0x0C: _parse_rlf,
    0x14: _parse_scg_failure,
}


def parse_gpeh_bytes(data: bytes, source_file: str = "gpeh") -> pd.DataFrame:
    """Parse raw GPEH binary data into a DataFrame."""
    records: List[dict] = []
    stream = io.BytesIO(data)
    total = len(data)

    # Attempt to read file header (12 bytes: magic 4 + rop_start_ms 8)
    header = stream.read(12)
    if len(header) < 12:
        logger.warning("GPEH file too short for header, attempting raw parse")
        stream.seek(0)

    current_cell_id = 0
    base_ts = 0

    while stream.tell() < total:
        header_bytes = stream.read(4)
        if len(header_bytes) < 4:
            break

        rtype = _read_be16(header_bytes, 0)
        rlen = _read_be16(header_bytes, 2)

        if rlen > 65535 or rlen < 0:
            logger.debug("Invalid record length %d at offset %d", rlen, stream.tell())
            stream.seek(1 - 4, io.SEEK_CUR)
            continue

        payload = stream.read(rlen)
        if len(payload) < rlen:
            break

        # Extract timestamp — first 8 bytes of payload if record has >= 8 bytes
        ts = base_ts
        if len(payload) >= 8:
            ts_raw = _read_be64(payload, 0)
            # Heuristic: if value looks like a valid epoch ms (year 2000-2100)
            if 946684800000 <= ts_raw <= 4102444800000:
                ts = ts_raw
                payload = payload[8:]
            else:
                # 4-byte relative timestamp in ms
                if len(payload) >= 4:
                    ts_rel = _read_be32(payload, 0)
                    ts = base_ts + ts_rel
                    payload = payload[4:]
        elif len(payload) >= 4:
            ts_rel = _read_be32(payload, 0)
            ts = base_ts + ts_rel
            payload = payload[4:]

        if rtype == 0x04:
            # Cell trace header: update current_cell_id and base_ts
            if len(payload) >= 4:
                current_cell_id = _read_be32(payload, 0)
            if ts > 0:
                base_ts = ts
            continue

        if rtype not in RECORD_TYPE_NAMES:
            continue

        parser = _PARSERS.get(rtype, _parse_generic_event)
        if rtype in _PARSERS:
            rec = parser(payload, ts, current_cell_id)
        else:
            rec = _parse_generic_event(rtype, payload, ts, current_cell_id)

        records.append(
            {
                "timestamp_utc": pd.Timestamp(ts, unit="ms", tz="UTC"),
                "cell_id": rec.cell_id,
                "rnti": rec.rnti,
                "pci": rec.pci,
                "earfcn": rec.earfcn,
                "rsrp_dbm": rec.rsrp_dbm,
                "rsrq_db": rec.rsrq_db,
                "sinr_db": rec.sinr_db,
                "cqi": rec.cqi if rec.cqi > 0 else np.nan,
                "ta_us": rec.ta * 16.0 / 1000.0 if rec.ta > 0 else np.nan,
                "ho_attempt": 1 if rec.event_type == "INTERNAL_HANDOVER_ATTEMPT" else 0,
                "ho_success": 1 if rec.event_type == "INTERNAL_HANDOVER_SUCCESS" else 0,
                "ho_failure": 1 if rec.event_type == "INTERNAL_HANDOVER_FAILURE" else 0,
                "rlf_flag": 1 if rec.event_type == "RRC_RLF" else 0,
                "scg_failure": 1 if rec.event_type == "NR_SCG_FAILURE" else 0,
                "a3_trigger": 1 if rec.event_type == "A3_TRIGGER" else 0,
                "a5_trigger": 1 if rec.event_type == "A5_TRIGGER" else 0,
                "event_type": rec.event_type,
                "event_cause": HO_CAUSE_CODES.get(rec.cause_code, str(rec.cause_code)),
                "source_file": source_file,
            }
        )

    if not records:
        logger.warning("No GPEH records parsed from %s", source_file)
        return _empty_df()

    df = pd.DataFrame(records)
    df["timestamp_ms"] = df["timestamp_utc"].astype("int64") // 1_000_000
    return df


def parse_gpeh_file(file_path: str) -> pd.DataFrame:
    with open(file_path, "rb") as f:
        data = f.read()
    return parse_gpeh_bytes(data, source_file=file_path)


def _empty_df() -> pd.DataFrame:
    cols = [
        "timestamp_utc", "timestamp_ms", "cell_id", "rnti", "pci", "earfcn",
        "rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "ta_us",
        "ho_attempt", "ho_success", "ho_failure", "rlf_flag", "scg_failure",
        "a3_trigger", "a5_trigger", "event_type", "event_cause", "source_file",
    ]
    return pd.DataFrame(columns=cols)
