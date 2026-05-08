# VDT Tool — Virtual Drive Test

A production-grade Virtual Drive Test tool for Ericsson RAN RF optimization engineers.

## Quick Start

```bash
cd vdt_tool
pip install -r requirements.txt
python app.py
```

Open **http://localhost:8050**

Click **Load Demo Data** for an instant fully-populated demo over Muscat, Oman.

---

## Supported Input Files

| Type | Format | Notes |
|------|--------|-------|
| Cell Trace | GPEH `.gpb` binary | Ericsson OSS export |
| Cell Trace | CTUM `.gz` gzip XML | Ericsson OSS export |
| Cell Trace | ROP XML | 3GPP PM format |
| Cell Trace | ZIP / TAR bundle | Auto-extracts and parses all contained files |
| L3 Messages | `.pcap` / `.pcapng` | Wireshark capture (requires pyshark + tshark) |
| L3 Messages | `.log` / `.txt` | Ericsson OSS RRC log export |
| Events | CTR binary | GPEH-compatible |
| Events | CSV export | Ericsson OSS event export tool |
| GPS Track | CSV | Columns: `timestamp_ms`, `latitude`, `longitude` |
| GPS Track | KML | Google Earth export |
| GPS Track | GPX | Standard GPS exchange format |
| Cell Metadata | CSV | See `data/cell_db.csv` for template |

---

## Visualization Modes

| Mode | Description |
|------|-------------|
| **Colored Dots** | Each GPS sample colored by RSRP bin (Excellent/Good/Fair/Poor/Bad) |
| **Heatmap** | Density heatmap weighted by RSRP / RSRQ / SINR / Throughput |
| **IDW Grid** | Inverse distance weighted interpolated raster overlay |
| **HexBin** | Uber H3 hexagonal aggregation choropleth |
| **Cluster Events** | Event markers (HO failure, RLF, SCG failure, A3/A5 triggers) |

---

## KPIs Computed

- Median / P5 / P95 RSRP, RSRQ, SINR
- HO Success Rate = HO_SUCCESS / HO_ATTEMPT × 100
- RLF Rate per 10,000 samples
- SCG Failure count (EN-DC)
- RAB Setup Success Rate
- Per-cell table: Median RSRP, P5 RSRP, HO SR, RLF Rate, Avg TA, CQI, DL Throughput

---

## RSRP Classification

| Class | Range | Color |
|-------|-------|-------|
| Excellent | ≥ −80 dBm | `#2ecc71` |
| Good | −80 to −90 dBm | `#a8d08d` |
| Fair | −90 to −100 dBm | `#f0a500` |
| Poor | −100 to −110 dBm | `#e06000` |
| Bad | < −110 dBm | `#c0392b` |

Conversions per 3GPP TS 36.133:
- `RSRP_dBm = IE_value − 140`
- `RSRQ_dB = IE_value / 2 − 19.5`
- `SINR_dB = IE_value × 0.5 − 23` (NR, TS 38.133)

---

## Exports

- **CSV** — full unified dataset
- **GeoJSON** — all samples as FeatureCollection
- **KMZ** — Google Earth file with color-coded dots, track, sector wedges
- **PDF Report** — KPI cards, RSRP CDF, worst cells table, coverage distribution

---

## Directory Structure

```
vdt_tool/
├── app.py                  # Dash entry point
├── demo_data.py            # Synthetic demo data generator
├── parsers/
│   ├── gpeh_parser.py      # GPEH .gpb binary decoder
│   ├── ctum_parser.py      # CTUM gzip XML parser
│   ├── l3_parser.py        # RRC log / pcap parser
│   ├── events_parser.py    # CTR/GPEH event decoder
│   └── gps_correlator.py   # GPS ↔ sample alignment
├── metrics/
│   ├── kpi_engine.py       # KPI derivations
│   └── classifier.py       # RSRP/RSRQ/SINR bin classification
├── visualization/
│   ├── dot_map.py          # Colored dots layer
│   ├── heatmap.py          # Weighted heatmap
│   ├── idw_grid.py         # IDW raster interpolation
│   ├── hexbin.py           # H3 hexagonal aggregation
│   ├── sector_overlay.py   # Antenna sector wedge builder
│   └── timeline.py         # Plotly time-series panel
├── export/
│   ├── kmz_exporter.py
│   ├── geojson_exporter.py
│   └── pdf_reporter.py
├── data/
│   └── cell_db.csv         # Cell metadata template
└── requirements.txt
```

---

## Cell Metadata CSV Template

```csv
CellID,eNB_ID,Sector,Site_Name,Lat,Lon,Azimuth,Tilt,Height,Band,EARFCN,PCI,Tech
10101,1010,1,SITE_A,23.5880,58.3829,0,6,30,B3,1300,42,LTE
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8050` | HTTP port |
| `VDT_DEBUG` | `0` | Set to `1` for Dash debug mode |

---

## Performance Notes

- Above 50,000 GPS samples: Douglas-Peucker decimation applied automatically
- Files > 500MB: Use Dask (import in `parse_*` functions)
- IDW grid: limited to 5,000 scatter points for interpolation speed
- HexBin: requires `h3-py` (`pip install h3`)
- PDF export: requires `reportlab`
- PCAP parsing: requires `pyshark` + `tshark` system binary

---

## License

MIT
