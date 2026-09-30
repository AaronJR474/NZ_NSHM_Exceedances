"""Utilities for NZ NSHM2010/NSHM2022 UHS and hazard disaggregation.

Purpose
-------
This module is intentionally narrow. It supports the hazard-side calculations needed
for an observed-ground-motion/design-spectrum exceedance study:

1. prepare OpenQuake site files (including the NZ NSHM2022 backarc flag),
2. build classical jobs for site-specific UHS,
3. build chunked disaggregation jobs,
4. run OpenQuake with the NSHM2022-required command-line overrides,
5. extract mean/quantile UHS and TRT-Magnitude-Distance-Epsilon disaggregation results, and
6. compute the NZS 1170.5:2004 design spectrum and equivalent return periods.

OpenQuake is imported lazily, so job preparation and validation work in a normal
NumPy/Pandas environment. Hazard calculations/extraction require OpenQuake.

The NSHM2022 workflow is designed for OpenQuake >= 3.23.4. For reproducibility,
3.23.4 itself is recommended unless a newer version has first been benchmarked
against the GNS NSHM web results.
"""

from __future__ import annotations

import ast
import configparser
import json
import ntpath
import os
import re
import shutil
import signal
import subprocess
import sys
import zipfile
from collections import deque
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from shapely.geometry import Point, shape
from scipy.io import loadmat

import numpy as np
import pandas as pd


__version__ = "0.3.14"
__version_tuple__ = (0, 3, 14)


def version_info() -> dict[str, str]:
    """Return the loaded module version and absolute file path.

    This is intentionally tiny but useful in notebooks, where Python's import cache
    can otherwise make it easy to run an older copy of this module by accident.
    """
    return {"version": __version__, "file": str(Path(__file__).resolve())}


# -----------------------------------------------------------------------------
# Recommended study settings
# -----------------------------------------------------------------------------

OQ_MIN_VERSION = (3, 23, 4)
NSHM2022_LOGIC_TREE_SAMPLES = 100_000
MAX_POTENTIAL_PATHS = 1_000_000

# Production return periods agreed for this NZ study. The paper-specific values
# below are retained separately for direct literature replication if needed.
DEFAULT_RETURN_PERIODS = (100.0, 250.0, 500.0, 1000.0, 2500.0)
# Production hazard/UHS calculations are mean-only unless quantiles are requested.
DEFAULT_QUANTILES: tuple[float, ...] = ()
IMAZEKI_RETURN_PERIODS = (100.0, 475.0, 975.0, 2475.0)
CALDERON_DISAGG_PERIODS = (0.1, 1.0, 5.0)

# Dense spectral-period set agreed for production hazard/UHS calculations.
# Keep model-specific benchmark/default period constants below unchanged; pass
# PRODUCTION_PERIODS explicitly for the final study calculation.
PRODUCTION_PERIODS = (
    0.01, 0.02, 0.03, 0.04, 0.05, 0.075,
    0.1, 0.12, 0.15, 0.17, 0.2, 0.25,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9,
    1.0, 1.2, 1.25, 1.5, 2.0, 2.5,
    3.0, 4.0, 5.0, 6.0, 7.5, 10.0,
)

NZGMDB_PERIODS = (
     0.01, 0.02, 0.022, 0.025, 0.029, 0.03,
     0.032, 0.035, 0.036, 0.04, 0.042, 0.044,
     0.045, 0.046, 0.048, 0.05, 0.055, 0.06, 0.065,
     0.067, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095,
     0.1, 0.11, 0.12, 0.13, 0.133, 0.14, 0.15, 0.16,
     0.17, 0.18, 0.19, 0.2, 0.22, 0.24, 0.25, 0.26,
     0.28, 0.29, 0.3, 0.32, 0.34, 0.35, 0.36, 0.38,
     0.4, 0.42, 0.44, 0.45, 0.46, 0.48, 0.5, 0.55,
     0.6, 0.65, 0.667, 0.7, 0.75, 0.8, 0.85, 0.9,
     0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
     1.8, 1.9, 2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0,
     3.2, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6,
     4.8, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
     9.0, 9.5, 10.0
)

# Native OpenQuake disaggregation bins used by this study. ``dist`` is the
# OpenQuake 3.23.4 disaggregation implementation bins ``ctx.rrup``. Explicit edges keep the
# final M-Rrup-epsilon contribution table independent of OQ's automatic binning.
DEFAULT_DISAGG_MAG_BIN_EDGES = tuple(float(x) for x in np.round(np.arange(5.0, 10.2, 0.2), 10))
DEFAULT_DISAGG_DIST_BIN_EDGES = (
    0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
    80.0, 100.0, 140.0, 180.0, 220.0, 260.0, 320.0, 380.0, 500.0,
)
DEFAULT_DISAGG_EPS_BIN_EDGES = tuple(float(x) for x in np.round(np.arange(-5.0, 5.2, 0.2), 10))
# 100 degrees effectively collapses lon/lat; those axes are not requested in
# TRT_Mag_Dist_Eps but OQ 3.23 still needs coordinate_bin_width to construct
# its internal disaggregation bin geometry.
DEFAULT_DISAGG_COORDINATE_BIN_WIDTH_DEG = 100.0

NSHM2010_DEFAULT_PERIODS = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0
)
NSHM2022_DEFAULT_PERIODS = (0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)

# Periods/statistics in the supplied GNS NSHM_v1.0.4 UHS benchmark.
NSHM2022_GNS_VALIDATION_PERIODS = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0,
    4.0, 5.0, 6.0, 7.5, 10.0,
)
NSHM2022_GNS_VALIDATION_QUANTILES = (0.05, 0.10, 0.90, 0.95)
NSHM2010_REFERENCE_POES = (0.002105, 0.000404)

# The supplied NSHM2010 job uses logscale(0.005, 5.0, 50). This grid must be
# retained when reproducing the attached hazard_uhs-mean_949_site.csv values.
NSHM2010_DEFAULT_IM_LEVELS = np.geomspace(0.005, 5.0, 50)

# Exact level grid supplied with the uploaded NSHM2022 model. Reusing it avoids
# silently changing the model team's numerical setup. It can also be used for
# NSHM2010 when a broader/bracket-safe grid than the old 0.005-5 g grid is wanted.
DEFAULT_IM_LEVELS = np.array(
    [
        0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004,
        0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
        2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0,
    ],
    dtype=float,
)
NSHM2022_DEFAULT_IM_LEVELS = DEFAULT_IM_LEVELS

_SITE_COLUMNS = (
    "custom_site_id", "lon", "lat", "vs30", "z1pt0", "z2pt5",
    "vs30measured", "backarc",
)

# -----------------------------------------------------------------------------
# NZ GMDB Rupture plane utility
# -----------------------------------------------------------------------------

def build_corners(df):
    """
    Construct corners arrays grouped by evid.
    Returns dict: {evid: np.ndarray of shape (4, 3, n_planes)}
    where axes are (plane, [lon, lat, depth], corner_index).
    """
    result = {}
    for evid, g in df.groupby("evid"):
        plane_blocks = []
        for _, row in g.iterrows():
            lats   = [row[f"corner_{i}_lat"]   for i in range(4)]
            lons   = [row[f"corner_{i}_lon"]   for i in range(4)]
            depths = [row[f"corner_{i}_depth"] for i in range(4)]
            arr = np.vstack([lons, lats, depths])  # (3,4)
            plane_blocks.append(arr)

        # stack planes → (4, 3, n_planes)
        arr = np.stack(plane_blocks)
        arr = np.transpose(arr, (2, 1, 0))

        result[evid] = arr
    return result

# -----------------------------------------------------------------------------
# load NGA data
# -----------------------------------------------------------------------------

def load_nga_exceedance(nga_dir, mat_name, variable, label):

    rsn_table_nga = pd.read_csv(nga_dir / "rsn_set.csv")

    rsn_selected = set(pd.to_numeric(rsn_table_nga["Record Sequence Number"], errors="raise").astype(int))

    nga_meta = loadmat(
        nga_dir / "NGA_W2_corr_meta_data.mat",
        variable_names=["closest_D", "magnitude"],
        squeeze_me=True,
    )

    closest_D_nga = np.asarray(nga_meta["closest_D"], dtype=float).ravel()
    magnitude_nga = np.asarray(nga_meta["magnitude"], dtype=float).ravel()

    mat = loadmat(nga_dir / mat_name, variable_names=[variable], squeeze_me=True)

    if variable not in mat:
        raise KeyError(f"{variable} not found in {mat_name}")

    intervals = np.atleast_2d(np.asarray(mat[variable], dtype=float))

    if intervals.shape[1] != 3:
        raise ValueError(
            f"{mat_name}:{variable} must have three columns "
            f"[RSN, T_min, T_max], got {intervals.shape}"
        )

    df = pd.DataFrame(intervals, columns=["RSN", "T_min", "T_max"]).dropna()
    df["RSN"] = df["RSN"].astype(int)
    df = df[df["RSN"].isin(rsn_selected)].copy()

    idx = df["RSN"].to_numpy(dtype=int) - 1

    if len(idx) and (
        idx.min() < 0
        or idx.max() >= len(closest_D_nga)
        or idx.max() >= len(magnitude_nga)
    ):
        raise IndexError(f"{label}: RSN exceeds NGA-West2 metadata-array bounds")

    df["r_rup"] = closest_D_nga[idx]
    df["mag"] = magnitude_nga[idx]
    df["dataset"] = label

    return df[
        np.isfinite(df["r_rup"])
        & np.isfinite(df["mag"])
        & (df["r_rup"] > 0)
    ].reset_index(drop=True)

# -----------------------------------------------------------------------------
# Probability / return-period utilities
# -----------------------------------------------------------------------------

def exceedance_probability(return_period: float, years: float = 1.0) -> float:
    """Poisson probability of >=1 exceedance in ``years`` for a return period."""
    rp = float(return_period)
    yrs = float(years)
    if not np.isfinite(rp) or rp <= 0:
        raise ValueError("return_period must be positive and finite")
    if not np.isfinite(yrs) or yrs <= 0:
        raise ValueError("years must be positive and finite")
    return float(1.0 - np.exp(-yrs / rp))


def return_period_from_probability(probability: float, years: float = 1.0) -> float:
    """Poisson return period corresponding to an exceedance probability."""
    p = float(probability)
    yrs = float(years)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be between 0 and 1")
    if not np.isfinite(yrs) or yrs <= 0:
        raise ValueError("years must be positive and finite")
    return float(-yrs / np.log1p(-p))


def annual_exceedance_rate(return_period: float) -> float:
    """Annual exceedance rate lambda = 1 / return period."""
    rp = float(return_period)
    if not np.isfinite(rp) or rp <= 0:
        raise ValueError("return_period must be positive and finite")
    return 1.0 / rp


def annual_poe(return_period: float) -> float:
    """One-year PoE corresponding to a return period (OpenQuake investigation_time=1)."""
    return exceedance_probability(return_period, years=1.0)


def annual_poe_from_probability(probability: float, years: float = 50.0) -> float:
    """Convert an exact probability in ``years`` to the equivalent one-year PoE."""
    p = float(probability)
    yrs = float(years)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be between 0 and 1")
    if not np.isfinite(yrs) or yrs <= 0:
        raise ValueError("years must be positive and finite")
    return float(1.0 - (1.0 - p) ** (1.0 / yrs))


# -----------------------------------------------------------------------------
# Backarc polygon handling (pure NumPy; no geopandas/qcore dependency)
# -----------------------------------------------------------------------------

def load_geojson_polygon(path: str | Path) -> np.ndarray:
    """Load the first Polygon ring from a GeoJSON file as [lon, lat] vertices."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("type") == "FeatureCollection":
        geom = obj["features"][0]["geometry"]
    elif obj.get("type") == "Feature":
        geom = obj["geometry"]
    else:
        geom = obj
    if geom["type"] != "Polygon":
        raise ValueError(f"Expected Polygon geometry, got {geom['type']!r}")
    xy = np.asarray(geom["coordinates"][0], dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Invalid polygon coordinates")
    return xy


def points_in_polygon(lon: Sequence[float], lat: Sequence[float], polygon: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test in lon/lat coordinates."""
    x = np.asarray(lon, dtype=float)
    y = np.asarray(lat, dtype=float)
    if x.shape != y.shape:
        raise ValueError("lon and lat must have the same shape")
    p = np.asarray(polygon, dtype=float)
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])

    inside = np.zeros(x.shape, dtype=bool)
    x0, y0 = p[:-1, 0], p[:-1, 1]
    x1, y1 = p[1:, 0], p[1:, 1]
    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        crosses = ((ya > y) != (yb > y))
        xcross = (xb - xa) * (y - ya) / ((yb - ya) + np.finfo(float).tiny) + xa
        inside ^= crosses & (x < xcross)
    return inside


def add_nshm2022_backarc(
    sites: pd.DataFrame,
    backarc_json: str | Path,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Add the NSHM2022 ``backarc`` flag using the supplied backarc polygon."""
    out = sites.copy()
    _require_columns(out, ("lon", "lat"))
    if "backarc" in out and not overwrite:
        return out
    polygon = load_geojson_polygon(backarc_json)
    out["backarc"] = points_in_polygon(out["lon"], out["lat"], polygon)
    return out


# -----------------------------------------------------------------------------
# Model discovery and site files
# -----------------------------------------------------------------------------

def extract_model_zip(zip_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> Path:
    """Extract an NSHM model archive, preserving its directory structure."""
    zip_path, output_dir = Path(zip_path), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            return output_dir
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    return output_dir


def resolve_model_root(model_dir: str | Path, model: str) -> Path:
    """Resolve a supplied/extracted NSHM2010 or NSHM2022 model directory."""
    model = _normalise_model(model)
    root = Path(model_dir).resolve()
    required = (
        ("gsim_model.xml", "sources/source_model_extended.xml", "backarc.json")
        if model == "2022"
        else ("gmpe_logic_tree_McVerry06_ONLY_MW_RotD50.xml", "source_models/source_model_logic_tree.xml")
    )
    if all((root / item).exists() for item in required):
        return root

    for candidate in [p for p in root.rglob("*") if p.is_dir()]:
        if all((candidate / item).exists() for item in required):
            return candidate
    raise FileNotFoundError(f"Could not find the required NSHM{model} model files below {root}")


def infer_basin_depths(vs30: Sequence[float], model: str = "2022") -> tuple[np.ndarray, np.ndarray]:
    """Infer ``z1pt0`` (m) and ``z2pt5`` (km) from Vs30 for the NZ inputs.

    These equations are not generic replacements for measured basin depths. They
    reproduce the conventions in the supplied NSHM inputs:

    * NSHM2022: the supplied workflow example uses the CY14-style z1 relation
      with 571 m/s (Vs30=400 -> z1pt0=355.7170357122867 m), plus the CB14-style
      z2 relation (Vs30=400 -> z2pt5=1.2646109912757877 km).
    * NSHM2010: ``site_table_aaron.csv`` is reproduced to floating-point precision
      by the same relations with 570.94 m/s in the z1 equation.

    User-supplied z1pt0/z2pt5 values always take precedence in :func:`prepare_sites`.
    """
    model = _normalise_model(model)
    v = np.asarray(vs30, dtype=float)
    if np.any(~np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("vs30 must contain positive finite values")
    c = 571.0 if model == "2022" else 570.94
    z1pt0 = np.exp((-7.15 / 4.0) * np.log((v**4 + c**4) / (1360.0**4 + c**4)))
    z2pt5 = np.exp(7.089 - 1.144 * np.log(v))
    return z1pt0, z2pt5


def prepare_sites(
    sites: pd.DataFrame,
    model: str,
    model_dir: str | Path,
    recompute_backarc: bool = True,
    infer_missing_basin_depths: bool = True,
    recompute_basin_depths: bool = False,
) -> pd.DataFrame:
    """Validate and normalize a site table for OpenQuake.

    ``lon``, ``lat`` and ``vs30`` are required for this study workflow. When
    ``infer_missing_basin_depths=True``, absent/NaN/non-positive ``z1pt0`` and
    ``z2pt5`` values are populated from Vs30 using the recovered NSHM conventions.
    Set ``recompute_basin_depths=True`` to overwrite all supplied basin depths with
    those Vs30-derived proxies.

    For NSHM2022, ``vs30measured`` defaults to 0 (inferred) when it is not
    supplied. This matches the supplied NSHM2022 custom-site convention. Backarc
    is recomputed from the model polygon by default.
    """
    model = _normalise_model(model)
    root = resolve_model_root(model_dir, model)
    out = sites.copy().reset_index(drop=True)
    _require_columns(out, ("lon", "lat", "vs30"))

    for col in ("lon", "lat", "vs30"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
        if not np.isfinite(out[col].to_numpy(float)).all():
            raise ValueError(f"{col} must contain finite values")
    if not out["lon"].between(-180.0, 180.0).all():
        raise ValueError("lon must be between -180 and 180 degrees")
    if not out["lat"].between(-90.0, 90.0).all():
        raise ValueError("lat must be between -90 and 90 degrees")
    if (out["vs30"] <= 0).any():
        raise ValueError("vs30 must contain positive values")

    if "custom_site_id" not in out:
        out.insert(0, "custom_site_id", [f"S{i:05d}" for i in range(len(out))])
    site_ids = out["custom_site_id"].astype(str)
    if site_ids.duplicated().any():
        raise ValueError("custom_site_id values must be unique")
    # OpenQuake 3.23.x stores custom_site_id in a fixed-width S8 field.
    id_lengths = site_ids.map(lambda s: len(s.encode("utf-8")))
    if (id_lengths > 8).any():
        bad = site_ids[id_lengths > 8].tolist()[:5]
        raise ValueError(
            "OpenQuake 3.23.x requires custom_site_id values to be at most 8 bytes; "
            f"offending value(s): {bad}"
        )

    if infer_missing_basin_depths or recompute_basin_depths:
        z1, z2 = infer_basin_depths(out["vs30"].to_numpy(float), model=model)

        if recompute_basin_depths or "z1pt0" not in out.columns:
            out["z1pt0"] = z1
        else:
            supplied = pd.to_numeric(out["z1pt0"], errors="coerce")
            missing = supplied.isna() | (supplied <= 0)
            out["z1pt0"] = supplied
            out.loc[missing, "z1pt0"] = z1[missing.to_numpy()]

        if recompute_basin_depths or "z2pt5" not in out.columns:
            out["z2pt5"] = z2
        else:
            supplied = pd.to_numeric(out["z2pt5"], errors="coerce")
            missing = supplied.isna() | (supplied <= 0)
            out["z2pt5"] = supplied
            out.loc[missing, "z2pt5"] = z2[missing.to_numpy()]

    for col in ("z1pt0", "z2pt5"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
            values = out[col].to_numpy(float)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(
                    f"{col} must contain positive finite values; use "
                    "infer_missing_basin_depths=True or recompute_basin_depths=True "
                    "for missing/non-positive basin depths"
                )

    if model == "2022":
        _require_columns(out, ("z1pt0", "z2pt5"))
        if "vs30measured" not in out.columns:
            out["vs30measured"] = 0
        if recompute_backarc or "backarc" not in out:
            out = add_nshm2022_backarc(out, root / "backarc.json", overwrite=True)

    # OpenQuake 3.23.x reads Boolean-like site parameters using integer dtypes.
    for col in ("backarc", "vs30measured"):
        if col in out.columns:
            out[col] = _oq_binary_site_param(out[col], col)

    keep = [c for c in _SITE_COLUMNS if c in out.columns]
    return out[keep].copy()


def write_site_model(
    sites: pd.DataFrame,
    path: str | Path,
    model: str,
    model_dir: str | Path,
    recompute_backarc: bool = True,
    recompute_basin_depths: bool = False,
) -> Path:
    """Write a compact OpenQuake site-model CSV."""
    out = prepare_sites(
        sites,
        model,
        model_dir,
        recompute_backarc=recompute_backarc,
        recompute_basin_depths=recompute_basin_depths,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


# -----------------------------------------------------------------------------
# NZS 1170.5:2004 design spectrum (ported from the supplied wrapper)
# -----------------------------------------------------------------------------
def nzs1170p5_assign_near_fault_distance(
    sites: pd.DataFrame,
    table_path: str | Path = Path(r"NZS1170_Z_D_TABLE.csv"),
    settlement_geojson_path: str | Path = Path(r"TS1170_data") / "TS1170-5_Figure3-2_2025.geojson",
) -> pd.DataFrame:
    """
    Assign NZS 1170.5:2004 near-fault distance information to site coordinates.

    Site latitude/longitude is first matched to the settlement polygons in the
    supplied Figure 3.2 GeoJSON. The corresponding NZS 1170.5 location is then
    matched to NZS1170_Z_D_TABLE.csv.

    D_min and D_max retain the tabulated distance range. D is the value used by
    nzs1170p5_spectrum():

        D = D_min       where a near-fault distance is tabulated
        D = np.inf      where the listed NZS location has no tabulated D,
                        corresponding to N(T,D) = 1
        D = np.nan      where the site cannot be associated with a listed
                        NZS 1170.5 location

    The minimum distance is used where a range is tabulated, providing the
    conservative near-fault factor for that listed locality.
    """
    table = pd.read_csv(table_path, encoding="cp1252")
    table["Location"] = (
        table["Location"].astype(str)
        .str.strip()
        .str.replace("–", "-", regex=False)
    )

    with open(settlement_geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    settlements = [
        (str(feature["properties"]["Name"]).strip(), shape(feature["geometry"]))
        for feature in geojson["features"]
    ]

    location_aliases = {
        "Whanganui": "Wanganui",
        "Foxton": "Foxton/Foxton Beach",
        "Foxton Beach": "Foxton/Foxton Beach",
        "Lower Hutt": "Hutt Valley-south of Taita Gorge",
        "Mt Cook Village": "Mt Cook",
    }

    records = []

    for row in sites.itertuples(index=False):
        point = Point(float(row.lon), float(row.lat))
        matches = [(name, geom.area) for name, geom in settlements if geom.covers(point)]

        if not matches:
            records.append({
                "NZS_location": None,
                "D_min": np.inf,
                "D_max": np.inf,
                "D": np.inf,
            })
            continue

        location = min(matches, key=lambda x: x[1])[0]
        table_location = location_aliases.get(location, location)

        match = table.loc[table["Location"] == table_location]

        if match.empty:
            records.append({
                "NZS_location": location,
                "D_min": np.inf,
                "D_max": np.inf,
                "D": np.inf,
            })
            continue

        match = match.iloc[0]
        d_min = match["D_min"]
        d_max = match["D_max"]

        records.append({
            "NZS_location": table_location,
            "D_min": d_min,
            "D_max": d_max,
            "D": np.inf if pd.isna(d_min) else float(d_min),
        })

    return pd.concat(
        [sites.reset_index(drop=True), pd.DataFrame(records)],
        axis=1,
    )

def nzs1170p5_return_period_factor(return_period: float) -> float:
    """
    Return the NZS 1170.5:2004 return-period factor R.

    Parameters
    ----------
    return_period : float
        Mean return period in years. NZS 1170.5 Table 3.5 tabulates R for
        20, 25, 50, 100, 250, 500, 1000, 2000 and 2500 years. Values between
        tabulated return periods are linearly interpolated in return period,
        matching the supplied NZCodeSpectra.m implementation.

    Returns
    -------
    float
        Return-period factor R (Rs or Ru, depending on the limit state in
        which it is subsequently used).

    Raises
    ------
    ValueError
        If return_period is not finite or lies outside 20--2500 years.

    Notes
    -----
    R is independent of site class and the geographic hazard factor Z.
    The NZS 1170.5 ULS limitation Z*Ru <= 0.7 is applied in
    nzs1170p5_spectrum(), not here.
    """
    data = np.array([
        [20.0, 0.20], [25.0, 0.25], [50.0, 0.35], [100.0, 0.50],
        [250.0, 0.75], [500.0, 1.00], [1000.0, 1.30],
        [2000.0, 1.70], [2500.0, 1.80],
    ], dtype=float)

    rp = float(return_period)
    if not np.isfinite(rp) or not data[0, 0] <= rp <= data[-1, 0]:
        raise ValueError("return_period must be finite and between 20 and 2500 years")

    return float(np.interp(rp, data[:, 0], data[:, 1]))


def _nzs1170p5_spectral_shape(periods: np.ndarray, soil_class: str, spectrum_form: str) -> np.ndarray:
    """Return the NZS 1170.5:2004 horizontal spectral shape factor Ch(T)."""
    T = np.asarray(periods, dtype=float)
    soil = soil_class.upper()
    form = spectrum_form.lower()

    if soil not in {"A", "B", "C", "D", "E"}:
        raise ValueError("soil_class must be one of A, B, C, D or E")
    if form not in {"modal", "general"}:
        raise ValueError("spectrum_form must be 'modal' or 'general'")

    Ch = np.empty_like(T)

    for i, t in enumerate(T):
        if soil in {"A", "B"}:
            if form == "modal" and t < 0.1:
                Ch[i] = 1.0 + 1.35 * t / 0.1
            elif form == "modal" and t < 0.3:
                Ch[i] = 2.35
            elif form == "general" and t < 0.4:
                Ch[i] = 1.89
            elif t < 1.5:
                Ch[i] = 1.6 * (0.5 / t) ** 0.75
            elif t < 3.0:
                Ch[i] = 1.05 / t
            else:
                Ch[i] = 3.15 / t**2

        elif soil == "C":
            if form == "modal" and t < 0.1:
                Ch[i] = 1.33 + 1.6 * t / 0.1
            elif form == "modal" and t < 0.3:
                Ch[i] = 2.93
            elif form == "general" and t < 0.4:
                Ch[i] = 2.36
            elif t < 1.5:
                Ch[i] = 2.0 * (0.5 / t) ** 0.75
            elif t < 3.0:
                Ch[i] = 1.32 / t
            else:
                Ch[i] = 3.96 / t**2

        elif soil == "D":
            if form == "modal" and t < 0.1:
                Ch[i] = 1.12 + 1.88 * t / 0.1
            elif t < 0.56:
                Ch[i] = 3.0
            elif t < 1.5:
                Ch[i] = 2.4 * (0.75 / t) ** 0.75
            elif t < 3.0:
                Ch[i] = 2.14 / t
            else:
                Ch[i] = 6.42 / t**2

        else:  # E
            if form == "modal" and t < 0.1:
                Ch[i] = 1.12 + 1.88 * t / 0.1
            elif t < 1.0:
                Ch[i] = 3.0
            elif t < 1.5:
                Ch[i] = 3.0 * (1.0 / t) ** 0.75
            elif t < 3.0:
                Ch[i] = 3.32 / t
            else:
                Ch[i] = 9.96 / t**2

    return Ch

def _nzs1170p5_component_factor(periods: np.ndarray, component: str) -> np.ndarray:
    """Return factor converting the native NZS Larger component to the requested component."""
    T = np.asarray(periods, dtype=float)
    comp = component.lower()

    if comp == "larger":
        return np.ones_like(T)

    if comp != "rotd50":
        raise ValueError("component must be 'larger' or 'rotd50'")

    larger_gm = np.full_like(T, 1.14)
    rotd50_gm = np.full_like(T, 1.01)

    mid = (T >= 0.1) & (T < 2.0)
    high = T >= 2.0

    x = np.log(T[mid] / 0.1) / np.log(2.0 / 0.1)

    larger_gm[mid] = 1.14 + (1.25 - 1.14) * x
    rotd50_gm[mid] = 1.01 + (1.06 - 1.01) * x

    larger_gm[high] = 1.25
    rotd50_gm[high] = 1.06

    # Native NZS is Larger; convert Larger -> RotD50
    return rotd50_gm / larger_gm

def nzs1170p5_spectrum(
    periods: Sequence[float],
    Z: float,
    return_period: float = 500.0,
    near_fault_distance_km: float | None = None,
    soil_class: str = "C",
    spectrum_form: str = "modal",
    limit_state: str = "ULS",
    enforce_uls_cap: bool = True,
    component: str = "rotd50",
) -> pd.DataFrame:
    """
    Compute the NZS 1170.5:2004 horizontal elastic site hazard spectrum.

    The elastic site hazard spectrum is

        C(T) = Ch(T) * Z * R * N(T,D)

    with the NZS 1170.5 ultimate-limit-state restriction Z*Ru <= 0.7 applied
    when limit_state="ULS" and enforce_uls_cap=True.

    Parameters
    ----------
    periods : sequence of float
        Oscillator periods in seconds. Values must be finite and non-negative.
        Include 0.0 when the T=0 spectral ordinate is required.

    Z : float
        NZS 1170.5 geographic hazard factor for the site. Z is obtained from
        Table 3.3 or the applicable NZS 1170.5 hazard-factor map. It is a code
        geographic hazard parameter; it is not PGA, Vs30, or a value derived
        from an NSHM2010 or NSHM2022 UHS.

    return_period : float, default 500.0
        Mean return period in years, between 20 and 2500. The corresponding
        return-period factor R is obtained using
        nzs1170p5_return_period_factor().

    near_fault_distance_km : float or None, default None
        Shortest distance D, in km, to a major fault for which NZS 1170.5
        requires the near-fault factor. For return periods <=250 years,
        N(T,D)=1 and D is not required. For return periods >250 years, D must
        be supplied. Use np.inf when the site is known not to require
        near-fault amplification, including sites at D >=20 km.

    soil_class : {"A", "B", "C", "D", "E"}, default "C"
        NZS 1170.5 site subsoil class. No automatic conversion of an unknown
        class "U" to Class C is performed.

    spectrum_form : {"modal", "general"}, default "modal"
        Spectral-shape form from NZS 1170.5 Table 3.1. "modal" uses the
        bracketed short-period values applicable to modal response-spectrum
        and numerical-integration time-history analyses. "general" uses the
        general spectral-shape values.

    limit_state : {"ULS", "SLS"}, default "ULS"
        Limit state associated with the return-period factor. The Z*Ru <= 0.7
        restriction applies only to ULS calculations.

    enforce_uls_cap : bool, default True
        Enforce the NZS 1170.5 ULS restriction Z*Ru <= 0.7. Set False only
        when deliberately reproducing a legacy calculation, such as the
        supplied NZCodeSpectra.m implementation, which does not apply it.

    Returns
    -------
    pandas.DataFrame
        One row per oscillator period containing:

        period
            Oscillator period, s.
        C
            Horizontal elastic site hazard spectrum coefficient.
        Ch
            Spectral shape factor.
        Z
            Input NZS 1170.5 geographic hazard factor.
        R
            Return-period factor.
        N
            Near-fault factor.
        ZR_raw
            Product Z*R before application of the ULS cap.
        ZR_used
            Product Z*R actually used to calculate C(T).
        uls_cap_applied
            True where the ULS Z*Ru <= 0.7 restriction has modified Z*R.
        soil_class
            NZS 1170.5 site subsoil class.
        spectrum_form
            Spectral-shape form used.
        limit_state
            Limit state used.

    Notes
    -----
    C(T) is dimensionless and is numerically equivalent to spectral
    acceleration expressed in units of g, so it can be compared directly
    with UHS ordinates stored in g.

    The optional Class-D site-period interpolation introduced in Amendment 1
    is not applied here. Class D therefore uses the standard Table 3.1 Class-D
    spectral shape.

    The Class-E long-period expression uses 9.96/T**2, consistent with the
    tabulated NZS 1170.5 values. The supplied NZCodeSpectra.m implementation
    uses 9.66/T**2, which is inconsistent with Table 3.1.
    """
    T = np.asarray(periods, dtype=float)
    if T.ndim != 1:
        raise ValueError("periods must be a one-dimensional sequence")
    if np.any(~np.isfinite(T)) or np.any(T < 0):
        raise ValueError("periods must contain only finite, non-negative values")

    z = float(Z)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Z must be a finite positive NZS 1170.5 hazard factor")

    soil = soil_class.upper()
    form = spectrum_form.lower()
    state = limit_state.upper()
    comp = component.lower()

    if soil not in {"A", "B", "C", "D", "E"}:
        raise ValueError("soil_class must be one of A, B, C, D or E")
    if form not in {"modal", "general"}:
        raise ValueError("spectrum_form must be 'modal' or 'general'")
    if state not in {"ULS", "SLS"}:
        raise ValueError("limit_state must be 'ULS' or 'SLS'")
    if comp not in {"larger", "rotd50"}:
        raise ValueError("component must be 'larger' or 'rotd50'")

    R = nzs1170p5_return_period_factor(return_period)

    if return_period <= 250:
        N = np.ones_like(T)
    else:
        if near_fault_distance_km is None:
            raise ValueError(
                "near_fault_distance_km is required for return_period > 250 years; "
                "use np.inf when N(T,D)=1 is known to apply"
            )

        d = float(near_fault_distance_km)
        if np.isnan(d) or d < 0:
            raise ValueError("near_fault_distance_km must be non-negative or np.inf")

        if d >= 20:
            N = np.ones_like(T)
        else:
            nmax_table = np.array([
                [0.0, 1.00], [1.5, 1.00], [2.0, 1.12],
                [3.0, 1.36], [4.0, 1.60], [5.0, 1.72],
            ], dtype=float)
            nmax = np.interp(T, nmax_table[:, 0], nmax_table[:, 1])
            N = nmax if d <= 2 else 1.0 + (nmax - 1.0) * (20.0 - d) / 18.0

    Ch = _nzs1170p5_spectral_shape(T, soil, form)

    ZR_raw = z * R
    ZR_used = min(ZR_raw, 0.7) if state == "ULS" and enforce_uls_cap else ZR_raw
    cap_applied = bool(state == "ULS" and enforce_uls_cap and ZR_raw > 0.7)

    C_larger = Ch * ZR_used * N
    component_factor = _nzs1170p5_component_factor(T, comp)
    C = C_larger * component_factor

    return pd.DataFrame({
        "period": T,
        "C": C,
        "C_larger": C_larger,
        "component_factor": component_factor,
        "Ch": Ch,
        "Z": z,
        "R": R,
        "N": N,
        "ZR_raw": ZR_raw,
        "ZR_used": ZR_used,
        "uls_cap_applied": cap_applied,
        "soil_class": soil,
        "spectrum_form": form,
        "limit_state": state,
        "component": comp,
    })


def equivalent_return_period(
    return_periods: Sequence[float],
    uhs_sa: Sequence[float],
    design_sa: float,
) -> float:
    """Imazeki-style log-log interpolation of RP where UHS/design = 1.

    Returns NaN if the design level is not bracketed. UHS ordinates must be
    non-decreasing with return period; a reversal is treated as a data/calculation
    problem rather than being silently hidden by sorting on amplitude.
    """
    rp = np.asarray(return_periods, dtype=float)
    sa = np.asarray(uhs_sa, dtype=float)
    design = float(design_sa)
    if rp.size != sa.size or rp.size < 2:
        raise ValueError("return_periods and uhs_sa must have the same length >= 2")
    if not np.isfinite(design) or design <= 0:
        raise ValueError("design_sa must be positive and finite")

    valid = np.isfinite(sa) & (sa > 0) & np.isfinite(rp) & (rp > 0)
    rp, sa = rp[valid], sa[valid]
    if rp.size < 2:
        return float("nan")

    order = np.argsort(rp)
    rp, sa = rp[order], sa[order]
    if np.any(np.diff(rp) <= 0):
        raise ValueError("return_periods must be unique")
    if np.any(np.diff(sa) < -1e-12 * np.maximum(1.0, np.abs(sa[:-1]))):
        raise ValueError("UHS ordinates must be non-decreasing with return period")

    ratio = sa / design
    if not (ratio.min() <= 1.0 <= ratio.max()):
        return float("nan")

    exact = np.flatnonzero(np.isclose(ratio, 1.0, rtol=1e-12, atol=1e-14))
    if exact.size:
        return float(rp[exact[0]])

    hi = int(np.flatnonzero(ratio > 1.0)[0])
    lo = hi - 1
    return float(np.exp(np.interp(
        0.0,
        np.log(ratio[[lo, hi]]),
        np.log(rp[[lo, hi]]),
    )))


# -----------------------------------------------------------------------------
# OpenQuake job generation
# -----------------------------------------------------------------------------

def build_uhs_job(
    model: str,
    model_dir: str | Path,
    sites: pd.DataFrame,
    run_dir: str | Path,
    periods: Sequence[float] | None = None,
    return_periods: Sequence[float] | None = DEFAULT_RETURN_PERIODS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    include_pga: bool = True,
    im_levels: Sequence[float] | None = None,
    n_logic_tree_samples: int | None = None,
    description: str | None = None,
    poes: Sequence[float] | None = None,
    individual_rlzs: bool = False,
    max_sites_disagg: int | None = None,
    recompute_basin_depths: bool = False,
) -> Path:
    """Build one classical job for all study sites.

    For 300-400 sites this should normally be ONE OpenQuake job, not one job per
    site. Source/rupture work is then shared across the entire site set.

    ``individual_rlzs=True`` is required when the UC/GNS
    ``nshm_2022.get_hcurves_stats`` workflow will be used. It is deliberately
    opt-in for generic multi-site jobs because realization curves can make the
    OpenQuake datastore very large.

    ``max_sites_disagg`` is retained for explicit OpenQuake configuration, but it
    does not force a many-site classical calculation to store one rate chunk per
    site. Therefore it is NOT sufficient to make a large classical calculation
    reusable as a native disaggregation parent via ``--hc``. For multi-site
    disaggregation, run the disaggregation job directly and let OpenQuake execute
    its internal classical precalculation in disaggregation mode.
    """
    model = _normalise_model(model)
    root = resolve_model_root(model_dir, model)
    run_dir = Path(run_dir).resolve()
    _validate_oq_run_directory(root, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sites_csv = write_site_model(
        sites,
        run_dir / "sites.csv",
        model,
        root,
        recompute_basin_depths=recompute_basin_depths,
    )

    if periods is None:
        periods = NSHM2022_DEFAULT_PERIODS if model == "2022" else NSHM2010_DEFAULT_PERIODS
    if im_levels is None:
        im_levels = NSHM2022_DEFAULT_IM_LEVELS if model == "2022" else NSHM2010_DEFAULT_IM_LEVELS
    samples = _logic_tree_samples(model, n_logic_tree_samples)
    imtls = _format_imtls(periods, im_levels, include_pga=include_pga)
    target_poes = _resolve_uhs_poes(return_periods=return_periods, poes=poes)
    poes_text = " ".join(_fmt(p) for p in target_poes)

    sections = _base_sections(model, root, sites_csv, samples, description or f"NSHM{model} UHS", run_dir)
    sections["calculation"]["intensity_measure_types_and_levels"] = imtls
    if max_sites_disagg is not None:
        max_sites_disagg = int(max_sites_disagg)
        if max_sites_disagg < 1:
            raise ValueError("max_sites_disagg must be >= 1 or None")
        sections.setdefault("disaggregation", {})["max_sites_disagg"] = str(max_sites_disagg)
    sections["output"] = {
        "mean": "true",
        "individual_rlzs": "true" if individual_rlzs else "false",
        "uniform_hazard_spectra": "true",
        "poes": poes_text,
    }
    if quantiles:
        q = np.asarray(quantiles, dtype=float)
        if np.any((q <= 0) | (q >= 1)):
            raise ValueError("quantiles must be strictly between 0 and 1")
        sections["output"]["quantiles"] = " ".join(_fmt(x) for x in q)
    job = run_dir / f"job_uhs_nshm{model}.ini"
    _write_ini(sections, job)
    _write_poe_manifest(run_dir / "return_periods.csv", target_poes)
    return job


def build_nshm2022_uhs_validation_jobs(
    model_dir: str | Path,
    reference_csv: str | Path,
    run_root: str | Path,
    sample_sizes: Sequence[int] = (10_000, 100_000),
    include_quantiles: bool = False,
) -> list[Path]:
    """Build one-site NSHM2022 jobs matching the supplied GNS UHS query.

    The site, periods and exact 50-year PoE are always read from the benchmark CSV.
    Validation is mean-only by default, matching the production study. Set
    ``include_quantiles=True`` only when the benchmark's epistemic quantiles are
    deliberately required; that also stores individual realizations for the repo's
    realization-based statistics workflow.
    """
    ref = read_nshm2022_uhs_reference(reference_csv)
    site_cols = [c for c in ("lon", "lat", "vs30") if c in ref.columns]
    site = ref.iloc[[0]][site_cols].copy()
    site["vs30measured"] = 0
    # Do not inject a descriptive long ID here: OQ 3.23.x restricts
    # custom_site_id to 8 bytes. prepare_sites() will assign S00000.
    periods = sorted({_imt_period(c) for c in ref.columns if c.startswith("SA(")})
    quantiles = (
        sorted(
            float(str(x).replace("quantile-", ""))
            for x in ref["statistic"].astype(str).unique()
            if str(x) != "mean"
        )
        if include_quantiles else ()
    )
    p50 = float(ref["PoE (% in 50 years)"].iloc[0]) / 100.0
    target_poe = annual_poe_from_probability(p50, years=50.0)

    jobs = []
    for n in sample_sizes:
        run_dir = Path(run_root) / f"n{int(n):06d}"
        jobs.append(
            build_uhs_job(
                "2022", model_dir, site, run_dir, periods=periods,
                return_periods=None, poes=[target_poe], quantiles=quantiles,
                include_pga="PGA" in ref.columns,
                im_levels=NSHM2022_DEFAULT_IM_LEVELS,
                n_logic_tree_samples=int(n),
                description=f"NSHM2022 GNS mean-UHS validation, {int(n):,} samples",
                individual_rlzs=bool(include_quantiles),
            )
        )
    return jobs


def build_nshm2010_uhs_validation_job(
    model_dir: str | Path,
    reference_csv: str | Path,
    run_dir: str | Path,
    sites: pd.DataFrame | None = None,
) -> Path:
    """Build the exact full-enumeration NSHM2010 mean-UHS validation job.

    The benchmark/model/site configuration is validated before any job is written.
    This reproduces the supplied 2010 RotD50 McVerry job: one deterministic logic-tree
    path, the native 0.005--5 g/50-level grid, PGA + 0.1--3 s SA periods, and the two
    supplied annual PoEs.
    """
    root = resolve_model_root(model_dir, "2010")
    if sites is None:
        default_sites = root / "site_table_aaron.csv"
        if not default_sites.exists():
            raise FileNotFoundError(
                "sites was not supplied and site_table_aaron.csv is not present in the NSHM2010 model directory"
            )
        sites = pd.read_csv(default_sites)
    else:
        sites = sites.copy()

    setup = validate_nshm2010_uhs_validation_inputs(root, reference_csv, sites=sites)
    run_dir = Path(run_dir).resolve()
    job = build_uhs_job(
        "2010",
        root,
        sites,
        run_dir,
        periods=setup["periods"],
        return_periods=None,
        poes=setup["poes"],
        quantiles=(),
        include_pga=bool(setup["include_pga"]),
        im_levels=NSHM2010_DEFAULT_IM_LEVELS,
        n_logic_tree_samples=0,
        description="NSHM2010 supplied mean-UHS validation (McVerry06 MW RotD50)",
        individual_rlzs=False,
    )
    manifest = {
        **setup,
        "job_ini": str(job),
        "number_of_logic_tree_samples": 0,
        "im_levels_min_g": float(NSHM2010_DEFAULT_IM_LEVELS[0]),
        "im_levels_max_g": float(NSHM2010_DEFAULT_IM_LEVELS[-1]),
        "im_levels_count": int(len(NSHM2010_DEFAULT_IM_LEVELS)),
    }
    (run_dir / "nshm2010_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return job


def _validated_bin_edges(values: Sequence[float], name: str) -> list[float]:
    """Return finite, strictly increasing disaggregation bin edges as floats."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a 1-D sequence containing at least two edges")
    if np.any(~np.isfinite(arr)) or np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} must contain finite, strictly increasing values")
    return [float(x) for x in arr]


def build_disagg_job(
    model: str,
    model_dir: str | Path,
    sites: pd.DataFrame,
    run_dir: str | Path,
    periods: Sequence[float] = CALDERON_DISAGG_PERIODS,
    return_periods: Sequence[float] = DEFAULT_RETURN_PERIODS,
    include_pga: bool = False,
    im_levels: Sequence[float] | None = None,
    n_logic_tree_samples: int | None = None,
    num_rlzs_disagg: int | None = 0,
    mag_bin_edges: Sequence[float] | None = None,
    dist_bin_edges: Sequence[float] | None = None,
    eps_bin_edges: Sequence[float] | None = None,
    coordinate_bin_width_deg: float = DEFAULT_DISAGG_COORDINATE_BIN_WIDTH_DEG,
    mag_bin_width: float | None = None,
    distance_bin_width_km: float | None = None,
    num_epsilon_bins: int | None = None,
    description: str | None = None,
    recompute_basin_depths: bool = False,
) -> Path:
    """Build a multi-site native OpenQuake disaggregation job.

    By default the calculation uses explicit study bins for magnitude, OpenQuake
    distance (Rrup), and epsilon. The scalar ``mag_bin_width``,
    ``distance_bin_width_km`` and ``num_epsilon_bins`` arguments are retained as
    fallback controls for compatibility: supplying one replaces the corresponding
    default explicit edge set unless explicit edges are also supplied.

    ``num_rlzs_disagg=0`` follows OpenQuake >=3.17 semantics and disaggregates
    all sampled realizations. For NSHM2022 production this means all members of the
    configured Monte-Carlo sample (100,000 by default). A positive value selects
    that many realizations whose hazard is closest to the mean.

    For multi-site calculations this job should normally be run directly, without
    ``hazard_calculation_id``/``--hc``. OpenQuake then performs the required
    classical precalculation internally using the disaggregation job's site-wise
    chunking. Reusing a conventional many-site classical parent can fail because
    its rate data are stored in fewer chunks than there are sites.
    """
    model = _normalise_model(model)
    root = resolve_model_root(model_dir, model)
    run_dir = Path(run_dir).resolve()
    _validate_oq_run_directory(root, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sites_csv = write_site_model(
        sites,
        run_dir / "sites.csv",
        model,
        root,
        recompute_basin_depths=recompute_basin_depths,
    )
    if im_levels is None:
        im_levels = NSHM2022_DEFAULT_IM_LEVELS if model == "2022" else NSHM2010_DEFAULT_IM_LEVELS
    samples = _logic_tree_samples(model, n_logic_tree_samples)

    if not np.isfinite(coordinate_bin_width_deg) or float(coordinate_bin_width_deg) <= 0:
        raise ValueError("coordinate_bin_width_deg must be a positive finite value")
    if num_rlzs_disagg is not None and int(num_rlzs_disagg) < 0:
        raise ValueError("num_rlzs_disagg must be >= 0 or None")

    # Explicit edges are the production defaults. A supplied scalar fallback is
    # only used when its corresponding explicit edge set is omitted.
    if mag_bin_edges is None and mag_bin_width is None:
        mag_bin_edges = DEFAULT_DISAGG_MAG_BIN_EDGES
    if dist_bin_edges is None and distance_bin_width_km is None:
        dist_bin_edges = DEFAULT_DISAGG_DIST_BIN_EDGES
    if eps_bin_edges is None and num_epsilon_bins is None:
        eps_bin_edges = DEFAULT_DISAGG_EPS_BIN_EDGES

    explicit_edges: dict[str, list[float]] = {}
    if mag_bin_edges is not None:
        explicit_edges["mag"] = _validated_bin_edges(mag_bin_edges, "mag_bin_edges")
    elif mag_bin_width is not None:
        if not np.isfinite(mag_bin_width) or float(mag_bin_width) <= 0:
            raise ValueError("mag_bin_width must be a positive finite value")

    if dist_bin_edges is not None:
        explicit_edges["dist"] = _validated_bin_edges(dist_bin_edges, "dist_bin_edges")
    elif distance_bin_width_km is not None:
        if not np.isfinite(distance_bin_width_km) or float(distance_bin_width_km) <= 0:
            raise ValueError("distance_bin_width_km must be a positive finite value")

    if eps_bin_edges is not None:
        explicit_edges["eps"] = _validated_bin_edges(eps_bin_edges, "eps_bin_edges")
    elif num_epsilon_bins is not None:
        if int(num_epsilon_bins) < 1:
            raise ValueError("num_epsilon_bins must be >= 1")

    sections = _base_sections(model, root, sites_csv, samples, description or f"NSHM{model} disaggregation", run_dir)
    sections["general"]["calculation_mode"] = "disaggregation"
    sections["calculation"]["intensity_measure_types_and_levels"] = _format_imtls(periods, im_levels, include_pga=include_pga)
    # In OQ 3.23, poes_disagg is an alias for poes; use the current parameter name.
    target_poes = _resolve_uhs_poes(return_periods=return_periods, poes=None)
    sections["calculation"]["poes"] = " ".join(_fmt(p) for p in target_poes)

    disagg = {
        "max_sites_disagg": str(len(sites)),
        "coordinate_bin_width": _fmt(coordinate_bin_width_deg),
        "disagg_outputs": "TRT_Mag_Dist_Eps",
    }
    if explicit_edges:
        disagg["disagg_bin_edges"] = repr(explicit_edges)
    if mag_bin_edges is None and mag_bin_width is not None:
        disagg["mag_bin_width"] = _fmt(mag_bin_width)
    if dist_bin_edges is None and distance_bin_width_km is not None:
        disagg["distance_bin_width"] = _fmt(distance_bin_width_km)
    if eps_bin_edges is None and num_epsilon_bins is not None:
        disagg["num_epsilon_bins"] = str(int(num_epsilon_bins))
    if num_rlzs_disagg is not None:
        disagg["num_rlzs_disagg"] = str(int(num_rlzs_disagg))

    sections["disaggregation"] = disagg
    sections["output"] = {"mean": "true", "individual_rlzs": "false"}

    job = run_dir / f"job_disagg_nshm{model}.ini"
    _write_ini(sections, job)
    _write_return_period_manifest(run_dir / "return_periods.csv", return_periods)
    return job


def build_disagg_jobs(
    model: str,
    model_dir: str | Path,
    sites: pd.DataFrame,
    run_root: str | Path,
    chunk_size: int = 1,
    **kwargs,
) -> list[Path]:
    """Create memory-controlled disaggregation jobs for site chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    run_root = Path(run_root)
    jobs = []
    for i, start in enumerate(range(0, len(sites), chunk_size)):
        chunk = sites.iloc[start : start + chunk_size].copy()
        jobs.append(build_disagg_job(model, model_dir, chunk, run_root / f"chunk_{i:03d}", **kwargs))
    return jobs


# -----------------------------------------------------------------------------
# OpenQuake execution
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class OQRunResult:
    calc_id: int | None
    returncode: int
    command: tuple[str, ...]
    stdout: str
    log_file: str | None = None


def _parse_version3(text: str) -> tuple[int, int, int] | None:
    """Return the first semantic ``major.minor.patch`` tuple found in text."""
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)", str(text))
    return tuple(map(int, match.groups())) if match else None


def openquake_version(oq_command: str = "oq") -> tuple[int, int, int]:
    """Return the installed OpenQuake version as a 3-integer tuple.

    Package metadata is checked first. This avoids importing every OpenQuake CLI
    command merely to determine the version, which can obscure dependency problems
    (for example an incompatible pandas release). The CLI is retained as a fallback.
    """
    try:
        version_text = importlib_metadata.version("openquake.engine")
    except importlib_metadata.PackageNotFoundError:
        version_text = ""
    version = _parse_version3(version_text)
    if version is not None:
        return version

    attempts = ([oq_command, "--version"], [oq_command, "engine", "--version"])
    output = ""
    for cmd in attempts:
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"OpenQuake command {oq_command!r} was not found") from exc
        output = (result.stdout + "\n" + result.stderr).strip()
        version = _parse_version3(output)
        if version is not None:
            return version
    raise RuntimeError(f"Could not determine OpenQuake version from package metadata or CLI output: {output!r}")


def openquake_environment_info(oq_command: str = "oq") -> dict[str, object]:
    """Return a compact diagnostic of the Python/OpenQuake/pandas environment."""
    try:
        oq_version = openquake_version(oq_command)
    except Exception as exc:
        oq_version = None
        oq_error = f"{type(exc).__name__}: {exc}"
    else:
        oq_error = None

    pandas_version = _parse_version3(pd.__version__)
    problems: list[str] = []
    warnings: list[str] = []

    if oq_version is not None and oq_version[:2] == (3, 23):
        if pandas_version is not None and pandas_version[0] >= 3:
            problems.append(
                "OpenQuake 3.23 imports pandas.errors.SettingWithCopyWarning, which was removed in pandas 3.0; "
                "use the OpenQuake 3.23 Windows dependency set (pandas 2.0.3)."
            )
        if sys.version_info[:2] > (3, 11):
            warnings.append(
                "OpenQuake 3.23.4 is documented/tested with Python 3.9-3.11; for reproducible validation use Python 3.11."
            )

    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "python_executable": sys.executable,
        "openquake": ".".join(map(str, oq_version)) if oq_version is not None else None,
        "openquake_error": oq_error,
        "oq_command": shutil.which(oq_command),
        "pandas": pd.__version__,
        "problems": problems,
        "warnings": warnings,
    }


def require_openquake_version(
    oq_command: str = "oq",
    minimum: tuple[int, int, int] = OQ_MIN_VERSION,
) -> tuple[int, int, int]:
    """Raise if the installed engine is older than the required version or unusable."""
    version = openquake_version(oq_command)
    if version < minimum:
        raise RuntimeError(f"OpenQuake >= {'.'.join(map(str, minimum))} is required; found {'.'.join(map(str, version))}")

    env = openquake_environment_info(oq_command)
    if env["problems"]:
        details = "\n- ".join(str(x) for x in env["problems"])
        raise RuntimeError(f"OpenQuake environment is incompatible:\n- {details}")
    return version


def _parse_oq_calc_id(text: str) -> int | None:
    """Return the OpenQuake calculation ID from engine stdout.

    Successful runs end with a datastore path such as ``calc_5.hdf5``; that is
    the most reliable identifier and is therefore preferred.  If a run fails
    before the datastore is written, fall back to the ``[#ID LEVEL]`` prefix
    emitted by the engine log.  Deliberately do *not* match generic phrases
    such as ``regular calculation with 79 outputs``.
    """
    stored = re.findall(r"\bcalc_(\d+)\.hdf5\b", text, flags=re.I)
    if stored:
        return int(stored[-1])

    logged = re.findall(
        r"^\[[^]\r\n]*\s#(\d+)\s+(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\]",
        text,
        flags=re.I | re.M,
    )
    if logged:
        return int(logged[-1])
    return None


def _job_calculation_mode(job_ini: str | Path) -> str | None:
    """Read ``calculation_mode`` from a generated OpenQuake job.ini."""
    match = re.search(
        r"^\s*calculation_mode\s*=\s*([^#;\r\n]+)",
        Path(job_ini).read_text(encoding="utf-8", errors="replace"),
        flags=re.I | re.M,
    )
    return match.group(1).strip().lower() if match else None


def inspect_disagg_parent_chunks(hazard_calculation_id: int) -> dict[str, int | str]:
    """Inspect whether an OQ classical parent has one rate chunk per site.

    Native multi-site disaggregation requires one map getter/rate chunk per site.
    A conventional classical parent commonly stores only ``concurrent_tasks / 2``
    chunks, which cannot be reused through ``--hc`` for disaggregation even when
    ``max_sites_disagg`` is increased in the parent job.
    """
    try:
        from openquake.calculators import getters
        from openquake.commonlib.datastore import read
    except ImportError as exc:
        raise ImportError(
            "OpenQuake is required to inspect a disaggregation parent; use the "
            "same OQ environment used to run the hazard calculation"
        ) from exc

    calc_id = int(hazard_calculation_id)
    with read(calc_id) as ds:
        n_sites = int(len(ds["sitecol/sids"]))
        oq = ds["oqparam"]
        mode = str(getattr(oq, "calculation_mode", ""))
        full_lt = ds["full_lt"].init()
        n_chunks = int(getters.get_num_chunks(ds, full_lt))

    return {
        "calc_id": calc_id,
        "calculation_mode": mode,
        "n_sites": n_sites,
        "n_chunks": n_chunks,
    }


def _check_disagg_parent_compatibility(job_ini: str | Path, hazard_calculation_id: int) -> None:
    """Fail early when ``--hc`` would reproduce OQ's sites/chunks error."""
    if _job_calculation_mode(job_ini) != "disaggregation":
        return
    info = inspect_disagg_parent_chunks(hazard_calculation_id)
    if int(info["n_chunks"]) < int(info["n_sites"]):
        raise RuntimeError(
            "The requested OpenQuake parent cannot be reused for this multi-site "
            "disaggregation: parent calc {calc_id} has {n_sites} sites but only "
            "{n_chunks} rate chunks. OpenQuake's native disaggregation requires "
            "site-wise chunks here. Run this disaggregation job directly, i.e. "
            "shw.run_openquake(job_disagg, num_cores=...), WITHOUT "
            "hazard_calculation_id. OpenQuake will run its required classical "
            "precalculation internally in disaggregation mode. Increasing "
            "max_sites_disagg on a conventional classical parent does not change "
            "this chunking constraint."
            .format(**info)
        )

def _looks_like_openquake_memory_error(text: str) -> bool:
    """Recognize NumPy/OpenQuake memory failures, including 3.23.x wrapper errors."""
    lower = str(text).lower()
    markers = (
        "_arraymemoryerror",
        "unable to allocate",
        "cannot allocate memory",
        "out of memory",
        "memoryerror",
    )
    return any(marker in lower for marker in markers)


def _parse_oq_processpool_workers(text: str) -> int | None:
    """Return the latest process-pool worker count reported by OpenQuake.

    OpenQuake 3.23.4 emits ``Using N processpool workers`` during engine startup.
    Newer engines may append additional text (for example ``concurrent_tasks``),
    so only the stable leading phrase is parsed.
    """
    matches = re.findall(
        r"\bUsing\s+(\d+)\s+processpool\s+workers\b",
        str(text),
        flags=re.I,
    )
    return int(matches[-1]) if matches else None


def _openquake_subprocess_env(runtime_cfg: Path | None, num_cores: int | None) -> dict[str, str]:
    """Build the child environment used to start the OpenQuake executable.

    OpenQuake 3.23.4 initializes its process-pool size while importing the engine
    modules. Therefore a runtime ``openquake.cfg`` must be exposed through
    ``OQ_CONFIG_FILE`` *before* the ``oq`` process starts; supplying the same file
    later via ``oq engine --config-file`` is too late to reliably constrain the
    already-imported process-pool state.

    The caller's environment is copied and never mutated. When ``num_cores`` is
    explicit, nested BLAS/OpenMP threading is also forced to one thread per
    OpenQuake process so the requested process count is a meaningful CPU/memory
    bound rather than being multiplied by library-level threads.
    """
    env = os.environ.copy()
    if runtime_cfg is not None:
        if num_cores is None:
            raise ValueError("runtime_cfg requires an explicit num_cores")
        env["OQ_CONFIG_FILE"] = str(Path(runtime_cfg).resolve())
        env["OQ_DISTRIBUTE"] = "processpool"
        # Harmless on 3.23.4 and reinforces the same bound on newer engines.
        env["OQ_NUM_CORES"] = str(int(num_cores))
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def _terminate_oq_subprocess(proc: subprocess.Popen) -> None:
    """Terminate an OpenQuake process and, where practical, its child workers."""
    if proc.poll() is not None:
        return

    if os.name == "nt":
        # OpenQuake processpool workers are child processes on Windows. taskkill /T
        # prevents a worker-count validation failure from leaving orphaned workers.
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()


def run_openquake(
    job_ini: str | Path,
    oq_command: str = "oq",
    use_rates: bool = True,
    max_potential_paths: int = MAX_POTENTIAL_PATHS,
    check_version: bool = True,
    echo: bool = True,
    hazard_calculation_id: int | None = None,
    num_cores: int | None = None,
) -> OQRunResult:
    """Run one OpenQuake job and return its calculation ID.

    ``hazard_calculation_id`` adds OpenQuake's ``--hc`` option. Before using it
    for a disaggregation job, the parent is checked for the site-wise rate chunking
    required by OpenQuake.

    When ``num_cores`` is supplied, the subprocess is constrained with a temporary
    ``openquake.cfg`` exposed through ``OQ_CONFIG_FILE`` *before* OpenQuake starts.
    This ordering is required by OpenQuake 3.23.4, whose process-pool worker count
    is initialized during module import. The engine's startup message is parsed and
    the run is terminated immediately if its reported processpool worker count does
    not equal ``num_cores``. Thus ``num_cores=8`` is verified to mean eight actual
    OpenQuake worker processes rather than merely writing a configuration request.

    Console output is streamed to ``openquake_console.log`` in the job directory.
    Only a bounded tail is retained in memory, preventing long calculations from
    accumulating unbounded Python-side log text. OpenQuake/NumPy memory failures
    are re-raised as a clear ``MemoryError`` even when OQ 3.23.x masks the original
    ``_ArrayMemoryError`` with a secondary ``TypeError``.
    """
    if isinstance(job_ini, (list, tuple)):
        raise TypeError(
            "run_openquake() accepts one job path, but a sequence was supplied. "
            "Run generated jobs one at a time."
        )

    job_ini = Path(job_ini).resolve()
    if not job_ini.is_file():
        raise FileNotFoundError(f"OpenQuake job file does not exist: {job_ini}")

    if check_version:
        require_openquake_version(oq_command)

    if hazard_calculation_id is not None:
        hc = int(hazard_calculation_id)
        if hc < 0:
            raise ValueError("hazard_calculation_id must be >= 0 or None")
        _check_disagg_parent_compatibility(job_ini, hc)
    else:
        hc = None

    runtime_cfg: Path | None = None
    if num_cores is not None:
        num_cores = int(num_cores)
        if num_cores < 1:
            raise ValueError("num_cores must be >= 1 or None")
        runtime_cfg = job_ini.parent / "openquake_runtime.cfg"
        runtime_cfg.write_text(
            "[distribution]\n"
            "oq_distribute = processpool\n"
            f"num_cores = {num_cores}\n"
            "serialize_jobs = 1\n",
            encoding="utf-8",
        )

    if max_potential_paths is not None:
        max_potential_paths = int(max_potential_paths)
        if max_potential_paths < 1:
            raise ValueError("max_potential_paths must be >= 1 or None")

    cmd = [oq_command, "engine", "--run", job_ini.name]
    # Deliberately DO NOT append --config-file here. For OQ 3.23.4 the config must
    # be visible through OQ_CONFIG_FILE before the oq executable imports the engine.
    if hc is not None:
        cmd += ["--hc", str(hc)]
    if use_rates:
        cmd += ["-p", "use_rates=true"]
    if max_potential_paths is not None:
        cmd += ["-p", f"max_potential_paths={max_potential_paths}"]

    child_env = _openquake_subprocess_env(runtime_cfg, num_cores)
    console_log = job_ini.parent / "openquake_console.log"
    proc = subprocess.Popen(
        cmd,
        cwd=job_ini.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
        start_new_session=(os.name != "nt"),
    )

    tail: deque[str] = deque(maxlen=5000)
    if proc.stdout is None:
        _terminate_oq_subprocess(proc)
        raise RuntimeError("OpenQuake subprocess did not expose stdout")

    reported_workers: int | None = None
    worker_mismatch: tuple[int, int] | None = None

    with console_log.open("w", encoding="utf-8", errors="replace") as log:
        for line in proc.stdout:
            tail.append(line)
            log.write(line)
            log.flush()
            if echo:
                print(line, end="")

            workers = _parse_oq_processpool_workers(line)
            if workers is not None:
                reported_workers = workers
                if num_cores is not None and workers != num_cores:
                    worker_mismatch = (num_cores, workers)
                    msg = (
                        "OpenQuake worker-count mismatch: "
                        f"requested num_cores={num_cores}, but the engine reported "
                        f"{workers} processpool workers. Terminating before the "
                        "calculation can oversubscribe CPU/RAM.\n"
                    )
                    tail.append(msg)
                    log.write(msg)
                    log.flush()
                    if echo:
                        print(msg, end="")
                    _terminate_oq_subprocess(proc)
                    break

    rc = proc.wait()
    output_tail = "".join(tail)

    if worker_mismatch is not None:
        expected, actual = worker_mismatch
        raise RuntimeError(
            "OpenQuake processpool verification failed: "
            f"requested {expected} workers but OpenQuake reported {actual}. "
            f"The run was terminated. Inspect {console_log}."
        )

    calc_id = _parse_oq_calc_id(output_tail)

    if rc != 0:
        if _looks_like_openquake_memory_error(output_tail):
            mode = _job_calculation_mode(job_ini) or "unknown"
            raise MemoryError(
                "OpenQuake failed because a worker exhausted memory "
                f"(calculation_mode={mode!r}). OQ 3.23.x may mask NumPy's "
                "_ArrayMemoryError with a secondary TypeError. For disaggregation, "
                "split IMTs first and, if needed, site chunks; lowering num_cores "
                "reduces concurrent worker memory but cannot shrink a single task. "
                f"Full console log: {console_log}"
            )
        if (
            "versioningnotinstalled" in output_tail.lower()
            or "no such table: revision_info" in output_tail.lower()
        ):
            raise RuntimeError(
                "OpenQuake's own database is not initialized for this installation. "
                "The workflow does not modify or upgrade the OpenQuake database. "
                "Run `oq engine --upgrade-db --yes` once in the same OpenQuake "
                f"environment, then rerun the job. Full console log: {console_log}"
            )
        raise subprocess.CalledProcessError(rc, cmd, output=output_tail)

    if num_cores is not None and reported_workers is None:
        raise RuntimeError(
            "OpenQuake returned without reporting its processpool worker count, so "
            f"the requested num_cores={num_cores} could not be verified. "
            f"Inspect {console_log}."
        )

    if calc_id is None:
        raise RuntimeError(
            "OpenQuake returned success but no calculation ID could be parsed from "
            f"the console output. Inspect {console_log}"
        )

    return OQRunResult(
        calc_id=calc_id,
        returncode=rc,
        command=tuple(cmd),
        stdout=output_tail,
        log_file=str(console_log),
    )


# -----------------------------------------------------------------------------
# OpenQuake extraction
# -----------------------------------------------------------------------------

def extract_uhs(calc_id: int, output_csv: str | Path | None = None) -> pd.DataFrame:
    """Extract all configured statistical UHS (mean and requested quantiles).

    OpenQuake's ``uhs?...`` extractor returns a generator-backed ``ArrayWrapper``
    whose statistics are stored as attributes containing nested structured arrays.
    Such wrappers intentionally do *not* have ``shape_descr`` and therefore cannot
    be passed to ``ArrayWrapper.to_dframe()``.  This routine follows OpenQuake's
    own UHS plotting convention and decodes those nested arrays explicitly.
    """
    Extractor, read = _openquake_imports()
    with read(int(calc_id)) as ds:
        oq = ds["oqparam"]
        kinds = list(oq.hazard_stats())
        poes = np.asarray(oq.poes, dtype=float)
        imts = [imt.string for imt in oq.imt_periods()]

    if not kinds:
        raise ValueError("The OpenQuake calculation contains no configured UHS statistics")
    if poes.size == 0:
        raise ValueError("The OpenQuake calculation contains no UHS PoEs")

    frames = []
    with Extractor(int(calc_id)) as ex:
        for kind in kinds:
            wrapper = ex.get(f"uhs?kind={kind}")
            frames.append(_uhs_arraywrapper_to_frame(wrapper, kind, poes, imts))

    df = pd.concat(frames, ignore_index=True)
    df["return_period"] = [return_period_from_probability(float(p), 1.0) for p in df["poe"]]

    site_df = extract_site_collection(calc_id).reset_index(drop=True).reset_index(names="site_id")
    df = df.merge(site_df, on="site_id", how="left", suffixes=("", "_site"), validate="many_to_one")

    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def extract_site_collection(calc_id: int) -> pd.DataFrame:
    """Extract the OpenQuake site collection for robust site-id mapping.

    ``sitecol`` is a structured array, not a shape-described dense array, so it is
    converted directly with pandas rather than through ``ArrayWrapper.to_dframe``.
    """
    Extractor, _ = _openquake_imports()
    with Extractor(int(calc_id)) as ex:
        wrapper = ex.get("sitecol")
        if not hasattr(wrapper, "array"):
            raise TypeError("OpenQuake sitecol extractor did not return an array-backed object")
        arr = np.asarray(wrapper.array)
    if arr.dtype.names is None:
        raise TypeError("OpenQuake sitecol array is not a structured array")
    return _decode_object_columns(pd.DataFrame.from_records(arr))


def _resolve_disagg_spec(dstore, kind: str, spec: str = "auto") -> str:
    """Resolve ``stats`` versus ``rlzs`` native OpenQuake disaggregation output.

    ``num_rlzs_disagg=1`` stores a realization result (``disagg-rlzs``), whereas
    mean/all-realization production calculations normally expose
    ``disagg-stats``.  ``spec="auto"`` prefers statistical output when available
    and otherwise falls back to realization output.
    """
    requested = str(spec).strip().lower()
    if requested not in {"auto", "stats", "rlzs"}:
        raise ValueError("spec must be 'auto', 'stats', or 'rlzs'")

    available = []
    for candidate in ("stats", "rlzs"):
        try:
            dstore[f"disagg-{candidate}/{kind}"]
        except KeyError:
            continue
        else:
            available.append(candidate)

    if requested == "auto":
        if "stats" in available:
            return "stats"
        if "rlzs" in available:
            return "rlzs"
    elif requested in available:
        return requested

    raise KeyError(
        f"No native OpenQuake disaggregation output for kind={kind!r}, "
        f"spec={requested!r}; available specs are {available}"
    )


def _validate_disagg_min_contribution(value: float) -> float:
    """Validate the durable disaggregation contribution threshold.

    The extracted M-Rrup-epsilon table is the complete discrete causal-hazard
    distribution used to compute moments. Removing positive tail cells before
    :func:`summarize_disaggregation` changes the mean and, especially, the
    standard deviation. Consequently the durable scientific extraction must keep
    every positive-contribution cell. ``min_contribution`` is retained in the
    public API for backwards compatibility, but only zero is accepted.

    Apply any visual/display threshold to a *copy* after the summary statistics
    have been computed.
    """
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("min_contribution must be finite and >= 0")
    if threshold > 0:
        raise ValueError(
            "min_contribution must be 0 for moment-preserving disaggregation extraction; "
            "dropping positive cells changes Mw/Rrup/epsilon moments. Filter a copy "
            "only after summarize_disaggregation() when a display threshold is needed."
        )
    return 0.0


def _disagg_probabilities_to_annual_rates(
    probabilities: Sequence[float],
    investigation_time: float,
) -> np.ndarray:
    """Convert OpenQuake disaggregation cell probabilities to annual rates.

    OpenQuake stores statistical disaggregation PMFs in probability space after
    forming its disaggregation matrices. Rate space is recovered with the exact
    Poisson transform ``lambda = -log(1-p) / T`` before contributions are
    normalized. For NSHM2022 ``disagg-stats`` this therefore operates on the
    OpenQuake *mean disaggregation*, whose multi-realization averaging has already
    occurred upstream in OpenQuake rate space; it does not average normalized
    realization PMFs in this workflow.
    """
    p = np.asarray(probabilities, dtype=float)
    time = float(investigation_time)
    if not np.isfinite(time) or time <= 0:
        raise ValueError("investigation_time must be positive and finite")
    if not np.isfinite(p).all():
        raise ValueError("disaggregation probabilities contain non-finite values")

    # Permit only tiny floating-point excursions around the probability bounds.
    tol = 1e-12
    if np.any(p < -tol) or np.any(p > 1.0 + tol):
        raise ValueError("disaggregation probabilities must lie in [0, 1]")

    p = np.clip(p, 0.0, 1.0 - 1e-15)
    return -np.log1p(-p) / time


def extract_disaggregation(
    calc_id: int,
    output_csv: str | Path | None = None,
    kind: str = "TRT_Mag_Dist_Eps",
    marginalize_tectonic_region: bool = True,
    min_contribution: float = 0.0,
    spec: str = "auto",
) -> pd.DataFrame:
    """Extract the complete normalized OpenQuake M-Rrup-epsilon distribution.

    The scientific quantity returned here is one causal-hazard distribution for
    every site/IMT/PoE. OpenQuake cell probabilities are first converted back to
    additive annual-rate contributions using ``-log(1-p) / T`` and are then
    normalized so their sum is one.

    For NSHM2010 the production model has one deterministic logic-tree path. For
    NSHM2022, ``spec="auto"`` preferentially selects ``disagg-stats``; OpenQuake
    has already formed its realization-weighted *mean rate disaggregation* before
    saving that statistical PMF. The downstream normalization and all source
    moments are therefore identical for the two models. Realization-to-realization
    variability is not used to define ``mw_std``, ``r_rup_std`` or
    ``epsilon_std``.

    By default TRT is marginalized out because the durable study product is the
    M-Rrup-epsilon distribution. Every positive-contribution cell is retained.
    ``min_contribution`` remains in the signature for backwards compatibility but
    must be zero: pruning tails before computing moments would bias the reported
    means and standard deviations.

    ``spec="auto"`` transparently handles production ``disagg-stats`` output and
    single-realization ``disagg-rlzs`` output used by plumbing tests.
    """
    _validate_disagg_min_contribution(min_contribution)

    Extractor, read = _openquake_imports()
    with read(int(calc_id)) as ds:
        oq = ds["oqparam"]
        imtls = getattr(oq, "hazard_imtls", None)
        if imtls is None:
            imtls = getattr(oq, "imtls", None)
        if imtls is None:
            raise AttributeError("OpenQuake oqparam has neither hazard_imtls nor imtls")
        imts = list(imtls)
        poes = np.asarray(oq.poes, dtype=float)
        investigation_time = float(oq.investigation_time)
        disagg_bin_edges = dict(getattr(oq, "disagg_bin_edges", {}) or {})
        resolved_spec = _resolve_disagg_spec(ds, kind, spec=spec)

    if not np.isclose(investigation_time, 1.0):
        raise ValueError("This workflow expects investigation_time = 1.0 year")

    site_df = extract_site_collection(calc_id)
    n_sites = len(site_df)
    frames: list[pd.DataFrame] = []

    with Extractor(int(calc_id)) as ex:
        for site_id in range(n_sites):
            for imt in imts:
                for poe_id, poe in enumerate(poes):
                    aw = ex.get(
                        f"disagg?kind={kind}&spec={resolved_spec}"
                        f"&site_id={site_id}&imt={imt}&poe_id={poe_id}"
                    )
                    dims = list(aw.shape_descr[:-2])
                    coords = {}
                    for dim in dims:
                        values = np.asarray(aw[dim])
                        if dim == "trt":
                            values = values.astype(str)
                        coords[dim] = values

                    array = np.asarray(aw.array, dtype=float)
                    while array.ndim > len(dims):
                        if array.shape[-1] != 1:
                            raise RuntimeError(
                                "The selected disagg-rlzs extraction contains more than one "
                                f"trailing realization/statistic value: shape={array.shape}, "
                                f"dims={dims}. Use statistical disaggregation output for "
                                "multi-realization production extraction."
                            )
                        array = array[..., 0]

                    frame = _array_to_long_frame(array, dims, coords)
                    raw = frame.pop("value").to_numpy(float)
                    rates = _disagg_probabilities_to_annual_rates(raw, investigation_time)
                    total = float(rates.sum())
                    frame["contribution"] = rates / total if total > 0 else np.nan
                    frame["site_id"] = site_id
                    frame["imt"] = imt
                    frame["period"] = _imt_period(imt)
                    frame["poe"] = float(poe)
                    frame["return_period"] = return_period_from_probability(float(poe), years=1.0)
                    frames.append(frame)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return result

    result = result.rename(columns={
        "trt": "tectonic_region",
        "mag": "mw",
        "dist": "r_rup",
        "eps": "epsilon",
    })

    if marginalize_tectonic_region and "tectonic_region" in result.columns:
        group_cols = [c for c in result.columns if c not in {"tectonic_region", "contribution"}]
        result = (
            result.groupby(group_cols, dropna=False, as_index=False, sort=False)["contribution"]
            .sum()
        )

    # Durable scientific output keeps the complete positive-mass distribution.
    result = result[pd.to_numeric(result["contribution"], errors="coerce") > 0.0].copy()

    # A positive-mass cell with an invalid coordinate would silently corrupt the
    # first and second moments, so reject it rather than dropping it.
    moment_cols = ("mw", "r_rup", "epsilon")
    if all(c in result.columns for c in moment_cols):
        contribution = pd.to_numeric(result["contribution"], errors="coerce").to_numpy(float)
        positive = np.isfinite(contribution) & (contribution > 0)
        coords = result.loc[:, moment_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if positive.any() and not np.isfinite(coords[positive]).all():
            raise RuntimeError(
                "Extracted disaggregation contains a positive-contribution cell "
                "with non-finite Mw, Rrup or epsilon"
            )

    norm_groups = ["site_id", "imt", "poe"]
    totals = result.groupby(norm_groups, dropna=False)["contribution"].sum().to_numpy(float)
    if totals.size and not np.allclose(totals, 1.0, rtol=1e-8, atol=1e-10):
        raise RuntimeError("Extracted disaggregation contributions do not sum to one within site/IMT/PoE")

    result = result.merge(
        site_df.reset_index(drop=True).reset_index(names="site_id"),
        on="site_id",
        how="left",
        suffixes=("", "_site"),
        validate="many_to_one",
    )

    preferred = [
        "site_id", "custom_site_id", "lon", "lat", "vs30",
        "imt", "period", "poe", "return_period",
        "mw", "r_rup", "epsilon", "contribution", "tectonic_region",
    ]
    result = result[[c for c in preferred if c in result.columns] + [c for c in result.columns if c not in preferred]]
    result.attrs["distance_metric"] = "Rrup"
    result.attrs["disagg_bin_edges"] = disagg_bin_edges
    result.attrs["openquake_disagg_spec"] = resolved_spec
    result.attrs["contribution_semantics"] = (
        "normalized annual-rate fraction within each site/IMT/PoE; all positive cells retained"
    )
    result.attrs["dispersion_semantics"] = (
        "population standard deviation of M-Rrup-epsilon bin coordinates weighted by "
        "the final normalized hazard-contribution distribution"
    )
    result.attrs["logic_tree_semantics"] = (
        "OpenQuake statistical mean disaggregation; multi-realization averaging occurs "
        "upstream in OpenQuake rate space before this normalization"
        if resolved_spec == "stats"
        else "single/selected realization disaggregation"
    )

    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
    return result


def _weighted_population_mean_std(
    values: Sequence[float],
    weights: Sequence[float],
    name: str,
) -> tuple[float, float]:
    """Return the weighted population mean and standard deviation.

    ``weights`` represent fractions of the complete discrete hazard-contribution
    distribution, not a random sample. The variance therefore uses the population
    definition ``sum(w * (x - mean)**2) / sum(w)`` with no Bessel correction.
    The centered second moment is used instead of ``E[x**2] - E[x]**2`` to reduce
    cancellation error when the distribution is narrow relative to its location.
    """
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.ndim != 1 or x.size != w.size:
        raise ValueError(f"{name} values and weights must be aligned 1-D arrays")
    if x.size == 0:
        raise ValueError(f"{name} distribution is empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    if not np.isfinite(w).all():
        raise ValueError(f"{name} contains non-finite contribution weights")
    if np.any(w < 0):
        raise ValueError(f"{name} contains negative contribution weights")

    total = float(w.sum())
    if total <= 0:
        raise ValueError(f"{name} contribution weights must sum to a positive value")

    wn = w / total
    mean = float(np.sum(wn * x))
    variance = float(np.sum(wn * np.square(x - mean)))
    variance = max(variance, 0.0)
    return mean, float(np.sqrt(variance))


def summarize_disaggregation(disagg: pd.DataFrame) -> pd.DataFrame:
    """Summarize the final causal-hazard distribution for each site/IMT/RP.

    The input is the complete positive M-Rrup-epsilon contribution distribution.
    For NSHM2010 this comes from its single logic-tree path. For NSHM2022 it should
    be the OpenQuake ``disagg-stats`` mean distribution: OpenQuake has already
    realization-weighted the underlying disaggregation in rate space, so the same
    downstream equations apply to both models.

    Three established NSHM-style point summaries are returned:

    ``mean_source``
        Contribution-weighted mean Mw, Rrup and epsilon. ``contribution`` is NaN
        because no individual cell represents the mean source.

    ``mode_mag_dist``
        Highest-contributing Mw-Rrup cell after marginalizing epsilon. Epsilon is
        NaN and ``contribution`` is that Mw-Rrup cell's total fraction.

    ``mode_mag_dist_eps``
        Highest-contributing joint Mw-Rrup-epsilon cell and its contribution.

    The columns ``mw_std``, ``r_rup_std`` and ``epsilon_std`` describe the spread
    of that same final causal distribution. For coordinate ``x`` and normalized
    cell weights ``w`` they use the population second central moment

        mu_x    = sum_i w_i x_i
        sigma_x = sqrt(sum_i w_i (x_i - mu_x)**2).

    There is no Bessel correction: the weighted cells constitute the discrete
    distribution being summarized rather than a random sample from an unknown
    population. These are distributional dispersions, not standard errors or
    epistemic logic-tree uncertainties. In particular, NSHM2022 ``*_std`` is not
    the standard deviation of realization-specific mean or modal sources.

    OpenQuake exposes disaggregation coordinates as bin representatives; hence the
    reported means and standard deviations are bin-based numerical summaries and
    inherit the resolution of the configured Mw, Rrup and epsilon bins.

    The same ``*_std`` values are repeated on all three statistic rows so any
    filtered point-summary table remains self-contained. A mode itself does not
    have a separate standard deviation.
    """
    data = disagg.rename(columns={
        "mag": "mw",
        "dist": "r_rup",
        "distance": "r_rup",
        "r_jb": "r_rup",  # backwards-compatible alias from v0.3.2
        "eps": "epsilon",
    }).copy()
    _require_columns(data, ("site_id", "return_period", "mw", "r_rup", "epsilon", "contribution"))

    identity_cols = [
        c for c in (
            "model", "site_id", "custom_site_id", "lon", "lat", "vs30",
            "imt", "period", "poe", "return_period",
        )
        if c in data.columns
    ]
    rows: list[dict[str, object]] = []

    for key, g in data.groupby(identity_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(identity_cols, key))

        # Summing duplicate coordinate triplets makes the result invariant to row
        # splitting and to whether TRT has already been marginalized upstream.
        cells = (
            g.groupby(["mw", "r_rup", "epsilon"], dropna=False, as_index=False, sort=False)["contribution"]
            .sum()
        )
        w = pd.to_numeric(cells["contribution"], errors="coerce").to_numpy(float)
        valid = np.isfinite(w) & (w > 0)
        if not valid.any() or w[valid].sum() <= 0:
            continue

        cells = cells.loc[valid].reset_index(drop=True)
        w = w[valid]
        w = w / w.sum()
        cells["contribution"] = w

        mw_values = pd.to_numeric(cells["mw"], errors="coerce").to_numpy(float)
        rrup_values = pd.to_numeric(cells["r_rup"], errors="coerce").to_numpy(float)
        epsilon_values = pd.to_numeric(cells["epsilon"], errors="coerce").to_numpy(float)

        mw_mean, mw_std = _weighted_population_mean_std(mw_values, w, "Mw")
        rrup_mean, rrup_std = _weighted_population_mean_std(rrup_values, w, "Rrup")
        epsilon_mean, epsilon_std = _weighted_population_mean_std(epsilon_values, w, "epsilon")

        dispersion = {
            "mw_std": mw_std,
            "r_rup_std": rrup_std,
            "epsilon_std": epsilon_std,
        }

        rows.append(base | {
            "statistic": "mean_source",
            "mw": mw_mean,
            "r_rup": rrup_mean,
            "epsilon": epsilon_mean,
            "contribution": np.nan,
        } | dispersion)

        md = (
            cells.groupby(["mw", "r_rup"], dropna=False, as_index=False, sort=False)["contribution"]
            .sum()
        )
        md_mode = md.loc[md["contribution"].idxmax()]
        rows.append(base | {
            "statistic": "mode_mag_dist",
            "mw": float(md_mode["mw"]),
            "r_rup": float(md_mode["r_rup"]),
            "epsilon": np.nan,
            "contribution": float(md_mode["contribution"]),
        } | dispersion)

        mde_mode = cells.loc[cells["contribution"].idxmax()]
        rows.append(base | {
            "statistic": "mode_mag_dist_eps",
            "mw": float(mde_mode["mw"]),
            "r_rup": float(mde_mode["r_rup"]),
            "epsilon": float(mde_mode["epsilon"]),
            "contribution": float(mde_mode["contribution"]),
        } | dispersion)

    result = pd.DataFrame(rows)
    if not result.empty:
        order = [c for c in identity_cols if c in result.columns] + [
            "statistic",
            "mw", "mw_std",
            "r_rup", "r_rup_std",
            "epsilon", "epsilon_std",
            "contribution",
        ]
        result = result[order]
        result.attrs["dispersion_semantics"] = (
            "population standard deviation of the final normalized causal-hazard distribution; "
            "not epistemic realization-to-realization uncertainty"
        )
        result.attrs["distance_metric"] = "Rrup"
    return result


def plot_mag_dist_disaggregation(
    disagg: pd.DataFrame,
    period: float,
    return_period: float | None = None,
    summary: pd.DataFrame | None = None,
    mag_bin_edges: Sequence[float] = DEFAULT_DISAGG_MAG_BIN_EDGES,
    dist_bin_edges: Sequence[float] = DEFAULT_DISAGG_DIST_BIN_EDGES,
    elev: float = 28.0,
    azim: float = -55.0,
    figsize: tuple[float, float] = (10.0, 7.0),
    dpi: int = 180,
):
    """Plot the conventional 3-D Mw-Rrup disaggregation bars.

    Epsilon (and TRT, if still present) is marginalized. Bar height is percent
    contribution. The weighted mean source and Mw-Rrup mode are overlaid.
    """
    import matplotlib.pyplot as plt

    data = disagg.rename(columns={"distance": "r_rup", "dist": "r_rup", "r_jb": "r_rup", "mag": "mw", "eps": "epsilon"}).copy()
    _require_columns(data, ("period", "return_period", "mw", "r_rup", "contribution"))

    data = data[np.isclose(pd.to_numeric(data["period"], errors="coerce"), float(period))].copy()
    if return_period is not None:
        data = data[np.isclose(pd.to_numeric(data["return_period"], errors="coerce"), float(return_period))].copy()
    elif data["return_period"].nunique(dropna=True) != 1:
        raise ValueError("return_period must be supplied when the data contain multiple return periods")
    if data.empty:
        raise ValueError(f"No disaggregation rows found for period={period:g}")

    rp = float(data["return_period"].iloc[0])
    mag_edges = np.asarray(_validated_bin_edges(mag_bin_edges, "mag_bin_edges"), dtype=float)
    dist_edges = np.asarray(_validated_bin_edges(dist_bin_edges, "dist_bin_edges"), dtype=float)
    mag_centres = np.round(0.5 * (mag_edges[:-1] + mag_edges[1:]), 10)
    dist_centres = np.round(0.5 * (dist_edges[:-1] + dist_edges[1:]), 10)

    md = (
        data.groupby(["mw", "r_rup"], as_index=False, dropna=False, sort=False)["contribution"]
        .sum()
    )
    md["mw"] = np.round(pd.to_numeric(md["mw"], errors="raise").to_numpy(float), 10)
    md["r_rup"] = np.round(pd.to_numeric(md["r_rup"], errors="raise").to_numpy(float), 10)
    pivot = (
        md.pivot(index="mw", columns="r_rup", values="contribution")
        .reindex(index=mag_centres, columns=dist_centres)
        .fillna(0.0)
    )

    contribution = 100.0 * pivot.to_numpy(float)
    d0, m0 = np.meshgrid(dist_edges[:-1], mag_edges[:-1])
    dw, mw = np.meshgrid(np.diff(dist_edges), np.diff(mag_edges))
    x, y = d0.ravel(), m0.ravel()
    dx, dy = dw.ravel(), mw.ravel()
    dz = contribution.ravel()
    keep = dz > 0

    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        x[keep], y[keep], np.zeros(np.count_nonzero(keep)),
        dx[keep], dy[keep], dz[keep], shade=True,
    )

    if summary is None:
        summary = summarize_disaggregation(data)
    s = summary[
        np.isclose(pd.to_numeric(summary["period"], errors="coerce"), float(period))
        & np.isclose(pd.to_numeric(summary["return_period"], errors="coerce"), rp)
    ]
    z_marker = float(dz.max() * 1.05) if dz.size and dz.max() > 0 else 0.0

    mean = s[s["statistic"] == "mean_source"]
    if not mean.empty:
        row = mean.iloc[0]
        ax.scatter(row["r_rup"], row["mw"], z_marker, marker="o", s=70, label="Mean source")

    mode = s[s["statistic"] == "mode_mag_dist"]
    if not mode.empty:
        row = mode.iloc[0]
        ax.scatter(row["r_rup"], row["mw"], z_marker, marker="s", s=70, label="Mode (mag-dist)")

    ax.set_xlabel("Rrup (km)", labelpad=10)
    ax.set_ylabel("Magnitude, Mw", labelpad=10)
    ax.set_zlabel("Hazard contribution (%)", labelpad=8)
    ax.set_title(f"Disaggregation — T = {float(period):g} s, RP = {rp:.0f} yr")
    ax.view_init(elev=float(elev), azim=float(azim))
    if not s.empty:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def extract_hazard_curves(
    calc_id: int,
    statistic: str = "mean",
    output_csv: str | Path | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Extract hazard curves in long form for all sites and IMTs.

    Output columns include site metadata, IMT/period, intensity level in g,
    one-year PoE and annual exceedance rate. This is the durable hazard-curve
    product from which UHS values can be regenerated later.
    """
    Extractor, read = _openquake_imports()

    with read(int(calc_id)) as ds:
        oq = ds["oqparam"]
        imtls = getattr(oq, "hazard_imtls", None)
        if imtls is None:
            imtls = getattr(oq, "imtls", None)
        if imtls is None:
            raise AttributeError("OpenQuake oqparam has neither hazard_imtls nor imtls")
        imtls = {str(imt): np.asarray(levels, dtype=float) for imt, levels in imtls.items()}
        investigation_time = float(oq.investigation_time)

    site_df = extract_site_collection(calc_id)
    frames: list[pd.DataFrame] = []

    with Extractor(int(calc_id)) as ex:
        for site_id in range(len(site_df)):
            for imt, levels in imtls.items():
                aw = ex.get(f"hcurves?kind={statistic}&imt={imt}&site_id={site_id}")
                if not hasattr(aw, statistic):
                    available = sorted(k for k in vars(aw) if not k.startswith("_"))
                    raise KeyError(
                        f"Hazard-curve statistic {statistic!r} is absent; available fields are {available}"
                    )
                poes = np.asarray(getattr(aw, statistic), dtype=float).squeeze()
                if poes.ndim != 1 or poes.size != levels.size:
                    raise RuntimeError(
                        f"Unexpected hazard-curve shape for site={site_id}, imt={imt}: "
                        f"{poes.shape}; expected {levels.size} levels"
                    )
                rates = -np.log1p(-np.clip(poes, 0.0, 1.0 - 1e-15)) / investigation_time
                frames.append(pd.DataFrame({
                    "site_id": site_id,
                    "statistic": str(statistic),
                    "imt": imt,
                    "period": _imt_period(imt),
                    "im_level_g": levels,
                    "poe_1yr": poes,
                    "annual_exceedance_rate": rates,
                }))

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return result

    result = result.merge(
        site_df.reset_index(drop=True).reset_index(names="site_id"),
        on="site_id",
        how="left",
        suffixes=("", "_site"),
        validate="many_to_one",
    )
    if model is not None:
        result.insert(0, "model", str(model))

    preferred = [
        "model", "site_id", "custom_site_id", "lon", "lat", "vs30",
        "statistic", "imt", "period", "im_level_g", "poe_1yr",
        "annual_exceedance_rate",
    ]
    result = result[[c for c in preferred if c in result.columns] + [c for c in result.columns if c not in preferred]]

    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
    return result


def validate_hazard_curve_coverage(
    hcurves: pd.DataFrame,
    return_periods: Sequence[float] = DEFAULT_RETURN_PERIODS,
    monotonic_rtol: float = 1e-10,
    monotonic_atol: float = 1e-14,
) -> pd.DataFrame:
    """Check that extracted hazard curves bracket every requested return period.

    The extracted ``annual_exceedance_rate`` is compared with ``1 / RP``. A
    target is bracketed when it lies between the minimum and maximum finite rates
    on the supplied IM-level grid. The function also reports whether each hazard
    curve is non-increasing with increasing IM level.
    """
    _require_columns(
        hcurves,
        ("site_id", "imt", "im_level_g", "annual_exceedance_rate"),
    )
    rps = np.asarray(return_periods, dtype=float)
    if rps.ndim != 1 or rps.size == 0 or np.any(~np.isfinite(rps)) or np.any(rps <= 0):
        raise ValueError("return_periods must be a non-empty sequence of positive finite values")

    identity = [
        c for c in ("model", "site_id", "custom_site_id", "lon", "lat", "vs30", "imt", "period")
        if c in hcurves.columns
    ]
    group_cols = [c for c in ("site_id", "imt") if c in hcurves.columns]
    rows: list[dict[str, object]] = []

    for _, group in hcurves.groupby(group_cols, dropna=False, sort=False):
        group = group.sort_values("im_level_g")
        imls = pd.to_numeric(group["im_level_g"], errors="raise").to_numpy(float)
        rates = pd.to_numeric(group["annual_exceedance_rate"], errors="raise").to_numpy(float)
        finite = np.isfinite(imls) & np.isfinite(rates) & (imls > 0) & (rates >= 0)

        base = {c: group[c].iloc[0] for c in identity if c in group.columns}
        if np.count_nonzero(finite) < 2:
            for rp in rps:
                rows.append(base | {
                    "return_period": float(rp),
                    "target_annual_rate": 1.0 / float(rp),
                    "bracketed": False,
                    "monotonic": False,
                    "min_rate": np.nan,
                    "max_rate": np.nan,
                    "min_im_level_g": np.nan,
                    "max_im_level_g": np.nan,
                })
            continue

        imls = imls[finite]
        rates = rates[finite]
        scale = np.maximum(np.maximum(np.abs(rates[:-1]), np.abs(rates[1:])), 1.0)
        monotonic = bool(
            np.all(np.diff(rates) <= monotonic_atol + monotonic_rtol * scale)
        )
        min_rate = float(np.min(rates))
        max_rate = float(np.max(rates))

        for rp in rps:
            target = 1.0 / float(rp)
            rows.append(base | {
                "return_period": float(rp),
                "target_annual_rate": target,
                "bracketed": bool(min_rate <= target <= max_rate),
                "monotonic": monotonic,
                "min_rate": min_rate,
                "max_rate": max_rate,
                "min_im_level_g": float(np.min(imls)),
                "max_im_level_g": float(np.max(imls)),
            })

    return pd.DataFrame(rows)


def format_uhs_wide(uhs: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """Return the final wide UHS table with machine-friendly ``pSA_*`` columns."""
    out = uhs.copy()
    rename = {}
    for col in out.columns:
        match = re.fullmatch(r"SA\(([^)]+)\)", str(col), flags=re.I)
        if match:
            period = float(match.group(1))
            label = f"{period:.1f}" if period.is_integer() else f"{period:g}"
            rename[col] = f"pSA_{label}"
    out = out.rename(columns=rename)
    if model is not None and "model" not in out.columns:
        out.insert(0, "model", str(model))

    meta = [
        "model", "site_id", "custom_site_id", "lon", "lat", "vs30",
        "z1pt0", "z2pt5", "backarc", "statistic", "return_period", "poe",
    ]
    spectral = ["PGA"] if "PGA" in out.columns else []
    spectral += sorted(
        [c for c in out.columns if str(c).startswith("pSA_")],
        key=lambda c: float(str(c).split("_", 1)[1]),
    )
    ordered = [c for c in meta if c in out.columns] + spectral
    ordered += [c for c in out.columns if c not in ordered]
    return out[ordered].copy()


# -----------------------------------------------------------------------------
# UHS reference parsing and validation
# -----------------------------------------------------------------------------

def read_nshm2022_uhs_reference(path: str | Path) -> pd.DataFrame:
    """Read the GNS NSHM2022 UHS CSV (metadata line + table header)."""
    df = pd.read_csv(path, skiprows=1)
    required = ("lat", "lon", "vs30", "PoE (% in 50 years)", "statistic")
    _require_columns(df, required)
    df["statistic"] = df["statistic"].map(_normalise_statistic)
    return df


def read_nshm2010_uhs_reference(path: str | Path) -> pd.DataFrame:
    """Read and validate the supplied NSHM2010 mean-UHS benchmark export."""
    df = pd.read_csv(path, skiprows=1)
    _require_columns(df, ("custom_site_id", "lon", "lat", "vs30"))
    if df["custom_site_id"].astype(str).duplicated().any():
        raise ValueError("NSHM2010 reference custom_site_id values must be unique")
    for col in ("lon", "lat", "vs30"):
        values = pd.to_numeric(df[col], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"NSHM2010 reference {col} values must be finite")
    if (pd.to_numeric(df["vs30"], errors="raise") <= 0).any():
        raise ValueError("NSHM2010 reference vs30 values must be positive")
    uhs_cols = [c for c in df.columns if re.match(r"^[0-9.eE+-]+~(?:PGA|SA\([^)]+\))$", str(c))]
    if not uhs_cols:
        raise ValueError("No OpenQuake UHS columns of the form '<poe>~PGA/SA(T)' were found")
    return df


def validate_nshm2010_uhs_validation_inputs(
    model_dir: str | Path,
    reference_csv: str | Path,
    sites: pd.DataFrame | None = None,
    coordinate_tolerance_deg: float = 5.1e-6,
) -> dict[str, object]:
    """Validate the exact NSHM2010 benchmark configuration before OpenQuake runs.

    This is deliberately NSHM2010-specific. It verifies the supplied RotD50
    McVerry logic tree, single-path source/GMM configuration, benchmark IMTs/PoEs,
    site table, and the supplied native 2010 job INI when present. It does not
    inspect or modify OpenQuake's database or queue.
    """
    root = resolve_model_root(model_dir, "2010")
    ref = read_nshm2010_uhs_reference(reference_csv)

    source_lt = root / "source_models/source_model_logic_tree.xml"
    gsim_lt = root / "gmpe_logic_tree_McVerry06_ONLY_MW_RotD50.xml"
    source_paths = int(logic_tree_branch_count(source_lt))
    gsim_paths = int(logic_tree_branch_count(gsim_lt))
    potential_paths = source_paths * gsim_paths
    if potential_paths != 1:
        raise ValueError(
            "NSHM2010 validation expects one deterministic source/GMM logic-tree path; "
            f"found {potential_paths} ({source_paths} source x {gsim_paths} GMM)"
        )

    imt_cols = [c for c in ref.columns if re.match(r"^[0-9.eE+-]+~(?:PGA|SA\([^)]+\))$", str(c))]
    poes = tuple(sorted({float(str(c).split("~", 1)[0]) for c in imt_cols}, reverse=True))
    expected_poes = tuple(float(x) for x in NSHM2010_REFERENCE_POES)
    if len(poes) != len(expected_poes) or not np.allclose(poes, expected_poes, rtol=0, atol=5e-10):
        raise ValueError(f"NSHM2010 benchmark PoEs {poes} do not match expected {expected_poes}")

    include_pga = any(str(c).endswith("~PGA") for c in imt_cols)
    periods = tuple(sorted({_imt_period(str(c).split("~", 1)[1]) for c in imt_cols if "~SA(" in str(c)}))
    expected_periods = tuple(float(x) for x in NSHM2010_DEFAULT_PERIODS)
    if not include_pga:
        raise ValueError("NSHM2010 benchmark must contain PGA")
    if len(periods) != len(expected_periods) or not np.allclose(periods, expected_periods, rtol=0, atol=1e-12):
        raise ValueError(f"NSHM2010 benchmark SA periods {periods} do not match expected {expected_periods}")

    if sites is None:
        site_file = root / "site_table_aaron.csv"
        if not site_file.exists():
            raise FileNotFoundError(
                "sites was not supplied and site_table_aaron.csv is not present in the NSHM2010 model directory"
            )
        sites = pd.read_csv(site_file)
    else:
        sites = sites.copy()

    _require_columns(sites, ("custom_site_id", "lon", "lat", "vs30"))
    prepared = prepare_sites(sites, "2010", root)
    merged = ref[["custom_site_id", "lon", "lat", "vs30"]].merge(
        prepared[["custom_site_id", "lon", "lat", "vs30"]],
        on="custom_site_id",
        suffixes=("_ref", "_site"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if len(merged) != len(ref) or len(merged) != len(prepared) or not (merged["_merge"] == "both").all():
        raise ValueError(
            "NSHM2010 validation site IDs do not match the supplied benchmark exactly "
            f"(reference={len(ref)}, sites={len(prepared)})"
        )
    tol = float(coordinate_tolerance_deg)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("coordinate_tolerance_deg must be finite and >= 0")
    if not (
        np.allclose(merged["lon_ref"], merged["lon_site"], rtol=0, atol=tol)
        and np.allclose(merged["lat_ref"], merged["lat_site"], rtol=0, atol=tol)
        and np.allclose(merged["vs30_ref"], merged["vs30_site"], rtol=0, atol=0)
    ):
        raise ValueError("NSHM2010 benchmark coordinates/Vs30 do not match the validation site table")

    native_job = root / "job_nshm_all_trts_mcv.ini"
    native_job_matched = False
    if native_job.exists():
        cp = configparser.ConfigParser(interpolation=None)
        cp.read(native_job, encoding="utf-8")
        native_imtls = cp["calculation"]["intensity_measure_types_and_levels"]
        native_imts = re.findall(r'["\'](PGA|SA\([^)]+\))["\']\s*:', native_imtls)
        native_periods = tuple(_imt_period(x) for x in native_imts if x.startswith("SA("))
        native_logscales = re.findall(
            r"logscale\(\s*0\.005\s*,\s*5(?:\.0+)?\s*,\s*50\s*\)",
            native_imtls,
            flags=re.I,
        )
        expected_maxdist = ast.literal_eval(
            _base_sections("2010", root, root / "site_table_aaron.csv", 0, "validation", root)["calculation"]["maximum_distance"]
        )
        native_maxdist = ast.literal_eval(cp["calculation"]["maximum_distance"])
        native_job_matched = bool(
            cp.getint("general", "random_seed") == 1024
            and cp.getint("logic_tree", "number_of_logic_tree_samples") == 0
            and np.isclose(cp.getfloat("erf", "rupture_mesh_spacing"), 4.0)
            and np.isclose(cp.getfloat("erf", "width_of_mfd_bin"), 0.1)
            and np.isclose(cp.getfloat("erf", "area_source_discretization"), 10.0)
            and Path(cp["site_params"]["site_model_file"]).name == "site_table_aaron.csv"
            and Path(cp["calculation"]["source_model_logic_tree_file"]).name == source_lt.name
            and Path(cp["calculation"]["gsim_logic_tree_file"]).name == gsim_lt.name
            and np.isclose(cp.getfloat("calculation", "investigation_time"), 1.0)
            and np.isclose(cp.getfloat("calculation", "truncation_level"), 4.0)
            and native_maxdist == expected_maxdist
            and native_imts[0] == "PGA"
            and len(native_periods) == len(expected_periods)
            and np.allclose(native_periods, expected_periods, rtol=0, atol=1e-12)
            and len(native_logscales) == len(native_imts) == 1 + len(expected_periods)
        )
        if not native_job_matched:
            raise ValueError(
                "The supplied NSHM2010 native job_nshm_all_trts_mcv.ini does not match "
                "the validation configuration expected by this workflow"
            )

    return {
        "model_root": str(root),
        "reference_csv": str(Path(reference_csv).resolve()),
        "source_model_logic_tree": str(source_lt),
        "gsim_logic_tree": str(gsim_lt),
        "source_paths": source_paths,
        "gsim_paths": gsim_paths,
        "potential_paths": potential_paths,
        "n_sites": int(len(prepared)),
        "poes": poes,
        "periods": periods,
        "include_pga": include_pga,
        "native_job_present": native_job.exists(),
        "native_job_matched": native_job_matched if native_job.exists() else None,
    }


def normalise_uhs(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Convert GNS, OQ-export or :func:`extract_uhs` tables to one long schema.

    Returned columns include site identifiers/coordinates when available plus
    ``poe``, ``statistic``, ``imt`` and ``value``.
    """
    if isinstance(data, (str, Path)):
        path = Path(data)
        first = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
        if first.lstrip().startswith("#"):
            df = pd.read_csv(path, skiprows=1)
        elif "date-time:" in first and "NSHM model version:" in first:
            df = pd.read_csv(path, skiprows=1)
        else:
            df = pd.read_csv(path)
    else:
        df = data.copy()

    # GNS web UHS format.
    if "PoE (% in 50 years)" in df.columns:
        _require_columns(df, ("statistic", "PoE (% in 50 years)"))
        imts = [c for c in df.columns if c == "PGA" or re.fullmatch(r"SA\([^)]+\)", str(c))]
        ids = [c for c in ("custom_site_id", "lon", "lat", "vs30") if c in df.columns]
        out = df.melt(id_vars=ids + ["PoE (% in 50 years)", "statistic"], value_vars=imts, var_name="imt", value_name="value")
        out["poe"] = [annual_poe_from_probability(float(x) / 100.0, 50.0) for x in out["PoE (% in 50 years)"]]
        out["statistic"] = out["statistic"].map(_normalise_statistic)
        out["imt"] = out["imt"].map(_normalise_imt_name)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.drop(columns=["PoE (% in 50 years)"])

    # Native OQ UHS CSV export with '<poe>~<imt>' columns.
    uhs_cols = [c for c in df.columns if re.match(r"^[0-9.eE+-]+~(?:PGA|SA\([^)]+\))$", str(c))]
    if uhs_cols:
        ids = [c for c in df.columns if c not in uhs_cols]
        pieces = []
        for c in uhs_cols:
            poe_text, imt = str(c).split("~", 1)
            cur = df[ids].copy()
            cur["poe"] = float(poe_text)
            cur["statistic"] = "mean"
            cur["imt"] = _normalise_imt_name(imt)
            cur["value"] = pd.to_numeric(df[c], errors="coerce")
            pieces.append(cur)
        return pd.concat(pieces, ignore_index=True)

    # DataFrame returned by extract_uhs.
    imts = [c for c in df.columns if c == "PGA" or re.fullmatch(r"SA\([^)]+\)", str(c))]
    if not imts:
        raise ValueError("Could not identify UHS IMT columns")
    poe_col = next((c for c in ("poe", "poes") if c in df.columns), None)
    if poe_col is None:
        raise ValueError("Calculated UHS table has no 'poe'/'poes' column")
    if "statistic" not in df.columns:
        df["statistic"] = "mean"
    ids = [c for c in df.columns if c not in imts]
    out = df.melt(id_vars=ids, value_vars=imts, var_name="imt", value_name="value")
    if poe_col != "poe":
        out = out.rename(columns={poe_col: "poe"})
    out["poe"] = pd.to_numeric(out["poe"], errors="coerce")
    out["statistic"] = out["statistic"].map(_normalise_statistic)
    out["imt"] = out["imt"].map(_normalise_imt_name)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def compare_uhs(
    calculated: pd.DataFrame | str | Path,
    reference: pd.DataFrame | str | Path,
    coordinate_tolerance_deg: float = 1e-4,
    poe_tolerance: float = 5e-7,
) -> pd.DataFrame:
    """Compare UHS values and return one row per matched site/statistic/IMT/PoE.

    ``relative_error_pct`` is ``100 * (calculated-reference) / reference``.
    A missing benchmark match raises immediately rather than silently dropping rows.
    """
    calc = normalise_uhs(calculated).reset_index(drop=True)
    ref = normalise_uhs(reference).reset_index(drop=True)
    rows = []
    for _, rr in ref.iterrows():
        cand = calc[(calc["imt"] == rr["imt"]) & (calc["statistic"] == rr["statistic"])]
        cand = cand[np.abs(pd.to_numeric(cand["poe"], errors="coerce") - float(rr["poe"])) <= poe_tolerance]
        if "custom_site_id" in ref.columns and pd.notna(rr.get("custom_site_id")) and "custom_site_id" in cand.columns:
            cand = cand[cand["custom_site_id"].astype(str) == str(rr["custom_site_id"])]
        elif all(c in ref.columns for c in ("lon", "lat")) and all(c in cand.columns for c in ("lon", "lat")):
            cand = cand[(np.abs(pd.to_numeric(cand["lon"], errors="coerce") - float(rr["lon"])) <= coordinate_tolerance_deg) &
                        (np.abs(pd.to_numeric(cand["lat"], errors="coerce") - float(rr["lat"])) <= coordinate_tolerance_deg)]
        elif len(ref) > 1:
            raise ValueError("Cannot match multi-site UHS tables: no common site ID or coordinates")
        if len(cand) != 1:
            raise ValueError(
                f"Expected one calculated match for statistic={rr['statistic']}, imt={rr['imt']}, poe={rr['poe']:.12g}; found {len(cand)}"
            )
        cc = cand.iloc[0]
        ref_value = float(rr["value"])
        calc_value = float(cc["value"])
        rows.append({
            "custom_site_id": rr.get("custom_site_id", cc.get("custom_site_id", np.nan)),
            "lon": rr.get("lon", cc.get("lon", np.nan)),
            "lat": rr.get("lat", cc.get("lat", np.nan)),
            "poe": float(rr["poe"]),
            "statistic": rr["statistic"],
            "imt": rr["imt"],
            "reference": ref_value,
            "calculated": calc_value,
            "difference": calc_value - ref_value,
            "relative_error_pct": 100.0 * (calc_value - ref_value) / ref_value if ref_value != 0 else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_uhs_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize absolute UHS relative error by statistic."""
    _require_columns(comparison, ("statistic", "relative_error_pct"))
    tmp = comparison.copy()
    tmp["abs_relative_error_pct"] = np.abs(tmp["relative_error_pct"].to_numpy(float))
    return (
        tmp.groupby("statistic", dropna=False)["abs_relative_error_pct"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .rename(columns={"mean": "mean_abs_error_pct", "median": "median_abs_error_pct", "max": "max_abs_error_pct"})
    )


def validate_sample_convergence(
    candidate: pd.DataFrame | str | Path,
    baseline: pd.DataFrame | str | Path,
    tolerance_pct: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Compare a lower-sample UHS to a baseline and apply an absolute-% tolerance."""
    comp = compare_uhs(candidate, baseline)
    summary = summarize_uhs_comparison(comp)
    ok = bool(np.nanmax(np.abs(comp["relative_error_pct"].to_numpy(float))) <= float(tolerance_pct))
    return comp, summary, ok


def validate_uhs_calculation(
    calc_id: int,
    reference_csv: str | Path,
    output_dir: str | Path | None = None,
    tolerance_pct: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Extract an OQ UHS and compare it directly with a supplied benchmark CSV.

    Returns ``(calculated, comparison, summary, ok)``. If ``output_dir`` is
    supplied, all three tables are also written to disk. The pass criterion is
    a maximum absolute relative UHS error no larger than ``tolerance_pct``.
    """
    calculated = extract_uhs(int(calc_id))
    comparison = compare_uhs(calculated, reference_csv)
    summary = summarize_uhs_comparison(comparison)
    errors = np.abs(comparison["relative_error_pct"].to_numpy(float))
    ok = bool(np.isfinite(errors).all() and errors.max(initial=0.0) <= float(tolerance_pct))
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        calculated.to_csv(out / "uhs_calculated.csv", index=False)
        comparison.to_csv(out / "uhs_comparison.csv", index=False)
        summary.to_csv(out / "uhs_comparison_summary.csv", index=False)
    return calculated, comparison, summary, ok


# -----------------------------------------------------------------------------
# Internal validation (does not require OpenQuake)
# -----------------------------------------------------------------------------

def run_internal_validation(
    nshm2022_model_dir: str | Path | None = None,
    nshm2010_model_dir: str | Path | None = None,
    nshm2022_reference_csv: str | Path | None = None,
    nshm2010_reference_csv: str | Path | None = None,
) -> dict[str, object]:
    """Run deterministic checks on probability maths, model inputs and references.

    This deliberately distinguishes checks that can be completed without OpenQuake
    from numerical benchmark comparison, which requires a completed OQ calculation.
    """
    checks: dict[str, object] = {}

    rp_2in50 = return_period_from_probability(0.02, 50.0)
    checks["2pct_in_50_return_period"] = rp_2in50
    checks["2pct_in_50_ok"] = bool(np.isclose(rp_2in50, 2474.916, rtol=2e-6))
    checks["10pct_in_50_annual_poe"] = annual_poe_from_probability(0.10, 50.0)
    checks["2pct_in_50_annual_poe"] = annual_poe_from_probability(0.02, 50.0)
    checks["reference_poes_ok"] = bool(
        np.isclose(checks["10pct_in_50_annual_poe"], 0.002105, rtol=5e-6)
        and np.isclose(checks["2pct_in_50_annual_poe"], 0.000404, rtol=1e-4)
    )
    expected_2010_levels = np.geomspace(0.005, 5.0, 50)
    checks["nshm2010_im_grid_ok"] = bool(np.allclose(NSHM2010_DEFAULT_IM_LEVELS, expected_2010_levels, rtol=0, atol=0))

    mag_edges = np.asarray(DEFAULT_DISAGG_MAG_BIN_EDGES, float)
    dist_edges = np.asarray(DEFAULT_DISAGG_DIST_BIN_EDGES, float)
    eps_edges = np.asarray(DEFAULT_DISAGG_EPS_BIN_EDGES, float)
    checks["disagg_mag_bins"] = int(mag_edges.size - 1)
    checks["disagg_dist_bins"] = int(dist_edges.size - 1)
    checks["disagg_eps_bins"] = int(eps_edges.size - 1)
    checks["disagg_bins_ok"] = bool(
        mag_edges.size == 26
        and dist_edges.size == 18
        and eps_edges.size == 51
        and np.all(np.diff(mag_edges) > 0)
        and np.all(np.diff(dist_edges) > 0)
        and np.all(np.diff(eps_edges) > 0)
        and np.isclose(mag_edges[[0, -1]], [5.0, 10.0]).all()
        and np.isclose(dist_edges[[0, -1]], [0.0, 500.0]).all()
        and np.isclose(eps_edges[[0, -1]], [-5.0, 5.0]).all()
        and np.isclose(DEFAULT_DISAGG_COORDINATE_BIN_WIDTH_DEG, 100.0)
    )

    checks["production_period_count"] = len(PRODUCTION_PERIODS)
    checks["production_periods_ok"] = bool(
        len(PRODUCTION_PERIODS) == 32
        and np.all(np.diff(np.asarray(PRODUCTION_PERIODS, float)) > 0)
        and np.isclose(PRODUCTION_PERIODS[0], 0.01)
        and np.isclose(PRODUCTION_PERIODS[-1], 10.0)
        and all(any(np.isclose(t, p) for t in PRODUCTION_PERIODS) for p in CALDERON_DISAGG_PERIODS)
    )

    synthetic_disagg = pd.DataFrame({
        "site_id": [0, 0, 0, 0],
        "return_period": [2500.0] * 4,
        "period": [1.0] * 4,
        "mw": [6.5, 6.5, 7.0, 7.0],
        "r_rup": [7.5, 7.5, 17.5, 17.5],
        "epsilon": [0.9, 1.1, 0.9, 1.1],
        "contribution": [0.30, 0.20, 0.10, 0.40],
    })
    synthetic_summary = summarize_disaggregation(synthetic_disagg)
    mean_source = synthetic_summary.loc[
        synthetic_summary["statistic"] == "mean_source"
    ].iloc[0]

    expected_eps_std = np.sqrt(0.0096)
    checks["disagg_summary_statistics_ok"] = bool(
        set(synthetic_summary["statistic"]) ==
        {"mean_source", "mode_mag_dist", "mode_mag_dist_eps"}
        and np.isclose(mean_source["mw"], 6.75, rtol=0, atol=1e-14)
        and np.isclose(mean_source["r_rup"], 12.5, rtol=0, atol=1e-14)
        and np.isclose(mean_source["epsilon"], 1.02, rtol=0, atol=1e-14)
        and np.isclose(
            synthetic_summary.loc[
                synthetic_summary["statistic"] == "mode_mag_dist", "contribution"
            ].iloc[0],
            0.5,
        )
        and np.isclose(
            synthetic_summary.loc[
                synthetic_summary["statistic"] == "mode_mag_dist_eps", "contribution"
            ].iloc[0],
            0.4,
        )
    )
    checks["disagg_dispersion_moments_ok"] = bool(
        {"mw_std", "r_rup_std", "epsilon_std"}.issubset(synthetic_summary.columns)
        and np.isclose(mean_source["mw_std"], 0.25, rtol=0, atol=1e-14)
        and np.isclose(mean_source["r_rup_std"], 5.0, rtol=0, atol=1e-14)
        and np.isclose(mean_source["epsilon_std"], expected_eps_std, rtol=0, atol=1e-14)
    )
    checks["disagg_dispersion_repeated_by_statistic_ok"] = bool(
        synthetic_summary["mw_std"].nunique(dropna=False) == 1
        and synthetic_summary["r_rup_std"].nunique(dropna=False) == 1
        and synthetic_summary["epsilon_std"].nunique(dropna=False) == 1
    )

    scaled_disagg = synthetic_disagg.copy()
    scaled_disagg["contribution"] *= 17.0
    scaled_summary = summarize_disaggregation(scaled_disagg)
    compare_cols = ["mw", "mw_std", "r_rup", "r_rup_std", "epsilon_std", "contribution"]
    left = synthetic_summary.sort_values("statistic").reset_index(drop=True)
    right = scaled_summary.sort_values("statistic").reset_index(drop=True)
    checks["disagg_dispersion_weight_scaling_ok"] = bool(
        np.allclose(
            left[compare_cols].to_numpy(float),
            right[compare_cols].to_numpy(float),
            rtol=0,
            atol=1e-14,
            equal_nan=True,
        )
    )

    duplicated = pd.concat(
        [
            synthetic_disagg.assign(contribution=synthetic_disagg["contribution"] / 2.0),
            synthetic_disagg.assign(contribution=synthetic_disagg["contribution"] / 2.0),
        ],
        ignore_index=True,
    )
    duplicated_summary = summarize_disaggregation(duplicated)
    dup_right = duplicated_summary.sort_values("statistic").reset_index(drop=True)
    checks["disagg_dispersion_duplicate_cell_invariance_ok"] = bool(
        np.allclose(
            left[compare_cols].to_numpy(float),
            dup_right[compare_cols].to_numpy(float),
            rtol=0,
            atol=1e-14,
            equal_nan=True,
        )
    )

    single_cell = synthetic_disagg.iloc[[0]].copy()
    single_summary = summarize_disaggregation(single_cell)
    checks["disagg_dispersion_degenerate_zero_ok"] = bool(
        np.allclose(
            single_summary[["mw_std", "r_rup_std", "epsilon_std"]].to_numpy(float),
            0.0,
            rtol=0,
            atol=0,
        )
    )

    # TRT marginalization must not alter moments when each causal cell is split
    # across tectonic-region rows and then recombined at identical coordinates.
    trt_parts = []
    for trt, fraction in (("Active Shallow Crust", 0.35), ("Subduction", 0.65)):
        part = synthetic_disagg.copy()
        part["tectonic_region"] = trt
        part["contribution"] *= fraction
        trt_parts.append(part)
    trt_disagg = pd.concat(trt_parts, ignore_index=True)
    trt_summary = summarize_disaggregation(trt_disagg).sort_values("statistic").reset_index(drop=True)
    checks["disagg_trt_marginalization_invariance_ok"] = bool(
        np.allclose(
            left[compare_cols].to_numpy(float),
            trt_summary[compare_cols].to_numpy(float),
            rtol=0,
            atol=1e-14,
            equal_nan=True,
        )
    )

    # OpenQuake/NSHM2022 semantics: realizations are combined in *rate space*
    # upstream, after which the final mean-rate distribution is normalized once.
    # This is generally different from normalizing each realization first and then
    # averaging those PMFs. The synthetic case deliberately has unequal total
    # hazard rates so the two procedures cannot accidentally coincide.
    rlz_rates = np.array([
        [0.09, 0.01],   # total rate 0.10
        [0.02, 0.18],   # total rate 0.20
    ], dtype=float)
    rlz_weights = np.array([0.75, 0.25], dtype=float)
    mean_rates = np.sum(rlz_weights[:, None] * rlz_rates, axis=0)

    # Mimic OpenQuake storage (rate -> probability) and our extraction
    # (probability -> rate), then normalize the recovered mean rates.
    mean_probs = -np.expm1(-mean_rates)  # T = 1 year
    recovered_mean_rates = _disagg_probabilities_to_annual_rates(mean_probs, 1.0)
    oq_style_contrib = recovered_mean_rates / recovered_mean_rates.sum()

    mean_rate_disagg = pd.DataFrame({
        "site_id": [0, 0],
        "return_period": [2500.0, 2500.0],
        "period": [1.0, 1.0],
        "mw": [6.0, 8.0],
        "r_rup": [10.0, 50.0],
        "epsilon": [-1.0, 1.0],
        "contribution": oq_style_contrib,
    })
    mean_rate_summary = summarize_disaggregation(mean_rate_disagg)
    mean_rate_source = mean_rate_summary.loc[
        mean_rate_summary["statistic"] == "mean_source"
    ].iloc[0]

    expected_mw = float(np.sum(oq_style_contrib * np.array([6.0, 8.0])))
    expected_rrup = float(np.sum(oq_style_contrib * np.array([10.0, 50.0])))
    expected_eps = float(np.sum(oq_style_contrib * np.array([-1.0, 1.0])))
    expected_mw_std = float(np.sqrt(np.sum(oq_style_contrib * (np.array([6.0, 8.0]) - expected_mw) ** 2)))
    expected_rrup_std = float(np.sqrt(np.sum(oq_style_contrib * (np.array([10.0, 50.0]) - expected_rrup) ** 2)))
    expected_eps_std = float(np.sqrt(np.sum(oq_style_contrib * (np.array([-1.0, 1.0]) - expected_eps) ** 2)))

    checks["disagg_mean_rate_probability_roundtrip_ok"] = bool(
        np.allclose(recovered_mean_rates, mean_rates, rtol=0, atol=1e-15)
        and np.isclose(oq_style_contrib.sum(), 1.0, rtol=0, atol=1e-15)
    )
    checks["disagg_nshm2022_mean_rate_moments_ok"] = bool(
        np.isclose(mean_rate_source["mw"], expected_mw, rtol=0, atol=1e-14)
        and np.isclose(mean_rate_source["r_rup"], expected_rrup, rtol=0, atol=1e-14)
        and np.isclose(mean_rate_source["epsilon"], expected_eps, rtol=0, atol=1e-14)
        and np.isclose(mean_rate_source["mw_std"], expected_mw_std, rtol=0, atol=1e-14)
        and np.isclose(mean_rate_source["r_rup_std"], expected_rrup_std, rtol=0, atol=1e-14)
        and np.isclose(mean_rate_source["epsilon_std"], expected_eps_std, rtol=0, atol=1e-14)
    )

    # Deliberately demonstrate the incorrect alternative: normalize each
    # realization first, then logic-tree-average the normalized PMFs.
    rlz_pmfs = rlz_rates / rlz_rates.sum(axis=1, keepdims=True)
    mean_of_normalized_pmfs = np.sum(rlz_weights[:, None] * rlz_pmfs, axis=0)
    checks["disagg_mean_rate_not_mean_normalized_pmfs_ok"] = bool(
        not np.allclose(oq_style_contrib, mean_of_normalized_pmfs, rtol=0, atol=1e-12)
        and not np.isclose(
            expected_mw,
            float(np.sum(mean_of_normalized_pmfs * np.array([6.0, 8.0]))),
            rtol=0,
            atol=1e-12,
        )
    )

    # Moment-preserving extraction must reject pre-summary tail pruning.
    checks["disagg_min_contribution_zero_ok"] = bool(
        _validate_disagg_min_contribution(0.0) == 0.0
    )
    try:
        _validate_disagg_min_contribution(1e-6)
    except ValueError:
        checks["disagg_min_contribution_pruning_rejected_ok"] = True
    else:
        checks["disagg_min_contribution_pruning_rejected_ok"] = False

    synthetic_uhs = pd.DataFrame({
        "site_id": [0],
        "return_period": [2500.0],
        "statistic": ["mean"],
        "PGA": [0.5],
        "SA(0.1)": [1.0],
        "SA(1.0)": [0.4],
    })
    synthetic_uhs_wide = format_uhs_wide(synthetic_uhs)
    checks["uhs_wide_columns_ok"] = bool(
        {"PGA", "pSA_0.1", "pSA_1.0"}.issubset(synthetic_uhs_wide.columns)
    )

    # Basin-depth conventions recovered directly from the supplied inputs.
    z1_22, z2_22 = infer_basin_depths([400.0], model="2022")
    checks["nshm2022_z1_v400"] = float(z1_22[0])
    checks["nshm2022_z2_v400"] = float(z2_22[0])
    checks["nshm2022_basin_depth_formula_ok"] = bool(
        np.isclose(z1_22[0], 355.7170357122867, rtol=0, atol=1e-10)
        and np.isclose(z2_22[0], 1.2646109912757877, rtol=0, atol=1e-12)
    )

    checks["nzs_R500"] = nzs1170p5_return_period_factor(500)
    checks["nzs_R2500"] = nzs1170p5_return_period_factor(2500)
    checks["nzs_factors_ok"] = bool(np.isclose(checks["nzs_R500"], 1.0) and np.isclose(checks["nzs_R2500"], 1.8))

    # Synthetic log-log interpolation with an exact known crossing at RP=1000.
    rps = np.array([100, 500, 1000, 2500], float)
    uhs = np.sqrt(rps / 1000.0)
    checks["equivalent_rp_synthetic"] = equivalent_return_period(rps, uhs, 1.0)
    checks["equivalent_rp_ok"] = bool(np.isclose(checks["equivalent_rp_synthetic"], 1000.0))

    if nshm2022_model_dir is not None:
        root22 = resolve_model_root(nshm2022_model_dir, "2022")
        reference_sites = pd.read_csv(root22 / "sites_all.csv")
        computed = add_nshm2022_backarc(reference_sites[["lon", "lat"]], root22 / "backarc.json")
        expected = reference_sites["backarc"].astype(bool).to_numpy()
        matches = computed["backarc"].to_numpy(bool) == expected
        checks["nshm2022_backarc_matches"] = int(matches.sum())
        checks["nshm2022_backarc_total"] = int(matches.size)
        checks["nshm2022_backarc_ok"] = bool(matches.all())
        checks["nshm2022_files_ok"] = True
        src_paths = logic_tree_branch_count(root22 / "sources/source_model_extended.xml")
        gsim_paths = logic_tree_branch_count(root22 / "gsim_model.xml")
        checks["nshm2022_source_paths"] = src_paths
        checks["nshm2022_gsim_paths"] = gsim_paths
        checks["nshm2022_potential_paths"] = int(src_paths * gsim_paths)
        checks["nshm2022_path_count_ok"] = bool(checks["nshm2022_potential_paths"] == 979_776)

        # Validate site preparation against the supplied NSHM2022 custom-site convention.
        test_site = pd.DataFrame({
            "custom_site_id": ["TEST"],
            "lon": [172.63],
            "lat": [-43.53],
            "vs30": [400.0],
        })
        prepared = prepare_sites(test_site, "2022", root22)
        checks["nshm2022_site_prepare_ok"] = bool(
            np.isclose(prepared["z1pt0"].iloc[0], 355.7170357122867, rtol=0, atol=1e-10)
            and np.isclose(prepared["z2pt5"].iloc[0], 1.2646109912757877, rtol=0, atol=1e-12)
            and int(prepared["vs30measured"].iloc[0]) == 0
            and int(prepared["backarc"].iloc[0]) in (0, 1)
        )

        supplied_bad_depths = test_site.assign(z1pt0=0.0, z2pt5=-1.0)
        repaired = prepare_sites(
            supplied_bad_depths,
            "2022",
            root22,
            infer_missing_basin_depths=True,
        )
        checks["nshm2022_nonpositive_basin_repair_ok"] = bool(
            np.isclose(repaired["z1pt0"].iloc[0], 355.7170357122867, rtol=0, atol=1e-10)
            and np.isclose(repaired["z2pt5"].iloc[0], 1.2646109912757877, rtol=0, atol=1e-12)
        )

        supplied_positive_depths = test_site.assign(z1pt0=999.0, z2pt5=9.0)
        recomputed = prepare_sites(
            supplied_positive_depths,
            "2022",
            root22,
            recompute_basin_depths=True,
        )
        checks["nshm2022_recompute_basin_depths_ok"] = bool(
            np.isclose(recomputed["z1pt0"].iloc[0], 355.7170357122867, rtol=0, atol=1e-10)
            and np.isclose(recomputed["z2pt5"].iloc[0], 1.2646109912757877, rtol=0, atol=1e-12)
        )

        # Compare our baseline settings with the supplied model team's job_classical.ini.
        native_job = root22 / "job_classical.ini"
        if native_job.exists():
            cp = configparser.ConfigParser(interpolation=None)
            cp.read(native_job, encoding="utf-8")
            native_imtls = ast.literal_eval(cp["calculation"]["intensity_measure_types_and_levels"])
            native_pga = np.asarray(native_imtls["PGA"], dtype=float)
            checks["nshm2022_native_job_samples_ok"] = bool(
                cp.getint("logic_tree", "number_of_logic_tree_samples") == NSHM2022_LOGIC_TREE_SAMPLES
            )
            checks["nshm2022_native_job_seed_ok"] = bool(
                cp.getint("general", "random_seed") == 25
            )
            checks["nshm2022_native_im_grid_ok"] = bool(
                np.array_equal(native_pga, NSHM2022_DEFAULT_IM_LEVELS)
                and all(np.array_equal(np.asarray(v, float), native_pga) for v in native_imtls.values())
            )
            base = _base_sections(
                "2022",
                root22,
                root22 / "sites_all.csv",
                NSHM2022_LOGIC_TREE_SAMPLES,
                "validation",
                root22,
            )
            checks["nshm2022_base_settings_ok"] = bool(
                base["general"]["random_seed"] == cp["general"]["random_seed"]
                and base["general"]["ps_grid_spacing"] == cp["general"]["ps_grid_spacing"]
                and base["logic_tree"]["number_of_logic_tree_samples"]
                == cp["logic_tree"]["number_of_logic_tree_samples"]
                and base["erf"]["rupture_mesh_spacing"] == cp["erf"]["rupture_mesh_spacing"]
                and base["erf"]["width_of_mfd_bin"] == cp["erf"]["width_of_mfd_bin"]
                and float(base["erf"]["complex_fault_mesh_spacing"])
                == cp.getfloat("erf", "complex_fault_mesh_spacing")
                and float(base["erf"]["area_source_discretization"])
                == cp.getfloat("erf", "area_source_discretization")
                and base["calculation"]["investigation_time"] == cp["calculation"]["investigation_time"]
                and base["calculation"]["truncation_level"] == cp["calculation"]["truncation_level"]
            )

    if nshm2010_model_dir is not None:
        root10 = resolve_model_root(nshm2010_model_dir, "2010")
        checks["nshm2010_files_ok"] = bool((root10 / "gmpe_logic_tree_McVerry06_ONLY_MW_RotD50.xml").exists())
        src_paths = logic_tree_branch_count(root10 / "source_models/source_model_logic_tree.xml")
        gsim_paths = logic_tree_branch_count(root10 / "gmpe_logic_tree_McVerry06_ONLY_MW_RotD50.xml")
        checks["nshm2010_potential_paths"] = int(src_paths * gsim_paths)
        checks["nshm2010_single_path_ok"] = bool(checks["nshm2010_potential_paths"] == 1)
        checks["nshm2010_source_paths"] = int(src_paths)
        checks["nshm2010_gsim_paths"] = int(gsim_paths)

        native_job10 = root10 / "job_nshm_all_trts_mcv.ini"
        checks["nshm2010_native_job_present_ok"] = bool(native_job10.exists())
        if native_job10.exists():
            cp10 = configparser.ConfigParser(interpolation=None)
            cp10.read(native_job10, encoding="utf-8")
            native_imtls10 = cp10["calculation"]["intensity_measure_types_and_levels"]
            native_imts10 = re.findall(r'["\'](PGA|SA\([^)]+\))["\']\s*:', native_imtls10)
            native_periods10 = tuple(_imt_period(x) for x in native_imts10 if x.startswith("SA("))
            native_logscales10 = re.findall(
                r"logscale\(\s*0\.005\s*,\s*5(?:\.0+)?\s*,\s*50\s*\)",
                native_imtls10,
                flags=re.I,
            )
            base10 = _base_sections(
                "2010", root10, root10 / "site_table_aaron.csv", 0, "validation", root10
            )
            checks["nshm2010_native_job_settings_ok"] = bool(
                cp10.getint("general", "random_seed") == 1024
                and cp10.getint("logic_tree", "number_of_logic_tree_samples") == 0
                and np.isclose(cp10.getfloat("erf", "rupture_mesh_spacing"), 4.0)
                and np.isclose(cp10.getfloat("erf", "width_of_mfd_bin"), 0.1)
                and np.isclose(cp10.getfloat("erf", "area_source_discretization"), 10.0)
                and Path(cp10["site_params"]["site_model_file"]).name == "site_table_aaron.csv"
                and Path(cp10["calculation"]["source_model_logic_tree_file"]).name == "source_model_logic_tree.xml"
                and Path(cp10["calculation"]["gsim_logic_tree_file"]).name == "gmpe_logic_tree_McVerry06_ONLY_MW_RotD50.xml"
                and np.isclose(cp10.getfloat("calculation", "investigation_time"), 1.0)
                and np.isclose(cp10.getfloat("calculation", "truncation_level"), 4.0)
                and ast.literal_eval(cp10["calculation"]["maximum_distance"])
                == ast.literal_eval(base10["calculation"]["maximum_distance"])
                and native_imts10[0] == "PGA"
                and len(native_periods10) == len(NSHM2010_DEFAULT_PERIODS)
                and np.allclose(native_periods10, NSHM2010_DEFAULT_PERIODS, rtol=0, atol=1e-12)
                and len(native_logscales10) == len(native_imts10) == 1 + len(NSHM2010_DEFAULT_PERIODS)
                and set(base10["site_params"]) == {"site_model_file"}
            )
        supplied_sites10 = root10 / "site_table_aaron.csv"
        if supplied_sites10.exists():
            s10 = pd.read_csv(supplied_sites10)
            z1_calc, z2_calc = infer_basin_depths(s10["vs30"].to_numpy(float), model="2010")
            checks["nshm2010_z1_formula_max_abs_m"] = float(np.max(np.abs(z1_calc - s10["z1pt0"].to_numpy(float))))
            checks["nshm2010_z2_formula_max_abs_km"] = float(np.max(np.abs(z2_calc - s10["z2pt5"].to_numpy(float))))
            checks["nshm2010_basin_depth_formula_ok"] = bool(
                checks["nshm2010_z1_formula_max_abs_m"] < 1e-9
                and checks["nshm2010_z2_formula_max_abs_km"] < 1e-12
            )

    if nshm2022_reference_csv is not None:
        ref22 = read_nshm2022_uhs_reference(nshm2022_reference_csv)
        imts = [c for c in ref22.columns if c == "PGA" or re.fullmatch(r"SA\([^)]+\)", str(c))]
        expected_stats = {"mean", "quantile-0.05", "quantile-0.1", "quantile-0.9", "quantile-0.95"}
        checks["nshm2022_reference_statistics_ok"] = bool(set(ref22["statistic"]) == expected_stats)
        checks["nshm2022_reference_periods_ok"] = bool(
            tuple(_imt_period(c) for c in imts if c.startswith("SA(")) == NSHM2022_GNS_VALIDATION_PERIODS
        )
        checks["nshm2022_reference_site"] = (float(ref22["lon"].iloc[0]), float(ref22["lat"].iloc[0]), float(ref22["vs30"].iloc[0]))
        norm22 = normalise_uhs(ref22)
        checks["nshm2022_reference_rows"] = int(len(norm22))
        pivot = norm22.pivot_table(index="imt", columns="statistic", values="value", aggfunc="first")
        checks["nshm2022_quantile_order_ok"] = bool(
            np.all(pivot["quantile-0.05"] <= pivot["quantile-0.1"])
            and np.all(pivot["quantile-0.1"] <= pivot["mean"])
            and np.all(pivot["mean"] <= pivot["quantile-0.9"])
            and np.all(pivot["quantile-0.9"] <= pivot["quantile-0.95"])
        )

    if nshm2010_reference_csv is not None:
        ref10 = read_nshm2010_uhs_reference(nshm2010_reference_csv)
        checks["nshm2010_reference_sites"] = int(len(ref10))
        poes10 = sorted({float(c.split("~", 1)[0]) for c in ref10.columns if "~" in c and re.match(r"^[0-9.eE+-]+~", str(c))}, reverse=True)
        checks["nshm2010_reference_poes"] = tuple(poes10)
        checks["nshm2010_reference_poes_ok"] = bool(np.allclose(poes10, NSHM2010_REFERENCE_POES, rtol=0, atol=5e-10))
        if nshm2010_model_dir is not None and (root10 / "site_table_aaron.csv").exists():
            sites10 = pd.read_csv(root10 / "site_table_aaron.csv")
            merged = ref10[["custom_site_id", "lon", "lat", "vs30"]].merge(
                sites10[["custom_site_id", "lon", "lat", "vs30"]], on="custom_site_id", suffixes=("_ref", "_site"), how="outer", indicator=True
            )
            checks["nshm2010_reference_site_count_ok"] = bool(len(merged) == len(ref10) == len(sites10) and (merged["_merge"] == "both").all())
            checks["nshm2010_reference_site_values_ok"] = bool(
                np.allclose(merged["lon_ref"], merged["lon_site"], atol=5.1e-6)
                and np.allclose(merged["lat_ref"], merged["lat_site"], atol=5.1e-6)
                and np.array_equal(merged["vs30_ref"].to_numpy(), merged["vs30_site"].to_numpy())
            )
            setup10 = validate_nshm2010_uhs_validation_inputs(
                root10, nshm2010_reference_csv, sites=sites10
            )
            checks["nshm2010_validation_setup_ok"] = bool(
                setup10["potential_paths"] == 1
                and setup10["n_sites"] == len(ref10)
                and np.allclose(setup10["poes"], NSHM2010_REFERENCE_POES, rtol=0, atol=5e-10)
                and np.allclose(setup10["periods"], NSHM2010_DEFAULT_PERIODS, rtol=0, atol=1e-12)
                and bool(setup10["include_pga"])
            )

    # OpenQuake UHS extractor wrappers are generator-backed and have no shape_descr.
    # Exercise the exact nested structured-array layout used by calc.make_uhs().
    _imt_dt = np.dtype([("PGA", np.float32), ("SA(0.1)", np.float32)])
    _uhs_dt = np.dtype([("0.000404", _imt_dt), ("0.002105", _imt_dt)])
    _uhs_arr = np.zeros(2, dtype=_uhs_dt)
    _uhs_arr["0.000404"]["PGA"] = [0.4, 0.5]
    _uhs_arr["0.000404"]["SA(0.1)"] = [0.8, 0.9]
    _uhs_arr["0.002105"]["PGA"] = [0.2, 0.3]
    _uhs_arr["0.002105"]["SA(0.1)"] = [0.4, 0.5]
    _dummy_wrapper = type("DummyUHSWrapper", (), {})()
    _dummy_wrapper.mean = _uhs_arr
    _parsed = _uhs_arraywrapper_to_frame(_dummy_wrapper, "mean", [0.000404, 0.002105], ["PGA", "SA(0.1)"])
    checks["uhs_arraywrapper_parser_ok"] = bool(
        len(_parsed) == 4
        and set(_parsed.columns) >= {"site_id", "poe", "statistic", "PGA", "SA(0.1)"}
        and np.isclose(_parsed.loc[(_parsed.site_id == 0) & np.isclose(_parsed.poe, 0.000404), "SA(0.1)"].iloc[0], 0.8)
        and np.isclose(_parsed.loc[(_parsed.site_id == 1) & np.isclose(_parsed.poe, 0.002105), "PGA"].iloc[0], 0.3)
    )

    checks["version_tuple_ok"] = bool(__version_tuple__ == tuple(int(x) for x in __version__.split(".")))
    checks["production_return_periods_ok"] = bool(
        DEFAULT_RETURN_PERIODS == (100.0, 250.0, 500.0, 1000.0, 2500.0)
    )
    checks["production_mean_only_default_ok"] = bool(DEFAULT_QUANTILES == ())

    # Execution/path safety introduced after Windows cross-drive and queue failures.
    checks["nshm2022_default_samples_ok"] = bool(
        _logic_tree_samples("2022", None) == NSHM2022_LOGIC_TREE_SAMPLES
        and _logic_tree_samples("2022", 0) == 0
        and _logic_tree_samples("2010", None) == 0
    )
    try:
        _logic_tree_samples("2022", -1)
    except ValueError:
        checks["negative_logic_tree_samples_rejected_ok"] = True
    else:
        checks["negative_logic_tree_samples_rejected_ok"] = False

    import inspect as _inspect
    checks["oq_execution_wrapper_simple_ok"] = bool(
        tuple(_inspect.signature(run_openquake).parameters) == (
            "job_ini", "oq_command", "use_rates", "max_potential_paths",
            "check_version", "echo", "hazard_calculation_id", "num_cores",
        )
    )
    _probe_cfg = Path.cwd() / "oq_runtime_probe.cfg"
    _probe_env = _openquake_subprocess_env(_probe_cfg, 8)
    checks["oq_3234_preimport_config_env_ok"] = bool(
        _probe_env["OQ_CONFIG_FILE"] == str(_probe_cfg.resolve())
        and _probe_env["OQ_DISTRIBUTE"] == "processpool"
        and _probe_env["OQ_NUM_CORES"] == "8"
        and all(
            _probe_env[name] == "1"
            for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        )
    )
    checks["oq_processpool_worker_parser_ok"] = bool(
        _parse_oq_processpool_workers(
            "[2026-09-02 08:05:24 #3 WARNING] Using 16 processpool workers"
        ) == 16
        and _parse_oq_processpool_workers(
            "WARNING Using 8 processpool workers, concurrent_tasks=16"
        ) == 8
        and _parse_oq_processpool_workers("ordinary OpenQuake message") is None
    )
    checks["oq_memory_error_detection_ok"] = bool(
        _looks_like_openquake_memory_error(
            "TypeError: _ArrayMemoryError.__init__() missing 1 required positional argument: 'dtype'"
        )
        and _looks_like_openquake_memory_error("numpy.core._exceptions._ArrayMemoryError")
        and not _looks_like_openquake_memory_error("ordinary OpenQuake validation error")
    )
    checks["oq_windows_drive_parser_ok"] = bool(
        _windows_drive(r"C:\model") == "c:"
        and _windows_drive(r"D:\runs") == "d:"
    )
    try:
        _validate_oq_run_directory(r"C:\model", r"D:\runs")
    except ValueError:
        checks["oq_cross_drive_guard_ok"] = True
    else:
        checks["oq_cross_drive_guard_ok"] = False
    try:
        _validate_oq_run_directory(r"C:\model", r"C:\runs")
    except ValueError:
        checks["oq_same_drive_allowed_ok"] = False
    else:
        checks["oq_same_drive_allowed_ok"] = True

    try:
        _format_imtls((), [0.01, 0.1], include_pga=False)
    except ValueError:
        checks["empty_imt_job_rejected_ok"] = True
    else:
        checks["empty_imt_job_rejected_ok"] = False

    synthetic_hcurves = pd.DataFrame({
        "site_id": [0, 0, 0, 0],
        "custom_site_id": ["TEST"] * 4,
        "imt": ["SA(1.0)"] * 4,
        "period": [1.0] * 4,
        "im_level_g": [0.01, 0.1, 1.0, 10.0],
        "annual_exceedance_rate": [0.1, 0.01, 0.001, 0.0001],
    })
    coverage = validate_hazard_curve_coverage(
        synthetic_hcurves,
        return_periods=(100.0, 2500.0),
    )
    checks["hazard_curve_coverage_validator_ok"] = bool(
        coverage["bracketed"].all() and coverage["monotonic"].all()
    )

    try:
        equivalent_return_period(
            [100.0, 500.0, 1000.0],
            [0.2, 0.5, 0.4],
            0.45,
        )
    except ValueError:
        checks["equivalent_rp_nonmonotonic_rejected_ok"] = True
    else:
        checks["equivalent_rp_nonmonotonic_rejected_ok"] = False

    required_bools = [v for k, v in checks.items() if k.endswith("_ok")]
    checks["all_checks_pass"] = bool(all(required_bools))
    return checks


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------

def _oq_binary_site_param(values: pd.Series, name: str) -> pd.Series:
    """Normalize an OpenQuake Boolean-like site parameter to integer 0/1.

    OpenQuake 3.23.x parses ``backarc`` and ``vs30measured`` with integer
    dtypes in CSV site models. Accept common Boolean representations from user
    DataFrames but always serialize the result as 0/1.
    """
    s = pd.Series(values, copy=True)
    if s.isna().any():
        raise ValueError(f"{name} contains missing values")

    if pd.api.types.is_bool_dtype(s.dtype):
        return s.astype(np.uint8)

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        arr = numeric.to_numpy(float)
        if not np.isin(arr, [0.0, 1.0]).all():
            bad = sorted(set(arr[~np.isin(arr, [0.0, 1.0])].tolist()))
            raise ValueError(f"{name} must contain only 0/1 or Boolean values; found {bad[:5]}")
        return numeric.astype(np.uint8)

    mapping = {
        "true": 1, "t": 1, "yes": 1, "y": 1, "1": 1, "measured": 1,
        "false": 0, "f": 0, "no": 0, "n": 0, "0": 0, "inferred": 0,
    }
    mapped = s.astype(str).str.strip().str.lower().map(mapping)
    if mapped.isna().any():
        bad = sorted(s[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Could not interpret {name} values as Boolean/0-1: {bad[:5]}")
    return mapped.astype(np.uint8)


def _normalise_model(model: str) -> str:
    value = str(model).lower().replace("nshm", "").strip()
    if value not in {"2010", "2022"}:
        raise ValueError("model must be '2010' or '2022'")
    return value


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def logic_tree_branch_count(path: str | Path) -> int:
    """Return the product of branch counts across branch sets in an OQ logic tree.

    This is an exact path count for the supplied NZ NSHM2010/2022 trees, whose
    branch sets combine independently. It is primarily a validation/diagnostic helper.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    counts = []
    for branch_set in root.iter():
        if branch_set.tag.endswith("logicTreeBranchSet"):
            n = sum(1 for child in branch_set if child.tag.endswith("logicTreeBranch"))
            if n:
                counts.append(n)
    if not counts:
        raise ValueError(f"No logicTreeBranchSet elements found in {path}")
    return int(np.prod(counts, dtype=np.int64))


def _resolve_uhs_poes(
    return_periods: Sequence[float] | None,
    poes: Sequence[float] | None,
) -> np.ndarray:
    if poes is not None:
        out = np.asarray(poes, dtype=float)
    elif return_periods is not None:
        out = np.asarray([annual_poe(float(rp)) for rp in return_periods], dtype=float)
    else:
        raise ValueError("Provide return_periods or poes")
    if out.ndim != 1 or out.size == 0 or np.any((out <= 0) | (out >= 1)):
        raise ValueError("UHS PoEs must be a non-empty 1-D sequence strictly between 0 and 1")
    return out


def _normalise_imt_name(value: object) -> str:
    """Canonicalize PGA/SA labels so ``SA(1)`` and ``SA(1.0)`` match."""
    text = str(value).strip()
    if text.upper() == "PGA":
        return "PGA"
    match = re.fullmatch(r"SA\(([^)]+)\)", text, flags=re.IGNORECASE)
    if match:
        return f"SA({float(match.group(1)):g})"
    return text


def _normalise_statistic(value: object) -> str:
    text = str(value).strip().lower().replace("quantile_", "quantile-")
    if text == "mean":
        return "mean"
    if text.startswith("quantile-"):
        q = float(text.split("-", 1)[1])
    else:
        q = float(text)
    return f"quantile-{q:g}"


def _logic_tree_samples(model: str, samples: int | None) -> int:
    if samples is None:
        return NSHM2022_LOGIC_TREE_SAMPLES if model == "2022" else 0
    value = int(samples)
    if value < 0:
        raise ValueError("n_logic_tree_samples must be >= 0 or None")
    return value


def _windows_drive(path: str | Path) -> str:
    """Return a normalized Windows drive letter, or an empty string."""
    return ntpath.splitdrive(str(path))[0].casefold()


def _validate_oq_run_directory(model_root: str | Path, run_dir: str | Path) -> None:
    """Fail before writing files when an OQ job would cross Windows drives.

    OpenQuake 3.23.x requires file-valued job.ini inputs to be relative to the
    job file. Windows cannot represent a relative path between different drives.
    """
    model_drive = _windows_drive(model_root)
    run_drive = _windows_drive(run_dir)
    if model_drive and run_drive and model_drive != run_drive:
        model_root = Path(model_root)
        suggested = model_root / "_oq_runs"
        raise ValueError(
            "OpenQuake model inputs and run_dir are on different Windows drives: "
            f"model={model_root} ({model_drive.upper()}), run_dir={run_dir} "
            f"({run_drive.upper()}). OpenQuake 3.23.x requires relative input paths. "
            f"Use a run directory on the model drive, for example {suggested}, and "
            "save extracted Parquet/CSV results to any drive afterwards."
        )


def _oq_relative_path(target: str | Path, base_dir: str | Path) -> str:
    """Return an OpenQuake-safe path relative to the job INI directory."""
    _validate_oq_run_directory(target, base_dir)
    target = Path(target).resolve()
    base_dir = Path(base_dir).resolve()
    return Path(os.path.relpath(target, start=base_dir)).as_posix()


def _base_sections(
    model: str,
    root: Path,
    sites_csv: Path,
    samples: int,
    description: str,
    job_dir: Path,
) -> dict[str, dict[str, str]]:
    site_model_file = _oq_relative_path(sites_csv, job_dir)
    if model == "2022":
        return {
            "general": {
                "description": description,
                "random_seed": "25",
                "calculation_mode": "classical",
                "ps_grid_spacing": "30",
            },
            "logic_tree": {"number_of_logic_tree_samples": str(samples)},
            "erf": {
                "rupture_mesh_spacing": "4",
                "width_of_mfd_bin": "0.1",
                "complex_fault_mesh_spacing": "10.0",
                "area_source_discretization": "10.0",
            },
            "site_params": {
                "site_model_file": site_model_file,
            },
            "calculation": {
                "source_model_logic_tree_file": _oq_relative_path(root / "sources/source_model_extended.xml", job_dir),
                "gsim_logic_tree_file": _oq_relative_path(root / "gsim_model.xml", job_dir),
                "investigation_time": "1.0",
                "truncation_level": "4",
                "maximum_distance": "{'Active Shallow Crust': [(4.0, 0), (5.0, 100.0), (6.0, 200.0), (9.5, 300.0)], 'Subduction Interface': [(5.0, 0), (6.0, 200.0), (10, 500.0)], 'Subduction Intraslab': [(5.0, 0), (6.0, 200.0), (10, 500.0)]}",
            },
        }

    return {
        "general": {
            "description": description,
            "random_seed": "1024",
            "calculation_mode": "classical",
        },
        "logic_tree": {"number_of_logic_tree_samples": str(samples)},
        "erf": {
            "rupture_mesh_spacing": "4.0",
            "width_of_mfd_bin": "0.1",
            "area_source_discretization": "10.0",
        },
        "site_params": {
            # Match the supplied NSHM2010 native job exactly: all site parameters
            # come from the site model, so no reference-site values are added.
            "site_model_file": site_model_file,
        },
        "calculation": {
            "source_model_logic_tree_file": _oq_relative_path(root / "source_models/source_model_logic_tree.xml", job_dir),
            "gsim_logic_tree_file": _oq_relative_path(root / "gmpe_logic_tree_McVerry06_ONLY_MW_RotD50.xml", job_dir),
            "investigation_time": "1.0",
            "truncation_level": "4",
            "maximum_distance": "{'Active Shallow Crust': [(5.25, 0), (5.26, 400), (9.5, 400.0)], 'Subduction Interface': [(5.25, 0), (5.26, 400), (10.0, 400.0)], 'Subduction Intraslab': [(5.25, 0), (5.26, 400), (10.0, 400.0)], 'Volcanic': [(5.25, 0), (5.26, 400), (10.0, 400.0)]}",
        },
    }


def _format_imtls(periods: Sequence[float], levels: Sequence[float], include_pga: bool) -> str:
    levels = np.asarray(levels, dtype=float)
    if (
        levels.ndim != 1
        or levels.size < 2
        or np.any(~np.isfinite(levels))
        or np.any(levels <= 0)
        or np.any(np.diff(levels) <= 0)
    ):
        raise ValueError("im_levels must be a strictly increasing positive finite 1-D sequence")

    vals = [float(x) for x in levels]
    data: dict[str, list[float]] = {}
    if include_pga:
        data["PGA"] = vals

    for period in sorted(set(float(t) for t in periods)):
        if not np.isfinite(period) or period <= 0:
            raise ValueError("SA periods must be positive and finite")
        period_text = f"{period:.1f}" if period.is_integer() else f"{period:g}"
        data[f"SA({period_text})"] = vals

    if not data:
        raise ValueError("At least one IMT is required: include PGA and/or one SA period")
    return repr(data)


def _write_ini(sections: dict[str, dict[str, str]], path: Path) -> None:
    lines = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {value}" for key, value in values.items())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_poe_manifest(path: Path, poes: Sequence[float]) -> None:
    p = np.asarray(poes, dtype=float)
    return_periods = np.array([return_period_from_probability(x, 1.0) for x in p])
    df = pd.DataFrame(
        {
            "return_period": return_periods,
            "annual_poe": p,
            "poe_50yr": 1.0 - (1.0 - p) ** 50,
        }
    )
    df.to_csv(path, index=False)


def _write_return_period_manifest(path: Path, return_periods: Sequence[float]) -> None:
    _write_poe_manifest(path, [annual_poe(rp) for rp in return_periods])


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _openquake_imports():
    try:
        from openquake.calculators.extract import Extractor
        from openquake.commonlib.datastore import read
    except ImportError as exc:
        raise ImportError("OpenQuake is required for result extraction; use an OQ >=3.23.4 environment") from exc
    return Extractor, read


def _decode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="object"):
        out[col] = out[col].map(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
    return out


def _uhs_arraywrapper_to_frame(wrapper, statistic: str, poes: Sequence[float], imts: Sequence[str]) -> pd.DataFrame:
    """Decode an OpenQuake UHS ``ArrayWrapper`` into one row per site and PoE.

    The wrapped statistic is a nested structured array with outer fields named by
    PoE (formatted by OpenQuake to six decimal places) and inner fields named by
    IMT.  This helper is kept independent of OpenQuake imports so its behaviour can
    be unit-tested with synthetic structured arrays.
    """
    if not hasattr(wrapper, statistic):
        available = sorted(k for k in vars(wrapper) if not k.startswith("_") and k != "extra")
        raise KeyError(f"UHS statistic {statistic!r} is absent; available fields are {available}")

    arr = np.asarray(getattr(wrapper, statistic))
    if arr.dtype.names is None:
        raise TypeError(f"UHS statistic {statistic!r} is not a structured array")

    poe_fields = list(arr.dtype.names)
    numeric_poe_fields = {}
    for field in poe_fields:
        try:
            numeric_poe_fields[field] = float(field)
        except (TypeError, ValueError):
            continue

    rows: list[dict[str, object]] = []
    for site_id in range(len(arr)):
        for poe in np.asarray(poes, dtype=float):
            preferred = f"{poe:.6f}"
            if preferred in poe_fields:
                poe_field = preferred
            elif numeric_poe_fields:
                poe_field = min(numeric_poe_fields, key=lambda f: abs(numeric_poe_fields[f] - poe))
                if not np.isclose(numeric_poe_fields[poe_field], poe, rtol=0.0, atol=5e-7):
                    raise KeyError(f"Could not match requested UHS PoE {poe:.12g} to fields {poe_fields}")
            else:
                raise KeyError(f"UHS array contains no numeric PoE fields: {poe_fields}")

            spectrum = arr[poe_field][site_id]
            spectrum_names = spectrum.dtype.names
            if spectrum_names is None:
                raise TypeError(f"UHS PoE field {poe_field!r} does not contain an IMT-structured spectrum")

            row: dict[str, object] = {"site_id": site_id, "poe": float(poe), "statistic": str(statistic)}
            for imt in imts:
                if imt in spectrum_names:
                    row[imt] = float(spectrum[imt])
            rows.append(row)

    return pd.DataFrame(rows)


def _array_to_long_frame(array: np.ndarray, dims: Sequence[str], coords: dict[str, np.ndarray]) -> pd.DataFrame:
    if array.ndim != len(dims):
        raise ValueError(f"Array ndim {array.ndim} does not match dimensions {dims}")
    index = pd.MultiIndex.from_product([coords[d] for d in dims], names=list(dims))
    return pd.DataFrame({"value": array.reshape(-1)}, index=index).reset_index()


def _imt_period(imt: str) -> float:
    if str(imt).upper() == "PGA":
        return 0.0
    match = re.match(r"SA\(([^)]+)\)", str(imt), flags=re.I)
    return float(match.group(1)) if match else float("nan")


__all__ = [
    "__version__", "__version_tuple__", "version_info",
    "OQ_MIN_VERSION", "NSHM2022_LOGIC_TREE_SAMPLES", "MAX_POTENTIAL_PATHS",
    "DEFAULT_RETURN_PERIODS", "DEFAULT_QUANTILES", "IMAZEKI_RETURN_PERIODS",
    "CALDERON_DISAGG_PERIODS", "PRODUCTION_PERIODS",
    "DEFAULT_DISAGG_MAG_BIN_EDGES", "DEFAULT_DISAGG_DIST_BIN_EDGES",
    "DEFAULT_DISAGG_EPS_BIN_EDGES", "DEFAULT_DISAGG_COORDINATE_BIN_WIDTH_DEG",
    "NSHM2010_DEFAULT_PERIODS", "NSHM2022_DEFAULT_PERIODS",
    "NSHM2022_GNS_VALIDATION_PERIODS", "NSHM2022_GNS_VALIDATION_QUANTILES",
    "NSHM2010_REFERENCE_POES", "NSHM2010_DEFAULT_IM_LEVELS", "NSHM2022_DEFAULT_IM_LEVELS",
    "exceedance_probability", "return_period_from_probability", "annual_exceedance_rate",
    "annual_poe", "annual_poe_from_probability",
    "load_geojson_polygon", "points_in_polygon", "add_nshm2022_backarc",
    "extract_model_zip", "resolve_model_root", "infer_basin_depths", "prepare_sites", "write_site_model",
    "nzs1170p5_return_period_factor", "nzs1170p5_spectrum", "equivalent_return_period",
    "build_uhs_job", "build_nshm2022_uhs_validation_jobs", "build_nshm2010_uhs_validation_job",
    "build_disagg_job", "build_disagg_jobs",
    "openquake_version", "openquake_environment_info", "require_openquake_version", "inspect_disagg_parent_chunks",
    "run_openquake", "OQRunResult",
    "extract_uhs", "format_uhs_wide", "extract_hazard_curves", "validate_hazard_curve_coverage", "extract_site_collection",
    "extract_disaggregation", "summarize_disaggregation", "plot_mag_dist_disaggregation",
    "read_nshm2022_uhs_reference", "read_nshm2010_uhs_reference", "validate_nshm2010_uhs_validation_inputs", "normalise_uhs",
    "compare_uhs", "summarize_uhs_comparison", "validate_sample_convergence",
    "validate_uhs_calculation", "run_internal_validation", "logic_tree_branch_count",
]
