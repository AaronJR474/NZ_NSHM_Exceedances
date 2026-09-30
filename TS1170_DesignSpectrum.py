"""SNZ TS 1170.5:2025 horizontal elastic site spectra.

This module reads the official digital data distributed with SNZ TS 1170.5:2025
and evaluates the horizontal elastic site spectrum from Section 3.1.

Expected data directory
-----------------------
By default, the module reads the following files from ``Path('TS1170_data')``::

    TS1170-5_Figure3-2_2025.geojson
    TS1170-5_Table3-1_2025.csv
    TS1170-5_Table3-2_2025.csv

Figure 3.2 is used to determine whether a latitude/longitude falls within a
listed urban or rural settlement. Table 3.1 is used inside those boundaries;
otherwise the nearest 0.1 x 0.1 degree Table 3.2 grid point is used.

The tabulated horizontal spectral parameters are PGA, Sa,s, Tc and Td. The
spectrum returned here is the elastic site spectrum C(T) = Sa(T), in units of g.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from shapely.geometry import Point, shape
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TS1170_DesignSpectrum requires shapely for the Figure 3.2 settlement lookup"
    ) from exc


DEFAULT_DATA_DIR = Path("TS1170_data")
SUPPORTED_RETURN_PERIODS = (25, 50, 100, 250, 500, 1000, 2500)
SUPPORTED_SITE_CLASSES = ("I", "II", "III", "IV", "V", "VI")

__all__ = [
    "DEFAULT_DATA_DIR",
    "SUPPORTED_RETURN_PERIODS",
    "SUPPORTED_SITE_CLASSES",
    "ts1170_site_class_from_vs30",
    "ts1170_locate_site",
    "ts1170_site_demand_parameters",
    "ts1170_spectrum_from_parameters",
    "ts1170_spectrum",
]


def _normalise_site_class(site_class: str | int) -> str:
    """Normalize a Site Class identifier to Roman numerals I--VI."""
    if isinstance(site_class, (int, np.integer)):
        mapping = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
        if int(site_class) not in mapping:
            raise ValueError("site_class integer must be between 1 and 6")
        return mapping[int(site_class)]

    value = str(site_class).strip().upper()
    aliases = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}
    value = aliases.get(value, value)
    if value not in SUPPORTED_SITE_CLASSES:
        raise ValueError("site_class must be one of I, II, III, IV, V or VI")
    return value


def ts1170_site_class_from_vs30(vs30: float) -> str:
    """Assign the simplified SNZ TS 1170.5:2025 Site Class from Vs30.

    Parameters
    ----------
    vs30 : float
        Time-averaged shear-wave velocity over the upper 30 m, in m/s.

    Returns
    -------
    str
        Site Class ``I`` through ``VII`` using the Vs30 boundaries in Table 3.3.

    Notes
    -----
    This is deliberately a Vs30-only classification for large-scale hazard-spectrum
    comparisons. Table 3.3 includes additional geological/geotechnical criteria for
    several classes. Those additional criteria are not evaluated here.

    The Vs30 boundaries used are::

        I    Vs30 > 750 m/s
        II   450 < Vs30 <= 750 m/s
        III  300 < Vs30 <= 450 m/s
        IV   250 < Vs30 <= 300 m/s
        V    200 < Vs30 <= 250 m/s
        VI   150 < Vs30 <= 200 m/s
        VII  Vs30 <= 150 m/s

    Site Class VII requires site-specific dynamic site-response analysis and cannot
    be generated from the tabulated Site Class I--VI parameters used by this module.
    """
    value = float(vs30)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("vs30 must be a finite positive value in m/s")
    if value > 750:
        return "I"
    if value > 450:
        return "II"
    if value > 300:
        return "III"
    if value > 250:
        return "IV"
    if value > 200:
        return "V"
    if value > 150:
        return "VI"
    return "VII"


def _validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    lat = float(latitude)
    lon = float(longitude)
    if not np.isfinite(lat) or not -90 <= lat <= 90:
        raise ValueError("latitude must be finite and between -90 and 90 degrees")
    if not np.isfinite(lon) or not -180 <= lon <= 180:
        raise ValueError("longitude must be finite and between -180 and 180 degrees")
    return lat, lon


def _validate_return_period(return_period: float) -> int:
    """Validate and snap a return period to the supported TS1170 table values."""
    rp = float(return_period)

    if not np.isfinite(rp):
        raise ValueError(f"return_period must be one of {SUPPORTED_RETURN_PERIODS} years")

    supported = np.asarray(SUPPORTED_RETURN_PERIODS, dtype=float)
    idx = np.argmin(np.abs(supported - rp))

    if not np.isclose(rp, supported[idx], rtol=0.0, atol=1e-6):
        raise ValueError(f"return_period must be one of {SUPPORTED_RETURN_PERIODS} years")

    return int(supported[idx])


def _required_path(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required TS1170 data file not found: {path}")
    return path


@lru_cache(maxsize=8)
def _load_data_cached(data_dir_string: str):
    data_dir = Path(data_dir_string)
    table31_path = _required_path(data_dir, "TS1170-5_Table3-1_2025.csv")
    table32_path = _required_path(data_dir, "TS1170-5_Table3-2_2025.csv")
    figure32_path = _required_path(data_dir, "TS1170-5_Figure3-2_2025.geojson")

    table31 = pd.read_csv(table31_path)
    table32 = pd.read_csv(table32_path)

    with figure32_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    settlements = []
    for feature in geojson.get("features", []):
        name = str(feature.get("properties", {}).get("Name", "")).strip()
        if not name:
            raise ValueError(f"Figure 3.2 feature without a Name in {figure32_path}")
        settlements.append((name, shape(feature["geometry"])))

    required31 = {"location", "location_ascii", "apoe", "M", "D"}
    required32 = {"location", "latitude", "longitude", "apoe", "M", "D"}
    spectral = {f"{site}-{name}" for site in SUPPORTED_SITE_CLASSES for name in ("PGA", "Sas", "Tc", "Td")}
    missing31 = (required31 | spectral) - set(table31.columns)
    missing32 = (required32 | spectral) - set(table32.columns)
    if missing31:
        raise ValueError(f"Table 3.1 is missing required columns: {sorted(missing31)}")
    if missing32:
        raise ValueError(f"Table 3.2 is missing required columns: {sorted(missing32)}")

    table31 = table31.copy()
    table32 = table32.copy()
    table31["return_period"] = table31["apoe"].astype(str).str.split("/").str[-1].astype(int)
    table32["return_period"] = table32["apoe"].astype(str).str.split("/").str[-1].astype(int)

    polygon_names = {name for name, _ in settlements}
    table31_names = set(table31["location"].astype(str)) | set(table31["location_ascii"].astype(str))
    missing_names = sorted(polygon_names - table31_names)
    if missing_names:
        raise ValueError(f"Figure 3.2 names missing from Table 3.1: {missing_names[:10]}")

    return table31, table32, settlements


def _load_data(data_dir: str | Path):
    return _load_data_cached(str(Path(data_dir).resolve()))


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance from one point to arrays of points, in km."""
    radius_km = 6371.0088
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def ts1170_locate_site(
    latitude: float,
    longitude: float,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict:
    """Locate a site in the SNZ TS 1170.5:2025 digital demand-parameter data.

    Parameters
    ----------
    latitude, longitude : float
        Site coordinates in decimal degrees (WGS84 longitude/latitude convention).
    data_dir : path-like, default ``Path('TS1170_data')``
        Directory containing the official TS1170 digital CSV and GeoJSON files.

    Returns
    -------
    dict
        Location metadata describing which source applies. If the point lies inside a
        Figure 3.2 urban/rural settlement polygon, ``source`` is ``"Table 3.1"`` and
        ``location`` is the named settlement. Otherwise ``source`` is ``"Table 3.2"``
        and the nearest 0.1 x 0.1 degree grid point is returned.

    Notes
    -----
    This implements Section 3.1.2: Table 3.1 applies within the Figure 3.2
    settlement boundaries; Table 3.2 applies elsewhere using the nearest grid point.
    Polygon boundaries are treated as belonging to the settlement.
    """
    lat, lon = _validate_coordinates(latitude, longitude)
    table31, table32, settlements = _load_data(data_dir)
    point = Point(lon, lat)

    matches = [(name, geom.area) for name, geom in settlements if geom.covers(point)]
    if matches:
        matches.sort(key=lambda item: item[1])
        name = matches[0][0]
        return {
            "source": "Table 3.1",
            "location": name,
            "input_latitude": lat,
            "input_longitude": lon,
            "grid_latitude": np.nan,
            "grid_longitude": np.nan,
            "grid_distance_km": np.nan,
        }

    grid_lat = table32["latitude"].to_numpy(dtype=float)
    grid_lon = table32["longitude"].to_numpy(dtype=float)
    unique_mask = ~table32[["latitude", "longitude"]].duplicated().to_numpy()
    grid_lat = grid_lat[unique_mask]
    grid_lon = grid_lon[unique_mask]
    distances = _haversine_km(lat, lon, grid_lat, grid_lon)
    idx = int(np.argmin(distances))
    glat = float(grid_lat[idx])
    glon = float(grid_lon[idx])

    row = table32[(table32["latitude"] == glat) & (table32["longitude"] == glon)].iloc[0]
    return {
        "source": "Table 3.2",
        "location": str(row["location"]),
        "input_latitude": lat,
        "input_longitude": lon,
        "grid_latitude": glat,
        "grid_longitude": glon,
        "grid_distance_km": float(distances[idx]),
    }


def ts1170_site_demand_parameters(
    latitude: float,
    longitude: float,
    return_period: float,
    site_class: str | int | None = None,
    vs30: float | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict:
    """Return the tabulated TS1170 horizontal spectral parameters for a site.

    Parameters
    ----------
    latitude, longitude : float
        Site coordinates in decimal degrees.
    return_period : float
        Mean return period in years. Supported values are 25, 50, 100, 250, 500,
        1000 and 2500, corresponding directly to the APoE rows in Tables 3.1--3.2.
    site_class : {"I", "II", "III", "IV", "V", "VI"}, optional
        Site Class to query. Integer values 1--6 are also accepted.
    vs30 : float, optional
        Vs30 in m/s. Used to assign a simplified Site Class when ``site_class`` is not
        supplied. If both ``site_class`` and ``vs30`` are supplied, ``site_class`` is
        used and ``vs30`` is retained only as metadata.
    data_dir : path-like, default ``Path('TS1170_data')``
        Directory containing the official TS1170 digital data files.

    Returns
    -------
    dict
        The selected PGA, Sa,s, Tc and Td values together with earthquake magnitude M,
        distance D, return period, Site Class, and Table 3.1/Table 3.2 source metadata.

    Notes
    -----
    The Vs30-only classification is intentionally simplified. It follows the Table 3.3
    Vs30 boundaries but does not evaluate the additional geotechnical criteria stated
    for several classes. Site Class VII is not tabulated and requires site-specific
    dynamic site-response analysis.
    """
    rp = _validate_return_period(return_period)
    if site_class is None and vs30 is None:
        raise ValueError("supply either site_class or vs30")

    if site_class is None:
        site = ts1170_site_class_from_vs30(float(vs30))
        if site == "VII":
            raise ValueError("Vs30 <= 150 m/s gives Site Class VII, which requires site-specific analysis")
    else:
        site = _normalise_site_class(site_class)

    location = ts1170_locate_site(latitude, longitude, data_dir=data_dir)
    table31, table32, _ = _load_data(data_dir)

    if location["source"] == "Table 3.1":
        name = location["location"]
        mask_name = (table31["location"].astype(str) == name) | (table31["location_ascii"].astype(str) == name)
        rows = table31[mask_name & (table31["return_period"] == rp)]
    else:
        rows = table32[
            (table32["latitude"] == location["grid_latitude"])
            & (table32["longitude"] == location["grid_longitude"])
            & (table32["return_period"] == rp)
        ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one TS1170 row for {location['location']}, RP={rp}; found {len(rows)}"
        )

    row = rows.iloc[0]
    prefix = f"{site}-"
    result = {
        **location,
        "return_period": rp,
        "apoe": f"1/{rp}",
        "site_class": site,
        "vs30": np.nan if vs30 is None else float(vs30),
        "M": float(row["M"]),
        "D": row["D"],
        "PGA": float(row[prefix + "PGA"]),
        "Sas": float(row[prefix + "Sas"]),
        "Tc": float(row[prefix + "Tc"]),
        "Td": float(row[prefix + "Td"]),
    }
    return result


def ts1170_spectrum_from_parameters(
    periods: Sequence[float],
    PGA: float,
    Sas: float,
    Tc: float,
    Td: float,
    short_period_method: str = "linear",
) -> np.ndarray:
    """Evaluate the SNZ TS 1170.5:2025 parametric horizontal spectrum.

    Parameters
    ----------
    periods : sequence of float
        Oscillator periods in seconds. Values must be finite and non-negative.
    PGA : float
        Peak ground acceleration in g.
    Sas : float
        Short-period spectral acceleration Sa,s in g.
    Tc : float
        Spectral-acceleration-plateau corner period in seconds.
    Td : float
        Spectral-velocity-plateau corner period in seconds.
    short_period_method : {"linear", "equivalent_static"}, default "linear"
        Treatment for 0 < T <= 0.1 s. ``"linear"`` interpolates between PGA at
        T=0 and Sa,s at T=0.1 s, as permitted for analysis methods other than the
        equivalent-static method. ``"equivalent_static"`` uses Sa,s throughout
        0 < T <= 0.1 s.

    Returns
    -------
    numpy.ndarray
        Spectral acceleration Sa(T), in g, at each requested period.

    Notes
    -----
    For T > 0.1 s the function implements the Section 3.1.2 plateau, constant-
    velocity and long-period branches defined by Sa,s, Tc and Td. The function is
    independent of the location lookup and is useful for direct numerical validation.
    """
    T = np.asarray(periods, dtype=float)
    if T.ndim != 1:
        raise ValueError("periods must be a one-dimensional sequence")
    if np.any(~np.isfinite(T)) or np.any(T < 0):
        raise ValueError("periods must contain only finite, non-negative values")

    pga = float(PGA)
    sas = float(Sas)
    tc = float(Tc)
    td = float(Td)
    if not all(np.isfinite(x) for x in (pga, sas, tc, td)):
        raise ValueError("PGA, Sas, Tc and Td must be finite")
    if pga < 0 or sas < 0 or tc <= 0.1 or td <= tc:
        raise ValueError("require PGA >= 0, Sas >= 0, Tc > 0.1 s and Td > Tc")

    method = str(short_period_method).strip().lower()
    if method not in {"linear", "equivalent_static"}:
        raise ValueError("short_period_method must be 'linear' or 'equivalent_static'")

    Sa = np.empty_like(T)
    m0 = T == 0
    m_short = (T > 0) & (T <= 0.1)
    m_plateau = (T > 0.1) & (T <= tc)
    m_velocity = (T > tc) & (T <= td)
    m_long = T > td

    Sa[m0] = pga
    if method == "linear":
        Sa[m_short] = pga + (sas - pga) * T[m_short] / 0.1
    else:
        Sa[m_short] = sas
    Sa[m_plateau] = sas
    Sa[m_velocity] = sas * tc / T[m_velocity]
    Sa[m_long] = sas * tc / T[m_long] * np.sqrt(td / T[m_long])
    return Sa


def ts1170_spectrum(
    periods: Sequence[float],
    latitude: float,
    longitude: float,
    return_period: float,
    site_class: str | int | None = None,
    vs30: float | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    short_period_method: str = "linear",
) -> pd.DataFrame:
    """Compute the SNZ TS 1170.5:2025 horizontal elastic site spectrum C(T).

    Parameters
    ----------
    periods : sequence of float
        Oscillator periods in seconds.
    latitude, longitude : float
        Site coordinates in decimal degrees.
    return_period : float
        Mean return period in years. Supported values are 25, 50, 100, 250, 500,
        1000 and 2500.
    site_class : {"I", "II", "III", "IV", "V", "VI"}, optional
        Site Class. Supply either ``site_class`` or ``vs30``.
    vs30 : float, optional
        Vs30 in m/s. If ``site_class`` is omitted, the simplified Table 3.3 Vs30
        boundaries are used to assign Site Class I--VII. Site Class VII is rejected
        because it requires site-specific dynamic response analysis.
    data_dir : path-like, default ``Path('TS1170_data')``
        Directory containing the official TS1170 digital data.
    short_period_method : {"linear", "equivalent_static"}, default "linear"
        Short-period treatment for 0 < T <= 0.1 s. ``"linear"`` is appropriate for
        response-spectrum comparison and reproduces the supplied getTS1170Spectra.m
        formulation. ``"equivalent_static"`` uses Sa,s over this interval.

    Returns
    -------
    pandas.DataFrame
        One row per requested period. ``Sa`` and ``C`` are the elastic horizontal
        spectral acceleration in g. The selected PGA, Sa,s, Tc, Td, M, D, Site Class,
        return period, and Table 3.1/Table 3.2 location source are retained as metadata.

    Notes
    -----
    Section 3.1 defines C(T) = Sa(T). For sites inside a Figure 3.2 settlement
    boundary, the demand parameters are taken from Table 3.1. For all other sites,
    they are taken from the nearest 0.1 x 0.1 degree Table 3.2 grid point.
    """
    params = ts1170_site_demand_parameters(
        latitude, longitude, return_period, site_class=site_class, vs30=vs30, data_dir=data_dir
    )
    T = np.asarray(periods, dtype=float)
    Sa = ts1170_spectrum_from_parameters(
        T, params["PGA"], params["Sas"], params["Tc"], params["Td"], short_period_method
    )

    return pd.DataFrame({
        "period": T,
        "Sa": Sa,
        "C": Sa,
        "PGA": params["PGA"],
        "Sas": params["Sas"],
        "Tc": params["Tc"],
        "Td": params["Td"],
        "M": params["M"],
        "D": params["D"],
        "return_period": params["return_period"],
        "site_class": params["site_class"],
        "vs30": params["vs30"],
        "source": params["source"],
        "source_location": params["location"],
        "source_latitude": params["grid_latitude"],
        "source_longitude": params["grid_longitude"],
        "grid_distance_km": params["grid_distance_km"],
        "short_period_method": str(short_period_method).strip().lower(),
    })
