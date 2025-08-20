# engine/geo.py
# Build nodes from sheets, convert flows to geo arcs, and manage Locations.
# Prefers the 'Locations' sheet; falls back to a small built-in gazetteer.

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# Minimal offline gazetteer for validated examples
_BASE_COORDS: Dict[str, Tuple[float, float]] = {
    "Bucharest": (44.4268, 26.1025),
    "Paris": (48.8566, 2.3522),
    "Ankara": (39.9334, 32.8597),
    "Beirut": (33.8938, 35.5018),
    "Belgrade": (44.8125, 20.4612),
    "Berlin": (52.5200, 13.4050),
    "Bratislava": (48.1486, 17.1077),
    "Brussels": (50.8503, 4.3517),
    "Aalborg": (57.0488, 9.9217),
    "Aarhus": (56.1629, 10.2039),
    "Abu Dhabi": (24.4539, 54.3773),
    "Ad Damman": (26.4207, 50.0888),
    "Adana": (37.0000, 35.3213),
    "Aden": (12.7855, 45.0187),
    "Agadir": (30.4278, -9.5981),
    "Al Ahmadi": (29.0760, 48.0836),
    "Al Aqabah": (29.5328, 35.0063),
    "Al Ayn": (24.1302, 55.8023),
    "Antalya": (36.8969, 30.7133),
    "Krems": (48.4102, 15.6148),
}
_COORDS: Dict[str, Tuple[float, float]] = dict(_BASE_COORDS)

# Allow exact tokens like Bucharest_CDC, Berlin_LDC, Antalya_FG, etc.
for city in ["Bucharest","Paris","Ankara","Beirut","Belgrade","Berlin","Bratislava","Brussels"]:
    _CORDS_CITY = _BASE_COORDS.get(city)
    if _CORDS_CITY:
        _COORDS[f"{city}_CDC"] = _CORDS_CITY
        _COORDS[f"{city}_LDC"] = _CORDS_CITY
for sup in ["Antalya","Krems"]:
    if sup in _BASE_COORDS:
        _COORDS[f"{sup}_FG"] = _BASE_COORDS[sup]

def _safe_df(df) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _external_coords(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
    """Prefer 'Locations' sheet: Location, Latitude, Longitude."""
    out: Dict[str, Tuple[float,float]] = {}
    loc = _safe_df(dfs.get("Locations"))
    if not loc.empty and set(["Location","Latitude","Longitude"]).issubset(loc.columns):
        for _, r in loc.iterrows():
            name = str(r.get("Location","")).strip()
            lat = r.get("Latitude", None)
            lon = r.get("Longitude", None)
            try:
                if name and pd.notna(lat) and pd.notna(lon):
                    out[name] = (float(lat), float(lon))
            except Exception:
                continue
    return out

def resolve_latlon(name: str, ext: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[Optional[float], Optional[float]]:
    """Return (lat, lon) if known, else (None, None). Tries Locations sheet, exact token, then prefix before '_'."""
    if not name:
        return (None, None)
    if ext and name in ext:
        lat, lon = ext[name]
        return float(lat), float(lon)
    if name in _COORDS:
        lat, lon = _COORDS[name]
        return float(lat), float(lon)
    if "_" in name:
        base = name.split("_")[0]
        if ext and base in ext:
            lat, lon = ext[base]
            return float(lat), float(lon)
        if base in _COORDS:
            lat, lon = _COORDS[base]
            return float(lat), float(lon)
    return (None, None)

def build_nodes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return df with columns: name, type ('warehouse'|'customer'), location, lat, lon."""
    rows: List[Dict[str, Any]] = []
    ext = _external_coords(dfs)
    wh = _safe_df(dfs.get("Warehouse"))
    if not wh.empty:
        for _, r in wh.iterrows():
            w = str(r.get("Warehouse","")).strip()
            if not w:
                continue
            loc = str(r.get("Location", w)) if pd.notna(r.get("Location", w)) else w
            lat, lon = resolve_latlon(loc, ext)
            if lat is not None and lon is not None:
                rows.append({"name": w, "type": "warehouse", "location": loc, "lat": lat, "lon": lon})
    cu = _safe_df(dfs.get("Customers"))
    if not cu.empty:
        for _, r in cu.iterrows():
            c = str(r.get("Customer","")).strip()
            if not c:
                continue
            loc = str(r.get("Location", c)) if pd.notna(r.get("Location", c)) else c
            lat, lon = resolve_latlon(loc, ext)
            if lat is not None and lon is not None:
                rows.append({"name": c, "type": "customer", "location": loc, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)

def flows_to_geo(flows: List[Dict[str, Any]], dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert optimizer flow list to geocoded arcs with lat/lon for from/to.
    Returns columns: from, to, product, qty, from_lat, from_lon, to_lat, to_lon
    """
    wh = _safe_df(dfs.get("Warehouse"))
    wloc = {}
    if not wh.empty and "Warehouse" in wh.columns:
        for _, r in wh.iterrows():
            wloc[str(r.get("Warehouse",""))] = str(r.get("Location", "")) or str(r.get("Warehouse",""))

    ext = _external_coords(dfs)
    rows: List[Dict[str, Any]] = []
    for f in flows or []:
        w = str(f.get("warehouse",""))
        c = str(f.get("customer",""))
        p = f.get("product")
        q = float(f.get("qty", 0.0) or 0.0)
        from_loc = wloc.get(w, w)
        to_loc = c
        lat1, lon1 = resolve_latlon(from_loc, ext)
        lat2, lon2 = resolve_latlon(to_loc, ext)
        if None in (lat1, lon1, lat2, lon2):
            continue
        rows.append({
            "from": w, "to": c, "product": p, "qty": q,
            "from_lat": lat1, "from_lon": lon1, "to_lat": lat2, "to_lon": lon2
        })
    return pd.DataFrame(rows)

def guess_map_center(nodes_df: pd.DataFrame, default: Tuple[float, float] = (30.0, 15.0)) -> Tuple[float, float]:
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return default
    try:
        lat = float(nodes_df["lat"].mean())
        lon = float(nodes_df["lon"].mean())
        if np.isfinite(lat) and np.isfinite(lon):
            return (lat, lon)
    except Exception:
        pass
    return default

def ensure_location_row(dfs: Dict[str, pd.DataFrame], location: str, country: str, lat: float, lon: float) -> Dict[str, pd.DataFrame]:
    """
    Upsert a row in 'Locations' sheet with given coordinates.
    """
    newdfs = {k: (v.copy() if isinstance(v, pd.DataFrame) else pd.DataFrame()) for k, v in (dfs or {}).items()}
    loc = _safe_df(newdfs.get("Locations"))
    if loc.empty:
        loc = pd.DataFrame(columns=["Location","Country","Latitude","Longitude"])
    if not set(["Location","Country","Latitude","Longitude"]).issubset(loc.columns):
        for c in ["Location","Country","Latitude","Longitude"]:
            if c not in loc.columns:
                loc[c] = np.nan
    mask = (loc["Location"].astype(str) == str(location))
    if mask.any():
        loc.loc[mask, "Country"] = country
        loc.loc[mask, "Latitude"] = float(lat)
        loc.loc[mask, "Longitude"] = float(lon)
    else:
        loc = pd.concat([loc, pd.DataFrame([{
            "Location": location, "Country": country, "Latitude": float(lat), "Longitude": float(lon)
        }])], ignore_index=True)
    newdfs["Locations"] = loc
    return newdfs
