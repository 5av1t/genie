from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd

DEFAULT_CENTER = (28.0, 20.0)  # broad EMEA-ish center

# Minimal offline geocoder for known examples; fall back to prefix before '_' for city.
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

# also allow tokens with suffixes
for city in ["Bucharest","Paris","Ankara","Beirut","Belgrade","Berlin","Bratislava","Brussels"]:
    _COORDS[f"{city}_CDC"] = _BASE_COORDS[city]
    _COORDS[f"{city}_LDC"] = _BASE_COORDS[city]
for sup in ["Antalya","Krems"]:
    _COORDS[f"{sup}_FG"] = _BASE_COORDS[sup]

def _external_coords(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
    """If a 'Locations' (or 'locations') sheet exists with Location, Latitude, Longitude, prefer it."""
    for name in ["Locations", "locations", "Geo", "Coordinates"]:
        df = dfs.get(name)
        if isinstance(df, pd.DataFrame) and {"Location", "Latitude", "Longitude"}.issubset(set(df.columns)):
            out = {}
            for _, r in df.iterrows():
                loc = str(r.get("Location")) if pd.notna(r.get("Location")) else None
                lat = r.get("Latitude"); lon = r.get("Longitude")
                if loc and pd.notna(lat) and pd.notna(lon):
                    try:
                        out[loc] = (float(lat), float(lon))
                    except Exception:
                        pass
            return out
    return {}

def resolve_latlon(name: str, ext: Dict[str, Tuple[float, float]] = None) -> Tuple[float, float]:
    """Return (lat, lon) if known, else (None, None). Tries external map, exact, then prefix before '_'"""
    if not name:
        return (None, None)
    if ext and name in ext:
        lat, lon = ext[name]; return float(lat), float(lon)
    if name in _COORDS:
        lat, lon = _COORDS[name]; return float(lat), float(lon)
    if "_" in name:
        base = name.split("_")[0]
        if ext and base in ext:
            lat, lon = ext[base]; return float(lat), float(lon)
        if base in _COORDS:
            lat, lon = _COORDS[base]; return float(lat), float(lon)
    return (None, None)

def build_nodes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ext = _external_coords(dfs)
    wh = dfs.get("Warehouse", pd.DataFrame())
    if not wh.empty:
        for _, r in wh.iterrows():
            w = str(r.get("Warehouse", "")).strip()
            if not w: continue
            loc = str(r.get("Location", "")).strip() or w
            lat, lon = resolve_latlon(loc, ext)
            if lat is not None and lon is not None:
                rows.append({"name": w, "type": "warehouse", "location": loc, "lat": lat, "lon": lon})
    custs = dfs.get("Customers", pd.DataFrame())
    if not custs.empty:
        for _, r in custs.iterrows():
            c = str(r.get("Customer", "")).strip()
            if not c: continue
            loc = str(r.get("Location", "")).strip() or c
            lat, lon = resolve_latlon(loc, ext)
            if lat is not None and lon is not None:
                rows.append({"name": c, "type": "customer", "location": loc, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)

def flows_to_geo(flows: List[Dict[str, Any]], dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert optimizer flows to geocoded arcs with lat/lon for from/to."""
    wh = dfs.get("Warehouse", pd.DataFrame())
    wloc = {}
    if not wh.empty:
        for _, r in wh.iterrows():
            wname = str(r.get("Warehouse", "")).strip()
            if not wname: continue
            wloc[wname] = str(r.get("Location", "")).strip() or wname

    ext = _external_coords(dfs)
    rows = []
    for f in flows:
        w = f.get("warehouse"); c = f.get("customer"); p = f.get("product"); q = float(f.get("qty", 0) or 0)
        from_loc = wloc.get(str(w), str(w))
        to_loc = str(c)
        lat1, lon1 = resolve_latlon(from_loc, ext)
        lat2, lon2 = resolve_latlon(to_loc, ext)
        if None in (lat1, lon1, lat2, lon2):
            continue
        rows.append({
            "from": w, "to": c, "product": p, "qty": q,
            "from_lat": lat1, "from_lon": lon1, "to_lat": lat2, "to_lon": lon2,
        })
    return pd.DataFrame(rows)

def guess_map_center(nodes_df: pd.DataFrame) -> Tuple[float, float]:
    if nodes_df is None or nodes_df.empty:
        return DEFAULT_CENTER
    lat = nodes_df["lat"].astype(float).mean()
    lon = nodes_df["lon"].astype(float).mean()
    if pd.isna(lat) or pd.isna(lon):
        return DEFAULT_CENTER
    return float(lat), float(lon)

def ensure_location_row(dfs: Dict[str, pd.DataFrame], loc: str, country: str, lat: float, lon: float) -> Dict[str, pd.DataFrame]:
    newdfs = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in dfs.items()}
    locs = newdfs.get("Locations", pd.DataFrame())
    if locs.empty:
        locs = pd.DataFrame(columns=["Location", "Country", "Latitude", "Longitude"])
    # upsert
    mask = locs["Location"].astype(str).str.lower() == str(loc).lower()
    if mask.any():
        locs.loc[mask, ["Country","Latitude","Longitude"]] = [country, float(lat), float(lon)]
    else:
        locs = pd.concat([locs, pd.DataFrame([{"Location": loc, "Country": country, "Latitude": float(lat), "Longitude": float(lon)}])], ignore_index=True)
    newdfs["Locations"] = locs
    return newdfs
