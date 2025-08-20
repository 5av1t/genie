# engine/geo.py
# Robust geocoding for GENIE: NEVER trust an LLM for lat/lon.
# Resolution order:
#   1) Locations sheet in the workbook (authoritative)
#   2) Offline gazetteer CSV (data/geo_gazetteer.csv) and/or built-in seed map
#   3) Optional online geocoding (OpenStreetMap Nominatim) with caching & rate limits
#
# Also provides build_nodes(), flows_to_geo(), guess_map_center() for pydeck maps.

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import os
import json
import time
import csv

import pandas as pd

# --------------------- Config ---------------------

_GAZETTEER_CSV = os.environ.get("GENIE_GAZETTEER_CSV", "data/geo_gazetteer.csv")
_CACHE_FILE     = os.environ.get("GENIE_GEOCODE_CACHE", "data/geocode_cache.json")
_ENABLE_ONLINE  = os.environ.get("GENIE_GEOCODE_ONLINE", "false").lower() in {"1","true","yes"}
_NOMINATIM_URL  = "https://nominatim.openstreetmap.org/search"
_USER_AGENT     = os.environ.get("GENIE_HTTP_UA", "GENIE-SCND/1.0 (support@example.com)")

# Create data/ if not exists (safe on Streamlit Cloud; ephemeral but fine)
os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)

# --------------------- Built-in seed (tiny, safe defaults) ---------------------

# These are *only* for immediate UX. Prefer Locations sheet and gazetteer CSV.
_BUILTIN_SEED: Dict[str, Tuple[float, float, str]] = {
    # name             (lat, lon, country)
    "Bucharest":       (44.4268, 26.1025, "Romania"),
    "Paris":           (48.8566, 2.3522,  "France"),
    "Ankara":          (39.9334, 32.8597, "Türkiye"),
    "Beirut":          (33.8938, 35.5018, "Lebanon"),
    "Belgrade":        (44.8125, 20.4612, "Serbia"),
    "Berlin":          (52.52,   13.405,  "Germany"),
    "Bratislava":      (48.1486, 17.1077, "Slovakia"),
    "Brussels":        (50.8503, 4.3517,  "Belgium"),
    "Aalborg":         (57.0488, 9.9217,  "Denmark"),
    "Aarhus":          (56.1629, 10.2039, "Denmark"),
    "Abu Dhabi":       (24.4539, 54.3773, "United Arab Emirates"),
    "Ad Damman":       (26.4207, 50.0888, "Saudi Arabia"),
    "Adana":           (37.0,    35.3213, "Türkiye"),
    "Aden":            (12.7855, 45.0187, "Yemen"),
    "Agadir":          (30.4278, -9.5981, "Morocco"),
    "Al Ahmadi":       (29.0760, 48.0836, "Kuwait"),
    "Al Aqabah":       (29.5328, 35.0063, "Jordan"),
    "Al Ayn":          (24.1302, 55.8023, "United Arab Emirates"),
    "Antalya":         (36.8969, 30.7133, "Türkiye"),
    "Krems":           (48.4102, 15.6148, "Austria"),
}

# Also allow common suffixes we see (CDC/LDC/FG)
def _seed_lookup(name: str) -> Optional[Tuple[float, float, str]]:
    if not name:
        return None
    if name in _BUILTIN_SEED:
        return _BUILTIN_SEED[name]
    if "_" in name:
        base = name.split("_")[0]
        return _BUILTIN_SEED.get(base)
    return None

# --------------------- Cache helpers ---------------------

def _load_cache() -> Dict[str, Dict[str, Any]]:
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --------------------- Gazetteer loader ---------------------

def _load_gazetteer() -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping {name_lower: {name, country, lat, lon}} from CSV if present.
    CSV headers expected: Name, Country, Latitude, Longitude
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        if os.path.exists(_GAZETTEER_CSV):
            with open(_GAZETTEER_CSV, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    name = (row.get("Name") or "").strip()
                    if not name:
                        continue
                    try:
                        lat = float(row.get("Latitude"))
                        lon = float(row.get("Longitude"))
                    except Exception:
                        continue
                    out[name.lower()] = {
                        "name": name,
                        "country": (row.get("Country") or "").strip(),
                        "lat": lat,
                        "lon": lon,
                    }
    except Exception:
        # Ignore bad CSV; app still runs with seed/online
        return {}
    return out

_GAZ = _load_gazetteer()

# --------------------- Online geocoding (optional) ---------------------

def _nominatim_search(q: str) -> Optional[Tuple[float, float, str]]:
    """Call OpenStreetMap Nominatim for 'q' if online geocoding is enabled."""
    if not _ENABLE_ONLINE:
        return None
    try:
        import requests  # Only import if needed
        params = {"q": q, "format": "json", "limit": 1}
        headers = {"User-Agent": _USER_AGENT}
        resp = requests.get(_NOMINATIM_URL, params=params, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                lat = float(data[0]["lat"]); lon = float(data[0]["lon"])
                # Try to extract country from display_name if present
                disp = data[0].get("display_name") or ""
                country = ""
                # naive parse: last token often country
                parts = [p.strip() for p in disp.split(",") if p.strip()]
                if parts:
                    country = parts[-1]
                # Nominatim usage policy: throttle a bit
                time.sleep(1.0)
                return (lat, lon, country)
    except Exception:
        return None
    return None

# --------------------- Public geocoding API ---------------------

def resolve_from_locations_sheet(dfs: Dict[str, pd.DataFrame], name: str) -> Optional[Tuple[float, float, str]]:
    """Return (lat, lon, country) from workbook Locations sheet, if found."""
    locs = dfs.get("Locations")
    if not isinstance(locs, pd.DataFrame) or locs.empty:
        return None
    df = locs
    # Try exact match in Location column
    match = df[df["Location"].astype(str) == str(name)]
    if match.empty and "_" in name:
        base = name.split("_")[0]
        match = df[df["Location"].astype(str) == base]
    if match.empty:
        return None
    row = match.iloc[0]
    lat = row.get("Latitude"); lon = row.get("Longitude")
    if pd.isna(lat) or pd.isna(lon):
        return None
    country = row.get("Country") if "Country" in df.columns else ""
    return (float(lat), float(lon), str(country or ""))

def geocode_location(name: str,
                     dfs: Optional[Dict[str, pd.DataFrame]] = None,
                     city_hint: Optional[str] = None,
                     country_hint: Optional[str] = None,
                     allow_online: Optional[bool] = None) -> Optional[Tuple[float, float, str]]:
    """
    Deterministic resolver:
      - Locations sheet (by Location)
      - Gazetteer CSV (case-insensitive) / Built-in seed
      - Online Nominatim (if enabled)
    Returns (lat, lon, country) or None if unresolved.
    """
    if not name:
        return None

    # 1) Locations sheet
    if dfs:
        hit = resolve_from_locations_sheet(dfs, name)
        if hit:
            return hit

    # Prepare cache key
    key_tokens = [name]
    if city_hint: key_tokens.append(city_hint)
    if country_hint: key_tokens.append(country_hint)
    cache_key = "|".join(key_tokens).lower()

    cache = _load_cache()
    if cache_key in cache:
        rec = cache[cache_key]
        return (rec["lat"], rec["lon"], rec.get("country",""))

    # 2) Gazetteer CSV
    # Try exact, base token, and hints
    def _gaz_lookup(q: str) -> Optional[Tuple[float,float,str]]:
        ent = _GAZ.get(q.lower())
        if ent:
            return (float(ent["lat"]), float(ent["lon"]), ent.get("country",""))
        return None

    # exact
    hit = _gaz_lookup(name)
    if not hit and "_" in name:
        hit = _gaz_lookup(name.split("_")[0])
    # try hints combination
    if not hit and city_hint:
        hit = _gaz_lookup(city_hint)
    if not hit:
        seed = _seed_lookup(name)
        if seed:
            lat, lon, country = seed
            hit = (lat, lon, country)

    # 3) Online (optional, with hints)
    if not hit:
        q = name
        if city_hint and city_hint.lower() not in q.lower():
            q = f"{city_hint}, {q}"
        if country_hint:
            q = f"{q}, {country_hint}"
        hit = _nominatim_search(q)

    # Cache & return
    if hit:
        lat, lon, country = hit
        cache[cache_key] = {"lat": lat, "lon": lon, "country": country}
        _save_cache(cache)
        return hit

    return None

# --------------------- Map helpers ---------------------

def _ext_locations_map(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float,float,str]]:
    """Build a fast dict from Locations sheet for plotting."""
    out: Dict[str, Tuple[float,float,str]] = {}
    locs = dfs.get("Locations")
    if isinstance(locs, pd.DataFrame) and not locs.empty:
        for _, r in locs.iterrows():
            loc = str(r.get("Location") or "").strip()
            if not loc: continue
            lat = r.get("Latitude"); lon = r.get("Longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue
            country = str(r.get("Country") or "") if "Country" in locs.columns else ""
            out[loc] = (float(lat), float(lon), country)
    return out

def resolve_latlon(name: str, dfs: Dict[str, pd.DataFrame]) -> Tuple[Optional[float], Optional[float]]:
    """Convenience: (lat, lon) only (for mapping); consults Locations sheet, gazetteer, seed, online if enabled."""
    rec = geocode_location(name, dfs=dfs, allow_online=_ENABLE_ONLINE)
    if rec:
        return rec[0], rec[1]
    return None, None

def build_nodes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Returns DataFrame with columns: name, type ('warehouse'/'customer'), location, lat, lon, country
    """
    rows: List[Dict[str, Any]] = []

    wh = dfs.get("Warehouse")
    if isinstance(wh, pd.DataFrame) and not wh.empty:
        for _, r in wh.iterrows():
            w = str(r.get("Warehouse") or "").strip()
            if not w: continue
            loc = str(r.get("Location") or w).strip()
            hit = geocode_location(loc, dfs=dfs, allow_online=_ENABLE_ONLINE)
            if hit:
                lat, lon, country = hit
                rows.append({"name": w, "type": "warehouse", "location": loc, "lat": lat, "lon": lon, "country": country})

    cu = dfs.get("Customers")
    if isinstance(cu, pd.DataFrame) and not cu.empty:
        for _, r in cu.iterrows():
            c = str(r.get("Customer") or "").strip()
            if not c: continue
            loc = str(r.get("Location") or c).strip()
            hit = geocode_location(loc, dfs=dfs, allow_online=_ENABLE_ONLINE)
            if hit:
                lat, lon, country = hit
                rows.append({"name": c, "type": "customer", "location": loc, "lat": lat, "lon": lon, "country": country})

    return pd.DataFrame(rows)

def flows_to_geo(flows: List[Dict[str, Any]], dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert optimizer flow list to geocoded arcs: columns:
      from, to, product, qty, from_lat, from_lon, to_lat, to_lon
    """
    # Warehouse name -> location name map
    wloc = {}
    wh = dfs.get("Warehouse")
    if isinstance(wh, pd.DataFrame) and not wh.empty:
        for _, r in wh.iterrows():
            wloc[str(r.get("Warehouse"))] = str(r.get("Location") or r.get("Warehouse"))

    rows = []
    for f in flows or []:
        w = str(f.get("warehouse") or "")
        c = str(f.get("customer") or "")
        p = f.get("product")
        q = float(f.get("qty", 0) or 0)
        from_loc = wloc.get(w, w)
        to_loc = c
        lat1, lon1 = resolve_latlon(from_loc, dfs)
        lat2, lon2 = resolve_latlon(to_loc, dfs)
        if None in (lat1, lon1, lat2, lon2):
            continue
        rows.append({
            "from": w, "to": c, "product": p, "qty": q,
            "from_lat": float(lat1), "from_lon": float(lon1),
            "to_lat": float(lat2), "to_lon": float(lon2),
        })
    return pd.DataFrame(rows)

def guess_map_center(nodes: pd.DataFrame, default: Tuple[float,float]=(30.0, 15.0)) -> Tuple[float,float]:
    """Return (lat, lon) average for view."""
    if not isinstance(nodes, pd.DataFrame) or nodes.empty:
        return default
    lat = pd.to_numeric(nodes["lat"], errors="coerce").dropna()
    lon = pd.to_numeric(nodes["lon"], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return default
    return (float(lat.mean()), float(lon.mean()))

# --------------------- Assistance for CRUD: add/update Locations ---------------------

def ensure_location_row(dfs: Dict[str, pd.DataFrame], location: str, country: str, lat: float, lon: float) -> Dict[str, pd.DataFrame]:
    """
    Insert or update a row in the Locations sheet with given coordinates.
    """
    locs = dfs.get("Locations")
    row = {"Location": str(location), "Country": str(country or ""), "Latitude": float(lat), "Longitude": float(lon)}
    if not isinstance(locs, pd.DataFrame) or locs.empty:
        dfs["Locations"] = pd.DataFrame([row], columns=["Location","Country","Latitude","Longitude"])
        return dfs
    df = locs.copy()
    mask = df["Location"].astype(str) == str(location)
    if mask.any():
        df.loc[mask, ["Country","Latitude","Longitude"]] = [row["Country"], row["Latitude"], row["Longitude"]]
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    dfs["Locations"] = df
    return dfs
