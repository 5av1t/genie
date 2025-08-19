from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# ----------------------------
# Built-in fallback coordinates
# ----------------------------
_BASE_COORDS = {
    "Bucharest": (44.4268, 26.1025),
    "Paris": (48.8566, 2.3522),
    "Ankara": (39.9334, 32.8597),
    "Beirut": (33.8938, 35.5018),
    "Belgrade": (44.8125, 20.4612),
    "Berlin": (52.5200, 13.4050),
    "Bratislava": (48.1486, 17.1077),
    "Brussels": (50.8503, 4.3517),
    # Customers
    "Aalborg": (57.0488, 9.9217),
    "Aarhus": (56.1629, 10.2039),
    "Abu Dhabi": (24.4539, 54.3773),
    "Ad Damman": (26.4207, 50.0888),
    "Adana": (37.0, 35.3213),
    "Aden": (12.7855, 45.0187),
    "Agadir": (30.4278, -9.5981),
    "Al Ahmadi": (29.0760, 48.0836),
    "Al Aqabah": (29.5328, 35.0063),
    "Al Ayn": (24.1302, 55.8023),
    # Suppliers
    "Antalya": (36.8969, 30.7133),
    "Krems": (48.4102, 15.6148),
}

# Expand with suffix forms commonly used in sheets
_COORDS: Dict[str, Tuple[float, float]] = dict(_BASE_COORDS)
for city in ["Bucharest", "Paris", "Ankara", "Beirut", "Belgrade", "Berlin", "Bratislava", "Brussels"]:
    _COORDS[f"{city}_CDC"] = _BASE_COORDS[city]
    _COORDS[f"{city}_LDC"] = _BASE_COORDS[city]
for sup in ["Antalya", "Krems"]:
    _COORDS[f"{sup}_FG"] = _BASE_COORDS[sup]


# ----------------------------
# Helpers
# ----------------------------
def _alias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column aliases for Locations sheets.
    Accepts: Location, Latitude/lat, Longitude/lon
    """
    if df is None or df.empty:
        return df
    cols = {c.lower().strip(): c for c in df.columns}
    rename = {}
    # Location
    for cand in ["location", "site", "name"]:
        if cand in cols:
            rename[cols[cand]] = "Location"
            break
    # Latitude
    for cand in ["latitude", "lat", "y"]:
        if cand in cols:
            rename[cols[cand]] = "Latitude"
            break
    # Longitude
    for cand in ["longitude", "lon", "lng", "x"]:
        if cand in cols:
            rename[cols[cand]] = "Longitude"
            break
    if rename:
        df = df.rename(columns=rename)
    return df


def _external_coords(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
    """
    If a Locations-like sheet exists with columns (Location, Latitude, Longitude),
    prefer it over built-in fallbacks.
    """
    for name in ["Locations", "Location", "Geo", "Coordinates"]:
        df = dfs.get(name)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _alias(df)
            if {"Location", "Latitude", "Longitude"}.issubset(df.columns):
                out: Dict[str, Tuple[float, float]] = {}
                for _, r in df.iterrows():
                    loc = r.get("Location")
                    lat = r.get("Latitude")
                    lon = r.get("Longitude")
                    if pd.notna(loc) and pd.notna(lat) and pd.notna(lon):
                        try:
                            out[str(loc)] = (float(lat), float(lon))
                        except Exception:
                            continue
                if out:
                    return out
    return {}


def resolve_latlon(name: Optional[str], ext: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (lat, lon) if known, else (None, None).
    Tries: exact match in `ext`, exact match in built-in, token-before-underscore in either.
    """
    if not name:
        return (None, None)
    # Exact
    if ext and name in ext:
        lat, lon = ext[name]
        return (float(lat), float(lon))
    if name in _COORDS:
        lat, lon = _COORDS[name]
        return (float(lat), float(lon))
    # Token before underscore
    if "_" in name:
        base = name.split("_", 1)[0]
        if ext and base in ext:
            lat, lon = ext[base]
            return (float(lat), float(lon))
        if base in _COORDS:
            lat, lon = _COORDS[base]
            return (float(lat), float(lon))
    return (None, None)


# ----------------------------
# Public API
# ----------------------------
def build_nodes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construct a table of network nodes for mapping.

    Returns columns:
      - name: display name
      - type: "warehouse" | "customer"
      - location: the location token used for geocoding
      - lat, lon
    """
    rows: List[Dict[str, Any]] = []
    ext = _external_coords(dfs)

    # Warehouses
    wh = dfs.get("Warehouse")
    if isinstance(wh, pd.DataFrame) and not wh.empty:
        for _, r in wh.iterrows():
            w = str(r.get("Warehouse")) if pd.notna(r.get("Warehouse")) else None
            if not w:
                continue
            loc = r.get("Location")
            loc_s = str(loc) if pd.notna(loc) else w  # fall back to name
            lat, lon = resolve_latlon(loc_s, ext)
            if lat is not None and lon is not None:
                rows.append({"name": w, "type": "warehouse", "location": loc_s, "lat": lat, "lon": lon})

    # Customers from Customers sheet (preferred)
    custs = dfs.get("Customers")
    if isinstance(custs, pd.DataFrame) and not custs.empty:
        for _, r in custs.iterrows():
            c = r.get("Customer")
            if pd.isna(c):
                continue
            c_s = str(c)
            loc = r.get("Location")
            loc_s = str(loc) if pd.notna(loc) else c_s
            lat, lon = resolve_latlon(loc_s, ext)
            if lat is not None and lon is not None:
                rows.append({"name": c_s, "type": "customer", "location": loc_s, "lat": lat, "lon": lon})

    # Fallback: derive customers from CPD if Customers sheet missing/empty
    if not isinstance(custs, pd.DataFrame) or custs.empty:
        cpd = dfs.get("Customer Product Data")
        if isinstance(cpd, pd.DataFrame) and not cpd.empty:
            # Prefer CPD.Location if present, else Customer name
            for _, r in cpd.iterrows():
                cust_name = r.get("Customer")
                if pd.isna(cust_name):
                    continue
                cust_name = str(cust_name)
                loc = r.get("Location")
                loc_s = str(loc) if pd.notna(loc) else cust_name
                lat, lon = resolve_latlon(loc_s, ext)
                if lat is not None and lon is not None:
                    rows.append({"name": cust_name, "type": "customer", "location": loc_s, "lat": lat, "lon": lon})

    # Deduplicate (keep first)
    if rows:
        df = pd.DataFrame(rows)
        return df.drop_duplicates(subset=["name", "type"], keep="first", ignore_index=True)
    return pd.DataFrame(columns=["name", "type", "location", "lat", "lon"])


def flows_to_geo(flows: List[Dict[str, Any]], dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert optimizer flow list into geocoded arcs with lat/lon for from/to.

    `flows` items expected: {"warehouse": W, "customer": C, "product": P, "qty": q}

    Returns columns:
      - from, to, product, qty
      - from_lat, from_lon, to_lat, to_lon
    """
    if not flows:
        return pd.DataFrame(columns=["from", "to", "product", "qty", "from_lat", "from_lon", "to_lat", "to_lon"])

    # Map warehouse -> location token
    wloc: Dict[str, str] = {}
    wh = dfs.get("Warehouse")
    if isinstance(wh, pd.DataFrame) and not wh.empty:
        for _, r in wh.iterrows():
            w = r.get("Warehouse")
            if pd.isna(w):
                continue
            w_s = str(w)
            loc = r.get("Location")
            wloc[w_s] = str(loc) if pd.notna(loc) else w_s

    ext = _external_coords(dfs)

    rows: List[Dict[str, Any]] = []
    for f in flows:
        w = f.get("warehouse")
        c = f.get("customer")
        p = f.get("product")
        q = f.get("qty", 0)

        if w is None or c is None:
            continue

        from_loc = wloc.get(str(w), str(w))
        to_loc = str(c)

        lat1, lon1 = resolve_latlon(from_loc, ext)
        lat2, lon2 = resolve_latlon(to_loc, ext)

        if None in (lat1, lon1, lat2, lon2):
            # Skip arcs that we can't place on the map
            continue

        try:
            qf = float(q)
        except Exception:
            qf = 0.0

        rows.append({
            "from": str(w),
            "to": str(c),
            "product": str(p) if p is not None else None,
            "qty": qf,
            "from_lat": lat1,
            "from_lon": lon1,
            "to_lat": lat2,
            "to_lon": lon2,
        })

    if not rows:
        return pd.DataFrame(columns=["from", "to", "product", "qty", "from_lat", "from_lon", "to_lat", "to_lon"])
    return pd.DataFrame(rows)


def guess_map_center(nodes: pd.DataFrame) -> Tuple[float, float]:
    """
    Simple mean center for convenience in callers.
    """
    if isinstance(nodes, pd.DataFrame) and not nodes.empty and {"lat", "lon"}.issubset(nodes.columns):
        return float(nodes["lat"].mean()), float(nodes["lon"].mean())
    # Default to Europe-ish if nothing known
    return (48.0, 14.0)
