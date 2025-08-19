from typing import Dict, Any, List, Tuple
import pandas as pd

# Minimal offline geocoder for validated examples + external sheet support
_COORDS = {
    "Bucharest": (44.4268, 26.1025), "Paris": (48.8566, 2.3522), "Ankara": (39.9334, 32.8597),
    "Beirut": (33.8938, 35.5018), "Belgrade": (44.8125, 20.4612), "Berlin": (52.5200, 13.4050),
    "Bratislava": (48.1486, 17.1077), "Brussels": (50.8503, 4.3517),
    "Aalborg": (57.0488, 9.9217), "Aarhus": (56.1629, 10.2039), "Abu Dhabi": (24.4539, 54.3773),
    "Ad Damman": (26.4207, 50.0888), "Adana": (37.0, 35.3213), "Aden": (12.7855, 45.0187),
    "Agadir": (30.4278, -9.5981), "Al Ahmadi": (29.076, 48.0836), "Al Aqabah": (29.5328, 35.0063),
    "Al Ayn": (24.1302, 55.8023), "Antalya": (36.8969, 30.7133), "Krems": (48.4102, 15.6148),
}
for city in ["Bucharest","Paris","Ankara","Beirut","Belgrade","Berlin","Bratislava","Brussels"]:
    _COORDS[f"{city}_CDC"] = _COORDS[city]; _COORDS[f"{city}_LDC"] = _COORDS[city]
for sup in ["Antalya","Krems"]:
    _COORDS[f"{sup}_FG"] = _COORDS[sup]

def _external_coords(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
    for name in ["Locations", "Coordinates", "Geo"]:
        df = dfs.get(name)
        if isinstance(df, pd.DataFrame) and {"Location", "Latitude", "Longitude"}.issubset(df.columns):
            out = {}
            for _, r in df.iterrows():
                loc = str(r["Location"]) if pd.notna(r.get("Location")) else None
                lat = r.get("Latitude"); lon = r.get("Longitude")
                if loc and pd.notna(lat) and pd.notna(lon):
                    out[loc] = (float(lat), float(lon))
            return out
    return {}

def resolve_latlon(name: str, ext: Dict[str, Tuple[float, float]] = None) -> Tuple[float, float]:
    if not name:
        return (None, None)
    if ext and name in ext:
        lat, lon = ext[name]; return (float(lat), float(lon))
    if name in _COORDS:
        lat, lon = _COORDS[name]; return (float(lat), float(lon))
    if "_" in name:
        base = name.split("_")[0]
        if ext and base in ext:
            lat, lon = ext[base]; return (float(lat), float(lon))
        if base in _COORDS:
            lat, lon = _COORDS[base]; return (float(lat), float(lon))
    return (None, None)

def build_nodes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ext = _external_coords(dfs)
    wh = dfs.get("Warehouse")
    if isinstance(wh, pd.DataFrame):
        for _, r in wh.iterrows():
            w = str(r.get("Warehouse"))
            loc = str(r.get("Location")) if pd.notna(r.get("Location")) else w
            lat, lon = resolve_latlon(loc if loc else w, ext)
            if lat is not None:
                rows.append({"name": w, "type": "warehouse", "location": loc, "lat": lat, "lon": lon})
    custs = dfs.get("Customers")
    if isinstance(custs, pd.DataFrame):
        for _, r in custs.iterrows():
            c = str(r.get("Customer"))
            loc = str(r.get("Location", c)) if pd.notna(r.get("Location", c)) else c
            lat, lon = resolve_latlon(loc, ext)
            if lat is not None:
                rows.append({"name": c, "type": "customer", "location": loc, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)

def flows_to_geo(flows, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    wh = dfs.get("Warehouse")
    wloc = {}
    if isinstance(wh, pd.DataFrame):
        for _, r in wh.iterrows():
            wloc[str(r.get("Warehouse"))] = str(r.get("Location"))
    ext = _external_coords(dfs)
    rows = []
    for f in flows:
        w = f["warehouse"]; c = f["customer"]; p = f.get("product"); q = float(f.get("qty", 0))
        from_loc = wloc.get(w, w); to_loc = c
        lat1, lon1 = resolve_latlon(from_loc, ext); lat2, lon2 = resolve_latlon(to_loc, ext)
        if None in (lat1, lon1, lat2, lon2):
            continue
        rows.append({"from": w, "to": c, "product": p, "qty": q, "from_lat": lat1, "from_lon": lon1, "to_lat": lat2, "to_lon": lon2})
    return pd.DataFrame(rows)
