from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os
import pandas as pd

# Optional: only needed if you enable online geocoding fallback in suggest_location_candidates
try:
    import requests  # Streamlit Cloud generally allows outbound; if blocked, we handle gracefully
except Exception:  # keep module import-safe
    requests = None  # type: ignore

# Reuse your deterministic parser to avoid hallucinations for now
from engine.parser import parse_rules, DEFAULT_PERIOD
# Use geo helpers for offline coord resolution
from engine import geo as geo_mod


# -----------------------------------------------------------------------------
# LLM parsing shim: keep deterministic by routing to rules (safe & grounded)
# -----------------------------------------------------------------------------
def parse_with_llm(
    text: str,
    dfs: Dict[str, pd.DataFrame],
    default_period: int = DEFAULT_PERIOD,
    provider: str = "none",
) -> Dict[str, Any]:
    """
    For stability, we map NL -> scenario JSON using the deterministic rules parser.
    Later, you can replace this with real Gemini parsing, but ALWAYS post-validate
    entities against what's present in `dfs` (products/customers/warehouses/modes).
    """
    return parse_rules(text, dfs, default_period=default_period)


# -----------------------------------------------------------------------------
# Q&A: grounded summaries over the loaded model and solver diagnostics
# -----------------------------------------------------------------------------
def _unserved_pairs(diag: Dict[str, Any], dfs: Dict[str, pd.DataFrame]) -> List[str]:
    flows = diag.get("flows", [])
    served = {(f.get("customer"), f.get("product")) for f in flows}
    cpd = dfs.get("Customer Product Data", pd.DataFrame()).copy()
    if cpd.empty:
        return []
    cpd["Demand"] = pd.to_numeric(cpd.get("Demand", 0), errors="coerce").fillna(0.0)
    cpd = cpd[cpd["Demand"] > 0]
    out = []
    for _, r in cpd.iterrows():
        key = (str(r.get("Customer")), str(r.get("Product")))
        if key not in served:
            out.append(f"{key[0]} ({key[1]})")
    return sorted(set(out))


def answer_question(
    question: str,
    kpis: Dict[str, Any],
    diag: Dict[str, Any],
    provider: str = "none",
    dfs: Dict[str, pd.DataFrame] = None,
    model_index: Dict[str, Any] = None,
) -> str:
    """
    Lightweight, reliable answers based on your data (no hallucinations).
    Extend with Gemini later by passing compact structured context.
    """
    q = (question or "").strip().lower()
    if not q:
        return "Please ask a question about the model or solution."

    # Unserved demand pairs
    if "unserved" in q or "not served" in q:
        un = _unserved_pairs(diag, dfs or {})
        if un:
            return "Unserved customer–product pairs:\n- " + "\n- ".join(un)
        return "All demanded customer–product pairs appear served in the current solution."

    # Lowest demand customer (aggregated)
    if "least demand" in q or "lowest demand" in q:
        cpd = (dfs or {}).get("Customer Product Data", pd.DataFrame()).copy()
        if cpd.empty:
            return "I can't find demand data."
        cpd["Demand"] = pd.to_numeric(cpd.get("Demand", 0), errors="coerce").fillna(0.0)
        grp = cpd.groupby("Customer")["Demand"].sum().sort_values()
        if grp.empty:
            return "No demand rows found."
        c, d = grp.index[0], grp.iloc[0]
        return f"Lowest aggregated demand by customer: {c} = {int(d)} units."

    # KPI summary
    if "kpi" in q or "status" in q or "service" in q:
        return f"Status: {kpis.get('status')}, Service: {kpis.get('service_pct')}%, Total Cost: {kpis.get('total_cost')}."

    # Default help
    return (
        "I can answer grounded questions about unserved pairs, KPIs, and basic demand summaries. "
        "Try: 'Which customers are unserved?' or 'What is the total cost and service %?'"
    )


# -----------------------------------------------------------------------------
# Location suggestions for geocoding / upserting into Locations sheet
# -----------------------------------------------------------------------------
def _locations_sheet(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    for name in ["Locations", "locations", "Geo", "Coordinates"]:
        df = dfs.get(name)
        if isinstance(df, pd.DataFrame) and {"Location", "Latitude", "Longitude"}.issubset(set(df.columns)):
            return df.copy()
    return pd.DataFrame(columns=["Location", "Latitude", "Longitude"])


def _try_online_geocode(q: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Optional online lookup via Nominatim (OpenStreetMap). Only used if allow_online=True."""
    if requests is None:
        return []
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "json", "limit": str(limit)}
        headers = {"User-Agent": "genie-network-app/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if not r.ok:
            return []
        data = r.json()
        out = []
        for item in data:
            try:
                lat = float(item.get("lat"))
                lon = float(item.get("lon"))
                name = item.get("display_name") or q
                out.append({"name": name, "lat": lat, "lon": lon, "source": "online"})
            except Exception:
                continue
        return out
    except Exception:
        return []


def suggest_location_candidates(
    query: str,
    dfs: Dict[str, pd.DataFrame],
    provider: str = "none",
    allow_online: bool = False,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Returns up to `limit` coordinate candidates for `query`:
      1) Exact/contains matches in the Locations sheet
      2) Offline resolver (engine.geo.resolve_latlon incl. 'City_XYZ' prefix)
      3) Optional online Nominatim suggestions (if allow_online=True)
    """
    q = (query or "").strip()
    out: List[Dict[str, Any]] = []
    if not q:
        return out

    # 1) Locations sheet — exact, then substring
    locs = _locations_sheet(dfs)
    if not locs.empty:
        # exact
        m_exact = locs[locs["Location"].astype(str).str.lower() == q.lower()]
        for _, r in m_exact.head(limit).iterrows():
            try:
                out.append({"name": str(r["Location"]), "lat": float(r["Latitude"]), "lon": float(r["Longitude"]), "source": "locations"})
            except Exception:
                pass
        # contains
        if len(out) < limit:
            m_cont = locs[locs["Location"].astype(str).str.lower().str.contains(q.lower(), na=False)]
            for _, r in m_cont.head(limit - len(out)).iterrows():
                try:
                    cand = {"name": str(r["Location"]), "lat": float(r["Latitude"]), "lon": float(r["Longitude"]), "source": "locations"}
                    if cand not in out:
                        out.append(cand)
                except Exception:
                    pass

    # 2) Offline resolver (uses known cities and prefix-before-underscore)
    if len(out) < limit:
        lat, lon = geo_mod.resolve_latlon(q, ext=None)
        if lat is not None and lon is not None:
            out.append({"name": q, "lat": float(lat), "lon": float(lon), "source": "offline"})
        if "_" in q and len(out) < limit:
            base = q.split("_")[0]
            lat2, lon2 = geo_mod.resolve_latlon(base, ext=None)
            if lat2 is not None and lon2 is not None:
                cand = {"name": base, "lat": float(lat2), "lon": float(lon2), "source": "offline"}
                if cand not in out:
                    out.append(cand)

    # 3) Online (optional)
    if allow_online and len(out) < limit:
        out.extend(_try_online_geocode(q, limit=limit - len(out)))

    # Deduplicate
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in out:
        key = (c["name"], round(c["lat"], 6), round(c["lon"], 6))
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    return uniq[:limit]
