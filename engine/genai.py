# engine/genai.py
# LLM helpers: parse scenarios and answer questions, both grounded.
# Falls back gracefully if API key/provider unavailable.

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import math
import pandas as pd
import numpy as np

DEFAULT_PERIOD = 2023

# Optional providers
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

def _safe_df(df) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

# ------------------------- Parsing with LLM -------------------------

def parse_with_llm(prompt: str, dfs: Dict[str, pd.DataFrame], default_period: int = DEFAULT_PERIOD, provider: str = "none") -> Dict[str, Any]:
    """
    Use Gemini/OpenAI (Gemini implemented) to parse arbitrary text into our scenario JSON.
    If provider is missing/unavailable, we return {} (no edits) to avoid hallucination.
    """
    if not prompt or not isinstance(prompt, str):
        return {}

    # Light guard: if provider isn't configured, bail out to avoid random behavior
    if provider != "gemini":
        return {}

    if genai is None or not os.environ.get("GEMINI_API_KEY"):
        return {}

    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Provide entities so model doesn't invent names
        ents = _entities_from_dfs(dfs)
        sys = (
            "You are a strict function-filler that converts the user's natural language into a scenario JSON.\n"
            "Only use entity names provided. Do not invent.\n"
            "If a statement is ambiguous or references unknown entities, skip that part.\n"
            "Schema:\n"
            "{\n"
            '  "period": 2023,\n'
            '  "demand_updates": [{"product": "...","customer": "...","location": "...","delta_pct": 10,"set": {"Lead Time": 8}}],\n'
            '  "warehouse_changes": [{"warehouse": "...","field": "Maximum Capacity","new_value": 25000}],\n'
            '  "supplier_changes": [{"product": "...","supplier": "...","location": "...","field": "Available","new_value": 1}],\n'
            '  "transport_updates": [{"mode":"...","product":"...","from_location":"...","to_location":"...","period":2023,"fields":{"Cost Per UOM":9.5,"Available":1}}]\n'
            "}\n"
            "Default period is 2023 unless specified."
        )
        context = {
            "entities": ents,
            "default_period": default_period
        }
        msg = f"{sys}\n\nEntities:\n{json.dumps(context, ensure_ascii=False)}\n\nUser:\n{prompt}\n\nReturn ONLY JSON."
        resp = model.generate_content(msg)
        txt = resp.text or ""
        # Attempt to parse JSON substring
        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            js = json.loads(txt[start:end+1])
            # sanitize keys present
            out = {
                "period": int(js.get("period", default_period)),
                "demand_updates": js.get("demand_updates", []),
                "warehouse_changes": js.get("warehouse_changes", []),
                "supplier_changes": js.get("supplier_changes", []),
                "transport_updates": js.get("transport_updates", []),
            }
            return out
        return {}
    except Exception:
        # Fail closed (no edits) rather than guessing
        return {}

def _entities_from_dfs(dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    ents = {
        "products": [],
        "warehouses": [],
        "customers": [],
        "suppliers": [],
        "modes": [],
        "locations": [],
    }
    prod = _safe_df(dfs.get("Products"))
    if "Product" in prod.columns:
        ents["products"] = sorted(list(set(prod["Product"].dropna().astype(str).tolist())))
    wh = _safe_df(dfs.get("Warehouse"))
    if "Warehouse" in wh.columns:
        ents["warehouses"] = sorted(list(set(wh["Warehouse"].dropna().astype(str).tolist())))
    if "Location" in wh.columns:
        ents["locations"].extend(wh["Location"].dropna().astype(str).tolist())
    cu = _safe_df(dfs.get("Customers"))
    if "Customer" in cu.columns:
        ents["customers"] = sorted(list(set(cu["Customer"].dropna().astype(str).tolist())))
    if "Location" in cu.columns:
        ents["locations"].extend(cu["Location"].dropna().astype(str).tolist())
    sp = _safe_df(dfs.get("Supplier Product"))
    if "Supplier" in sp.columns:
        ents["suppliers"] = sorted(list(set(sp["Supplier"].dropna().astype(str).tolist())))
    if "Location" in sp.columns:
        ents["locations"].extend(sp["Location"].dropna().astype(str).tolist())
    mot = _safe_df(dfs.get("Mode of Transport"))
    if "Mode of Transport" in mot.columns:
        ents["modes"] = sorted(list(set(mot["Mode of Transport"].dropna().astype(str).tolist())))
    loc = _safe_df(dfs.get("Locations"))
    if "Location" in loc.columns:
        ents["locations"].extend(loc["Location"].dropna().astype(str).tolist())
    ents["locations"] = sorted(list(set([x for x in ents["locations"] if x])))
    return ents

# ------------------------- Q&A -------------------------

def answer_question(q: str,
                    kpis: Dict[str, Any],
                    diag: Dict[str, Any],
                    provider: str = "none",
                    dfs: Optional[Dict[str, pd.DataFrame]] = None,
                    model_index: Optional[Dict[str, Any]] = None,
                    force_llm: bool = False) -> str:
    """
    Grounded Q&A: compute deterministic answers for common analytics,
    optionally let Gemini paraphrase/augment if configured.
    """
    if not q or not isinstance(q, str):
        return "Please type a question."

    # Deterministic analytics
    base = _deterministic_answer(q, kpis, diag, dfs, model_index)

    # If Gemini set and the user chose Gemini, ask it to explain using our computed base + context
    if provider == "gemini" and genai is not None and os.environ.get("GEMINI_API_KEY"):
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-1.5-flash")
            context = {
                "kpis": kpis,
                "flows": diag.get("flows", []),
                "customers": (model_index or {}).get("customers", {}),
                "warehouses": (model_index or {}).get("warehouses", {}),
                "lanes": (model_index or {}).get("lanes", []),
            }
            prompt = (
                "You are a supply chain analyst. Answer using ONLY the provided context.\n"
                "If the question asks something not computable from the context, say what you can do instead.\n"
                f"QUESTION: {q}\n"
                f"BASE_FINDINGS: {json.dumps(base, ensure_ascii=False)}\n"
                f"CONTEXT: {json.dumps(context, ensure_ascii=False)[:18000]}\n"
                "Return a short, direct answer for an executive user."
            )
            resp = model.generate_content(prompt)
            txt = (resp.text or "").strip()
            if txt:
                return txt
        except Exception:
            pass

    # Fallback to deterministic answer text
    return base.get("answer_text", "Sorry, I couldn't compute that.")

def _deterministic_answer(q: str,
                          kpis: Dict[str, Any],
                          diag: Dict[str, Any],
                          dfs: Optional[Dict[str, pd.DataFrame]],
                          model_index: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ql = q.lower()
    out: Dict[str, Any] = {}

    # Served vs total
    if "service" in ql or "served" in ql:
        out["answer_text"] = f"Service: {kpis.get('service_pct', 0)}% ({kpis.get('served', 0)} of {kpis.get('total_demand', 0)})."
        return out

    # Least customer demand
    if "least" in ql and "demand" in ql and "customer" in ql:
        cpd = _safe_df((dfs or {}).get("Customer Product Data"))
        if not cpd.empty and "Demand" in cpd.columns and "Customer" in cpd.columns:
            agg = cpd.groupby("Customer", dropna=True)["Demand"].sum(numeric_only=True)
            if not agg.empty:
                cmin = agg.idxmin()
                out["answer_text"] = f"Lowest total demand: {cmin} ({int(agg.loc[cmin])})."
                return out
        out["answer_text"] = "I couldn't find demand data."
        return out

    # Unserved customers
    if "unserved" in ql or ("not" in ql and "served" in ql):
        flows = diag.get("flows", [])
        served_by_c = {}
        for f in flows:
            served_by_c[f["customer"]] = served_by_c.get(f["customer"], 0.0) + float(f.get("qty", 0.0) or 0.0)
        cpd = _safe_df((dfs or {}).get("Customer Product Data"))
        if not cpd.empty and set(["Customer","Demand"]).issubset(cpd.columns):
            dem = cpd.groupby("Customer", dropna=True)["Demand"].sum(numeric_only=True)
            unserved = [c for c, d in dem.items() if float(served_by_c.get(c, 0.0)) + 1e-6 < float(d)]
            if unserved:
                out["answer_text"] = f"Unserved customers: {', '.join(unserved[:10])}" + (" ..." if len(unserved) > 10 else "")
            else:
                out["answer_text"] = "All customers appear fully served."
            return out
        out["answer_text"] = "I couldn't compute unserved customers (missing CPD or flows)."
        return out

    # Why no supplier in <country>?
    if "supplier" in ql and ("why" in ql or "no" in ql) and ("india" in ql or "country" in ql):
        suppliers = (model_index or {}).get("suppliers", {})
        locs = (model_index or {}).get("locations", {})
        countries = {}
        for s, info in suppliers.items():
            loc = info.get("location")
            country = (locs.get(loc, {}) or {}).get("country")
            countries.setdefault(country or "(unknown)", 0)
            countries[country or "(unknown)"] += 1
        cnt = countries.get("India", 0) or countries.get("INDIA", 0)
        if cnt == 0:
            out["answer_text"] = "There are no suppliers mapped to India in the current model. You can add a supplier in India via the Manual Edits tab or an LLM scenario ('Enable <Product> at <Supplier>')."
        else:
            out["answer_text"] = f"There are {cnt} supplier entries mapped to India."
        return out

    # Default
    out["answer_text"] = "Here's what I know: " \
                         f"Status={kpis.get('status','n/a')}, Service={kpis.get('service_pct',0)}%, " \
                         f"Open WH={kpis.get('open_warehouses',0)}. Ask about unserved customers, least demand, or suppliers by country."
    return out

# ------------------------- Location Candidates -------------------------

def suggest_location_candidates(query: str,
                                dfs: Dict[str, pd.DataFrame],
                                provider: str = "none",
                                allow_online: bool = False) -> List[Dict[str, Any]]:
    """
    Deterministic mapping:
      1) Check 'Locations' sheet
      2) Try base city tokens
      3) Optionally, online geocoding (Nominatim)
    Returns list of candidates sorted best-first:
      [{"location": "...", "country": "...", "lat": 0.0, "lon": 0.0}]
    """
    if not query:
        return []

    # 1) Exact in Locations
    loc = _safe_df(dfs.get("Locations"))
    hits: List[Dict[str, Any]] = []
    if not loc.empty and set(["Location","Latitude","Longitude"]).issubset(loc.columns):
        for _, r in loc.iterrows():
            name = str(r.get("Location",""))
            if not name:
                continue
            if name.lower() in query.lower():
                lat = r.get("Latitude"); lon = r.get("Longitude")
                if pd.notna(lat) and pd.notna(lon):
                    hits.append({
                        "location": name,
                        "country": str(r.get("Country","")),
                        "lat": float(lat),
                        "lon": float(lon),
                    })
    if hits:
        return hits

    # 2) Try known base tokens from geo.py by importing resolve (no net calls)
    try:
        from .geo import resolve_latlon
        lat, lon = resolve_latlon(query, None)
        if lat is not None and lon is not None:
            return [{"location": query, "country": "", "lat": lat, "lon": lon}]
    except Exception:
        pass

    # 3) Optional: online geocoding
    if allow_online:
        try:
            import requests  # type: ignore
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "genie-app"},
                timeout=10,
            )
            if r.status_code == 200:
                js = r.json()
                if js:
                    lat = float(js[0]["lat"]); lon = float(js[0]["lon"])
                    disp = js[0].get("display_name","")
                    return [{"location": query, "country": disp, "lat": lat, "lon": lon}]
        except Exception:
            pass

    return []
