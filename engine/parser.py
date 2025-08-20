# engine/parser.py
# Deterministic rules parser that converts plain English statements into
# a schema-constrained scenario JSON for GENIE.
#
# Supported intents (case-insensitive; multiple statements separated by ';' or newlines):
# - Increase/Decrease <Product> demand at <Customer> by <pct>% [for/in <year>]
#   [and set Lead Time to <days>]
# - Cap <Warehouse> Maximum Capacity at <value>
# - Force close <Warehouse>
# - Force open <Warehouse>
# - Enable <Product> at <Supplier> [for/in <year>]
# - Set <Mode of Transport> lane <From> -> <To> for <Product> to Cost Per UOM = <value> [in <year>]
#
# Returns scenario dict:
# {
#   "period": 2023,
#   "demand_updates": [...],
#   "warehouse_changes": [...],
#   "supplier_changes": [...],
#   "transport_updates": [...],
#   "_notes": ["... skipped due to ...", ...]
# }
#
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import re
import pandas as pd

DEFAULT_PERIOD = 2023

# -------------------------- Utilities --------------------------

def _safe_df(df) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _collect_entities(dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """Collect canonical entity names from the workbook for exact matching."""
    ent: Dict[str, List[str]] = {
        "products": [],
        "warehouses": [],
        "customers": [],
        "suppliers": [],
        "modes": [],
        "locations": [],
    }
    # Products
    prod = _safe_df(dfs.get("Products"))
    if "Product" in prod.columns:
        ent["products"] = sorted(list({str(x) for x in prod["Product"].dropna().astype(str)}))

    # Warehouses & their locations
    wh = _safe_df(dfs.get("Warehouse"))
    if "Warehouse" in wh.columns:
        ent["warehouses"] = sorted(list({str(x) for x in wh["Warehouse"].dropna().astype(str)}))
    if "Location" in wh.columns:
        ent["locations"].extend([str(x) for x in wh["Location"].dropna().astype(str)])

    # Customers & their locations
    cu = _safe_df(dfs.get("Customers"))
    if "Customer" in cu.columns:
        ent["customers"] = sorted(list({str(x) for x in cu["Customer"].dropna().astype(str)}))
    if "Location" in cu.columns:
        ent["locations"].extend([str(x) for x in cu["Location"].dropna().astype(str)])

    # Locations sheet (optional)
    loc = _safe_df(dfs.get("Locations"))
    if "Location" in loc.columns:
        ent["locations"].extend([str(x) for x in loc["Location"].dropna().astype(str)])

    # Suppliers (from Supplier Product)
    sp = _safe_df(dfs.get("Supplier Product"))
    if "Supplier" in sp.columns:
        ent["suppliers"] = sorted(list({str(x) for x in sp["Supplier"].dropna().astype(str)}))
    if "Location" in sp.columns:
        ent["locations"].extend([str(x) for x in sp["Location"].dropna().astype(str)])

    # Modes of Transport
    mot = _safe_df(dfs.get("Mode of Transport"))
    if "Mode of Transport" in mot.columns:
        ent["modes"] = sorted(list({str(x) for x in mot["Mode of Transport"].dropna().astype(str)}))

    # De-duplicate locations
    ent["locations"] = sorted(list({x for x in ent["locations"] if x}))

    return ent

def _find_entity(text: str, candidates: List[str]) -> Optional[str]:
    """
    Match any candidate by case-insensitive exact substring, preferring the longest match.
    We avoid fuzzy guesses to prevent hallucinations.
    """
    if not text or not candidates:
        return None
    t = text.lower()
    hits: List[Tuple[int, str]] = []
    for c in candidates:
        cl = c.lower()
        if cl in t:
            hits.append((len(c), c))
    if not hits:
        return None
    # prefer longest name (avoids partials like 'Paris' matching 'Paris_CDC' unexpectedly)
    hits.sort(key=lambda x: (-x[0], x[1]))
    return hits[0][1]

def _extract_year(text: str, default_year: int) -> int:
    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return default_year
    return default_year

def _extract_number(text: str) -> Optional[float]:
    m = re.search(r"(-?\d+(?:\.\d+)?)", text.replace(",", ""))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def _split_statements(text: str) -> List[str]:
    # Split on semicolons or newlines
    parts = re.split(r"[;\n]+", text)
    return [p.strip() for p in parts if p.strip()]

# -------------------------- Intent Parsers --------------------------

def _parse_demand(stmt: str, ent: Dict[str, List[str]], default_period: int) -> Optional[Dict[str, Any]]:
    """
    Patterns:
      Increase <Product> demand at <Customer> by <pct>% [for 2023] [and set Lead Time to <days>]
      Decrease <Product> demand at <Customer> by <pct>%
    """
    if not re.search(r"\bdemand\b", stmt, flags=re.I):
        return None
    if not re.search(r"\b(increase|decrease|reduce|raise)\b", stmt, flags=re.I):
        return None

    product = _find_entity(stmt, ent["products"])
    customer = _find_entity(stmt, ent["customers"])
    if not product or not customer:
        return None

    # location for customer: default to customer name unless CPD uses a separate Location
    location = customer  # safe default; updater can map to CPD row by (Product, Customer, Location, Period)
    # If uploaded Customers sheet has a distinct location, use that
    # (We don't have direct access here; updater will handle creation if missing)

    # pct
    pct = _extract_number(stmt)
    if pct is None:
        return None
    if re.search(r"\b(decrease|reduce)\b", stmt, flags=re.I):
        pct = -abs(pct)
    else:
        pct = abs(pct)

    # year
    period = _extract_year(stmt, default_period)

    # optional lead time set
    lt = None
    mlt = re.search(r"set\s+lead\s*time\s+to\s+(-?\d+)", stmt, flags=re.I)
    if mlt:
        try:
            lt = int(mlt.group(1))
        except Exception:
            lt = None

    upd: Dict[str, Any] = {
        "product": product,
        "customer": customer,
        "location": location,
        "delta_pct": pct
    }
    if lt is not None:
        upd["set"] = {"Lead Time": lt}

    return {"period": period, "demand_updates": [upd]}

def _parse_warehouse(stmt: str, ent: Dict[str, List[str]], default_period: int) -> Optional[Dict[str, Any]]:
    """
    Patterns:
      Cap <Warehouse> Maximum Capacity at <value>
      Force close <Warehouse>
      Force open <Warehouse>
    """
    wh = _find_entity(stmt, ent["warehouses"])
    if not wh:
        return None

    # Cap capacity
    if re.search(r"\bcap\b.*\bmaximum\s+capacity\b", stmt, flags=re.I) or re.search(r"\bmaximum\s+capacity\b.*\bto|at\b", stmt, flags=re.I):
        val = _extract_number(stmt)
        if val is None:
            return None
        return {"warehouse_changes": [{"warehouse": wh, "field": "Maximum Capacity", "new_value": float(val)}]}

    # Force close
    if re.search(r"\bforce\s+close\b", stmt, flags=re.I):
        return {"warehouse_changes": [{"warehouse": wh, "field": "Force Close", "new_value": 1},
                                      {"warehouse": wh, "field": "Force Open", "new_value": 0}]}

    # Force open
    if re.search(r"\bforce\s+open\b", stmt, flags=re.I):
        return {"warehouse_changes": [{"warehouse": wh, "field": "Force Open", "new_value": 1},
                                      {"warehouse": wh, "field": "Force Close", "new_value": 0}]}

    return None

def _parse_supplier(stmt: str, ent: Dict[str, List[str]], default_period: int) -> Optional[Dict[str, Any]]:
    """
    Pattern:
      Enable <Product> at <Supplier> [for 2023]
    """
    if not re.search(r"\benable\b", stmt, flags=re.I):
        return None
    product = _find_entity(stmt, ent["products"])
    supplier = _find_entity(stmt, ent["suppliers"])
    if not product or not supplier:
        return None
    period = _extract_year(stmt, default_period)
    # Default location to supplier token (common in your files: Antalya_FG)
    location = supplier
    return {
        "period": period,
        "supplier_changes": [
            {"product": product, "supplier": supplier, "location": location, "field": "Available", "new_value": 1}
        ]
    }

def _parse_transport(stmt: str, ent: Dict[str, List[str]], default_period: int) -> Optional[Dict[str, Any]]:
    """
    Pattern:
      Set <Mode> lane <From> -> <To> for <Product> to Cost Per UOM = <value> [in <year>]
    Accept variants like '→', 'to', 'cpu', 'cost per uom', 'cost/uom'.
    """
    if not re.search(r"\bset\b.*\blane\b", stmt, flags=re.I):
        return None

    mode = _find_entity(stmt, ent["modes"])
    product = _find_entity(stmt, ent["products"])
    # From/To locations or warehouse/customer names (both become locations)
    # Try to find a 'from' candidate among warehouses/locations first
    from_loc = None
    to_loc = None

    # Detect arrow or 'to' direction
    arrow = re.search(r"lane\s+(.+?)\s*(?:->|→|to)\s*(.+?)\s+(?:for\b|cost\b|cpu\b|with\b)", stmt, flags=re.I)
    if arrow:
        left = arrow.group(1).strip()
        right = arrow.group(2).strip()
        # choose best matches
        from_loc = _find_entity(left, ent["warehouses"]) or _find_entity(left, ent["locations"]) or _find_entity(left, ent["customers"])
        to_loc   = _find_entity(right, ent["customers"]) or _find_entity(right, ent["locations"]) or _find_entity(right, ent["warehouses"])

    if not (mode and product and from_loc and to_loc):
        return None

    # CPU extraction
    m_cost = re.search(r"(?:cost\s*per\s*uom|cost\/uom|cpu)\s*(?:=|to)?\s*(-?\d+(?:\.\d+)?)", stmt, flags=re.I)
    if not m_cost:
        return None
    try:
        cpu = float(m_cost.group(1))
    except Exception:
        return None

    period = _extract_year(stmt, default_period)
    return {
        "transport_updates": [{
            "mode": mode,
            "product": product,
            "from_location": from_loc,
            "to_location": to_loc,
            "period": period,
            "fields": {"Cost Per UOM": cpu, "Available": 1}
        }]
    }

# -------------------------- Merge & Public API --------------------------

def _merge_scenarios(base: Dict[str, Any], add: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two scenario dicts safely."""
    out = dict(base)
    # period: keep earliest explicit mention or default
    if "period" in add:
        out["period"] = add.get("period", out.get("period", DEFAULT_PERIOD))

    for key in ("demand_updates", "warehouse_changes", "supplier_changes", "transport_updates"):
        if add.get(key):
            out.setdefault(key, [])
            out[key].extend(add[key])

    # propagate notes
    if add.get("_notes"):
        out.setdefault("_notes", [])
        out["_notes"].extend(add["_notes"])

    return out

def parse_rules(text: str, dfs: Dict[str, pd.DataFrame], default_period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    """
    Parse free text into a scenario JSON using deterministic rules and entities from dfs.
    Skips statements that don't match or reference unknown entities; records reasons in _notes.
    """
    scenario: Dict[str, Any] = {"period": int(default_period)}
    notes: List[str] = []

    if not text or not isinstance(text, str):
        scenario["_notes"] = ["Empty prompt; no edits."]
        return scenario

    ent = _collect_entities(dfs)
    statements = _split_statements(text)

    for raw in statements:
        stmt = raw.strip()
        if not stmt:
            continue
        matched = False

        # Try each parser in priority order
        for fn in (_parse_demand, _parse_warehouse, _parse_supplier, _parse_transport):
            try:
                out = fn(stmt, ent, default_period)
            except Exception:
                out = None
            if out:
                scenario = _merge_scenarios(scenario, out)
                matched = True
                break

        if not matched:
            notes.append(f"Skipped: '{stmt}' (no matching rule or unknown entities).")

    if notes:
        scenario["_notes"] = notes

    # If nothing parsed, keep empty lists for consistency
    for key in ("demand_updates", "warehouse_changes", "supplier_changes", "transport_updates"):
        scenario.setdefault(key, [])

    return scenario
