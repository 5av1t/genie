from __future__ import annotations
from typing import Dict, Any, List
import re
import pandas as pd

DEFAULT_PERIOD = 2023

def _entities(dfs):
    products = set(dfs.get("Products", pd.DataFrame()).get("Product", pd.Series(dtype=str)).dropna().astype(str))
    warehouses = set(dfs.get("Warehouse", pd.DataFrame()).get("Warehouse", pd.Series(dtype=str)).dropna().astype(str))
    customers = set(dfs.get("Customers", pd.DataFrame()).get("Customer", pd.Series(dtype=str)).dropna().astype(str))
    suppliers = set(dfs.get("Supplier Product", pd.DataFrame()).get("Supplier", pd.Series(dtype=str)).dropna().astype(str))
    locations = set(dfs.get("Supplier Product", pd.DataFrame()).get("Location", pd.Series(dtype=str)).dropna().astype(str))
    modes = set(dfs.get("Mode of Transport", pd.DataFrame()).get("Mode of Transport", pd.Series(dtype=str)).dropna().astype(str))
    return products, warehouses, customers, suppliers, locations, modes

def _find_in_text(candidates: List[str], text: str) -> List[str]:
    t = text.lower()
    out = []
    for c in candidates:
        if c and c.lower() in t:
            out.append(c)
    return out

def parse_rules(text: str, dfs, default_period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    scenario = {"period": default_period, "demand_updates": [], "warehouse_changes": [], "supplier_changes": [], "transport_updates": []}
    if not text or not isinstance(dfs, dict):
        return scenario

    products, warehouses, customers, suppliers, locations, modes = _entities(dfs)
    t = text.strip()

    # Demand change: Increase/Decrease X demand at Y by Z% (optional: set Lead Time to N)
    m = re.search(r"(increase|decrease)\s+(.+?)\s+demand\s+at\s+(.+?)\s+by\s+([0-9]+(?:\.[0-9]+)?)\s*%.*?(?:set\s+lead\s*time\s+to\s+([0-9]+))?", t, flags=re.I)
    if m:
        sign, prod_raw, cust_raw, pct, lead = m.groups()
        prod = next((p for p in products if p.lower() == prod_raw.lower()), None)
        cust = next((c for c in customers if c.lower() == cust_raw.lower()), None)
        if prod and cust:
            delta = float(pct)
            if sign.lower() == "decrease":
                delta = -delta
            upd = {"product": prod, "customer": cust, "location": cust, "delta_pct": delta}
            if lead:
                upd["set"] = {"Lead Time": int(lead)}
            scenario["demand_updates"].append(upd)

    # Cap warehouse maximum capacity at value
    for m in re.finditer(r"cap\s+(.+?)\s+maximum\s+capacity\s+at\s+([0-9]+)", t, flags=re.I):
        w_raw, cap = m.groups()
        w = next((w for w in warehouses if w.lower() == w_raw.lower()), None)
        if w:
            scenario["warehouse_changes"].append({"warehouse": w, "field": "Maximum Capacity", "new_value": int(cap)})

    # Force close/open
    for m in re.finditer(r"force\s+close\s+(.+?)\b", t, flags=re.I):
        w_raw = m.group(1)
        w = next((w for w in warehouses if w.lower() == w_raw.lower()), None)
        if w:
            scenario["warehouse_changes"].append({"warehouse": w, "field": "Force Close", "new_value": 1})
            scenario["warehouse_changes"].append({"warehouse": w, "field": "Force Open", "new_value": 0})
    for m in re.finditer(r"force\s+open\s+(.+?)\b", t, flags=re.I):
        w_raw = m.group(1)
        w = next((w for w in warehouses if w.lower() == w_raw.lower()), None)
        if w:
            scenario["warehouse_changes"].append({"warehouse": w, "field": "Force Open", "new_value": 1})
            scenario["warehouse_changes"].append({"warehouse": w, "field": "Force Close", "new_value": 0})

    # Enable product at supplier
    for m in re.finditer(r"enable\s+(.+?)\s+at\s+(.+?)\b", t, flags=re.I):
        prod_raw, sup_raw = m.groups()
        prod = next((p for p in products if p.lower() == prod_raw.lower()), None)
        sup = next((s for s in suppliers if s.lower() == sup_raw.lower()), None)
        if prod and sup:
            loc = sup if sup in locations else sup  # typical pattern Location==Supplier token
            scenario["supplier_changes"].append({"product": prod, "supplier": sup, "location": loc, "field": "Available", "new_value": 1})

    # Transport lane cost update
    m = re.search(r"set\s+(.+?)\s+lane\s+(.+?)\s*(?:->|→)\s*(.+?)\s+for\s+(.+?)\s+to\s+cost\s+per\s+uom\s*=\s*([0-9]+(?:\.[0-9]+)?)", t, flags=re.I)
    if m:
        mode_raw, from_raw, to_raw, prod_raw, cost = m.groups()
        mode = next((mm for mm in modes if mm.lower() == mode_raw.lower()), None)
        prod = next((p for p in products if p.lower() == prod_raw.lower()), None)
        from_loc = None
        to_loc = None
        # resolve from/to among warehouses locations & customer names/locations
        # we keep exact text as locations; updater will create if needed
        from_loc = from_raw
        to_loc = to_raw
        if mode and prod:
            scenario["transport_updates"].append({
                "mode": mode, "product": prod,
                "from_location": from_loc, "to_location": to_loc,
                "period": default_period,
                "fields": {"Cost Per UOM": float(cost), "Available": 1}
            })

    return scenario
