import re
from typing import Dict, Any, List, Set
from .schema import empty_scenario, validate_scenario, DEFAULT_PERIOD

def _vals(df, col) -> List[str]:
    if df is None or col not in df.columns:
        return []
    return sorted({str(x) for x in df[col].dropna().tolist()})

def build_catalog(dfs: Dict[str, Any]) -> Dict[str, Set[str]]:
    products = set(_vals(dfs.get("Products"), "Product"))
    customers = set(_vals(dfs.get("Customers"), "Customer"))
    warehouses = set(_vals(dfs.get("Warehouse"), "Warehouse"))
    suppliers = set(_vals(dfs.get("Supplier"), "Supplier"))
    modes = set(_vals(dfs.get("Mode of Transport"), "Mode of Transport"))
    locations = set(_vals(dfs.get("Warehouse"), "Location")) | set(_vals(dfs.get("Customers"), "Location"))
    return {
        "products": products,
        "customers": customers,
        "warehouses": warehouses,
        "suppliers": suppliers,
        "modes": modes,
        "locations": locations,
    }

def parse_prompt(prompt: str, dfs: Dict[str, Any], default_period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    s = empty_scenario(default_period)
    text = (prompt or "").strip()
    catalog = build_catalog(dfs)

    # Demand change with optional period + lead time
    dem_pat = re.compile(
        r"(increase|decrease)\s+([A-Za-z0-9\- ]+)\s+demand\s+at\s+([A-Za-z0-9\-_ ]+)\s+by\s+(\d+)\s*%?(?:.*?\b(\d{4})\b)?",
        re.IGNORECASE,
    )
    for m in dem_pat.finditer(text):
        direction, product, location, pct, period_opt = m.groups()
        product = product.strip(); location = location.strip()
        if product not in catalog["products"] or location not in catalog["customers"]:
            continue
        delta = int(pct)
        if direction.lower().startswith("decrease"):
            delta = -delta
        period_val = int(period_opt) if period_opt else default_period
        upd = {"product": product, "customer": location, "location": location, "delta_pct": delta}
        lt_m = re.search(r"set\s+lead\s*time\s*to\s*(\d+)", text, flags=re.IGNORECASE)
        if lt_m:
            upd["set"] = {"Lead Time": int(lt_m.group(1))}
        s["period"] = period_val
        s["demand_updates"].append(upd)

    # Warehouse capacity cap
    cap_pat = re.compile(r"cap\s+([A-Za-z0-9_\-]+)\s+maximum\s+capacity\s+at\s+(\d+)", re.IGNORECASE)
    for m in cap_pat.finditer(text):
        wh, val = m.groups()
        if wh in catalog["warehouses"]:
            s["warehouse_changes"].append({"warehouse": wh, "field": "Maximum Capacity", "new_value": int(val)})

    # Force open/close
    for phrase, field in [("force close", "Force Close"), ("force open", "Force Open")]:
        pat = re.compile(phrase + r"\s+([A-Za-z0-9_\-]+)", re.IGNORECASE)
        for m in pat.finditer(text):
            wh = m.group(1)
            if wh in catalog["warehouses"]:
                s["warehouse_changes"].append({"warehouse": wh, "field": field, "new_value": 1})

    # Supplier enable
    sup_pat = re.compile(r"enable\s+([A-Za-z0-9\- ]+)\s+at\s+([A-Za-z0-9_\-]+)", re.IGNORECASE)
    for m in sup_pat.finditer(text):
        prod, sup = m.groups()
        prod = prod.strip()
        if prod in catalog["products"] and sup in catalog["suppliers"]:
            s["supplier_changes"].append({"product": prod, "supplier": sup, "location": sup, "field": "Available", "new_value": 1})

    # Transport lane cost
    trans_pat = re.compile(
        r"set\s+([A-Za-z0-9_ \-]+)\s+lane\s+([A-Za-z0-9_\-]+)\s*→\s*([A-Za-z0-9_\-]+)\s+for\s+([A-Za-z0-9\- ]+)\s+to\s+cost\s+per\s+uom\s*=\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for m in trans_pat.finditer(text):
        mode, frm, to, prod, cost = m.groups()
        mode = mode.strip(); prod = prod.strip()
        if (mode in catalog["modes"] and frm in catalog["locations"] and to in catalog["locations"] and prod in catalog["products"]):
            s["transport_updates"].append({
                "mode": mode,
                "product": prod,
                "from_location": frm,
                "to_location": to,
                "period": default_period,
                "fields": {"Cost Per UOM": float(cost), "Available": 1},
            })

    return validate_scenario(s)
