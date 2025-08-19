# engine/parser.py
import re
from typing import Dict, Any, List, Set

DEFAULT_PERIOD = 2023

def _vals(df, col) -> List[str]:
    if df is None or col not in df.columns:
        return []
    return sorted({str(x) for x in df[col].dropna().tolist()})

def _catalog(dfs: Dict[str, Any]):
    products   = set(_vals(dfs.get("Products"), "Product"))
    customers  = set(_vals(dfs.get("Customers"), "Customer"))
    warehouses = set(_vals(dfs.get("Warehouse"), "Warehouse"))
    suppliers  = set(_vals(dfs.get("Supplier"), "Supplier"))
    modes      = set(_vals(dfs.get("Mode of Transport"), "Mode of Transport"))
    locations  = set(_vals(dfs.get("Warehouse"), "Location")) | set(_vals(dfs.get("Customers"), "Location"))
    return {
        "products": products, "customers": customers, "warehouses": warehouses,
        "suppliers": suppliers, "modes": modes, "locations": locations,
    }

def _empty_scenario(period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    return {
        "period": period,
        "demand_updates": [],
        "warehouse_changes": [],
        "supplier_changes": [],
        "transport_updates": [],
    }

def parse_prompt(prompt: str, dfs: Dict[str, Any], default_period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    """
    Rule-based, minimal parser that covers the hackathon examples:
      - Increase/Decrease demand at <Customer> by <pct> [%] [period]
      - set Lead Time to X (applies to the same demand update)
      - Cap <Warehouse> Maximum Capacity at <value>
      - Force close <Warehouse> / Force open <Warehouse>
      - Enable <Product> at <Supplier>
      - Set <Mode> lane <From> → <To> for <Product> to Cost Per UOM = <value>
    """
    s = _empty_scenario(default_period)
    text = (prompt or "").strip()
    cats = _catalog(dfs)

    # 1) Demand +/-% at customer (optional year)
    dem_pat = re.compile(
        r"(increase|decrease)\s+([A-Za-z0-9\- ]+)\s+demand\s+at\s+([A-Za-z0-9\-_ ]+)\s+by\s+(\d+)\s*%?(?:.*?\b(\d{4})\b)?",
        re.IGNORECASE,
    )
    for m in dem_pat.finditer(text):
        direction, product, place, pct, year = m.groups()
        product = product.strip()
        place   = place.strip()
        if product in cats["products"] and place in cats["customers"]:
            delta = int(pct)
            if direction.lower().startswith("decrease"):
                delta = -delta
            s["period"] = int(year) if year else default_period
            upd = {"product": product, "customer": place, "location": place, "delta_pct": delta}
            # optional: "set lead time to 8"
            lt = re.search(r"set\s+lead\s*time\s*to\s*(\d+)", text, re.IGNORECASE)
            if lt:
                upd["set"] = {"Lead Time": int(lt.group(1))}
            s["demand_updates"].append(upd)

    # 2) Warehouse capacity cap
    cap_pat = re.compile(r"cap\s+([A-Za-z0-9_\-]+)\s+maximum\s+capacity\s+at\s+(\d+)", re.IGNORECASE)
    for m in cap_pat.finditer(text):
        wh, val = m.groups()
        if wh in cats["warehouses"]:
            s["warehouse_changes"].append({"warehouse": wh, "field": "Maximum Capacity", "new_value": int(val)})

    # 3) Force close/open
    for phrase, field in [("force close", "Force Close"), ("force open", "Force Open")]:
        pat = re.compile(phrase + r"\s+([A-Za-z0-9_\-]+)", re.IGNORECASE)
        for m in pat.finditer(text):
            wh = m.group(1)
            if wh in cats["warehouses"]:
                s["warehouse_changes"].append({"warehouse": wh, "field": field, "new_value": 1})

    # 4) Enable product at supplier
    sup_pat = re.compile(r"enable\s+([A-Za-z0-9\- ]+)\s+at\s+([A-Za-z0-9_\-]+)", re.IGNORECASE)
    for m in sup_pat.finditer(text):
        prod, sup = m.groups()
        prod = prod.strip()
        if prod in cats["products"] and sup in cats["suppliers"]:
            s["supplier_changes"].append({
                "product": prod, "supplier": sup, "location": sup, "field": "Available", "new_value": 1
            })

    # 5) Transport lane cost
    trans_pat = re.compile(
        r"set\s+([A-Za-z0-9_ \-]+)\s+lane\s+([A-Za-z0-9_\-]+)\s*→\s*([A-Za-z0-9_\-]+)\s+for\s+([A-Za-z0-9\- ]+)\s+to\s+cost\s+per\s+uom\s*=\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for m in trans_pat.finditer(text):
        mode, frm, to, prod, cost = m.groups()
        mode = mode.strip(); prod = prod.strip()
        if (
            mode in cats["modes"]
            and frm in cats["locations"]
            and to in cats["locations"]
            and prod in cats["products"]
        ):
            s["transport_updates"].append({
                "mode": mode,
                "product": prod,
                "from_location": frm,
                "to_location": to,
                "period": default_period,
                "fields": {"Cost Per UOM": float(cost), "Available": 1},
            })

    return s
