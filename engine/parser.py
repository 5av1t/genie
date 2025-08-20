from typing import Dict, Any, List
import re
import pandas as pd

DEFAULT_PERIOD = 2023

def _vals(df, col) -> List[str]:
    if df is None or col not in df.columns:
        return []
    return sorted({str(x) for x in df[col].dropna().tolist()})

def _catalog(dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    return {
        "products": _vals(dfs.get("Products"), "Product"),
        "customers": _vals(dfs.get("Customers"), "Customer") or _vals(dfs.get("Customer Product Data"), "Customer"),
        "warehouses": _vals(dfs.get("Warehouse"), "Warehouse"),
        "suppliers": _vals(dfs.get("Supplier Product"), "Supplier"),
        "modes": _vals(dfs.get("Mode of Transport"), "Mode of Transport"),
        "locations": sorted(
            set(_vals(dfs.get("Warehouse"), "Location")) |
            set(_vals(dfs.get("Customers"), "Location")) |
            set(_vals(dfs.get("Customer Product Data"), "Location")) |
            set(_vals(dfs.get("Customers"), "Customer"))
        ),
    }

def _base_scenario(period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    return {"period": period, "demand_updates": [], "warehouse_changes": [], "supplier_changes": [], "transport_updates": []}

def parse_prompt(prompt: str, dfs: Dict[str, pd.DataFrame], default_period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    if not prompt:
        return _base_scenario(default_period)
    text = prompt.strip()

    # "run base model"
    if text.lower() in {"run base model", "run the base model"}:
        s = _base_scenario(default_period)
        s["meta"] = {"run_base": True}
        return s

    cats = _catalog(dfs)
    s = _base_scenario(default_period)

    # Period override if any 4-digit year appears
    m_year = re.search(r"\b(20\d{2})\b", text)
    if m_year:
        s["period"] = int(m_year.group(1))

    # Demand increase/decrease by %
    dm = re.search(r"(increase|decrease)\s+([A-Za-z0-9\-\s]+)\s+demand\s+at\s+([A-Za-z0-9\-\s]+)\s+by\s+(\d+(?:\.\d+)?)\s*%", text, re.I)
    if dm:
        kind, prod, cust, pct = dm.groups()
        prod = prod.strip()
        cust = cust.strip()
        if prod in cats["products"] and cust in cats["customers"]:
            delta = float(pct) * (1 if kind.lower()=="increase" else -1)
            upd = {"product": prod, "customer": cust, "location": cust, "delta_pct": delta}
            m_lt = re.search(r"set\s+lead\s*time\s+to\s+(\d+)", text, re.I)
            if m_lt:
                upd["set"] = {"Lead Time": float(m_lt.group(1))}
            s["demand_updates"].append(upd)

    # Warehouse changes
    m_cap = re.search(r"cap\s+([A-Za-z0-9\-_]+)\s+maximum\s+capacity\s+at\s+(\d+(?:\.\d+)?)", text, re.I)
    if m_cap:
        wh, val = m_cap.groups()
        wh = wh.strip()
        if wh in cats["warehouses"]:
            s["warehouse_changes"].append({"warehouse": wh, "field": "Maximum Capacity", "new_value": float(val)})

    for cmd, fld in [("force close", "Force Close"), ("force open", "Force Open")]:
        m_fc = re.search(rf"{cmd}\s+([A-Za-z0-9\-_]+)", text, re.I)
        if m_fc:
            wh = m_fc.group(1).strip()
            if wh in cats["warehouses"]:
                s["warehouse_changes"].append({"warehouse": wh, "field": fld, "new_value": 1})

    # Supplier enablement
    for sup in cats["suppliers"]:
        m_en = re.search(rf"enable\s+([A-Za-z0-9\-\s]+)\s+at\s+{re.escape(sup)}", text, re.I)
        if m_en:
            prod = m_en.group(1).strip()
            if prod in cats["products"]:
                s["supplier_changes"].append({"product": prod, "supplier": sup, "location": sup, "field": "Available", "new_value": 1})

    # Transport lane cost
    m_tc = re.search(
        r"set\s+([A-Za-z0-9\-\s_]+)\s+lane\s+([A-Za-z0-9\-_]+)\s*(?:→|->|to)\s*([A-Za-z0-9\-\s]+)\s+for\s+([A-Za-z0-9\-\s]+)\s+to\s+cost\s+per\s+uom\s*=?\s*(\d+(?:\.\d+)?)",
        text, re.I
    )
    if m_tc:
        mode, fr, to, prod, cost = m_tc.groups()
        mode = mode.strip(); fr = fr.strip(); to = to.strip(); prod = prod.strip()
        if mode in cats["modes"] and prod in cats["products"] and fr in cats["locations"] and to in cats["customers"]:
            s["transport_updates"].append({
                "mode": mode,
                "product": prod,
                "from_location": fr,
                "to_location": to,
                "period": s["period"],
                "fields": {"Cost Per UOM": float(cost), "Available": 1}
            })

    return s
