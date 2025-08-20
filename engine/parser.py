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
    return {
        "period": period,
        "demand_updates": [],
        "warehouse_changes": [],
        "supplier_changes": [],
        "transport_updates": [],
        "adds": {
            "customers": [],
            "customer_demands": [],
            "warehouses": [],
            "supplier_products": [],
            "transport_lanes": [],
        },
        "deletes": {
            "customers": [],
            "customer_product_rows": [],
            "warehouses": [],
            "supplier_products": [],
            "transport_lanes": [],
        }
    }

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

    # Period override (first 4-digit year wins)
    m_year = re.search(r"\b(20\d{2})\b", text)
    if m_year:
        s["period"] = int(m_year.group(1))

    # ========= Updates (existing behavior) =========

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

    # Supplier enablement (update Available=1)
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

    # ========= Adds (CRUD) =========

    # Add customer
    m_add_cust = re.search(r"add\s+customer\s+([A-Za-z0-9\-\s_]+)(?:\s+at\s+([A-Za-z0-9\-\s_]+))?", text, re.I)
    if m_add_cust:
        cust = m_add_cust.group(1).strip()
        loc  = (m_add_cust.group(2) or cust).strip()
        s["adds"]["customers"].append({"customer": cust, "location": loc})

    # Add demand row after add customer phrase or standalone
    m_add_dem = re.search(r"demand\s+(\d+(?:\.\d+)?)\s+of\s+([A-Za-z0-9\-\s_]+)", text, re.I)
    if m_add_dem:
        qty, prod = m_add_dem.groups()
        qty = float(qty); prod = prod.strip()
        # Try to infer customer from an earlier "add customer" or "...at <customer>"
        cust = None
        m_infer1 = re.search(r"add\s+customer\s+([A-Za-z0-9\-\s_]+)", text, re.I)
        if m_infer1:
            cust = m_infer1.group(1).strip()
        if not cust:
            m_infer2 = re.search(r"at\s+([A-Za-z0-9\-\s_]+)", text, re.I)
            if m_infer2:
                cust = m_infer2.group(1).strip()
        if cust:
            lead = None
            m_lead = re.search(r"lead\s*time\s*(?:=|to)?\s*(\d+)", text, re.I)
            if m_lead:
                lead = float(m_lead.group(1))
            s["adds"]["customer_demands"].append({
                "product": prod, "customer": cust, "location": cust, "period": s["period"], "demand": qty, **({"lead_time": lead} if lead is not None else {})
            })

    # Add warehouse
    m_add_wh = re.search(r"add\s+warehouse\s+([A-Za-z0-9\-\s_]+)(?:\s+at\s+([A-Za-z0-9\-\s_]+))?", text, re.I)
    if m_add_wh:
        wh = m_add_wh.group(1).strip()
        loc = (m_add_wh.group(2) or wh).strip()
        fields = {}
        m_cap2 = re.search(r"maximum\s+capacity\s*(?:=|at)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if m_cap2:
            fields["Maximum Capacity"] = float(m_cap2.group(1))
        if re.search(r"force\s+open", text, re.I):
            fields["Force Open"] = 1
            fields["Force Close"] = 0
        s["adds"]["warehouses"].append({"warehouse": wh, "location": loc, "fields": fields})

    # Add supplier product availability
    m_add_sp = re.search(r"enable\s+([A-Za-z0-9\-\s_]+)\s+at\s+([A-Za-z0-9\-\s_]+)", text, re.I)
    if m_add_sp:
        prod = m_add_sp.group(1).strip()
        sup  = m_add_sp.group(2).strip()
        s["adds"]["supplier_products"].append({"product": prod, "supplier": sup, "location": sup, "period": s["period"], "fields": {"Available": 1}})

    # Add transport lane
    m_add_lane = re.search(
        r"(?:add|create)\s+([A-Za-z0-9\-\s_]+)\s+lane\s+([A-Za-z0-9\-\s_]+)\s*(?:→|->|to)\s*([A-Za-z0-9\-\s_]+)\s+for\s+([A-Za-z0-9\-\s_]+)(?:\s+at\s+cost\s+per\s+uom\s*=?\s*(\d+(?:\.\d+)?))?",
        text, re.I
    )
    if m_add_lane:
        mode, fr, to, prod, cost = m_add_lane.groups()
        fields = {"Available": 1}
        if cost:
            fields["Cost Per UOM"] = float(cost)
        s["adds"]["transport_lanes"].append({
            "mode": mode.strip(), "product": prod.strip(), "from_location": fr.strip(), "to_location": to.strip(),
            "period": s["period"], "fields": fields
        })

    # ========= Deletes (CRUD) =========

    # Delete customer
    m_del_cust = re.search(r"delete\s+customer\s+([A-Za-z0-9\-\s_]+)", text, re.I)
    if m_del_cust:
        s["deletes"]["customers"].append({"customer": m_del_cust.group(1).strip()})

    # Delete warehouse
    m_del_wh = re.search(r"delete\s+warehouse\s+([A-Za-z0-9\-\s_]+)", text, re.I)
    if m_del_wh:
        s["deletes"]["warehouses"].append({"warehouse": m_del_wh.group(1).strip()})

    # Delete supplier product
    m_del_sp = re.search(r"delete\s+supplier\s+product\s+([A-Za-z0-9\-\s_]+)\s+at\s+([A-Za-z0-9\-\s_]+)", text, re.I)
    if m_del_sp:
        prod = m_del_sp.group(1).strip(); sup = m_del_sp.group(2).strip()
        s["deletes"]["supplier_products"].append({"product": prod, "supplier": sup})

    # Delete transport lane
    m_del_lane = re.search(
        r"delete\s+([A-Za-z0-9\-\s_]+)\s+lane\s+([A-Za-z0-9\-\s_]+)\s*(?:→|->|to)\s*([A-Za-z0-9\-\s_]+)\s+for\s+([A-Za-z0-9\-\s_]+)(?:\s+in\s+(20\d{2}))?",
        text, re.I
    )
    if m_del_lane:
        mode, fr, to, prod, yr = m_del_lane.groups()
        period = int(yr) if yr else s["period"]
        s["deletes"]["transport_lanes"].append({
            "mode": mode.strip(), "product": prod.strip(), "from_location": fr.strip(), "to_location": to.strip(), "period": period
        })

    # Delete specific CPD row
    m_del_cpd = re.search(r"delete\s+demand\s+for\s+([A-Za-z0-9\-\s_]+)\s+at\s+([A-Za-z0-9\-\s_]+)(?:\s+in\s+(20\d{2}))?", text, re.I)
    if m_del_cpd:
        prod, cust, yr = m_del_cpd.groups()
        period = int(yr) if yr else s["period"]
        s["deletes"]["customer_product_rows"].append({"product": prod.strip(), "customer": cust.strip(), "location": cust.strip(), "period": period})

    return s
