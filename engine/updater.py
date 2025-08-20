from typing import Dict, Any, List
import pandas as pd

def _ensure_df(dfs: Dict[str, pd.DataFrame], name: str, cols: List[str]) -> pd.DataFrame:
    df = dfs.get(name)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(columns=cols)
        dfs[name] = df
    return df

def _upsert(df: pd.DataFrame, match: Dict[str, Any], updates: Dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series([True] * len(df))
    for k, v in match.items():
        if k in df.columns:
            mask &= (df[k].astype(str) == str(v))
    idx = df.index[mask] if len(df) and mask.any() else []
    if len(idx) == 0:
        row = {**{c: None for c in df.columns}, **match, **updates}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        for i in idx:
            for k, v in updates.items():
                if k not in df.columns:
                    df[k] = None
                df.at[i, k] = v
    return df

def apply_scenario(dfs_in: Dict[str, pd.DataFrame], scenario: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    dfs = {k: v.copy() for k, v in dfs_in.items()}
    period = int(scenario.get("period", 2023))

    # Demand updates
    cpd = _ensure_df(dfs, "Customer Product Data", ["Product","Customer","Location","Period","UOM","Demand","Lead Time","Variable Cost"])
    for d in scenario.get("demand_updates", []):
        prod = d["product"]; cust = d["customer"]; loc = d.get("location", cust)
        # UOM inherit
        uom = "Each"
        if "UOM" in cpd.columns:
            found = cpd[(cpd["Product"].astype(str)==prod) & (cpd["Customer"].astype(str)==cust)]
            if not found.empty and "UOM" in found.columns and pd.notna(found.iloc[0].get("UOM")):
                uom = found.iloc[0]["UOM"]
        match = {"Product": prod, "Customer": cust, "Location": loc, "Period": period}
        existing = cpd[(cpd["Product"].astype(str)==prod) & (cpd["Customer"].astype(str)==cust) &
                       (cpd["Location"].astype(str)==loc) & (cpd["Period"].astype(int)==period)]
        base = float(existing.iloc[0]["Demand"]) if not existing.empty and pd.notna(existing.iloc[0].get("Demand")) else 0.0
        if "delta_pct" in d and d["delta_pct"] is not None:
            new_val = round(base * (1.0 + float(d["delta_pct"])/100.0))
        else:
            new_val = base
        updates = {"UOM": uom, "Demand": new_val}
        if "set" in d and isinstance(d["set"], dict):
            updates.update(d["set"])
        cpd = _upsert(cpd, match, updates)
    dfs["Customer Product Data"] = cpd

    # Warehouse changes
    wh = _ensure_df(dfs, "Warehouse", ["Warehouse","Location","Period","Available (Warehouse)","Minimum Capacity","Maximum Capacity","Fixed Cost","Variable Cost","Force Open","Force Close"])
    for w in scenario.get("warehouse_changes", []):
        match = {"Warehouse": w["warehouse"]}
        if "Period" in wh.columns:
            match["Period"] = period
        updates = {w["field"]: w["new_value"]}
        if w["field"] == "Force Close" and int(w["new_value"]) == 1:
            updates["Force Open"] = 0
        if w["field"] == "Force Open" and int(w["new_value"]) == 1:
            updates["Force Close"] = 0
        wh = _upsert(wh, match, updates)
    dfs["Warehouse"] = wh

    # Supplier changes
    sp = _ensure_df(dfs, "Supplier Product", ["Product","Supplier","Location","Step","Period","UOM","Available"])
    for s in scenario.get("supplier_changes", []):
        match = {"Product": s["product"], "Supplier": s["supplier"], "Location": s["location"], "Period": period}
        updates = {s["field"]: s["new_value"]}
        sp = _upsert(sp, match, updates)
    dfs["Supplier Product"] = sp

    # Transport updates
    tc = _ensure_df(dfs, "Transport Cost", ["Mode of Transport","Product","From Location","To Location","Period","UOM","Available","Retrieve Distance","Average Load Size","Cost Per UOM","Cost per Distance","Cost per Trip","Minimum Cost Per Trip"])
    for t in scenario.get("transport_updates", []):
        match = {
            "Mode of Transport": t["mode"],
            "Product": t["product"],
            "From Location": t["from_location"],
            "To Location": t["to_location"],
            "Period": int(t.get("period", period)),
        }
        updates = t.get("fields", {})
        if "Available" not in updates:
            updates["Available"] = 1
        if "UOM" not in updates:
            updates["UOM"] = "Each"
        tc = _upsert(tc, match, updates)
    dfs["Transport Cost"] = tc

    return dfs
