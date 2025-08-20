from typing import Dict, Any, List, Optional
import pandas as pd

def _ensure_df(dfs: Dict[str, pd.DataFrame], name: str, cols: List[str]) -> pd.DataFrame:
    df = dfs.get(name)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(columns=cols)
        dfs[name] = df
    # ensure all columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

def _upsert(df: pd.DataFrame, match: Dict[str, Any], updates: Dict[str, Any]) -> pd.DataFrame:
    if df is None:
        return df
    if df.empty:
        row = {**{c: None for c in df.columns}, **match, **updates}
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    mask = pd.Series([True] * len(df))
    for k, v in match.items():
        if k in df.columns:
            mask &= (df[k].astype(str) == str(v))
        else:
            df[k] = None
            mask &= (df[k].astype(str) == str(v))
    idx = df.index[mask] if mask.any() else []
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

def _delete_rows(df: pd.DataFrame, match: Dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mask = pd.Series([True] * len(df))
    for k, v in match.items():
        if k in df.columns:
            mask &= (df[k].astype(str) == str(v))
        else:
            # If column missing, nothing to delete for that criterion
            mask &= False
    keep = df[~mask]
    return keep.reset_index(drop=True)

def apply_scenario(dfs_in: Dict[str, pd.DataFrame], scenario: Dict[str, Any], allow_delete: bool = True) -> Dict[str, pd.DataFrame]:
    dfs = {k: v.copy() for k, v in dfs_in.items()}
    period = int(scenario.get("period", 2023))

    # ===== Adds (create rows) =====
    adds = scenario.get("adds") or {}

    # Add customers
    cu = _ensure_df(dfs, "Customers", ["Customer","Location"])
    for c in adds.get("customers", []):
        match = {"Customer": c["customer"]}
        updates = {"Location": c.get("location", c["customer"])}
        cu = _upsert(cu, match, updates)
    dfs["Customers"] = cu

    # Add customer demand rows
    cpd = _ensure_df(dfs, "Customer Product Data", ["Product","Customer","Location","Period","UOM","Demand","Lead Time","Variable Cost"])
    for d in adds.get("customer_demands", []):
        match = {"Product": d["product"], "Customer": d["customer"], "Location": d.get("location", d["customer"]), "Period": int(d.get("period", period))}
        updates = {"UOM": "Each", "Demand": float(d.get("demand", 0))}
        if d.get("lead_time") is not None:
            updates["Lead Time"] = float(d["lead_time"])
        cpd = _upsert(cpd, match, updates)
    dfs["Customer Product Data"] = cpd

    # Add warehouses
    wh = _ensure_df(dfs, "Warehouse", ["Warehouse","Location","Period","Available (Warehouse)","Minimum Capacity","Maximum Capacity","Fixed Cost","Variable Cost","Force Open","Force Close"])
    for w in adds.get("warehouses", []):
        match = {"Warehouse": w["warehouse"]}
        if "Period" in wh.columns:
            match["Period"] = period
        updates = {"Location": w.get("location", w["warehouse"])}
        fields = w.get("fields") or {}
        updates.update(fields)
        if "Available (Warehouse)" not in updates:
            updates["Available (Warehouse)"] = 1
        wh = _upsert(wh, match, updates)
    dfs["Warehouse"] = wh

    # Add supplier products
    sp = _ensure_df(dfs, "Supplier Product", ["Product","Supplier","Location","Step","Period","UOM","Available"])
    for s in adds.get("supplier_products", []):
        match = {"Product": s["product"], "Supplier": s["supplier"], "Location": s.get("location", s["supplier"]), "Period": int(s.get("period", period))}
        updates = s.get("fields", {}) or {}
        if "Available" not in updates:
            updates["Available"] = 1
        sp = _upsert(sp, match, updates)
    dfs["Supplier Product"] = sp

    # Add transport lanes
    tc = _ensure_df(dfs, "Transport Cost", ["Mode of Transport","Product","From Location","To Location","Period","UOM","Available","Retrieve Distance","Average Load Size","Cost Per UOM","Cost per Distance","Cost per Trip","Minimum Cost Per Trip"])
    for t in adds.get("transport_lanes", []):
        match = {
            "Mode of Transport": t["mode"],
            "Product": t["product"],
            "From Location": t["from_location"],
            "To Location": t["to_location"],
            "Period": int(t.get("period", period)),
        }
        updates = t.get("fields", {}) or {}
        if "UOM" not in updates:
            updates["UOM"] = "Each"
        if "Available" not in updates:
            updates["Available"] = 1
        tc = _upsert(tc, match, updates)
    dfs["Transport Cost"] = tc

    # ===== Updates (existing sections) =====

    # Demand updates (pct)
    for d in scenario.get("demand_updates", []):
        prod = d["product"]; cust = d["customer"]; loc = d.get("location", cust)
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

    # Warehouse field updates
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

    # Supplier changes (e.g., Available)
    for srow in scenario.get("supplier_changes", []):
        match = {"Product": srow["product"], "Supplier": srow["supplier"], "Location": srow["location"], "Period": period}
        updates = {srow["field"]: srow["new_value"]}
        sp = _upsert(sp, match, updates)
    dfs["Supplier Product"] = sp

    # Transport updates (cost, available, etc.)
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

    # ===== Deletes (if allowed) =====
    if allow_delete:
        dels = scenario.get("deletes") or {}

        # Delete customer product rows
        for d in dels.get("customer_product_rows", []):
            match = {
                "Product": d["product"], "Customer": d["customer"],
                "Location": d.get("location", d["customer"]), "Period": int(d.get("period", period))
            }
            cpd = _delete_rows(cpd, match)
        dfs["Customer Product Data"] = cpd

        # Delete customers
        for c in dels.get("customers", []):
            cu = _delete_rows(cu, {"Customer": c["customer"]})
        dfs["Customers"] = cu

        # Delete warehouses
        for w in dels.get("warehouses", []):
            wh = _delete_rows(wh, {"Warehouse": w["warehouse"]})
        dfs["Warehouse"] = wh

        # Delete supplier products
        for sdel in dels.get("supplier_products", []):
            sp = _delete_rows(sp, {"Product": sdel["product"], "Supplier": sdel["supplier"]})
        dfs["Supplier Product"] = sp

        # Delete transport lanes
        for tdel in dels.get("transport_lanes", []):
            match = {
                "Mode of Transport": tdel["mode"],
                "Product": tdel["product"],
                "From Location": tdel["from_location"],
                "To Location": tdel["to_location"],
                "Period": int(tdel.get("period", period)),
            }
            tc = _delete_rows(tc, match)
        dfs["Transport Cost"] = tc

    return dfs
