from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd

DEFAULT_PERIOD = 2023

def _ensure_sheet(dfs: Dict[str, pd.DataFrame], name: str, cols: List[str]) -> pd.DataFrame:
    df = dfs.get(name, pd.DataFrame())
    if df.empty:
        df = pd.DataFrame(columns=cols)
    # ensure columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

def _first_uom(cpd: pd.DataFrame, product: str) -> str:
    if "UOM" in cpd.columns:
        m = cpd[cpd["Product"] == product]["UOM"].dropna().astype(str)
        if not m.empty:
            return m.iloc[0]
    return "each"

def apply_scenario_edits(dfs: Dict[str, pd.DataFrame], scenario: Dict[str, Any], default_period: int = DEFAULT_PERIOD) -> Dict[str, pd.DataFrame]:
    out = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in dfs.items()}
    period = int(scenario.get("period", default_period))

    # Demand updates -> Customer Product Data
    cpd = _ensure_sheet(out, "Customer Product Data", ["Product","Customer","Location","Period","UOM","Demand","Lead Time","Variable Cost"])
    for upd in scenario.get("demand_updates", []):
        p = upd["product"]; c = upd["customer"]; loc = upd.get("location", c)
        mask = (cpd["Product"] == p) & (cpd["Customer"] == c) & (cpd["Location"] == loc) & (cpd["Period"].fillna(period) == period)
        if not mask.any():
            # create row with sensible defaults
            newrow = {"Product": p, "Customer": c, "Location": loc, "Period": period, "UOM": _first_uom(cpd, p), "Demand": 0}
            cpd = pd.concat([cpd, pd.DataFrame([newrow])], ignore_index=True)
            mask = (cpd["Product"] == p) & (cpd["Customer"] == c) & (cpd["Location"] == loc) & (cpd["Period"] == period)
        # apply delta
        cur = float(pd.to_numeric(cpd.loc[mask, "Demand"], errors="coerce").fillna(0.0).iloc[0])
        delta = float(upd.get("delta_pct", 0.0))
        newd = round(cur * (1.0 + delta/100.0))
        cpd.loc[mask, "Demand"] = newd
        # set extras
        for k, v in (upd.get("set", {}) or {}).items():
            if k not in cpd.columns:
                cpd[k] = None
            cpd.loc[mask, k] = v
    out["Customer Product Data"] = cpd

    # Warehouse changes
    wh = _ensure_sheet(out, "Warehouse", ["Warehouse","Location","Period","Available (Warehouse)","Minimum Capacity","Maximum Capacity","Fixed Cost","Variable Cost","Force Open","Force Close"])
    for ch in scenario.get("warehouse_changes", []):
        w = ch["warehouse"]; field = ch["field"]; val = ch["new_value"]
        mask = wh["Warehouse"] == w
        if not mask.any():
            # create bare warehouse row
            newrow = {"Warehouse": w, "Location": w, "Period": period}
            wh = pd.concat([wh, pd.DataFrame([newrow])], ignore_index=True)
            mask = wh["Warehouse"] == w
        if field not in wh.columns:
            wh[field] = None
        wh.loc[mask, field] = val
        # force mutual exclusivity
        if field == "Force Close" and int(val) == 1:
            if "Force Open" not in wh.columns: wh["Force Open"] = 0
            wh.loc[mask, "Force Open"] = 0
        if field == "Force Open" and int(val) == 1:
            if "Force Close" not in wh.columns: wh["Force Close"] = 0
            wh.loc[mask, "Force Close"] = 0
    out["Warehouse"] = wh

    # Supplier changes
    sp = _ensure_sheet(out, "Supplier Product", ["Product","Supplier","Location","Period","Available"])
    for ch in scenario.get("supplier_changes", []):
        p = ch["product"]; s = ch["supplier"]; loc = ch.get("location", s); field = ch["field"]; val = ch["new_value"]
        mask = (sp["Product"] == p) & (sp["Supplier"] == s) & (sp["Location"] == loc) & (sp["Period"].fillna(period) == period)
        if not mask.any():
            newrow = {"Product": p, "Supplier": s, "Location": loc, "Period": period, field: val}
            sp = pd.concat([sp, pd.DataFrame([newrow])], ignore_index=True)
        else:
            if field not in sp.columns:
                sp[field] = None
            sp.loc[mask, field] = val
    out["Supplier Product"] = sp

    # Transport updates
    tc = _ensure_sheet(out, "Transport Cost", ["Mode of Transport","Product","From Location","To Location","Period","UOM","Available","Retrieve Distance","Average Load Size","Cost Per UOM","Cost per Distance","Cost per Trip","Minimum Cost Per Trip"])
    for t in scenario.get("transport_updates", []):
        mode = t["mode"]; prod = t["product"]; fr = t["from_location"]; to = t["to_location"]; p = int(t.get("period", period))
        fields = t.get("fields", {})
        mask = (tc["Mode of Transport"] == mode) & (tc["Product"] == prod) & (tc["From Location"] == fr) & (tc["To Location"] == to) & (tc["Period"].fillna(p) == p)
        if not mask.any():
            newrow = {"Mode of Transport": mode, "Product": prod, "From Location": fr, "To Location": to, "Period": p}
            for k, v in fields.items(): newrow[k] = v
            tc = pd.concat([tc, pd.DataFrame([newrow])], ignore_index=True)
        else:
            for k, v in fields.items():
                if k not in tc.columns:
                    tc[k] = None
                tc.loc[mask, k] = v
    out["Transport Cost"] = tc

    return out

def _diff_df(before: pd.DataFrame, after: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if before is None: before = pd.DataFrame(columns=keys)
    if after is None: after = pd.DataFrame(columns=keys)
    b = before.copy(); a = after.copy()
    for df in (b, a):
        df.columns = [str(c) for c in df.columns]
    # outer merge to catch inserts/deletes
    merged = b.merge(a, how="outer", on=keys, suffixes=("_before", "_after"), indicator=True)
    # For non-key columns present in both frames, show only changes
    out_rows = []
    for _, row in merged.iterrows():
        if row["_merge"] == "both":
            for col in a.columns:
                if col in keys: continue
                before_val = row.get(f"{col}_before", None)
                after_val = row.get(f"{col}_after", None)
                if (pd.isna(before_val) and pd.isna(after_val)) or (before_val == after_val):
                    continue
                out_rows.append({**{k: row[k] for k in keys}, "Field": col, "Before": before_val, "After": after_val})
        elif row["_merge"] == "left_only":
            out_rows.append({**{k: row[k] for k in keys}, "Field": "__row__", "Before": "present", "After": "deleted"})
        elif row["_merge"] == "right_only":
            out_rows.append({**{k: row[k] for k in keys}, "Field": "__row__", "Before": "missing", "After": "added"})
    return pd.DataFrame(out_rows)

def diff_tables(before: Dict[str, pd.DataFrame], after: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    out["Customer Product Data"] = _diff_df(before.get("Customer Product Data", pd.DataFrame()),
                                            after.get("Customer Product Data", pd.DataFrame()),
                                            ["Product","Customer","Location","Period"])
    out["Warehouse"] = _diff_df(before.get("Warehouse", pd.DataFrame()),
                                after.get("Warehouse", pd.DataFrame()),
                                ["Warehouse"])
    out["Supplier Product"] = _diff_df(before.get("Supplier Product", pd.DataFrame()),
                                       after.get("Supplier Product", pd.DataFrame()),
                                       ["Product","Supplier","Location","Period"])
    out["Transport Cost"] = _diff_df(before.get("Transport Cost", pd.DataFrame()),
                                     after.get("Transport Cost", pd.DataFrame()),
                                     ["Mode of Transport","Product","From Location","To Location","Period"])
    return out
