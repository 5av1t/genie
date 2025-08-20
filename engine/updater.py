# engine/updater.py
# Applies scenario JSON edits to workbook dataframes and computes diffs.
# Defensive (safe on missing sheets/columns), targeted updates only.

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import copy

DEFAULT_PERIOD = 2023

# --------------------------- Utilities ---------------------------

def _safe_df(df) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _deepcopy_dfs(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for k, v in (dfs or {}).items():
        out[k] = v.copy(deep=True) if isinstance(v, pd.DataFrame) else pd.DataFrame()
    return out

def _infer_uom(cpd: pd.DataFrame, product: str) -> str:
    if isinstance(cpd, pd.DataFrame) and "UOM" in cpd.columns:
        sub = cpd[cpd.get("Product","").astype(str) == str(product)]
        vals = [x for x in sub.get("UOM", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist() if x]
        if vals:
            return vals[0]
    return "each"

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _round_demand(x: float) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return 0

# --------------------------- Updaters ---------------------------

def _apply_demand_updates(dfs: Dict[str, pd.DataFrame], updates: List[Dict[str, Any]], default_period: int) -> None:
    if not updates:
        return
    sheet = "Customer Product Data"
    cpd = _safe_df(dfs.get(sheet)).copy()
    # Ensure core columns
    cpd = _ensure_columns(cpd, ["Product","Customer","Location","Period","Demand","UOM","Lead Time","Variable Cost"])
    for u in updates:
        product  = str(u.get("product",""))
        customer = str(u.get("customer",""))
        location = str(u.get("location", customer))
        period   = int(u.get("period", default_period))
        delta    = float(u.get("delta_pct", 0.0))
        setters  = dict(u.get("set", {}))  # e.g., {"Lead Time": 8}

        if not product or not customer:
            continue

        mask = (
            (cpd["Product"].astype(str) == product) &
            (cpd["Customer"].astype(str) == customer) &
            (cpd["Location"].astype(str) == location) &
            (pd.to_numeric(cpd["Period"], errors="coerce").fillna(default_period).astype(int) == period)
        )
        if mask.any():
            # Update existing row(s)
            idxs = cpd.index[mask].tolist()
            for i in idxs:
                base = pd.to_numeric(cpd.at[i, "Demand"], errors="coerce")
                base = 0.0 if pd.isna(base) else float(base)
                newd = _round_demand(base * (1.0 + delta / 100.0))
                cpd.at[i, "Demand"] = newd
                # Optional setters
                for k, v in setters.items():
                    if k not in cpd.columns:
                        cpd[k] = np.nan
                    cpd.at[i, k] = v
        else:
            # Create a new row with sensible defaults
            uom = _infer_uom(cpd, product)
            base = 0.0
            newd = _round_demand(base * (1.0 + delta / 100.0))
            row = {
                "Product": product,
                "Customer": customer,
                "Location": location,
                "Period": period,
                "UOM": uom,
                "Demand": newd,
            }
            # Set optional fields if provided
            for k, v in setters.items():
                row[k] = v
            cpd = pd.concat([cpd, pd.DataFrame([row])], ignore_index=True)

    dfs[sheet] = cpd

def _apply_warehouse_changes(dfs: Dict[str, pd.DataFrame], changes: List[Dict[str, Any]]) -> None:
    if not changes:
        return
    sheet = "Warehouse"
    wh = _safe_df(dfs.get(sheet)).copy()
    if wh.empty:
        # Create minimal structure if missing
        wh = pd.DataFrame(columns=["Warehouse","Location","Maximum Capacity","Fixed Cost","Variable Cost","Force Open","Force Close"])
    for ch in changes:
        w   = str(ch.get("warehouse",""))
        fld = str(ch.get("field",""))
        val = ch.get("new_value", None)
        if not w or not fld:
            continue
        if fld not in wh.columns:
            wh[fld] = np.nan
        mask = (wh["Warehouse"].astype(str) == w)
        if mask.any():
            wh.loc[mask, fld] = val
        else:
            # If warehouse doesn't exist, we won't invent a new warehouse silently.
            # You can choose to append a new warehouse with minimal info (commented):
            # wh = pd.concat([wh, pd.DataFrame([{"Warehouse": w, "Location": w, fld: val}])], ignore_index=True)
            pass
    dfs[sheet] = wh

def _apply_supplier_changes(dfs: Dict[str, pd.DataFrame], changes: List[Dict[str, Any]], default_period: int) -> None:
    if not changes:
        return
    sheet = "Supplier Product"
    sp = _safe_df(dfs.get(sheet)).copy()
    sp = _ensure_columns(sp, ["Product","Supplier","Location","Period","Available"])
    for ch in changes:
        product  = str(ch.get("product",""))
        supplier = str(ch.get("supplier",""))
        location = str(ch.get("location", supplier))
        period   = int(ch.get("period", default_period))
        fld      = str(ch.get("field",""))
        val      = ch.get("new_value", None)
        if not (product and supplier and fld):
            continue
        # find row
        mask = (
            (sp["Product"].astype(str) == product) &
            (sp["Supplier"].astype(str) == supplier) &
            (sp["Location"].astype(str) == location) &
            (pd.to_numeric(sp["Period"], errors="coerce").fillna(default_period).astype(int) == period)
        )
        if fld not in sp.columns:
            sp[fld] = np.nan
        if mask.any():
            sp.loc[mask, fld] = val
        else:
            row = {"Product": product, "Supplier": supplier, "Location": location, "Period": period, fld: val}
            sp = pd.concat([sp, pd.DataFrame([row])], ignore_index=True)
    dfs[sheet] = sp

def _apply_transport_updates(dfs: Dict[str, pd.DataFrame], updates: List[Dict[str, Any]], default_period: int) -> None:
    if not updates:
        return
    sheet = "Transport Cost"
    tc = _safe_df(dfs.get(sheet)).copy()
    tc = _ensure_columns(tc, ["Mode of Transport","Product","From Location","To Location","Period","Available","Cost Per UOM","Cost per Distance","Cost per Trip","Minimum Cost Per Trip","Average Load Size","Retrieve Distance","UOM"])
    for u in updates:
        mode  = str(u.get("mode",""))
        product = str(u.get("product",""))
        fr    = str(u.get("from_location",""))
        to    = str(u.get("to_location",""))
        period = int(u.get("period", default_period))
        fields = dict(u.get("fields", {}))
        if not (mode and fr and to):
            continue
        mask = (
            (tc["Mode of Transport"].astype(str) == mode) &
            (tc["Product"].astype(str) == product) &
            (tc["From Location"].astype(str) == fr) &
            (tc["To Location"].astype(str) == to) &
            (pd.to_numeric(tc["Period"], errors="coerce").fillna(default_period).astype(int) == period)
        )
        # Ensure columns for fields
        for k in fields.keys():
            if k not in tc.columns:
                tc[k] = np.nan
        if mask.any():
            for k, v in fields.items():
                tc.loc[mask, k] = v
        else:
            row = {
                "Mode of Transport": mode,
                "Product": product,
                "From Location": fr,
                "To Location": to,
                "Period": period,
            }
            row.update(fields)
            tc = pd.concat([tc, pd.DataFrame([row])], ignore_index=True)
    dfs[sheet] = tc

# --------------------------- Public API ---------------------------

def apply_scenario_edits(dfs: Dict[str, pd.DataFrame], scenario: Dict[str, Any], default_period: int = DEFAULT_PERIOD) -> Dict[str, pd.DataFrame]:
    """
    Apply scenario edits to a copy of the workbook dict and return the new dict.
    Only touches the intended sheets/rows/columns; leaves others untouched.
    """
    newdfs = _deepcopy_dfs(dfs)
    scn = scenario or {}
    period = int(scn.get("period", default_period))

    # Demand updates
    _apply_demand_updates(newdfs, scn.get("demand_updates", []), period)

    # Warehouse changes
    _apply_warehouse_changes(newdfs, scn.get("warehouse_changes", []))

    # Supplier changes
    _apply_supplier_changes(newdfs, scn.get("supplier_changes", []), period)

    # Transport updates
    _apply_transport_updates(newdfs, scn.get("transport_updates", []), period)

    # (Optional) Support for "adds" / "deletes" if your LLM flow uses them later
    # Adds: {"Customers": [ {...}, ... ], "Warehouse": [ {...} ], ...}
    if scn.get("adds"):
        for sheet, rows in scn["adds"].items():
            df = _safe_df(newdfs.get(sheet)).copy()
            if df.empty and rows:
                # create with union of keys
                cols = set()
                for r in rows:
                    cols.update(list((r or {}).keys()))
                df = pd.DataFrame(columns=sorted(list(cols)))
            for r in rows or []:
                df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
            newdfs[sheet] = df

    # Deletes: {"Transport Cost": [{"From Location":"X","To Location":"Y"}], ...}
    if scn.get("deletes"):
        for sheet, filters in scn["deletes"].items():
            df = _safe_df(newdfs.get(sheet)).copy()
            for f in filters or []:
                if not isinstance(f, dict) or not f:
                    continue
                mask = pd.Series([True] * len(df))
                for k, v in f.items():
                    if k in df.columns:
                        mask &= (df[k].astype(str) == str(v))
                df = df[~mask]
            newdfs[sheet] = df

    return newdfs

# --------------------------- Diffs ---------------------------

def _row_signature(df: pd.DataFrame, keys: List[str], i: int) -> Tuple:
    sig = []
    for k in keys:
        val = df.iloc[i][k] if k in df.columns else None
        sig.append(None if pd.isna(val) else val)
    return tuple(sig)

def _diff_one(before: pd.DataFrame, after: pd.DataFrame, keys: List[str], show_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Produce a tidy diff: rows where any non-key column changed, with _before/_after columns.
    """
    b = before.copy()
    a = after.copy()
    for k in keys:
        if k not in b.columns: b[k] = np.nan
        if k not in a.columns: a[k] = np.nan

    b["_sig"] = [ _row_signature(b, keys, i) for i in range(len(b)) ]
    a["_sig"] = [ _row_signature(a, keys, i) for i in range(len(a)) ]

    # align on signature
    b_indexed = b.set_index("_sig", drop=False)
    a_indexed = a.set_index("_sig", drop=False)

    # Rows present in both signatures → compare non-key cols
    common = set(b_indexed.index).intersection(set(a_indexed.index))
    diffs: List[pd.DataFrame] = []

    compare_cols = show_cols or sorted(list(set(a.columns) | set(b.columns)))
    compare_cols = [c for c in compare_cols if c not in (["_sig"] + keys)]

    for sig in common:
        br = b_indexed.loc[sig]
        ar = a_indexed.loc[sig]
        # br/ar can be Series or DataFrame if duplicates; convert to one row each (take first)
        if isinstance(br, pd.DataFrame): br = br.iloc[0]
        if isinstance(ar, pd.DataFrame): ar = ar.iloc[0]
        changed = {}
        for c in compare_cols:
            bv = br.get(c, np.nan)
            av = ar.get(c, np.nan)
            # normalize NaN
            bvn = None if pd.isna(bv) else bv
            avn = None if pd.isna(av) else av
            if bvn != avn:
                changed[c + "_before"] = bvn
                changed[c + "_after"] = avn
        if changed:
            for k in keys:
                changed[k] = ar.get(k, br.get(k))
            diffs.append(pd.DataFrame([changed]))

    # Rows only in 'after' (new rows)
    new_rows = a_indexed[~a_indexed.index.isin(common)]
    for _, ar in new_rows.iterrows():
        rec = {k: ar.get(k) for k in keys}
        for c in compare_cols:
            rec[c + "_before"] = None
            rec[c + "_after"] = ar.get(c)
        diffs.append(pd.DataFrame([rec]))

    # Rows only in 'before' (deleted rows)
    del_rows = b_indexed[~b_indexed.index.isin(common)]
    for _, br in del_rows.iterrows():
        rec = {k: br.get(k) for k in keys}
        for c in compare_cols:
            rec[c + "_before"] = br.get(c)
            rec[c + "_after"] = None
        diffs.append(pd.DataFrame([rec]))

    if not diffs:
        return pd.DataFrame()

    out = pd.concat(diffs, ignore_index=True)
    # Put keys first
    cols = list(keys) + [c for c in out.columns if c not in keys]
    return out[cols]

def diff_tables(before: Dict[str, pd.DataFrame], after: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of sheet -> DataFrame of changed rows with *_before/*_after columns.
    Only for sheets we typically edit.
    """
    out: Dict[str, pd.DataFrame] = {}
    # Customer Product Data
    out["Customer Product Data"] = _diff_one(
        _safe_df(before.get("Customer Product Data")),
        _safe_df(after.get("Customer Product Data")),
        keys=["Product","Customer","Location","Period"],
        show_cols=["Demand","Lead Time","UOM","Variable Cost"],
    )
    # Warehouse
    out["Warehouse"] = _diff_one(
        _safe_df(before.get("Warehouse")),
        _safe_df(after.get("Warehouse")),
        keys=["Warehouse"],
        show_cols=["Maximum Capacity","Fixed Cost","Variable Cost","Force Open","Force Close"],
    )
    # Supplier Product
    out["Supplier Product"] = _diff_one(
        _safe_df(before.get("Supplier Product")),
        _safe_df(after.get("Supplier Product")),
        keys=["Product","Supplier","Location","Period"],
        show_cols=["Available"],
    )
    # Transport Cost
    out["Transport Cost"] = _diff_one(
        _safe_df(before.get("Transport Cost")),
        _safe_df(after.get("Transport Cost")),
        keys=["Mode of Transport","Product","From Location","To Location","Period"],
        show_cols=["Available","Cost Per UOM","Cost per Distance","Cost per Trip","Minimum Cost Per Trip","Average Load Size","Retrieve Distance","UOM"],
    )
    return out
