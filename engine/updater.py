from typing import Dict, Any, Optional
import pandas as pd

DEFAULT_PERIOD = 2023

def _ensure_int(x, fallback=DEFAULT_PERIOD) -> int:
    try:
        return int(x)
    except Exception:
        return fallback

def _ensure_col(df: pd.DataFrame, col: str, dtype="object"):
    if col not in df.columns:
        df[col] = pd.Series(dtype=dtype)

def _guess_uom(cpd: pd.DataFrame, product: str) -> str:
    if isinstance(cpd, pd.DataFrame) and not cpd.empty:
        try:
            return cpd.loc[cpd["Product"] == product, "UOM"].dropna().astype(str).iloc[0]
        except Exception:
            pass
    return "Each"

def _create_or_update_cpd_row(
    cpd: pd.DataFrame,
    product: str,
    customer: str,
    location: str,
    period: int,
    delta_pct: float,
    set_map: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    set_map = set_map or {}
    for col, dt in [
        ("Product", "object"),
        ("Customer", "object"),
        ("Location", "object"),
        ("Period", "int64"),
        ("UOM", "object"),
        ("Demand", "float"),
        ("Lead Time", "float"),
        ("Variable Cost", "float"),
    ]:
        _ensure_col(cpd, col, dt)

    mask = (
        (cpd["Product"] == product)
        & (cpd["Customer"] == customer)
        & (cpd["Location"] == location)
        & (cpd["Period"] == period)
    )

    if mask.any():
        idx = cpd[mask].index
        base = pd.to_numeric(cpd.loc[idx, "Demand"], errors="coerce").fillna(0.0)
        cpd.loc[idx, "Demand"] = (base * (1.0 + float(delta_pct) / 100.0)).round()
        for k, v in set_map.items():
            if k in cpd.columns:
                cpd.loc[idx, k] = v
    else:
        uom = _guess_uom(cpd, product)
        baseline = 100.0
        demand = baseline * (1.0 + float(delta_pct) / 100.0) if delta_pct is not None else baseline
        new_row = {
            "Product": product, "Customer": customer, "Location": location, "Period": period,
            "UOM": uom, "Demand": round(demand), "Lead Time": set_map.get("Lead Time", 0), "Variable Cost": 0,
        }
        cpd = pd.concat([cpd, pd.DataFrame([new_row])], ignore_index=True)
    return cpd

def apply_scenario(dfs: Dict[str, pd.DataFrame], scenario: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Return new copies of DataFrames with the scenario applied across key sheets."""
    period = _ensure_int(scenario.get("period", DEFAULT_PERIOD))
    out = {name: df.copy() for name, df in dfs.items()}

    # 1) Demand updates
    cpd = out.get("Customer Product Data")
    if not isinstance(cpd, pd.DataFrame):
        cpd = pd.DataFrame(columns=["Product", "Customer", "Location", "Period", "UOM", "Demand", "Lead Time", "Variable Cost"])
    for upd in scenario.get("demand_updates", []):
        cpd = _create_or_update_cpd_row(
            cpd=cpd,
            product=str(upd["product"]),
            customer=str(upd["customer"]),
            location=str(upd["location"]),
            period=period,
            delta_pct=float(upd.get("delta_pct", 0.0)),
            set_map=upd.get("set", {}),
        )
    out["Customer Product Data"] = cpd

    # 2) Warehouse changes
    wh = out.get("Warehouse")
    if not isinstance(wh, pd.DataFrame):
        wh = pd.DataFrame(columns=["Warehouse", "Location", "Period"])
    for upd in scenario.get("warehouse_changes", []):
        name = upd["warehouse"]; field = upd["field"]; val = upd["new_value"]
        _ensure_col(wh, field, "float")
        if "Period" in wh.columns:
            mask = (wh["Warehouse"] == name) & (wh["Period"].fillna(period) == period)
        else:
            mask = (wh["Warehouse"] == name)
        if mask.any():
            wh.loc[mask, field] = val
    out["Warehouse"] = wh

    # 3) Supplier changes
    sp = out.get("Supplier Product")
    if not isinstance(sp, pd.DataFrame):
        sp = pd.DataFrame(columns=["Product", "Supplier", "Location", "Period", "Available"])
    for upd in scenario.get("supplier_changes", []):
        prod = upd["product"]; sup = upd["supplier"]; loc = upd["location"]
        field = upd["field"]; val = upd["new_value"]
        _ensure_col(sp, field, "float")
        mask = (
            (sp["Product"] == prod) & (sp["Supplier"] == sup) & (sp["Location"] == loc) &
            ((sp["Period"].fillna(period) == period) if "Period" in sp.columns else True)
        )
        if mask.any():
            sp.loc[mask, field] = val
    out["Supplier Product"] = sp

    # 4) Transport updates (create if missing)
    tc = out.get("Transport Cost")
    if not isinstance(tc, pd.DataFrame):
        tc = pd.DataFrame(columns=[
            "Mode of Transport", "Product", "From Location", "To Location", "Period",
            "UOM", "Available", "Retrieve Distance", "Average Load Size",
            "Cost Per UOM", "Cost per Distance", "Cost per Trip", "Minimum Cost Per Trip",
        ])
    for upd in scenario.get("transport_updates", []):
        mode = upd["mode"]; prod = upd["product"]; fr = upd["from_location"]; to = upd["to_location"]
        p = _ensure_int(upd.get("period", period)); fields = upd.get("fields", {})
        mask = (
            (tc.get("Mode of Transport") == mode) & (tc.get("Product") == prod) &
            (tc.get("From Location") == fr) & (tc.get("To Location") == to) &
            ((tc.get("Period").fillna(p) == p) if "Period" in tc.columns else True)
        )
        if mask.any():
            for k, v in fields.items():
                _ensure_col(tc, k, "float")
                tc.loc[mask, k] = v
        else:
            base = {
                "Mode of Transport": mode, "Product": prod, "From Location": fr, "To Location": to,
                "Period": p, "UOM": "Each", "Available": 1,
                "Retrieve Distance": 0, "Average Load Size": 1,
                "Cost Per UOM": 0.0, "Cost per Distance": 0.0, "Cost per Trip": 0.0, "Minimum Cost Per Trip": 0.0,
            }
            for k, v in fields.items():
                base[k] = v
            tc = pd.concat([tc, pd.DataFrame([base])], ignore_index=True)
    out["Transport Cost"] = tc

    return out
