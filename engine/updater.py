# engine/updater.py
from typing import Dict, Any, Optional
import pandas as pd

DEFAULT_PERIOD = 2023


def _ensure_int(x, fallback=DEFAULT_PERIOD) -> int:
    try:
        return int(x)
    except Exception:
        return fallback


def _guess_uom(cpd: pd.DataFrame, product: str) -> str:
    """Try to reuse a UOM already used for the product; else default to 'Each'."""
    if isinstance(cpd, pd.DataFrame) and not cpd.empty:
        try:
            u = (
                cpd.loc[cpd["Product"] == product, "UOM"]
                .dropna()
                .astype(str)
                .iloc[0]
            )
            return u
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
    """Create or update a Customer Product Data row and return the new DataFrame."""
    set_map = set_map or {}

    # Ensure columns exist if the input file is very bare
    for col in ["Product", "Customer", "Location", "Period", "UOM", "Demand", "Lead Time", "Variable Cost"]:
        if col not in cpd.columns:
            cpd[col] = pd.Series(dtype="object" if col in ["Product", "Customer", "Location", "UOM"] else "float")

    mask = (
        (cpd["Product"] == product)
        & (cpd["Customer"] == customer)
        & (cpd["Location"] == location)
        & (cpd["Period"] == period)
    )

    if mask.any():
        idx = cpd[mask].index
        # multiplicative demand update
        # treat missing/NaN as 0
        base = pd.to_numeric(cpd.loc[idx, "Demand"], errors="coerce").fillna(0.0)
        cpd.loc[idx, "Demand"] = (base * (1.0 + float(delta_pct) / 100.0)).round()
        # apply any 'set' fields (e.g., Lead Time)
        for k, v in set_map.items():
            if k in cpd.columns:
                cpd.loc[idx, k] = v
    else:
        # create a sensible new row; if there was no prior demand, we seed a small baseline
        uom = _guess_uom(cpd, product)
        baseline = 100.0  # arbitrary seed so a +10% becomes 10 units
        demand = baseline * (1.0 + float(delta_pct) / 100.0) if delta_pct is not None else baseline
        new_row = {
            "Product": product,
            "Customer": customer,
            "Location": location,
            "Period": period,
            "UOM": uom,
            "Demand": round(demand),
            "Lead Time": set_map.get("Lead Time", 0),
            "Variable Cost": 0,
        }
        cpd = pd.concat([cpd, pd.DataFrame([new_row])], ignore_index=True)

    return cpd


def apply_scenario(dfs: Dict[str, pd.DataFrame], scenario: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Returns **new copies** of the DataFrames with the scenario applied.
    MVP: implements demand updates only (warehouse/supplier/transport can be added later).
    """
    period = _ensure_int(scenario.get("period", DEFAULT_PERIOD))
    out = {name: df.copy() for name, df in dfs.items()}

    cpd = out.get("Customer Product Data")
    if not isinstance(cpd, pd.DataFrame):
        # if the sheet is missing, create a minimal one
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
    return out
