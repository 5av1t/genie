# engine/examples.py
# Build scenario prompt examples from the user's uploaded workbook,
# so we never reference products/warehouses that don't exist.

from __future__ import annotations
from typing import Dict, List
import pandas as pd

DEFAULT_PERIOD = 2023

def _safe_df(df) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _first(series: pd.Series, fallback: str = "") -> str:
    vals = [str(x) for x in series.dropna().tolist() if str(x).strip()]
    return vals[0] if vals else fallback

def rules_examples(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> List[str]:
    if not isinstance(dfs, dict) or not dfs:
        # generic if no file yet
        return [
            "Increase demand for a product by 10% at a customer and set lead time to 8",
            "Cap a warehouse Maximum Capacity at 25000; force close another warehouse",
            "Enable a product at a supplier",
            "Set a transport lane cost per uom to 9.5 for a product",
        ]

    products = _safe_df(dfs.get("Products"))
    wh = _safe_df(dfs.get("Warehouse"))
    cust = _safe_df(dfs.get("Customers"))
    sp = _safe_df(dfs.get("Supplier Product"))
    tc = _safe_df(dfs.get("Transport Cost"))
    mot = _safe_df(dfs.get("Mode of Transport"))

    prod_name = _first(products.get("Product", pd.Series(dtype=str)), "YourProduct")
    cust_name = _first(cust.get("Customer", pd.Series(dtype=str)), "YourCustomer")
    cust_loc  = _first(cust.get("Location", pd.Series(dtype=str)), cust_name)
    wh1 = _first(wh.get("Warehouse", pd.Series(dtype=str)), "YourWarehouse1")
    wh2 = _first(wh.get("Warehouse", pd.Series(dtype=str)).iloc[1:] if "Warehouse" in wh.columns and len(wh) > 1 else pd.Series(dtype=str), "YourWarehouse2")
    sup_name = _first(sp.get("Supplier", pd.Series(dtype=str)), "YourSupplier")
    sup_loc  = _first(sp.get("Location", pd.Series(dtype=str)), sup_name)
    mode     = _first(mot.get("Mode of Transport", pd.Series(dtype=str)), "YourMode")

    # Try to build a lane from TC; if not present, fall back to (wh1 -> cust_loc)
    from_loc = _first(tc.get("From Location", pd.Series(dtype=str)), wh1)
    to_loc   = _first(tc.get("To Location", pd.Series(dtype=str)), cust_loc)
    lane_prod = _first(tc.get("Product", pd.Series(dtype=str)), prod_name)

    examples = [
        f"Increase {prod_name} demand at {cust_name} by 10% and set Lead Time to 8",
        f"Cap {wh1} Maximum Capacity at 25000; force close {wh2}",
        f"Enable {prod_name} at {sup_name}",
        f"Set {mode} lane {from_loc} → {to_loc} for {lane_prod} to Cost Per UOM = 9.5",
    ]
    # Deduplicate and keep order
    seen = set(); out = []
    for e in examples:
        if e not in seen:
            seen.add(e); out.append(e)
    return out

def llm_examples(dfs: Dict[str, pd.DataFrame], provider: str = "none", period: int = DEFAULT_PERIOD) -> List[str]:
    """
    Optional: if a provider is configured, you could call it to paraphrase the rules_examples.
    For now, we return the deterministic rules_examples to avoid any hallucination.
    """
    return rules_examples(dfs, period=period)
