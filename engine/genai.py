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
    """
    Build prompt examples *from the user's uploaded workbook* so we never
    reference entities that don't exist. If no file yet, show generic placeholders.
    """
    if not isinstance(dfs, dict) or not dfs:
        return [
            "Increase <Product> demand at <Customer> by 10% and set Lead Time to 8",
            "Cap <Warehouse> Maximum Capacity at 25000; force close <Warehouse2>",
            "Enable <Product> at <Supplier>",
            "Set <Mode> lane <From> → <To> for <Product> to Cost Per UOM = 9.5",
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
    wh2 = _first(
        wh.get("Warehouse", pd.Series(dtype=str)).iloc[1:]
        if "Warehouse" in wh.columns and len(wh) > 1
        else pd.Series(dtype=str),
        "YourWarehouse2"
    )
    sup_name = _first(sp.get("Supplier", pd.Series(dtype=str)), "YourSupplier")
    mode     = _first(mot.get("Mode of Transport", pd.Series(dtype=str)), "YourMode")
    from_loc = _first(tc.get("From Location", pd.Series(dtype=str)), wh1)
    to_loc   = _first(tc.get("To Location", pd.Series(dtype=str)), cust_loc)
    lane_prod = _first(tc.get("Product", pd.Series(dtype=str)), prod_name)

    examples = [
        f"Increase {prod_name} demand at {cust_name} by 10% and set Lead Time to 8",
        f"Cap {wh1} Maximum Capacity at 25000; force close {wh2}",
        f"Enable {prod_name} at {sup_name}",
        f"Set {mode} lane {from_loc} → {to_loc} for {lane_prod} to Cost Per UOM = 9.5",
    ]
    # Deduplicate while preserving order
    seen = set(); out: List[str] = []
    for e in examples:
        if e not in seen:
            seen.add(e); out.append(e)
    return out

def llm_examples(dfs: Dict[str, pd.DataFrame], provider: str = "none", period: int = DEFAULT_PERIOD) -> List[str]:
    """
    Keep deterministic for now to avoid hallucinations.
    If you later enable Gemini paraphrasing, you can transform `rules_examples`.
    """
    return rules_examples(dfs, period=period)
