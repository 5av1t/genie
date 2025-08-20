from typing import Dict, Any, List
import pandas as pd
import random

def _first_values(df: pd.DataFrame, col: str, k: int = 3) -> List[str]:
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return []
    vals = [str(x) for x in df[col].dropna().astype(str).unique().tolist()]
    return vals[:k]

def examples_for_file(dfs: Dict[str, pd.DataFrame], provider: str = "gemini") -> List[str]:
    """
    Build natural-language examples ONLY from entities present in the uploaded file.
    """
    products = _first_values(dfs.get("Products"), "Product", 3)
    if not products:
        # fallback: infer from CPD
        cpd = dfs.get("Customer Product Data")
        if isinstance(cpd, pd.DataFrame) and not cpd.empty and "Product" in cpd.columns:
            products = _first_values(cpd, "Product", 3)
    customers = _first_values(dfs.get("Customers"), "Customer", 6)
    warehouses = _first_values(dfs.get("Warehouse"), "Warehouse", 4)
    modes = _first_values(dfs.get("Mode of Transport"), "Mode of Transport", 3)

    ex: List[str] = []
    ex.append("run the base model")

    # Demand tweak
    if products and customers:
        ex.append(f"Increase {products[0]} demand at {customers[0]} by 10% and set lead time to 8")

    # Warehouse capacity / force open-close
    if warehouses:
        ex.append(f"Cap {warehouses[0]} Maximum Capacity at 25000")
        if len(warehouses) > 1:
            ex.append(f"Force close {warehouses[1]}")

    # Transport lane cost
    if modes and products and customers and warehouses:
        ex.append(f"Set {modes[0]} lane {warehouses[0]} -> {customers[1]} for {products[0]} to cost per uom = 9.5")

    # Adds
    if customers and products:
        ex.append(f"Add customer {customers[-1]} at {customers[-1]}; demand 6000 of {products[0]}; lead time 7")
    if warehouses:
        ex.append(f"Add warehouse {warehouses[-1]} at {warehouses[-1]}; Maximum Capacity 30000; force open")

    # Deletes
    if modes and products and customers and warehouses:
        ex.append(f"Delete {modes[0]} lane {warehouses[0]} -> {customers[0]} for {products[0]} in 2023")

    # Shuffle a bit but keep base model as first
    base = ex[0]
    rest = ex[1:]
    random.shuffle(rest)
    return [base] + rest[:7]
