from __future__ import annotations
from typing import Dict, Tuple, Any, List
import pandas as pd

DEFAULT_PERIOD = 2023

# Canonical sheet names we expose to the rest of the app
CANON = {
    "Products": ["products", "product", "items"],
    "Warehouse": ["warehouse", "warehouses", "facilities", "nodes"],
    "Supplier Product": ["supplier product", "supplier_products", "supplierproduct", "suppliers"],
    "Transport Cost": ["transport cost", "transport", "lanes", "costs"],
    "Customers": ["customers", "customer"],
    "Customer Product Data": ["customer product data", "demand", "customer_demand"],
    "Mode of Transport": ["mode of transport", "modes"],
    "Periods": ["periods", "time", "time periods"],
    "Locations": ["locations", "location_map", "geo", "coordinates"],
}

REQUIRED_COLUMNS = {
    "Products": {"Product"},
    "Warehouse": {"Warehouse"},
    "Supplier Product": {"Product", "Supplier", "Location"},
    "Transport Cost": {"Mode of Transport", "Product", "From Location", "To Location"},
    "Customers": {"Customer"},
    "Customer Product Data": {"Product", "Customer", "Location", "Demand"},
    "Mode of Transport": {"Mode of Transport"},
    "Periods": {"Start Date", "End Date"},
    # "Locations" optional
}

def _canonicalize_sheet_name(name: str) -> str:
    n = (name or "").strip().lower()
    for canon, aliases in CANON.items():
        if n == canon.lower() or n in aliases:
            return canon
    # try simple exact title
    for canon in CANON.keys():
        if name.strip().lower() == canon.lower():
            return canon
    return name  # leave as-is; caller will decide if used

def _pick_canon_mapping(xl: pd.ExcelFile) -> Dict[str, str]:
    """Return mapping canonical_name -> actual_sheetname"""
    mapping = {}
    for sheet in xl.sheet_names:
        canon = _canonicalize_sheet_name(sheet)
        # prefer the first match
        if canon not in mapping:
            mapping[canon] = sheet
    return mapping

def load_and_validate_excel(file) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    xl = pd.ExcelFile(file)
    mapping = _pick_canon_mapping(xl)

    dfs: Dict[str, pd.DataFrame] = {}
    report: Dict[str, Any] = {}
    warnings: List[str] = []

    for canon, actual in mapping.items():
        try:
            df = xl.parse(actual)
            # strip column names
            df.columns = [str(c).strip() for c in df.columns]
            dfs[canon] = df
            report[canon] = {"num_rows": len(df), "num_columns": len(df.columns)}
        except Exception as e:
            warnings.append(f"Failed to load sheet '{actual}': {e}")

    # Ensure canonical keys exist even if not present
    for need in CANON.keys():
        if need not in dfs:
            dfs[need] = pd.DataFrame()
            report[need] = {"num_rows": 0, "num_columns": 0}

    # Validation
    for sheet, req in REQUIRED_COLUMNS.items():
        miss = [c for c in req if c not in dfs[sheet].columns]
        if miss:
            report[sheet]["missing_columns"] = miss

    # normalize Period
    if "Customer Product Data" in dfs and "Period" not in dfs["Customer Product Data"].columns:
        dfs["Customer Product Data"]["Period"] = DEFAULT_PERIOD

    report["_warnings"] = warnings
    return dfs, report

def build_model_index(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    idx: Dict[str, Any] = {}
    products = set(dfs.get("Products", pd.DataFrame()).get("Product", pd.Series(dtype=str)).dropna().astype(str))
    warehouses = set(dfs.get("Warehouse", pd.DataFrame()).get("Warehouse", pd.Series(dtype=str)).dropna().astype(str))
    customers = set(dfs.get("Customers", pd.DataFrame()).get("Customer", pd.Series(dtype=str)).dropna().astype(str))
    modes = set(dfs.get("Mode of Transport", pd.DataFrame()).get("Mode of Transport", pd.Series(dtype=str)).dropna().astype(str))

    idx["products"] = sorted(products)
    idx["warehouses"] = sorted(warehouses)
    idx["customers"] = sorted(customers)
    idx["modes"] = sorted(modes)

    # Demand by (customer, product)
    cpd = dfs.get("Customer Product Data", pd.DataFrame())
    if not cpd.empty:
        cpd_use = cpd.copy()
        if "Period" in cpd_use.columns:
            cpd_use = cpd_use[cpd_use["Period"].fillna(period) == period]
        cpd_use["Demand"] = pd.to_numeric(cpd_use.get("Demand", 0), errors="coerce").fillna(0.0)
        idx["demand_map"] = cpd_use.groupby(["Customer","Product"])["Demand"].sum().to_dict()
    else:
        idx["demand_map"] = {}

    return idx
