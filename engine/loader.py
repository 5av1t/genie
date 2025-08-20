from typing import Dict, Tuple, Any
import pandas as pd

DEFAULT_PERIOD = 2023

# Required columns per sheet
REQ = {
    "Customers": ["Customer", "Location"],
    "Warehouse": ["Warehouse", "Location"],
    "Customer Product Data": ["Product", "Customer", "Location", "Period", "Demand"],
    "Transport Cost": ["Mode of Transport", "Product", "From Location", "To Location", "Period", "Cost Per UOM"],
    "Products": ["Product"],
    "Supplier Product": ["Product", "Supplier", "Location", "Period", "Available"],
    "Mode of Transport": ["Mode of Transport"],
    "Periods": ["Start Date", "End Date"],  # permissive; we coerce Period elsewhere
}

ALIAS_MAP = {
    "cost per uom": "Cost Per UOM",
    "cost per distance": "Cost per Distance",
    "cost per trip": "Cost per Trip",
    "minimum cost per trip": "Minimum Cost Per Trip",
    "available (warehouse)": "Available (Warehouse)",
    "available(warehouse)": "Available (Warehouse)",
    "periods": "Periods",
    "mode of transport": "Mode of Transport",
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    ren = {}
    for c in df.columns:
        c2 = c.strip()
        low = c2.lower()
        ren[c] = ALIAS_MAP.get(low, c2)
    return df.rename(columns=ren)

def _coerce_types(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # Customer Product Data
    cpd = dfs.get("Customer Product Data")
    if isinstance(cpd, pd.DataFrame) and not cpd.empty:
        if "Period" not in cpd.columns:
            cpd["Period"] = DEFAULT_PERIOD
        for col in ["Demand", "Lead Time", "Variable Cost"]:
            if col in cpd.columns:
                cpd[col] = pd.to_numeric(cpd[col], errors="coerce")
        cpd["Period"] = pd.to_numeric(cpd["Period"], errors="coerce").fillna(DEFAULT_PERIOD).astype(int)
        dfs["Customer Product Data"] = cpd

    # Warehouse
    wh = dfs.get("Warehouse")
    if isinstance(wh, pd.DataFrame) and not wh.empty:
        if "Available (Warehouse)" not in wh.columns:
            wh["Available (Warehouse)"] = 1
        for col in ["Minimum Capacity", "Maximum Capacity", "Fixed Cost", "Variable Cost", "Force Open", "Force Close", "Available (Warehouse)"]:
            if col in wh.columns:
                wh[col] = pd.to_numeric(wh[col], errors="coerce")
        if "Period" in wh.columns:
            wh["Period"] = pd.to_numeric(wh["Period"], errors="coerce").fillna(DEFAULT_PERIOD).astype(int)
        dfs["Warehouse"] = wh

    # Supplier Product
    sp = dfs.get("Supplier Product")
    if isinstance(sp, pd.DataFrame) and not sp.empty:
        if "Period" not in sp.columns:
            sp["Period"] = DEFAULT_PERIOD
        if "Available" in sp.columns:
            sp["Available"] = pd.to_numeric(sp["Available"], errors="coerce")
        sp["Period"] = pd.to_numeric(sp["Period"], errors="coerce").fillna(DEFAULT_PERIOD).astype(int)
        dfs["Supplier Product"] = sp

    # Transport Cost
    tc = dfs.get("Transport Cost")
    if isinstance(tc, pd.DataFrame) and not tc.empty:
        if "Period" not in tc.columns:
            tc["Period"] = DEFAULT_PERIOD
        for col in ["Available", "Retrieve Distance", "Average Load Size", "Cost Per UOM", "Cost per Distance", "Cost per Trip", "Minimum Cost Per Trip"]:
            if col in tc.columns:
                tc[col] = pd.to_numeric(tc[col], errors="coerce")
        tc["Period"] = pd.to_numeric(tc["Period"], errors="coerce").fillna(DEFAULT_PERIOD).astype(int)
        dfs["Transport Cost"] = tc

    return dfs

def _validate(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for sheet, req_cols in REQ.items():
        df = dfs.get(sheet)
        if df is None:
            report[sheet] = {"missing_columns": req_cols, "num_rows": 0, "num_columns": 0, "sheet_missing": True}
            continue
        missing = [c for c in req_cols if c not in df.columns]
        report[sheet] = {
            "missing_columns": missing,
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "sheet_missing": False,
        }

    # Warnings
    warn = []

    # Negative capacities
    wh = dfs.get("Warehouse")
    if isinstance(wh, pd.DataFrame) and "Maximum Capacity" in wh.columns:
        neg = wh[pd.to_numeric(wh["Maximum Capacity"], errors="coerce").fillna(0) < 0]
        if not neg.empty:
            warn.append(f"Warehouse: {len(neg)} row(s) with negative Maximum Capacity.")

    # Non-positive demand
    cpd = dfs.get("Customer Product Data")
    if isinstance(cpd, pd.DataFrame) and "Demand" in cpd.columns:
        npd = cpd[pd.to_numeric(cpd["Demand"], errors="coerce").fillna(0) <= 0]
        if not npd.empty:
            warn.append(f"Customer Product Data: {len(npd)} row(s) with non-positive Demand.")

    # Orphan lanes: From must be Warehouse.Location; To can be Customers.Customer OR CPD.Customer/Location
    tc = dfs.get("Transport Cost")
    if isinstance(tc, pd.DataFrame) and not tc.empty:
        wh_locs = set()
        if isinstance(wh, pd.DataFrame) and not wh.empty and "Location" in wh.columns:
            wh_locs = set(str(x) for x in wh["Location"].dropna().unique())

        cust_names = set()
        custs = dfs.get("Customers")
        if isinstance(custs, pd.DataFrame) and not custs.empty and "Customer" in custs.columns:
            cust_names |= set(str(x) for x in custs["Customer"].dropna().unique())
        if isinstance(cpd, pd.DataFrame) and not cpd.empty:
            if "Customer" in cpd.columns:
                cust_names |= set(str(x) for x in cpd["Customer"].dropna().unique())
            if "Location" in cpd.columns:
                cust_names |= set(str(x) for x in cpd["Location"].dropna().unique())

        orphan_from = 0
        orphan_to = 0
        for _, r in tc.iterrows():
            fl = str(r.get("From Location"))
            tl = str(r.get("To Location"))
            if wh_locs and fl not in wh_locs:
                orphan_from += 1
            if cust_names and tl not in cust_names:
                orphan_to += 1
        if orphan_from or orphan_to:
            warn.append(
                f"Transport Cost: orphan lanes — From not in Warehouse.Location: {orphan_from}, "
                f"To not in Customers/CPD: {orphan_to}"
            )

    report["_warnings"] = warn
    return report

def load_excel(file_like) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(file_like)
    dfs: Dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name)
        df = _normalize_columns(df)
        dfs[name] = df
    dfs = _coerce_types(dfs)
    return dfs

def load_and_validate_excel(file_like) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    dfs = load_excel(file_like)
    report = _validate(dfs)
    return dfs, report
