# engine/loader.py
# Safe Excel loader + validator + model_index builder for GENIE.
# - Robust to missing sheets/columns
# - Returns partial results instead of crashing
# - Produces a compact model_index for grounded Q&A and tools

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd

DEFAULT_PERIOD = 2023

# Map common/variant sheet names to canonical names we use in the app
_CANON = {
    "products": "Products",
    "supplier definition": "Supplier Definition",
    "supplier": "Supplier Definition",         # tolerate 'Supplier'
    "supplier product": "Supplier Product",
    "warehouse": "Warehouse",
    "customers": "Customers",
    "customer product data": "Customer Product Data",
    "mode of transport": "Mode of Transport",
    "transport cost": "Transport Cost",
    "periods": "Periods",
    "locations": "Locations",                  # tolerate lowercase 'locations'
}

# Minimal required columns per sheet (kept pragmatic)
_REQ_COLS = {
    "Products": ["Product"],
    "Supplier Definition": ["Supplier"],
    "Supplier Product": ["Product", "Supplier", "Location"],
    "Warehouse": ["Warehouse", "Location"],
    "Customers": ["Customer", "Location"],
    "Customer Product Data": ["Product", "Customer", "Location", "Demand"],
    "Transport Cost": ["Mode of Transport", "From Location", "To Location"],
    "Periods": ["Start Date", "End Date"],
    "Locations": ["Location"],  # Country/Latitude/Longitude optional
}

# --------------------------- Helpers ---------------------------

def _canonicalize_sheets(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Rename incoming sheet names to our canonical names (case-insensitive)."""
    dfs: Dict[str, pd.DataFrame] = {}
    for name, df in (raw or {}).items():
        try:
            key = name.strip().lower()
        except Exception:
            key = str(name).lower()
        if key in _CANON:
            dfs[_CANON[key]] = df
        else:
            dfs[name] = df  # keep unknown sheets as-is
    return dfs

def _ensure_columns(df: Optional[pd.DataFrame], cols: List[str]) -> List[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return cols[:]  # all missing if no data
    present = set(df.columns)
    return [c for c in cols if c not in present]

def _num(df: Optional[pd.DataFrame]) -> Tuple[int, int]:
    return (0, 0) if df is None or not isinstance(df, pd.DataFrame) else (int(df.shape[0]), int(df.shape[1]))

def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return ""

def _num_or_none(x):
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None

def _as_df(dfs: Dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    d = dfs.get(name)
    return d if isinstance(d, pd.DataFrame) else pd.DataFrame()

# --------------------------- Loader + Validator ---------------------------

def load_and_validate_excel(xlsx_file) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Read the Excel workbook and return:
      dfs: dict of canonical sheet name -> DataFrame
      report: validation info (missing columns, row/col counts, warnings)
    Safeguards:
      - Missing sheets/columns reported (no crash)
      - Numeric sanity checks produce warnings only
    """
    try:
        xls = pd.read_excel(xlsx_file, sheet_name=None, engine="openpyxl")
    except Exception as e:
        # Return empty dfs with error in report
        return {}, {"_warnings": [f"Failed to read workbook: {e}"]}

    dfs = _canonicalize_sheets(xls or {})
    report: Dict[str, Any] = {"_warnings": []}

    # Per-sheet structure
    for sheet, req in _REQ_COLS.items():
        df = dfs.get(sheet)
        miss = _ensure_columns(df, req)
        rows, cols = _num(df)
        report[sheet] = {"missing_columns": miss, "num_rows": rows, "num_columns": cols}

    # Sanity checks (all guarded)
    try:
        wh = dfs.get("Warehouse")
        if isinstance(wh, pd.DataFrame) and "Maximum Capacity" in wh.columns:
            bad = pd.to_numeric(wh["Maximum Capacity"], errors="coerce").fillna(0) < 0
            if bad.any():
                report["_warnings"].append(f"Warehouse: {int(bad.sum())} rows with negative Maximum Capacity.")
    except Exception:
        report["_warnings"].append("Warehouse: unable to validate capacities (non-numeric or malformed).")

    try:
        cpd = dfs.get("Customer Product Data")
        if isinstance(cpd, pd.DataFrame) and "Demand" in cpd.columns:
            nonpos = pd.to_numeric(cpd["Demand"], errors="coerce").fillna(0) <= 0
            if nonpos.all():
                report["_warnings"].append("Customer Product Data: all demand rows are non-positive (model may produce no flows).")
            elif nonpos.any():
                report["_warnings"].append(f"Customer Product Data: {int(nonpos.sum())} rows have non-positive demand.")
    except Exception:
        report["_warnings"].append("Customer Product Data: unable to validate demand (non-numeric or malformed).")

    # Orphaned lanes
    try:
        tc = dfs.get("Transport Cost")
        known_locations = set()
        for she in ("Warehouse", "Customers", "Locations"):
            df = dfs.get(she)
            if isinstance(df, pd.DataFrame) and "Location" in df.columns:
                known_locations.update(df["Location"].dropna().astype(str).tolist())
        wh_df = dfs.get("Warehouse")
        if isinstance(wh_df, pd.DataFrame) and "Warehouse" in wh_df.columns:
            known_locations.update(wh_df["Warehouse"].dropna().astype(str).tolist())
        cu_df = dfs.get("Customers")
        if isinstance(cu_df, pd.DataFrame) and "Customer" in cu_df.columns:
            known_locations.update(cu_df["Customer"].dropna().astype(str).tolist())

        if isinstance(tc, pd.DataFrame) and not tc.empty:
            if "From Location" in tc.columns and "To Location" in tc.columns:
                mask_from = ~tc["From Location"].astype(str).isin(known_locations)
                mask_to = ~tc["To Location"].astype(str).isin(known_locations)
                n_orphan = int((mask_from | mask_to).sum())
                if n_orphan > 0:
                    report["_warnings"].append(
                        f"Transport Cost: {n_orphan} lane(s) reference unknown From/To. "
                        "Add them to 'Locations', 'Warehouse', or 'Customers'."
                    )
            else:
                report["_warnings"].append("Transport Cost: missing 'From Location' or 'To Location' columns.")
    except Exception:
        report["_warnings"].append("Transport Cost: unable to validate lanes.")

    return dfs, report

# --------------------------- Model Index (safe) ---------------------------

def build_model_index(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    """
    Build a compact, queryable index for grounded Q&A and tools.
    Safe on missing sheets/columns — returns partial info + no crashes.

    Keys:
      - locations: {location -> {country, lat, lon}}
      - suppliers: {supplier -> {location, products: list, available_by_period: {period: {product: available_flag}}}}
      - warehouses: {warehouse -> {location, capacity, force_open, force_close}}
      - customers: {customer -> {location, demand_by_period: {period: total}, products: list}}
      - lanes: list of dicts with keys: mode, product, from_location, to_location, period, available, cpu
      - products: list of product names
      - period: int
      - _warnings: list[str] (index-building warnings, non-fatal)
    """
    idx: Dict[str, Any] = {
        "locations": {},
        "suppliers": {},
        "warehouses": {},
        "customers": {},
        "lanes": [],
        "products": set(),
        "period": int(period),
        "_warnings": [],
    }

    # ---- Locations (optional) ----
    try:
        loc = _as_df(dfs, "Locations")
        if not loc.empty and "Location" in loc.columns:
            for _, r in loc.iterrows():
                name = _safe_str(r.get("Location"))
                if not name:
                    continue
                idx["locations"][name] = {
                    "country": _safe_str(r.get("Country")) if "Country" in loc.columns else "",
                    "lat": _num_or_none(r.get("Latitude")) if "Latitude" in loc.columns else None,
                    "lon": _num_or_none(r.get("Longitude")) if "Longitude" in loc.columns else None,
                }
        else:
            idx["_warnings"].append("Locations sheet missing or lacks 'Location' column; country/coords may be unknown.")
    except Exception:
        idx["_warnings"].append("Failed to parse Locations; continuing without coordinates.")

    # ---- Warehouses ----
    try:
        wh = _as_df(dfs, "Warehouse")
        if not wh.empty:
            if "Warehouse" not in wh.columns:
                idx["_warnings"].append("Warehouse sheet missing 'Warehouse' column; skipping warehouses.")
            else:
                for _, r in wh.iterrows():
                    w = _safe_str(r.get("Warehouse"))
                    if not w:
                        continue
                    idx["warehouses"][w] = {
                        "location": _safe_str(r.get("Location")) if "Location" in wh.columns else "",
                        "capacity": _num_or_none(r.get("Maximum Capacity")) if "Maximum Capacity" in wh.columns else None,
                        "force_open": int(_num_or_none(r.get("Force Open")) or 0) if "Force Open" in wh.columns else 0,
                        "force_close": int(_num_or_none(r.get("Force Close")) or 0) if "Force Close" in wh.columns else 0,
                    }
        else:
            idx["_warnings"].append("Warehouse sheet missing or empty.")
    except Exception:
        idx["_warnings"].append("Failed to parse Warehouse sheet; skipping warehouses.")

    # ---- Products (optional) ----
    try:
        prods = _as_df(dfs, "Products")
        if not prods.empty and "Product" in prods.columns:
            for p in prods["Product"].dropna().astype(str).tolist():
                idx["products"].add(p)
    except Exception:
        idx["_warnings"].append("Failed to parse Products; product list may be incomplete.")

    # ---- Customers (+ demand by period) ----
    try:
        customers = _as_df(dfs, "Customers")
        if not customers.empty:
            if "Customer" not in customers.columns:
                idx["_warnings"].append("Customers sheet missing 'Customer' column; skipping customer identities.")
            else:
                for _, r in customers.iterrows():
                    c = _safe_str(r.get("Customer"))
                    if not c:
                        continue
                    idx["customers"].setdefault(c, {
                        "location": _safe_str(r.get("Location")) if "Location" in customers.columns else c,
                        "demand_by_period": {},
                        "products": set(),
                    })
        cpd = _as_df(dfs, "Customer Product Data")
        if not cpd.empty and "Customer" in cpd.columns:
            for _, r in cpd.iterrows():
                c = _safe_str(r.get("Customer"))
                if not c:
                    continue
                p = _safe_str(r.get("Product")) if "Product" in cpd.columns else ""
                if p:
                    idx["products"].add(p)
                per = int(_num_or_none(r.get("Period")) or period) if "Period" in cpd.columns else int(period)
                dem = float(_num_or_none(r.get("Demand")) or 0.0) if "Demand" in cpd.columns else 0.0
                ent = idx["customers"].setdefault(c, {
                    "location": _safe_str(r.get("Location")) if "Location" in cpd.columns else c,
                    "demand_by_period": {},
                    "products": set(),
                })
                ent["demand_by_period"][per] = ent["demand_by_period"].get(per, 0.0) + dem
                if p:
                    ent["products"].add(p)
        elif cpd.empty:
            idx["_warnings"].append("Customer Product Data sheet missing or empty; total demand may be 0.")
        else:
            idx["_warnings"].append("Customer Product Data present but missing 'Customer' column; cannot aggregate demand.")
    except Exception:
        idx["_warnings"].append("Failed to parse Customers/Customer Product Data; demand view may be incomplete.")

    # ---- Suppliers (from Supplier Product) ----
    try:
        sup_prod = _as_df(dfs, "Supplier Product")
        if not sup_prod.empty:
            if "Supplier" not in sup_prod.columns:
                idx["_warnings"].append("Supplier Product missing 'Supplier' column; skipping suppliers.")
            else:
                for _, r in sup_prod.iterrows():
                    s = _safe_str(r.get("Supplier"))
                    if not s:
                        continue
                    p = _safe_str(r.get("Product")) if "Product" in sup_prod.columns else ""
                    locn = _safe_str(r.get("Location") or s) if "Location" in sup_prod.columns else s
                    per = int(_num_or_none(r.get("Period")) or period) if "Period" in sup_prod.columns else int(period)
                    avail = int(_num_or_none(r.get("Available")) or 0) if "Available" in sup_prod.columns else 0
                    ent = idx["suppliers"].setdefault(s, {"location": locn, "products": set(), "available_by_period": {}})
                    if p:
                        ent["products"].add(p)
                    ent["available_by_period"].setdefault(per, {})
                    if p:
                        ent["available_by_period"][per][p] = avail
        else:
            idx["_warnings"].append("Supplier Product sheet missing or empty; supplier availability unknown.")
    except Exception:
        idx["_warnings"].append("Failed to parse Supplier Product; supplier info may be incomplete.")

    # ---- Transport lanes ----
    try:
        tc = _as_df(dfs, "Transport Cost")
        if not tc.empty:
            # Column presence checks
            has_fl = "From Location" in tc.columns
            has_tl = "To Location" in tc.columns
            if not (has_fl and has_tl):
                idx["_warnings"].append("Transport Cost missing 'From Location' or 'To Location'; skipping lanes.")
            else:
                for _, r in tc.iterrows():
                    per = int(_num_or_none(r.get("Period")) or period) if "Period" in tc.columns else int(period)
                    lane = {
                        "mode": _safe_str(r.get("Mode of Transport")) if "Mode of Transport" in tc.columns else "",
                        "product": _safe_str(r.get("Product")) if "Product" in tc.columns else "",
                        "from_location": _safe_str(r.get("From Location")),
                        "to_location": _safe_str(r.get("To Location")),
                        "period": per,
                        "available": int(_num_or_none(r.get("Available")) or 1) if "Available" in tc.columns else 1,
                        "cpu": _num_or_none(r.get("Cost Per UOM")) if "Cost Per UOM" in tc.columns else None,
                    }
                    idx["lanes"].append(lane)
        else:
            idx["_warnings"].append("Transport Cost sheet missing or empty; optimizer may find no arcs.")
    except Exception:
        idx["_warnings"].append("Failed to parse Transport Cost; arcs view may be incomplete.")

    # Convert sets to lists (JSON-safe)
    try:
        for c, v in list(idx.get("customers", {}).items()):
            v["products"] = sorted(list(v.get("products", set())))
        for s, v in list(idx.get("suppliers", {}).items()):
            v["products"] = sorted(list(v.get("products", set())))
        idx["products"] = sorted(list(idx["products"]))
    except Exception:
        idx["_warnings"].append("Failed to finalize product lists; continuing.")

    return idx
