# engine/optimizer.py
# Quick single-echelon MILP: warehouse -> customer flows by product.
# Hardened against NaNs, missing arcs/costs, force open/close, and adds an unmet-demand penalty
# so you get a feasible solution with diagnostics instead of a hard failure.

from __future__ import annotations
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import pulp

DEFAULT_PERIOD = 2023
BIG_M = 1e9  # penalty for unmet demand (per unit)

def _safe_df(df) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _num(x, default=0.0) -> float:
    try:
        v = float(x)
        if pd.isna(v) or not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def _i(x, default=0) -> int:
    try:
        v = int(float(x))
        return v
    except Exception:
        return int(default)

def _warehouse_capacity(row: pd.Series) -> float:
    # Force Close overrides capacity to 0
    if _i(row.get("Force Close", 0), 0) == 1:
        return 0.0
    cap = _num(row.get("Maximum Capacity"), 0.0)
    if cap < 0 or not np.isfinite(cap):
        cap = 0.0
    return cap

def run_optimizer(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build and solve a small MILP:
      - Vars: x[w,c,p] >= 0 (qty), u[c,p] >= 0 (unmet)
      - Min: sum(cpu[w,c,p] * x[w,c,p]) + BIG_M * sum(u[c,p])
      - s.t. For each (c,p): sum_w x[w,c,p] + u[c,p] = demand[c,p]
            For each w:      sum_{c,p} x[w,c,p] <= capacity_w
    Uses Transport Cost as the arc set (available=1, finite CPU).
    Returns:
      kpis: {status, total_cost, total_demand, served, service_pct, open_warehouses}
      diag: {flows: [ {warehouse, customer, product, qty, unit_cost}... ],
             binding_warehouses: [(w, used, cap, slack)]}
    """
    # --- Gather demand by customer & product for the given period ---
    cpd = _safe_df(dfs.get("Customer Product Data"))
    dem_rows = []
    if not cpd.empty:
        # ensure columns
        for c in ["Product","Customer","Location","Period","Demand"]:
            if c not in cpd.columns:
                cpd[c] = np.nan
        # only this period (or rows with missing Period treated as period)
        cpd["_Period"] = pd.to_numeric(cpd["Period"], errors="coerce").fillna(period).astype(int)
        sel = cpd[cpd["_Period"] == int(period)].copy()
        sel["Demand"] = pd.to_numeric(sel["Demand"], errors="coerce").fillna(0.0)
        for _, r in sel.iterrows():
            prod = str(r.get("Product","") or "")
            cust = str(r.get("Customer","") or "")
            if not prod or not cust:
                continue
            dem_rows.append((cust, prod, float(r["Demand"])))
    # collapse by (customer, product)
    demand: Dict[Tuple[str,str], float] = {}
    for c, p, d in dem_rows:
        demand[(c,p)] = demand.get((c,p), 0.0) + float(d)

    total_demand = float(sum(demand.values()))
    # Edge case: no demand rows at all
    if total_demand <= 0:
        return (
            {
                "status": "no_demand",
                "total_cost": 0.0,
                "total_demand": 0,
                "served": 0,
                "service_pct": 100.0,
                "open_warehouses": 0,
            },
            {"flows": [], "binding_warehouses": [], "num_arcs": 0}
        )

    # --- Warehouses & capacities ---
    wh = _safe_df(dfs.get("Warehouse")).copy()
    warehouses: List[str] = []
    cap: Dict[str, float] = {}
    if not wh.empty and "Warehouse" in wh.columns:
        for _, r in wh.iterrows():
            w = str(r.get("Warehouse",""))
            if not w:
                continue
            if w in warehouses:
                # if duplicate rows, take max capacity (simple aggregation)
                cap[w] = max(cap.get(w, 0.0), _warehouse_capacity(r))
            else:
                warehouses.append(w)
                cap[w] = _warehouse_capacity(r)

    # If no Warehouse sheet, we allow a "virtual" unlimited depot? Safer is to produce no arcs.
    # We'll proceed with arcs; if none exist, we exit later with "no_feasible_arcs".

    # --- Arcs from Transport Cost ---
    tc = _safe_df(dfs.get("Transport Cost")).copy()
    arcs: Dict[Tuple[str,str,str], float] = {}  # (w,c,p) -> cpu
    if not tc.empty:
        for c in ["Mode of Transport","Product","From Location","To Location","Period","Available","Cost Per UOM"]:
            if c not in tc.columns:
                tc[c] = np.nan
        tc["_Period"] = pd.to_numeric(tc["Period"], errors="coerce").fillna(period).astype(int)
        available = pd.to_numeric(tc["Available"], errors="coerce").fillna(1).astype(int)
        same_period = (tc["_Period"] == int(period))
        good = same_period & (available == 1)
        sub = tc[good].copy()
        sub["Cost Per UOM"] = pd.to_numeric(sub["Cost Per UOM"], errors="coerce")
        sub = sub[np.isfinite(sub["Cost Per UOM"])]
        for _, r in sub.iterrows():
            w = str(r.get("From Location","") or "")
            cst = str(r.get("To Location","") or "")
            p = str(r.get("Product","") or "")
            cpu = float(r["Cost Per UOM"])
            if not w or not cst or not np.isfinite(cpu):
                continue
            # Treat From Location as warehouse name (common in your file)
            arcs[(w, cst, p)] = cpu

    if not arcs:
        # Without arcs, we cannot route; report diagnostics.
        return (
            {
                "status": "no_feasible_arcs",
                "total_cost": None,
                "total_demand": int(total_demand),
                "served": 0,
                "service_pct": 0.0,
                "open_warehouses": int(sum(1 for v in cap.values() if v > 0)),
            },
            {"flows": [], "binding_warehouses": [], "num_arcs": 0}
        )

    # --- Build model ---
    m = pulp.LpProblem("GenieNetworkDesign", pulp.LpMinimize)

    # Vars
    x: Dict[Tuple[str,str,str], pulp.LpVariable] = {}  # (w,c,p)
    for key in arcs.keys():
        x[key] = pulp.LpVariable(f"x_{hash(key)}", lowBound=0, cat="Continuous")

    # unmet demand vars
    u: Dict[Tuple[str,str], pulp.LpVariable] = {}
    for cp in demand.keys():
        u[cp] = pulp.LpVariable(f"u_{hash(cp)}", lowBound=0, cat="Continuous")

    # Objective
    m += pulp.lpSum(arcs[(w,c,p)] * var for (w,c,p), var in x.items()) + BIG_M * pulp.lpSum(u.values())

    # Demand constraints
    for (cst, p), dem in demand.items():
        # sum_w x[w,c,p] + u[c,p] = dem
        terms = [x[(w,cst,p)] for (w, cc, pp) in x.keys() if cc == cst and pp == p and (w,cst,p) in x]
        if terms:
            m += pulp.lpSum(terms) + u[(cst,p)] == float(dem)
        else:
            # No inbound arcs for this (c,p); unmet var will take the demand.
            m += u[(cst,p)] == float(dem)

    # Capacity constraints per warehouse
    for w in warehouses:
        cw = float(cap.get(w, 0.0))
        terms = [x[(ww, c, p)] for (ww, c, p) in x.keys() if ww == w]
        if terms:
            if cw <= 0 or not np.isfinite(cw):
                # Force zero outbound if no capacity
                m += pulp.lpSum(terms) <= 0
            else:
                m += pulp.lpSum(terms) <= cw

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[m.status]
    flows: List[Dict[str, Any]] = []
    served = 0.0
    total_cost = 0.0

    if status not in ("Optimal", "Feasible"):
        # Infeasible or other status
        return (
            {
                "status": status.lower(),
                "total_cost": None,
                "total_demand": int(total_demand),
                "served": 0,
                "service_pct": 0.0,
                "open_warehouses": int(sum(1 for v in cap.values() if v > 0)),
            },
            {"flows": [], "binding_warehouses": [], "num_arcs": len(x)}
        )

    # Extract flows & KPIs
    for (w, cst, p), var in x.items():
        q = float(var.value() or 0.0)
        if q > 1e-6:
            cpu = float(arcs[(w,cst,p)])
            flows.append({"warehouse": w, "customer": cst, "product": p, "qty": q, "unit_cost": cpu})
            served += q
            total_cost += cpu * q

    unmet = sum(float(v.value() or 0.0) for v in u.values())
    # Note: total_cost includes penalty inside solver, but we report only transport part here.
    service_pct = 0.0 if total_demand <= 0 else (served / total_demand) * 100.0

    # binding caps
    binding: List[Tuple[str, float, float, float]] = []
    for w in warehouses:
        cw = float(cap.get(w, 0.0))
        used = sum(q["qty"] for q in flows if q["warehouse"] == w)
        slack = max(cw - used, 0.0)
        # Consider "binding" if within 1% or 1 unit
        if cw > 0 and (slack <= max(0.01 * cw, 1.0)):
            binding.append((w, used, cw, slack))

    kpis = {
        "status": status.lower(),
        "total_cost": round(total_cost, 4),
        "total_demand": int(total_demand),
        "served": int(round(served)),
        "service_pct": round(service_pct, 2),
        "open_warehouses": int(sum(1 for v in cap.values() if v > 0)),
    }
    diag = {"flows": flows, "binding_warehouses": binding[:3], "num_arcs": len(x)}
    return kpis, diag
