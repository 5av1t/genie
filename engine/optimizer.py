from __future__ import annotations
from typing import Dict, Any, Tuple, List
import pandas as pd
import pulp

DEFAULT_PERIOD = 2023
BIGM = 1e12
UNMET_PENALTY = 1e6  # strong preference to serve demand when arcs exist

def _num(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default

def _warehouse_capacity(row: pd.Series) -> float:
    # Force Close or Available==0 => capacity 0
    force_close = int(_num(row.get("Force Close", 0), 0)) == 1
    available = int(_num(row.get("Available (Warehouse)", 1), 1))  # optional column
    if force_close or available == 0:
        return 0.0
    maxcap = _num(row.get("Maximum Capacity", None), None)
    if maxcap is None:
        # If missing, treat as very big capacity (acts as unbounded)
        return BIGM
    return maxcap if maxcap >= 0 else 0.0

def run_optimizer(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    wh = dfs.get("Warehouse", pd.DataFrame()).copy()
    cust = dfs.get("Customers", pd.DataFrame()).copy()
    cpd = dfs.get("Customer Product Data", pd.DataFrame()).copy()
    tc = dfs.get("Transport Cost", pd.DataFrame()).copy()

    if wh.empty or cust.empty or cpd.empty:
        return {"status": "missing_input", "total_cost": None, "total_demand": 0, "served": 0, "service_pct": 0, "open_warehouses": 0}, {"flows": []}

    # Normalize
    for df in [wh, cust, cpd, tc]:
        df.columns = [str(c).strip() for c in df.columns]

    # Period filter (cpd, tc)
    if "Period" in cpd.columns:
        cpd = cpd[cpd["Period"].fillna(period) == period].copy()
    else:
        cpd["Period"] = period
    if "Period" in tc.columns:
        tc = tc[(tc["Period"].isna()) | (tc["Period"] == period)].copy()

    # Demand by (c,p)
    cpd["Demand"] = pd.to_numeric(cpd.get("Demand", 0), errors="coerce").fillna(0.0)
    demand = cpd.groupby(["Customer","Product"], as_index=False)["Demand"].sum()
    demand = demand[demand["Demand"] > 0]
    total_demand = float(demand["Demand"].sum())

    # Warehouse data
    wh["cap"] = wh.apply(_warehouse_capacity, axis=1)
    wh_map = {str(r["Warehouse"]): float(r["cap"]) for _, r in wh.iterrows()}
    wh_loc = {str(r["Warehouse"]): str(r.get("Location", r["Warehouse"])) for _, r in wh.iterrows()}
    open_warehouses = int(sum(1 for w, cap in wh_map.items() if cap > 0))

    # Customer locations
    cust_loc = {str(r["Customer"]): str(r.get("Location", r["Customer"])) for _, r in cust.iterrows()}

    # Build arcs (w,c,p) from Transport Cost rows that match From/To to either names or locations
    arcs: Dict[Tuple[str,str,str], float] = {}
    if not tc.empty:
        # Only consider Available != 0 (or missing -> available)
        avail_col = "Available" if "Available" in tc.columns else None
        for _, r in tc.iterrows():
            if avail_col is not None and int(_num(r.get(avail_col, 1), 1)) == 0:
                continue
            mode = str(r.get("Mode of Transport", "")).strip()
            prod = str(r.get("Product", "")).strip()
            fr = str(r.get("From Location", "")).strip()
            to = str(r.get("To Location", "")).strip()
            cost = _num(r.get("Cost Per UOM", 0.0), 0.0)
            # match warehouse by name or location
            for wname, wloc in wh_loc.items():
                if fr == wname or fr == wloc:
                    # match customer by name or location
                    for cname, cloc in cust_loc.items():
                        if to == cname or to == cloc:
                            arcs[(wname, cname, prod)] = float(cost)

    # If no arcs at all, still build a model with unmet flows to explain infeasibility
    if not arcs:
        kpis = {
            "status": "no_feasible_arcs",
            "total_cost": None,
            "total_demand": total_demand,
            "served": 0,
            "service_pct": 0,
            "open_warehouses": open_warehouses,
        }
        return kpis, {"flows": [], "binding_warehouses": [], "num_arcs": 0}

    # PuLP model
    m = pulp.LpProblem("network", pulp.LpMinimize)
    x: Dict[Tuple[str,str,str], pulp.LpVariable] = {
        (w, c, p): pulp.LpVariable(f"x_{hash((w,c,p))}", lowBound=0)
        for (w, c, p) in arcs.keys()
    }

    # unmet demand variables
    u: Dict[Tuple[str,str], pulp.LpVariable] = {
        (row["Customer"], row["Product"]): pulp.LpVariable(f"u_{hash((row['Customer'],row['Product']))}", lowBound=0)
        for _, row in demand.iterrows()
    }

    # Objective
    m += pulp.lpSum(arcs[(w, c, p)] * var for (w, c, p), var in x.items()) + UNMET_PENALTY * pulp.lpSum(u.values())

    # Demand satisfaction
    for _, row in demand.iterrows():
        c = row["Customer"]; p = row["Product"]; d = float(row["Demand"])
        m += pulp.lpSum(x[(w, c, p)] for (w, cc, pp) in x.keys() if cc == c and pp == p) + u[(c, p)] == d

    # Warehouse capacity
    for wname, cap in wh_map.items():
        cap_safe = float(cap if cap is not None and cap == cap and cap >= 0 else 0.0)
        if cap_safe == 0.0:
            # Force zero outbound
            m += pulp.lpSum(x[(ww, c, p)] for (ww, c, p) in x.keys() if ww == wname) <= 0.0
        else:
            m += pulp.lpSum(x[(ww, c, p)] for (ww, c, p) in x.keys() if ww == wname) <= cap_safe

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[m.status]

    # Extract flows
    flows: List[Dict[str, Any]] = []
    served = 0.0
    for (w, c, p), var in x.items():
        v = float(var.value() or 0.0)
        if v > 1e-6:
            flows.append({"warehouse": w, "customer": c, "product": p, "qty": v})
            served += v

    total_cost_val = None
    try:
        total_cost_val = float(pulp.value(m.objective))
    except Exception:
        total_cost_val = None

    service_pct = 0.0
    if total_demand > 0:
        service_pct = 100.0 * served / total_demand

    # Binding warehouses
    binding = []
    for wname, cap in wh_map.items():
        cap_safe = float(cap if cap is not None and cap == cap and cap >= 0 else 0.0)
        used = sum(f["qty"] for f in flows if f["warehouse"] == wname)
        if cap_safe > 0 and used >= 0.999 * cap_safe:
            binding.append({"warehouse": wname, "used": used, "cap": cap_safe})

    kpis = {
        "status": status,
        "total_cost": round(total_cost_val, 2) if total_cost_val is not None else None,
        "total_demand": int(total_demand),
        "served": int(round(served)),
        "service_pct": round(service_pct, 2),
        "open_warehouses": open_warehouses,
    }
    diag = {"flows": flows, "binding_warehouses": binding[:3], "num_arcs": len(x)}
    return kpis, diag
