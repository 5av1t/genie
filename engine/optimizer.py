from typing import Dict, Any, Tuple
import math
import pandas as pd
import pulp

DEFAULT_PERIOD = 2023


def _get_warehouse_locations(wh: pd.DataFrame) -> Dict[str, str]:
    return {row["Warehouse"]: row.get("Location") for _, row in wh.iterrows()}


def _warehouse_capacity(row) -> float:
    available = row.get("Available (Warehouse)", 1)
    if int(row.get("Force Close", 0) or 0) == 1 or int(available or 1) == 0:
        return 0.0
    return float(row.get("Maximum Capacity", 0) or 0)


def _safe_cost(val) -> float:
    """
    Convert to a finite non-negative float.
    Returns None if val is NaN/inf/negative/unusable.
    """
    try:
        x = float(val)
    except Exception:
        return None
    if not math.isfinite(x):  # NaN or inf
        return None
    if x < 0:
        return None
    return x


def run_optimizer(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simple MILP: warehouse → customer flows by product using Transport Cost.

    - Decision: x[w,c,p] >= 0 on admissible arcs (warehouse location == From Location; To Location == Customer).
    - Demand satisfaction: sum_w x[w,c,p] >= demand[c,p].
    - Warehouse capacity: sum_{c,p} x[w,c,p] <= Maximum Capacity (respecting Force Close/Available).
    - Objective: minimize sum(cost_per_uom * x).

    Returns:
      kpis: dict with status, total_cost, demand, served, etc.
      diag: dict with binding capacities, model size, and a 'flows' list for mapping.
    """
    cpd = dfs.get("Customer Product Data")
    wh = dfs.get("Warehouse")
    tc = dfs.get("Transport Cost")

    if cpd is None or wh is None or tc is None:
        return {"status": "missing_data"}, {"note": "Required sheets missing", "flows": []}

    # Filter demand by period if present
    cpd_use = cpd.copy()
    if "Period" in cpd_use.columns:
        cpd_use = cpd_use[cpd_use["Period"] == period]
    if cpd_use.empty:
        return {"status": "no_demand"}, {"note": "No demand rows for selected period", "flows": []}

    # Build demand per (customer, product)
    dem = {}
    for _, row in cpd_use.iterrows():
        cust = str(row.get("Customer"))
        prod = str(row.get("Product"))
        qty = float(row.get("Demand", 0) or 0)
        if qty > 0:
            dem[(cust, prod)] = dem.get((cust, prod), 0.0) + qty
    if not dem:
        return {"status": "no_positive_demand"}, {"note": "All demands are non-positive", "flows": []}

    # Map warehouses to their location
    wh_locs = _get_warehouse_locations(wh)

    # Build admissible arcs with finite, non-negative cost per UOM (pick min across modes/duplicates)
    arcs: Dict[tuple, float] = {}  # (w, customer, product) -> cost
    for _, row in tc.iterrows():
        if int(row.get("Period", period)) != period:
            continue
        w_loc = row.get("From Location")
        to_loc = row.get("To Location")
        prod = row.get("Product")
        if pd.isna(w_loc) or pd.isna(to_loc) or pd.isna(prod):
            continue

        cost_val = _safe_cost(row.get("Cost Per UOM", 0.0))
        if cost_val is None:
            # skip lanes with NaN/inf/negative/missing cost
            continue

        # find warehouses whose Location equals From Location
        whs = [w for w, loc in wh_locs.items() if str(loc) == str(w_loc)]
        if not whs:
            continue

        for w in whs:
            key = (w, str(to_loc), str(prod))
            prev = arcs.get(key)
            arcs[key] = cost_val if prev is None else min(prev, cost_val)

    # Decision variables only for arcs that feed demanded (customer, product)
    x_vars = {
        (w, c, p): pulp.LpVariable(f"x_{w}_{c}_{p}", lowBound=0)
        for (w, c, p), cost in arcs.items()
        if (c, p) in dem
    }

    if not x_vars:
        return {
            "status": "no_feasible_arcs",
            "total_cost": None,
            "total_demand": sum(dem.values()),
            "served": 0.0,
            "service_pct": 0.0,
            "open_warehouses": int(sum(1 for _, r in wh.iterrows() if _warehouse_capacity(r) > 0)),
        }, {"note": "No admissible arcs with finite costs for the demanded pairs", "flows": [], "num_arcs": 0, "num_demands": len(dem)}

    # Model
    m = pulp.LpProblem("genie_network", pulp.LpMinimize)

    # Objective
    m += pulp.lpSum(arcs[(w, c, p)] * var for (w, c, p), var in x_vars.items())

    # Demand satisfaction
    for (c, p), qty in dem.items():
        incoming = [x_vars[(w, c, p)] for (w, cc, pp) in x_vars if cc == c and pp == p]
        if incoming:
            m += pulp.lpSum(incoming) >= qty, f"demand_{c}_{p}"

    # Warehouse capacity
    caps = {row["Warehouse"]: _warehouse_capacity(row) for _, row in wh.iterrows()}
    for w, cap in caps.items():
        outflow = [x_vars[(ww, c, p)] for (ww, c, p) in x_vars if ww == w]
        if outflow:
            m += pulp.lpSum(outflow) <= cap, f"cap_{w}"

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[m.status]

    total_cost = pulp.value(m.objective) if m.status == 1 else None
    total_demand = sum(dem.values())
    served = 0.0
    flows = []
    if m.status == 1:
        for (w, c, p), var in x_vars.items():
            val = var.value()
            if val and val > 1e-6:
                served += val
                flows.append({"warehouse": w, "customer": c, "product": p, "qty": float(val)})

    service_pct = (served / total_demand * 100.0) if total_demand > 0 else 0.0

    # Binding capacities (heuristic)
    binding_caps = []
    if m.status == 1:
        for w, cap in caps.items():
            outflow = sum(v.value() for (ww, c, p), v in x_vars.items() if ww == w) or 0.0
            if cap > 0 and abs(outflow - cap) <= 1e-6:
                binding_caps.append(w)

    kpis = {
        "status": status,
        "total_cost": total_cost,
        "total_demand": total_demand,
        "served": served,
        "service_pct": round(service_pct, 2),
        "open_warehouses": int(sum(1 for _, row in wh.iterrows() if _warehouse_capacity(row) > 0)),
    }
    diag = {"binding_warehouses": binding_caps[:3], "num_arcs": len(x_vars), "num_demands": len(dem), "flows": flows}
    return kpis, diag
