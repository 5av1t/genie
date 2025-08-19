from typing import Dict, Any, Tuple
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

def run_optimizer(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Simple MILP: warehouse → customer flows by product using Transport Cost."""
    cpd = dfs.get("Customer Product Data")
    wh = dfs.get("Warehouse")
    tc = dfs.get("Transport Cost")

    if cpd is None or wh is None or tc is None:
        return {"status": "missing_data"}, {"note": "Required sheets missing", "flows": []}

    cpd_use = cpd.copy()
    if "Period" in cpd_use.columns:
        cpd_use = cpd_use[cpd_use["Period"] == period]
    if cpd_use.empty:
        return {"status": "no_demand"}, {"note": "No demand rows for selected period", "flows": []}

    wh_locs = _get_warehouse_locations(wh)

    # Build admissible arcs with cost per UOM (pick min across modes)
    arcs = {}  # (w, customer, product) -> cost
    for _, row in tc.iterrows():
        if int(row.get("Period", period)) != period:
            continue
        w_loc = row.get("From Location")
        to_loc = row.get("To Location")
        prod = row.get("Product")
        if pd.isna(w_loc) or pd.isna(to_loc) or pd.isna(prod):
            continue
        whs = [w for w, loc in wh_locs.items() if str(loc) == str(w_loc)]
        if not whs:
            continue
        cost = float(row.get("Cost Per UOM", 0.0) or 0.0)
        for w in whs:
            key = (w, str(to_loc), str(prod))
            arcs[key] = min(arcs.get(key, cost), cost)

    # Demand per (customer, product)   NOTE: assumes Customer name == To Location
    dem = {}
    for _, row in cpd_use.iterrows():
        cust = str(row.get("Customer"))
        prod = str(row.get("Product"))
        qty = float(row.get("Demand", 0) or 0)
        if qty > 0:
            dem[(cust, prod)] = dem.get((cust, prod), 0.0) + qty
    if not dem:
        return {"status": "no_positive_demand"}, {"note": "All demands are non-positive", "flows": []}

    # Model
    m = pulp.LpProblem("genie_network", pulp.LpMinimize)

    x = {
        (w, c, p): pulp.LpVariable(f"x_{w}_{c}_{p}", lowBound=0)
        for (w, c, p) in arcs.keys()
        if (c, p) in dem
    }

    # Objective
    m += pulp.lpSum(arcs[(w, c, p)] * var for (w, c, p), var in x.items())

    # Demand satisfaction
    for (c, p), qty in dem.items():
        incoming = [x[(w, c, p)] for (w, cc, pp) in x if cc == c and pp == p]
        if incoming:
            m += pulp.lpSum(incoming) >= qty, f"demand_{c}_{p}"

    # Warehouse capacity
    caps = {row["Warehouse"]: _warehouse_capacity(row) for _, row in wh.iterrows()}
    for w, cap in caps.items():
        outflow = [x[(ww, c, p)] for (ww, c, p) in x if ww == w]
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
        for (w, c, p), var in x.items():
            val = var.value()
            if val and val > 1e-6:
                served += val
                flows.append({"warehouse": w, "customer": c, "product": p, "qty": float(val)})

    service_pct = (served / total_demand * 100.0) if total_demand > 0 and served is not None else 0.0

    binding_caps = []
    if m.status == 1:
        for w, cap in caps.items():
            outflow = sum(var.value() for (ww, c, p), var in x.items() if ww == w)
            if outflow is not None and abs(outflow - cap) <= 1e-6 and cap > 0:
                binding_caps.append(w)

    kpis = {
        "status": status,
        "total_cost": total_cost,
        "total_demand": total_demand,
        "served": served,
        "service_pct": round(service_pct, 2),
        "open_warehouses": int(sum(1 for _, row in wh.iterrows() if _warehouse_capacity(row) > 0)),
    }
    diag = {"binding_warehouses": binding_caps[:3], "num_arcs": len(x), "num_demands": len(dem), "flows": flows}
    return kpis, diag
