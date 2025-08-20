from typing import Dict, Any, Tuple, List
import math
import pandas as pd
import pulp

DEFAULT_PERIOD = 2023

def _warehouse_capacity(row: pd.Series) -> float:
    """Compute usable capacity respecting Force Close / Available flags."""
    available = row.get("Available (Warehouse)", row.get("Available", 1))
    try:
        if int(row.get("Force Close", 0) or 0) == 1 or int(available or 1) == 0:
            return 0.0
    except Exception:
        pass
    try:
        return float(row.get("Maximum Capacity", 0) or 0)
    except Exception:
        return 0.0

def run_optimizer(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simple MILP: warehouse → customer flows by product using Transport Cost (Cost Per UOM).
    - Nodes: warehouses (supply capacity), customers (demand)
    - Arcs: From Location (warehouse location) -> Customer
    - Capacity: Warehouse Maximum Capacity (sum over products)
    - Demand: from Customer Product Data
    - Objective: Min transport cost
    Returns:
      kpis: dict
      diag: dict with flows, binding caps, throughput, and arc counts
    """
    cpd = dfs.get("Customer Product Data")
    wh  = dfs.get("Warehouse")
    tc  = dfs.get("Transport Cost")

    kpis = {"status": "ok", "total_cost": None, "total_demand": 0, "served": 0, "service_pct": 0.0, "open_warehouses": 0}
    diag: Dict[str, Any] = {"flows": [], "binding_warehouses": [], "num_arcs": 0}

    # Basic checks
    if not isinstance(wh, pd.DataFrame) or wh.empty:
        kpis["status"] = "no_warehouses"
        return kpis, diag
    if not isinstance(cpd, pd.DataFrame) or cpd.empty:
        kpis["status"] = "no_demand"
        return kpis, diag
    if not isinstance(tc, pd.DataFrame) or tc.empty:
        kpis["status"] = "no_transport_cost"
        return kpis, diag

    # Filter by period
    cpdp = cpd.copy()
    if "Period" in cpdp.columns:
        try:
            cpdp = cpdp[cpdp["Period"] == period]
        except Exception:
            pass
    tcp = tc.copy()
    if "Period" in tcp.columns:
        try:
            tcp = tcp[tcp["Period"] == period]
        except Exception:
            pass

    # Demand > 0 per (customer, product)
    dem: Dict[Tuple[str, str], float] = {}
    for _, r in cpdp.iterrows():
        cust = str(r.get("Customer"))
        prod = str(r.get("Product"))
        try:
            d = float(r.get("Demand", 0) or 0)
        except Exception:
            d = 0.0
        if d > 0:
            dem[(cust, prod)] = dem.get((cust, prod), 0.0) + d
    total_demand = sum(dem.values())
    kpis["total_demand"] = total_demand
    if total_demand <= 0:
        kpis["status"] = "no_positive_demand"
        return kpis, diag

    # Warehouse capacities
    cap: Dict[str, float] = {}
    open_flags: Dict[str, int] = {}
    for _, r in wh.iterrows():
        w = str(r.get("Warehouse"))
        cap[w] = cap.get(w, 0.0) + _warehouse_capacity(r)
        try:
            open_flags[w] = 1 if int(r.get("Force Close", 0) or 0) == 0 and int(r.get("Available (Warehouse)", r.get("Available", 1)) or 1) != 0 else 0
        except Exception:
            open_flags[w] = 1

    # From Location (warehouse location)
    wloc: Dict[str, str] = {}
    for _, r in wh.iterrows():
        w = str(r.get("Warehouse"))
        loc = str(r.get("Location")) if pd.notna(r.get("Location")) else w
        if w not in wloc:
            wloc[w] = loc

    # Build arcs: For each lane with Available and finite Cost Per UOM, connect warehouse (by location match) to customer
    # We need inverse map: location -> warehouses
    loc_to_ws: Dict[str, List[str]] = {}
    for w, loc in wloc.items():
        loc_to_ws.setdefault(str(loc), []).append(w)

    arcs: Dict[Tuple[str, str, str], float] = {}  # (w, c, p) -> cost per unit
    for _, r in tcp.iterrows():
        try:
            if int(r.get("Available", 1) or 1) == 0:
                continue
        except Exception:
            pass
        try:
            cost = float(r.get("Cost Per UOM", 0))
            if math.isinf(cost) or math.isnan(cost):
                continue
        except Exception:
            continue
        from_loc = str(r.get("From Location"))
        to_cust  = str(r.get("To Location"))
        prod     = str(r.get("Product"))
        # map from_loc to all warehouses sharing that location
        for w in loc_to_ws.get(from_loc, []):
            arcs[(w, to_cust, prod)] = cost

    diag["num_arcs"] = len(arcs)

    if len(arcs) == 0:
        kpis["status"] = "no_feasible_arcs"
        return kpis, diag

    # Variables
    m = pulp.LpProblem("SimpleNetworkFlow", pulp.LpMinimize)
    x: Dict[Tuple[str, str, str], pulp.LpVariable] = {}
    for key in arcs.keys():
        x[key] = pulp.LpVariable(f"flow::{key[0]}->{key[1]}::{key[2]}", lowBound=0)

    # Objective
    m += pulp.lpSum(arcs[k] * var for k, var in x.items())

    # Demand satisfaction (<= demand)
    for (c, p), d in dem.items():
        m += pulp.lpSum(x[(w, c, p)] for (w, cc, pp) in x.keys() if cc == c and pp == p) <= d, f"demand_{c}_{p}"

    # Warehouse capacity (sum over all outgoing flows by product)
    for w, cmax in cap.items():
        if cmax <= 0:
            # Force zero outbound if closed or zero capacity
            m += pulp.lpSum(x[(ww, c, p)] for (ww, c, p) in x.keys() if ww == w) <= 0, f"cap_{w}"
        else:
            m += pulp.lpSum(x[(ww, c, p)] for (ww, c, p) in x.keys() if ww == w) <= cmax, f"cap_{w}"

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[m.status] not in {"Optimal", "Not Solved", "Infeasible", "Unbounded"}:
        kpis["status"] = pulp.LpStatus[m.status].lower()

    # Extract solution
    flows: List[Dict[str, Any]] = []
    served = 0.0
    total_cost = 0.0
    for (w, c, p), var in x.items():
        val = var.value() or 0.0
        if val > 1e-9:
            flows.append({"warehouse": w, "customer": c, "product": p, "qty": float(val), "unit_cost": float(arcs[(w, c, p)])})
            served += val
            total_cost += val * float(arcs[(w, c, p)])

    kpis["served"] = served
    kpis["total_cost"] = round(total_cost, 4) if served > 0 else None
    kpis["service_pct"] = round((served / total_demand) * 100.0, 2) if total_demand > 0 else 0.0
    kpis["open_warehouses"] = int(sum(1 for _, row in wh.iterrows() if _warehouse_capacity(row) > 0))

    if served <= 0:
        kpis["status"] = "no_feasible_arcs"

    # Diagnostics: binding warehouses (shadow: if outbound equals capacity), throughput, lanes by flow
    # Capacity used by warehouse
    used_by_w: Dict[str, float] = {}
    for f in flows:
        used_by_w[f["warehouse"]] = used_by_w.get(f["warehouse"], 0.0) + float(f["qty"])

    # Binding: near capacity with tolerance
    binding_caps = []
    for w, cmax in cap.items():
        used = used_by_w.get(w, 0.0)
        if cmax > 0 and used >= 0.999 * cmax:
            binding_caps.append({"warehouse": w, "used": used, "capacity": cmax})

    # Lanes aggregated by (warehouse, customer)
    lane_flow: Dict[Tuple[str, str], float] = {}
    for f in flows:
        key = (f["warehouse"], f["customer"])
        lane_flow[key] = lane_flow.get(key, 0.0) + float(f["qty"])

    # Products by warehouse throughput
    wh_prod: Dict[str, Dict[str, float]] = {}
    for f in flows:
        wh_prod.setdefault(f["warehouse"], {})
        wh_prod[f["warehouse"]][f["product"]] = wh_prod[f["warehouse"]].get(f["product"], 0.0) + float(f["qty"])

    # Sorted lists
    lane_flow_sorted = sorted(([w, c, q] for (w, c), q in lane_flow.items()), key=lambda x: x[2], reverse=True)
    wh_throughput_sorted = sorted(([w, q] for w, q in used_by_w.items()), key=lambda x: x[1])

    diag.update({
        "flows": flows,
        "binding_warehouses": binding_caps[:3],
        "lane_flow_top": lane_flow_sorted[:10],
        "warehouse_throughput": used_by_w,
        "warehouse_throughput_by_product": wh_prod,
        "lowest_throughput_warehouses": wh_throughput_sorted[:5],
        "cap_by_warehouse": cap,
    })

    return kpis, diag
