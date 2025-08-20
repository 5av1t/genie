from typing import Dict, Any, Tuple, List
import math
import pandas as pd
import pulp

DEFAULT_PERIOD = 2023

# ---------- helpers ----------

def _to_float(x, default: float = 0.0) -> float:
    """Coerce to finite float; return default on None/NaN/inf or conversion error."""
    try:
        val = float(x)
        if math.isfinite(val):
            return val
        return default
    except Exception:
        return default

def _to_int01(x, default: int = 0) -> int:
    """Coerce to 0/1 int; non-finite / errors => default."""
    try:
        val = int(x)
        return 1 if val != 0 else 0
    except Exception:
        return default

def _warehouse_capacity(row: pd.Series) -> float:
    """Usable capacity respecting Force Close / Available flags. Returns finite >= 0."""
    # availability flags; treat missing as available
    force_close = _to_int01(row.get("Force Close", 0), default=0)
    available_flag = _to_int01(row.get("Available (Warehouse)", row.get("Available", 1)), default=1)
    if force_close == 1 or available_flag == 0:
        return 0.0
    # capacity
    cap = _to_float(row.get("Maximum Capacity", 0), default=0.0)
    if cap < 0 or not math.isfinite(cap):
        return 0.0
    return cap

# ---------- main ----------

def run_optimizer(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simple MILP: warehouse → customer flows by product using Transport Cost (Cost Per UOM).
    - Nodes: warehouses (supply capacity), customers (demand)
    - Arcs: From Location (warehouse location) -> Customer
    - Capacity: Warehouse Maximum Capacity (sum over products, after availability & force-close)
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

    # Basic presence checks
    if not isinstance(wh, pd.DataFrame) or wh.empty:
        kpis["status"] = "no_warehouses"
        return kpis, diag
    if not isinstance(cpd, pd.DataFrame) or cpd.empty:
        kpis["status"] = "no_demand"
        return kpis, diag
    if not isinstance(tc, pd.DataFrame) or tc.empty:
        kpis["status"] = "no_transport_cost"
        return kpis, diag

    # Filter by period (best-effort)
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

    # Demand per (customer, product), keep only positive finite demand
    dem: Dict[Tuple[str, str], float] = {}
    for _, r in cpdp.iterrows():
        cust = str(r.get("Customer"))
        prod = str(r.get("Product"))
        dval = _to_float(r.get("Demand", 0), default=0.0)
        if dval > 0:
            dem[(cust, prod)] = dem.get((cust, prod), 0.0) + dval

    total_demand = sum(dem.values())
    kpis["total_demand"] = total_demand
    if total_demand <= 0:
        kpis["status"] = "no_positive_demand"
        return kpis, diag

    # Warehouse capacities (finite)
    cap: Dict[str, float] = {}
    for _, r in wh.iterrows():
        w = str(r.get("Warehouse"))
        cap[w] = cap.get(w, 0.0) + _warehouse_capacity(r)

    # Warehouse → Location mapping
    wloc: Dict[str, str] = {}
    for _, r in wh.iterrows():
        w = str(r.get("Warehouse"))
        loc = str(r.get("Location")) if pd.notna(r.get("Location")) else w
        if w not in wloc:
            wloc[w] = loc

    # Location → list of warehouses (inverse map)
    loc_to_ws: Dict[str, List[str]] = {}
    for w, loc in wloc.items():
        loc_to_ws.setdefault(str(loc), []).append(w)

    # Build arcs: (w, customer, product) with finite, non-negative cost & Available != 0
    arcs: Dict[Tuple[str, str, str], float] = {}
    for _, r in tcp.iterrows():
        # availability
        if _to_int01(r.get("Available", 1), default=1) == 0:
            continue
        # cost
        cost = _to_float(r.get("Cost Per UOM", 0), default=0.0)
        if cost < 0 or not math.isfinite(cost):
            continue
        from_loc = str(r.get("From Location"))
        to_cust  = str(r.get("To Location"))
        prod     = str(r.get("Product"))
        # map from location to all warehouses at that location
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

    # Demand satisfaction: sum_in <= demand
    for (c, p), d in dem.items():
        m += pulp.lpSum(x[(w, c, p)] for (w, cc, pp) in x.keys() if cc == c and pp == p) <= d, f"demand_{c}_{p}"

    # Warehouse capacity: sum_out <= finite capacity (treat invalid as 0)
    for w in set(wloc.keys()):
        cmax = _to_float(cap.get(w, 0.0), default=0.0)
        cmax = 0.0 if (not math.isfinite(cmax) or cmax < 0) else cmax
        m += pulp.lpSum(x[(ww, c, p)] for (ww, c, p) in x.keys() if ww == w) <= cmax, f"cap_{w}"

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract solution
    flows: List[Dict[str, Any]] = []
    served = 0.0
    total_cost = 0.0
    for (w, c, p), var in x.items():
        val = var.value() or 0.0
        if val > 1e-9:
            cost = _to_float(arcs[(w, c, p)], default=0.0)
            qty = _to_float(val, default=0.0)
            flows.append({"warehouse": w, "customer": c, "product": p, "qty": qty, "unit_cost": cost})
            served += qty
            total_cost += qty * cost

    # KPIs
    kpis["served"] = served
    kpis["total_cost"] = round(total_cost, 4) if served > 0 else None
    kpis["service_pct"] = round((served / total_demand) * 100.0, 2) if total_demand > 0 else 0.0
    kpis["open_warehouses"] = int(sum(1 for w, v in cap.items() if _to_float(v, 0.0) > 0))
    if served <= 0:
        kpis["status"] = "no_feasible_arcs"
    else:
        # status from solver
        try:
            kpis["status"] = pulp.LpStatus[m.status].lower()
        except Exception:
            pass

    # Diagnostics: binding capacities, lane summaries, throughput
    used_by_w: Dict[str, float] = {}
    for f in flows:
        used_by_w[f["warehouse"]] = used_by_w.get(f["warehouse"], 0.0) + _to_float(f["qty"], 0.0)

    binding_caps = []
    for w, cmax in cap.items():
        cmaxf = _to_float(cmax, 0.0)
        used = _to_float(used_by_w.get(w, 0.0), 0.0)
        if cmaxf > 0 and used >= 0.999 * cmaxf:
            binding_caps.append({"warehouse": w, "used": used, "capacity": cmaxf})

    # Lanes aggregated by (w, c)
    lane_flow: Dict[Tuple[str, str], float] = {}
    for f in flows:
        key = (f["warehouse"], f["customer"])
        lane_flow[key] = lane_flow.get(key, 0.0) + _to_float(f["qty"], 0.0)

    # Products by warehouse throughput
    wh_prod: Dict[str, Dict[str, float]] = {}
    for f in flows:
        w = f["warehouse"]; p = f["product"]
        qty = _to_float(f["qty"], 0.0)
        wh_prod.setdefault(w, {})
        wh_prod[w][p] = wh_prod[w].get(p, 0.0) + qty

    lane_flow_sorted = sorted(([w, c, q] for (w, c), q in lane_flow.items()), key=lambda x: x[2], reverse=True)
    wh_throughput_sorted = sorted(([w, q] for w, q in used_by_w.items()), key=lambda x: x[1])

    diag.update({
        "flows": flows,
        "binding_warehouses": binding_caps[:3],
        "lane_flow_top": lane_flow_sorted[:10],
        "warehouse_throughput": used_by_w,
        "warehouse_throughput_by_product": wh_prod,
        "lowest_throughput_warehouses": wh_throughput_sorted[:5],
        "cap_by_warehouse": {w: _to_float(v, 0.0) for w, v in cap.items()},
        "num_arcs": len(arcs),
    })

    return kpis, diag
