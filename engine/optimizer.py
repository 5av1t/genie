from typing import Dict, Any, Tuple, List, Iterable
import math
import pandas as pd
import pulp

DEFAULT_PERIOD = 2023
UNMET_PENALTY = 1_000_000.0  # large penalty to strongly prefer serving demand

# ----------------- numeric coercion helpers -----------------

def _to_float(x, default: float = 0.0) -> float:
    """Coerce to finite float; return default on None/NaN/inf or conversion error."""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _to_int01(x, default: int = 0) -> int:
    """Coerce to 0/1 int; non-finite / errors => default."""
    try:
        v = int(x)
        return 1 if v != 0 else 0
    except Exception:
        return default

# ----------------- capacity helper -----------------

def _warehouse_capacity(row: pd.Series) -> float:
    """Usable capacity respecting Force Close / Available flags. Returns finite >= 0."""
    force_close = _to_int01(row.get("Force Close", 0), default=0)
    available_flag = _to_int01(row.get("Available (Warehouse)", row.get("Available", 1)), default=1)
    if force_close == 1 or available_flag == 0:
        return 0.0
    cap = _to_float(row.get("Maximum Capacity", 0), default=0.0)
    return cap if cap > 0 else 0.0

# ----------------- flexible matching utilities -----------------

def _candidate_warehouses(from_loc: str, wh_df: pd.DataFrame) -> List[str]:
    """
    Match a TC 'From Location' to warehouses by either exact warehouse name OR warehouse Location.
    """
    if not isinstance(wh_df, pd.DataFrame) or wh_df.empty:
        return []
    from_loc = str(from_loc)
    cans = set()
    for _, r in wh_df.iterrows():
        wname = str(r.get("Warehouse"))
        wloc  = str(r.get("Location")) if pd.notna(r.get("Location")) else wname
        if from_loc == wname or from_loc == wloc:
            cans.add(wname)
    return list(cans)

def _customer_location_map(customers_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build map: location -> [customer names].
    If Customers sheet missing/empty, returns {}.
    """
    locmap: Dict[str, List[str]] = {}
    if not isinstance(customers_df, pd.DataFrame) or customers_df.empty:
        return locmap
    for _, r in customers_df.iterrows():
        cname = str(r.get("Customer"))
        cloc  = str(r.get("Location")) if pd.notna(r.get("Location")) else cname
        locmap.setdefault(cloc, []).append(cname)
    return locmap

def _candidate_customers(to_loc: str, customers_df: pd.DataFrame) -> List[str]:
    """
    Match a TC 'To Location' to customers by either exact customer name OR their Location.
    Fan-out if multiple customers share the same location.
    """
    to_loc = str(to_loc)
    out = set()
    if not isinstance(customers_df, pd.DataFrame) or customers_df.empty:
        # If no Customers sheet, best effort: assume To is the customer name.
        out.add(to_loc)
        return list(out)
    # exact customer match
    for _, r in customers_df.iterrows():
        cname = str(r.get("Customer"))
        if cname == to_loc:
            out.add(cname)
    # location-based match
    locmap = _customer_location_map(customers_df)
    for cust in locmap.get(to_loc, []):
        out.add(cust)
    return list(out)

# ----------------- optimizer -----------------

def run_optimizer(dfs: Dict[str, pd.DataFrame], period: int = DEFAULT_PERIOD) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simple MILP: warehouse → customer flows by product using Transport Cost.
    Flexible arc building rules:
      - From Location may equal warehouse name OR warehouse Location
      - To Location may equal customer name OR Customers.Location (fan-out)
      - Product blank in TC means 'applies to all demanded products'
      - Period blank in TC matches current period
      - Cost Per UOM may be computed as Retrieve Distance × Cost per Distance if missing

    Demand is enforced with unmet variables:
      sum_in(w->c,p) + u[c,p] = demand[c,p],  u[c,p] >= 0
      Objective includes UNMET_PENALTY * u[c,p], making service preferred whenever feasible.

    Returns:
      kpis: dict with status/served/total_cost/service_pct/open_warehouses/total_demand
      diag: dict with flows, binding caps, lane summaries, throughput, cap_by_warehouse, num_arcs, penalty_cost
    """
    cpd = dfs.get("Customer Product Data")
    wh  = dfs.get("Warehouse")
    tc  = dfs.get("Transport Cost")
    customers = dfs.get("Customers")

    kpis = {"status": "ok", "total_cost": None, "total_demand": 0, "served": 0, "service_pct": 0.0, "open_warehouses": 0}
    diag: Dict[str, Any] = {"flows": [], "binding_warehouses": [], "num_arcs": 0, "penalty_cost": 0.0}

    # Presence checks
    if not isinstance(wh, pd.DataFrame) or wh.empty:
        kpis["status"] = "no_warehouses"; return kpis, diag
    if not isinstance(cpd, pd.DataFrame) or cpd.empty:
        kpis["status"] = "no_demand"; return kpis, diag
    if not isinstance(tc, pd.DataFrame) or tc.empty:
        kpis["status"] = "no_transport_cost"; return kpis, diag

    # Demand by (customer, product) for the given period (or all if Period missing)
    cpdp = cpd.copy()
    if "Period" in cpdp.columns:
        try:
            cpdp = cpdp[cpdp["Period"] == period]
        except Exception:
            pass
    dem: Dict[Tuple[str, str], float] = {}
    products_with_demand: set = set()
    for _, r in cpdp.iterrows():
        cust = str(r.get("Customer"))
        prod = str(r.get("Product"))
        qty  = _to_float(r.get("Demand", 0), default=0.0)
        if qty > 0:
            dem[(cust, prod)] = dem.get((cust, prod), 0.0) + qty
            products_with_demand.add(prod)
    total_demand = sum(dem.values())
    kpis["total_demand"] = total_demand
    if total_demand <= 0:
        kpis["status"] = "no_positive_demand"; return kpis, diag

    # Warehouse capacities
    cap: Dict[str, float] = {}
    for _, r in wh.iterrows():
        w = str(r.get("Warehouse"))
        cap[w] = cap.get(w, 0.0) + _warehouse_capacity(r)

    # Build arcs (warehouse, customer, product) with flexible matching
    arcs: Dict[Tuple[str, str, str], float] = {}
    tcp = tc.copy()

    def _row_matches_period(row) -> bool:
        if "Period" not in row or pd.isna(row["Period"]) or str(row["Period"]).strip() == "":
            return True  # blank period accepts current period
        try:
            return int(row["Period"]) == int(period)
        except Exception:
            return False

    for _, r in tcp.iterrows():
        # period rule
        if not _row_matches_period(r):
            continue
        # availability (missing => available)
        if _to_int01(r.get("Available", 1), default=1) == 0:
            continue
        # cost per uom (fallback to distance × cost_per_distance)
        cpu = _to_float(r.get("Cost Per UOM", None), default=float("nan"))
        if not math.isfinite(cpu) or cpu < 0:
            dist = _to_float(r.get("Retrieve Distance", None), default=float("nan"))
            cpdst = _to_float(r.get("Cost per Distance", None), default=float("nan"))
            if math.isfinite(dist) and math.isfinite(cpdst) and dist >= 0 and cpdst >= 0:
                cpu = dist * cpdst
            else:
                cpu = float("nan")
        if not math.isfinite(cpu) or cpu < 0:
            continue

        from_loc = str(r.get("From Location"))
        to_loc   = str(r.get("To Location"))
        tc_prod  = str(r.get("Product")) if pd.notna(r.get("Product")) else ""

        w_cands = _candidate_warehouses(from_loc, wh)
        if not w_cands:
            continue

        c_cands = _candidate_customers(to_loc, customers) if isinstance(customers, pd.DataFrame) else [to_loc]
        if not c_cands:
            continue

        # product logic: blank product in TC applies to all products with demand
        if tc_prod.strip() == "" or tc_prod.lower() == "nan":
            prod_list: Iterable[str] = products_with_demand
        else:
            prod_list = [tc_prod]

        for w in w_cands:
            for c in c_cands:
                for p in prod_list:
                    # only create arc if (c,p) has demand in CPD
                    if (c, p) in dem:
                        arcs[(w, c, p)] = _to_float(cpu, default=0.0)

    diag["num_arcs"] = len(arcs)
    if len(arcs) == 0:
        kpis["status"] = "no_feasible_arcs"; return kpis, diag

    # Variables
    m = pulp.LpProblem("SimpleNetworkFlow", pulp.LpMinimize)
    x: Dict[Tuple[str, str, str], pulp.LpVariable] = {k: pulp.LpVariable(f"flow::{k[0]}->{k[1]}::{k[2]}", lowBound=0) for k in arcs.keys()}
    u: Dict[Tuple[str, str], pulp.LpVariable] = {k: pulp.LpVariable(f"unmet::{k[0]}::{k[1]}", lowBound=0) for k in dem.keys()}  # unmet by (customer,product)

    # Objective = transport cost + unmet penalty
    transport_cost = pulp.lpSum(arcs[k] * var for k, var in x.items())
    penalty_cost   = pulp.lpSum(UNMET_PENALTY * var for var in u.values())
    m += transport_cost + penalty_cost

    # Demand satisfaction with unmet: sum_in + u = demand
    for (c, p), d in dem.items():
        m += pulp.lpSum(x[(w, c, p)] for (w, cc, pp) in x.keys() if cc == c and pp == p) + u[(c, p)] == _to_float(d, 0.0), f"demand_{c}_{p}"

    # Warehouse capacity: sum_out <= capacity
    for w in set(k[0] for k in x.keys()):
        cmax = _to_float(cap.get(w, 0.0), 0.0)
        cmax = 0.0 if (not math.isfinite(cmax) or cmax < 0) else cmax
        m += pulp.lpSum(x[(ww, c, p)] for (ww, c, p) in x.keys() if ww == w) <= cmax, f"cap_{w}"

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    # Results
    flows: List[Dict[str, Any]] = []
    served = 0.0
    trans_cost_val = 0.0
    for (w, c, p), var in x.items():
        val = _to_float(var.value(), 0.0)
        if val > 1e-9:
            unit = _to_float(arcs[(w, c, p)], 0.0)
            flows.append({"warehouse": w, "customer": c, "product": p, "qty": val, "unit_cost": unit})
            served += val
            trans_cost_val += val * unit

    unmet_total = sum(_to_float(var.value(), 0.0) for var in u.values())
    penalty_val = unmet_total * UNMET_PENALTY

    # KPIs
    total_demand = kpis["total_demand"]
    kpis["served"] = served
    kpis["service_pct"] = round((served / total_demand) * 100.0, 2) if total_demand > 0 else 0.0
    kpis["open_warehouses"] = int(sum(1 for w, v in cap.items() if _to_float(v, 0.0) > 0))
    kpis["total_cost"] = round(trans_cost_val, 4) if served > 0 else 0.0  # transport only
    try:
        kpis["status"] = pulp.LpStatus[m.status].lower()
    except Exception:
        pass

    # Diagnostics
    used_by_w: Dict[str, float] = {}
    for f in flows:
        used_by_w[f["warehouse"]] = used_by_w.get(f["warehouse"], 0.0) + _to_float(f["qty"], 0.0)

    binding_caps = []
    for w, cmax in cap.items():
        cmaxf = _to_float(cmax, 0.0)
        used  = _to_float(used_by_w.get(w, 0.0), 0.0)
        if cmaxf > 0 and used >= 0.999 * cmaxf:
            binding_caps.append({"warehouse": w, "used": used, "capacity": cmaxf})

    lane_flow: Dict[Tuple[str, str], float] = {}
    wh_prod: Dict[str, Dict[str, float]] = {}
    for f in flows:
        w, c, p = f["warehouse"], f["customer"], f["product"]
        q = _to_float(f["qty"], 0.0)
        lane_flow[(w, c)] = lane_flow.get((w, c), 0.0) + q
        wh_prod.setdefault(w, {}); wh_prod[w][p] = wh_prod[w].get(p, 0.0) + q

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
        "unmet_total": unmet_total,
        "penalty_cost": penalty_val,
    })

    return kpis, diag
