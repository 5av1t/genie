from typing import Dict, Any, List

def _bulletize(items: List[str]) -> str:
    return "\n".join([f"- {x}" for x in items]) if items else "-"

def build_summary(prompt: str, scenario: Dict[str, Any], kpis: Dict[str, Any], diag: Dict[str, Any]) -> str:
    period = scenario.get("period", 2023)
    status = kpis.get("status", "unknown")
    cost = kpis.get("total_cost", None)
    dem  = kpis.get("total_demand", 0)
    srv  = kpis.get("served", 0)
    pct  = kpis.get("service_pct", 0)
    open_wh = kpis.get("open_warehouses", 0)
    bindings = diag.get("binding_warehouses", [])

    changes = []
    for d in scenario.get("demand_updates", []):
        msg = f"Demand: {d.get('product')} at {d.get('location')} Δ {d.get('delta_pct', 0)}%"
        if d.get("set"): msg += f" (set {d['set']})"
        changes.append(msg)
    for w in scenario.get("warehouse_changes", []):
        changes.append(f"Warehouse: {w.get('warehouse')} → {w.get('field')} = {w.get('new_value')}")
    for s in scenario.get("supplier_changes", []):
        changes.append(f"Supplier: {s.get('supplier')} {s.get('product')} → {s.get('field')} = {s.get('new_value')}")
    for t in scenario.get("transport_updates", []):
        changes.append(f"Transport: {t.get('mode')} {t.get('from_location')}→{t.get('to_location')} ({t.get('product')}) {t.get('fields')}")

    adds = scenario.get("adds", {})
    for c in adds.get("customers", []):
        changes.append(f"Add customer {c.get('customer')} at {c.get('location')}")
    for d in adds.get("customer_demands", []):
        changes.append(f"Add demand {d.get('demand')} of {d.get('product')} at {d.get('customer')} (period {d.get('period')})")
    for w in adds.get("warehouses", []):
        changes.append(f"Add warehouse {w.get('warehouse')} at {w.get('location')} {w.get('fields')}")
    for sp in adds.get("supplier_products", []):
        changes.append(f"Add supplier product {sp.get('product')} at {sp.get('supplier')} (Available={sp.get('fields',{}).get('Available')})")
    for tl in adds.get("transport_lanes", []):
        changes.append(f"Add lane {tl.get('mode')} {tl.get('from_location')}→{tl.get('to_location')} ({tl.get('product')})")

    dels = scenario.get("deletes", {})
    for c in dels.get("customers", []):
        changes.append(f"Delete customer {c.get('customer')}")
    for d in dels.get("customer_product_rows", []):
        changes.append(f"Delete demand row {d.get('product')} at {d.get('customer')} (period {d.get('period')})")
    for w in dels.get("warehouses", []):
        changes.append(f"Delete warehouse {w.get('warehouse')}")
    for sp in dels.get("supplier_products", []):
        changes.append(f"Delete supplier product {sp.get('product')} at {sp.get('supplier')}")
    for tl in dels.get("transport_lanes", []):
        changes.append(f"Delete lane {tl.get('mode')} {tl.get('from_location')}→{tl.get('to_location')} ({tl.get('product')})")

    bullet_changes = _bulletize(changes)

    lines = [
        f"**Scenario period**: {period}",
        f"**Status**: `{status}`",
        f"**Total demand**: {dem:,}",
        f"**Served**: {srv:,}  (**Service**: {pct:.2f}%)",
        f"**Open warehouses**: {open_wh}",
    ]
    if cost is not None:
        lines.insert(2, f"**Total cost**: {cost:,.2f}")
    if bindings:
        lines.append(f"**Binding capacity at**: {', '.join(bindings)}")

    rec = []
    if status in {"no_feasible_arcs"}:
        rec.append("Enable Auto‑connect or add Transport Cost lanes with finite Cost Per UOM.")
    if pct < 100 and dem > 0:
        rec.append("Consider raising capacities at binding warehouses or adding cheaper lanes near unmet demand.")

    summary = f"""
### Intent (parsed)
{bullet_changes}

### KPIs
{_bulletize(lines)}

### Recommendations
{_bulletize(rec)}
"""
    return summary
