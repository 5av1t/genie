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
