from typing import Dict, Any

def build_summary(prompt: str, scenario: Dict[str, Any], kpis: Dict[str, Any], diag: Dict[str, Any]) -> str:
    lines = []
    lines.append("### Scenario Summary")
    if prompt:
        lines.append(f"**Intent:** {prompt}")
    lines.append("")
    lines.append("**Key KPIs**")
    lines.append(f"- Status: `{kpis.get('status')}`")
    if kpis.get("total_cost") is not None:
        lines.append(f"- Total Cost: {kpis['total_cost']:.2f}")
    lines.append(f"- Total Demand: {kpis.get('total_demand', 0)}")
    lines.append(f"- Served: {kpis.get('served', 0)}")
    lines.append(f"- Service Level: {kpis.get('service_pct', 0)}%")
    lines.append(f"- Open Warehouses: {kpis.get('open_warehouses', 0)}")
    lines.append("")
    du = scenario.get("demand_updates", []); wc = scenario.get("warehouse_changes", [])
    sc = scenario.get("supplier_changes", []); tu = scenario.get("transport_updates", [])
    if any([du, wc, sc, tu]):
        lines.append("**Applied Changes**")
        for d in du[:5]:
            lines.append(f"- Demand: {d['product']} at {d['location']} Δ {d.get('delta_pct', 0)}%")
        for w in wc[:5]:
            lines.append(f"- Warehouse: {w['warehouse']} → {w['field']} = {w['new_value']}")
        for s in sc[:5]:
            lines.append(f"- Supplier: {s['supplier']} {s['product']} → {s['field']} = {s['new_value']}")
        for t in tu[:5]:
            lines.append(f"- Transport: {t['mode']} {t['from_location']}→{t['to_location']} ({t['product']}) fields {t['fields']}")
        lines.append("")
    if diag:
        lines.append("**Diagnostics**")
        if diag.get("binding_warehouses"):
            lines.append(f"- Binding capacities at: {', '.join(diag['binding_warehouses'])}")
        lines.append(f"- Model size: {diag.get('num_arcs', 0)} arcs, {diag.get('num_demands', 0)} demand pairs")
    return "\n".join(lines)
