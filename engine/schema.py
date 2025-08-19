from typing import Any, Dict

DEFAULT_PERIOD = 2023

def empty_scenario(period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    return {
        "period": period,
        "demand_updates": [],
        "warehouse_changes": [],
        "supplier_changes": [],
        "transport_updates": [],
    }

def validate_scenario(s: Dict[str, Any]) -> Dict[str, Any]:
    if "period" not in s:
        s["period"] = DEFAULT_PERIOD
    for k in ["demand_updates", "warehouse_changes", "supplier_changes", "transport_updates"]:
        s.setdefault(k, [])
        if not isinstance(s[k], list):
            raise ValueError(f"Scenario field '{k}' must be a list")
    return s
