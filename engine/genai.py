# engine/genai.py
from typing import Dict, Any, List, Set
import os
import json
import re

# We support OpenAI's python client v1.x (already in requirements)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DEFAULT_PERIOD = 2023


def _vals(df, col) -> List[str]:
    if df is None or col not in df.columns:
        return []
    return sorted({str(x) for x in df[col].dropna().tolist()})


def build_catalog(dfs: Dict[str, Any]) -> Dict[str, List[str]]:
    """Collect exact names to constrain the LLM's output."""
    products = _vals(dfs.get("Products"), "Product")
    customers = _vals(dfs.get("Customers"), "Customer")
    warehouses = _vals(dfs.get("Warehouse"), "Warehouse")
    suppliers = _vals(dfs.get("Supplier"), "Supplier")
    modes = _vals(dfs.get("Mode of Transport"), "Mode of Transport")
    locations = sorted(
        set(_vals(dfs.get("Warehouse"), "Location")) |
        set(_vals(dfs.get("Customers"), "Location")) |
        set(customers) | set(warehouses) | set(suppliers)
    )
    return {
        "products": products,
        "customers": customers,
        "warehouses": warehouses,
        "suppliers": suppliers,
        "modes": modes,
        "locations": locations,
    }


def scenario_schema_json() -> Dict[str, Any]:
    """JSON Schema used to validate and coerce the LLM output."""
    return {
        "type": "object",
        "properties": {
            "period": {"type": "integer"},
            "demand_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["product", "customer", "location"],
                    "properties": {
                        "product": {"type": "string"},
                        "customer": {"type": "string"},
                        "location": {"type": "string"},
                        "delta_pct": {"type": "number"},
                        "set": {
                            "type": "object",
                            "properties": {
                                "Lead Time": {"type": "number"},
                                "Variable Cost": {"type": "number"},
                            },
                            "additionalProperties": True
                        }
                    },
                    "additionalProperties": True
                }
            },
            "warehouse_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["warehouse", "field", "new_value"],
                    "properties": {
                        "warehouse": {"type": "string"},
                        "field": {"type": "string"},
                        "new_value": {}
                    },
                    "additionalProperties": False
                }
            },
            "supplier_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["product", "supplier", "location", "field", "new_value"],
                    "properties": {
                        "product": {"type": "string"},
                        "supplier": {"type": "string"},
                        "location": {"type": "string"},
                        "field": {"type": "string"},
                        "new_value": {}
                    },
                    "additionalProperties": False
                }
            },
            "transport_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["mode", "product", "from_location", "to_location", "fields"],
                    "properties": {
                        "mode": {"type": "string"},
                        "product": {"type": "string"},
                        "from_location": {"type": "string"},
                        "to_location": {"type": "string"},
                        "period": {"type": "integer"},
                        "fields": {"type": "object", "additionalProperties": True}
                    },
                    "additionalProperties": False
                }
            }
        },
        "required": ["period", "demand_updates", "warehouse_changes", "supplier_changes", "transport_updates"],
        "additionalProperties": False
    }


def _client():
    """Return OpenAI client or None if not configured."""
    if OpenAI is None:
        return None
    # Prefer Streamlit secrets if present
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # Streamlit secrets env var is also exposed as env for Streamlit Cloud
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def _default_scenario(period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    return {
        "period": period,
        "demand_updates": [],
        "warehouse_changes": [],
        "supplier_changes": [],
        "transport_updates": [],
    }


def summarize_scenario(s: Dict[str, Any]) -> List[str]:
    """Human-readable bullets for the UI."""
    bullets: List[str] = []
    for d in s.get("demand_updates", []):
        msg = f"Demand: {d.get('product')} at {d.get('location')} "
        dp = d.get("delta_pct")
        if dp is not None:
            msg += f"Δ {dp}%"
        if d.get("set"):
            sets = "; ".join([f"{k}={v}" for k, v in d["set"].items()])
            msg += f" (set {sets})"
        bullets.append(msg)
    for w in s.get("warehouse_changes", []):
        bullets.append(f"Warehouse: {w.get('warehouse')} → {w.get('field')} = {w.get('new_value')}")
    for sp in s.get("supplier_changes", []):
        bullets.append(f"Supplier: {sp.get('supplier')} {sp.get('product')} → {sp.get('field')} = {sp.get('new_value')}")
    for t in s.get("transport_updates", []):
        bullets.append(f"Transport: {t.get('mode')} {t.get('from_location')}→{t.get('to_location')} ({t.get('product')}) fields {t.get('fields')}")
    if not bullets:
        bullets.append("No actionable changes detected.")
    return bullets


def parse_with_llm(prompt: str, dfs: Dict[str, Any], default_period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    """
    Use GenAI to produce scenario JSON. Falls back to empty scenario if unavailable.
    """
    client = _client()
    if client is None:
        return _default_scenario(default_period)

    catalog = build_catalog(dfs)

    system = (
        "You convert natural language supply-chain what-if requests into a strict JSON that updates Excel sheets.\n"
        "Return ONLY valid JSON matching the provided schema. Use EXACT entity names from the allowed lists.\n"
        f"Default period is {default_period} unless the user specifies another year.\n"
    )
    schema = scenario_schema_json()

    # Give the LLM the allowed enumerations to reduce hallucinations
    guide = {
        "allowed_entities": catalog,
        "notes": [
            "Use exact case/spelling.",
            "Customer name equals delivery node for demand.",
            "From Location must equal a Warehouse.Location.",
            "To Location must equal a Customer.",
            "When setting transport fields, include {'Available': 1} when enabling lanes.",
        ],
    }

    try:
        # Use the responses API with JSON schema (works on latest OpenAI python client)
        resp = client.responses.create(
            model="gpt-4o-mini",
            temperature=0,
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Instruction:\n{prompt}"},
                    {"type": "text", "text": "Constraints & Allowed Entities:\n" + json.dumps(guide)}
                ]},
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "scenario_schema", "schema": schema}},
        )
        txt = resp.output[0].content[0].text  # JSON string
        scenario = json.loads(txt)
        # Ensure period default if missing
        scenario.setdefault("period", default_period)
        for k in ["demand_updates", "warehouse_changes", "supplier_changes", "transport_updates"]:
            scenario.setdefault(k, [])
        return scenario
    except Exception:
        # Fail closed: return empty scenario so app continues
        return _default_scenario(default_period)
