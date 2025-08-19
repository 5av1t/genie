from typing import Dict, Any, List
import os, json

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DEFAULT_PERIOD = 2023

def _vals(df, col) -> List[str]:
    if df is None or col not in df.columns:
        return []
    return sorted({str(x) for x in df[col].dropna().tolist()})

def build_catalog(dfs) -> Dict[str, List[str]]:
    products  = _vals(dfs.get("Products"), "Product")
    customers = _vals(dfs.get("Customers"), "Customer")
    warehouses= _vals(dfs.get("Warehouse"), "Warehouse")
    suppliers = _vals(dfs.get("Supplier Product"), "Supplier")
    modes     = _vals(dfs.get("Mode of Transport"), "Mode of Transport")
    locations = sorted(
        set(_vals(dfs.get("Warehouse"), "Location")) |
        set(_vals(dfs.get("Customers"), "Location")) |
        set(customers) | set(warehouses) | set(suppliers)
    )
    return {"products": products, "customers": customers, "warehouses": warehouses, "suppliers": suppliers, "modes": modes, "locations": locations}

def scenario_schema_json() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "period": {"type": "integer"},
            "demand_updates": {"type":"array","items":{"type":"object","properties":{
                "product":{"type":"string"},"customer":{"type":"string"},"location":{"type":"string"},
                "delta_pct":{"type":"number"},"set":{"type":"object","additionalProperties":True}
            },"required":["product","customer","location"],"additionalProperties":True}},
            "warehouse_changes":{"type":"array","items":{"type":"object","properties":{
                "warehouse":{"type":"string"},"field":{"type":"string"},"new_value":{}
            },"required":["warehouse","field","new_value"],"additionalProperties":False}},
            "supplier_changes":{"type":"array","items":{"type":"object","properties":{
                "product":{"type":"string"},"supplier":{"type":"string"},"location":{"type":"string"},"field":{"type":"string"},"new_value":{}
            },"required":["product","supplier","location","field","new_value"],"additionalProperties":False}},
            "transport_updates":{"type":"array","items":{"type":"object","properties":{
                "mode":{"type":"string"},"product":{"type":"string"},"from_location":{"type":"string"},"to_location":{"type":"string"},
                "period":{"type":"integer"},"fields":{"type":"object","additionalProperties":True}
            },"required":["mode","product","from_location","to_location","fields"],"additionalProperties":False}}
        },
        "required": ["period","demand_updates","warehouse_changes","supplier_changes","transport_updates"],
        "additionalProperties": False
    }

def _client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

def _default(period:int)->Dict[str,Any]:
    return {"period": period, "demand_updates": [], "warehouse_changes": [], "supplier_changes": [], "transport_updates": []}

def summarize_scenario(s: Dict[str, Any]) -> List[str]:
    bullets = []
    for d in s.get("demand_updates", []):
        msg = f"Demand: {d.get('product')} at {d.get('location')} "
        dp = d.get("delta_pct")
        if dp is not None: msg += f"Δ {dp}%"
        if d.get("set"):
            sets = "; ".join([f"{k}={v}" for k, v in d["set"].items()])
            msg += f" (set {sets})"
        bullets.append(msg)
    for w in s.get("warehouse_changes", []):
        bullets.append(f"Warehouse: {w.get('warehouse')} → {w.get('field')} = {w.get('new_value')}")
    for sp in s.get("supplier_changes", []):
        bullets.append(f"Supplier: {sp.get('supplier')} {sp.get('product')} → {sp.get('field')} = {sp.get('new_value')}")
    for t in s.get("transport_updates", []):
        bullets.append(f"Transport: {t.get('mode')} {t.get('from_location')}→{t.get('to_location')} ({t.get('product')}) {t.get('fields')}")
    return bullets or ["No actionable changes detected."]

def parse_with_llm(prompt: str, dfs, default_period: int = DEFAULT_PERIOD) -> Dict[str, Any]:
    client = _client()
    if client is None:
        return _default(default_period)

    catalog = build_catalog(dfs)
    system = (
        "Convert the user's supply-chain what-if into a strict JSON that updates Excel sheets. "
        "Return ONLY JSON matching the given schema. Use EXACT entity names from the allowed lists. "
        f"Default period is {default_period} unless the user specifies another year."
    )
    schema = scenario_schema_json()
    guide = {"allowed_entities": catalog}

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            temperature=0,
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type":"text","text": f"Instruction:\n{prompt}"},
                    {"type":"text","text": "Constraints & Allowed Entities:\n" + json.dumps(guide)}
                ]}
            ],
            response_format={"type":"json_schema","json_schema":{"name":"scenario_schema","schema": schema}},
        )
        txt = resp.output[0].content[0].text
        data = json.loads(txt)
        data.setdefault("period", default_period)
        for k in ["demand_updates","warehouse_changes","supplier_changes","transport_updates"]:
            data.setdefault(k, [])
        return data
    except Exception:
        return _default(default_period)
