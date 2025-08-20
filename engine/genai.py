from typing import Dict, Any, List, Optional
import os, json
import math

# Optional deps; handled at runtime
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

DEFAULT_PERIOD = 2023

# ---------- Catalog Builders ----------
def _vals(df, col) -> List[str]:
    if df is None or col not in df.columns:
        return []
    return [str(x) for x in df[col].dropna().tolist()]

def build_catalog(dfs) -> Dict[str, List[str]]:
    products  = _vals(dfs.get("Products"), "Product")
    customers = _vals(dfs.get("Customers"), "Customer") or _vals(dfs.get("Customer Product Data"), "Customer")
    warehouses= _vals(dfs.get("Warehouse"), "Warehouse")
    suppliers = _vals(dfs.get("Supplier Product"), "Supplier")
    modes     = _vals(dfs.get("Mode of Transport"), "Mode of Transport")
    locations = sorted(
        set(_vals(dfs.get("Warehouse"), "Location")) |
        set(_vals(dfs.get("Customers"), "Location")) |
        set(_vals(dfs.get("Customer Product Data"), "Location")) |
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
            },"required":["warehouse","field","new_value"],"additionalProperties":True}},
            "supplier_changes":{"type":"array","items":{"type":"object","properties":{
                "product":{"type":"string"},"supplier":{"type":"string"},"location":{"type":"string"},"field":{"type":"string"},"new_value":{}
            },"required":["product","supplier","location","field","new_value"],"additionalProperties":True}},
            "transport_updates":{"type":"array","items":{"type":"object","properties":{
                "mode":{"type":"string"},"product":{"type":"string"},"from_location":{"type":"string"},"to_location":{"type":"string"},
                "period":{"type":"integer"},"fields":{"type":"object","additionalProperties":True}
            },"required":["mode","product","from_location","to_location","fields"],"additionalProperties":True}}
        },
        "required": ["period","demand_updates","warehouse_changes","supplier_changes","transport_updates"],
        "additionalProperties": False
    }

# ---------- Provider Clients ----------
def _client_openai():
    key = os.environ.get("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

def _client_gemini():
    key = os.environ.get("GEMINI_API_KEY")
    if not key or genai is None:
        return None
    try:
        genai.configure(api_key=key)
        return genai
    except Exception:
        return None

def _choose_provider(provider: Optional[str] = None) -> str:
    """Return 'gemini', 'openai', or 'none'."""
    if provider in ("gemini", "openai", "none"):
        return provider
    # Auto: prefer Gemini if available, then OpenAI
    if _client_gemini() is not None:
        return "gemini"
    if _client_openai() is not None:
        return "openai"
    return "none"

# ---------- Shared JSON LLM helper ----------
def llm_json(instruction: str, schema: Dict[str, Any], system: Optional[str] = None, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
    prov = _choose_provider(provider)

    if prov == "openai":
        client = _client_openai()
        if client is None:
            return None
        try:
            resp = client.responses.create(
                model="gpt-4o-mini",
                temperature=0,
                input=[
                    {"role": "system", "content": system or "Return valid JSON only."},
                    {"role": "user", "content": [{"type":"text","text": instruction}]}
                ],
                response_format={"type":"json_schema","json_schema":{"name":"genie_schema","schema": schema}},
            )
            txt = resp.output[0].content[0].text
            return json.loads(txt)
        except Exception:
            return None

    if prov == "gemini":
        g = _client_gemini()
        if g is None:
            return None
        try:
            model = g.GenerativeModel("gemini-1.5-flash")
            generation_config = {
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": schema,
            }
            parts = []
            if system:
                parts.append(system)
            parts.append(instruction)
            resp = model.generate_content(parts, generation_config=generation_config)
            txt = resp.text
            return json.loads(txt)
        except Exception:
            # Fallback: try parsing any JSON substring
            try:
                import re
                m = re.search(r"\{.*\}", resp.text, flags=re.S)
                if m:
                    return json.loads(m.group(0))
            except Exception:
                pass
            return None

    return None  # provider none

# ---------- High-level APIs ----------
def _default_scenario(period:int)->Dict[str,Any]:
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

def parse_with_llm(prompt: str, dfs, default_period: int = DEFAULT_PERIOD, provider: Optional[str] = None) -> Dict[str, Any]:
    catalog = build_catalog(dfs)
    system = (
        "Convert the user's supply-chain what-if into a strict JSON that updates Excel sheets. "
        "Return ONLY JSON matching the given schema. Use EXACT entity names from the allowed lists. "
        f"Default period is {default_period} unless the user specifies another year."
    )
    schema = scenario_schema_json()
    guide = {"allowed_entities": catalog, "default_period": default_period, "rules": [
        "Do not invent entities not in allowed_entities.",
        "Prefer existing fields: Demand, Lead Time, Maximum Capacity, Force Open/Close, Available, Cost Per UOM.",
        "If a year is missing, use default period.",
    ]}
    instruction = "Instruction:\n" + prompt + "\n\nConstraints:\n" + json.dumps(guide)
    data = llm_json(instruction, schema, system=system, provider=provider)
    if not isinstance(data, dict):
        return _default_scenario(default_period)
    # Ensure required keys
    data.setdefault("period", default_period)
    for k in ["demand_updates","warehouse_changes","supplier_changes","transport_updates"]:
        data.setdefault(k, [])
    return data

# ---------- Example generation JSON ----------
def example_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "minItems": 4,
                "maxItems": 8,
                "items": {"type": "string"}
            }
        },
        "required": ["examples"],
        "additionalProperties": False
    }

def generate_examples_with_llm(catalog: Dict[str, List[str]], stats: Dict[str, Any], provider: Optional[str] = None) -> List[str]:
    system = (
        "Write concise, copy-pasteable what-if prompts for a supply-chain network design app. "
        "Use ONLY entity names from the provided lists. Output JSON only."
    )
    guidance = {
        "allowed_entities": catalog,
        "has_lead_time": stats.get("has_lead_time", True),
        "coverage": [
            "one 'run the base model'",
            "1-2 demand changes (with or without lead time)",
            "1 warehouse capacity or force toggle",
            "1 supplier enable/disable",
            "1 transport lane cost (Mode, From, To, Product)",
        ],
        "style": ["short, direct", "no extra commentary", "ASCII arrow '->' is fine"],
        "defaults": {"period": DEFAULT_PERIOD},
        "tips": ["Do not invent entities", "Respect exact spellings/case/underscores"],
    }
    instruction = "Produce 4-8 example prompts.\nContext:\n" + json.dumps(guidance)
    schema = example_schema()
    data = llm_json(instruction, schema, system=system, provider=provider)
    if not isinstance(data, dict):
        return []
    ex = data.get("examples") or []
    # Basic dedupe/strip
    seen = set()
    out = []
    for e in ex:
        e = str(e).strip()
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    return out[:8]
