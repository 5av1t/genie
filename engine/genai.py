"""
engine/genai.py
---------------
GenAI helpers for GENIE — Google Gemini first (free tier), OpenAI as optional fallback.

What this file provides:
- build_catalog(dfs): extracts allowed entity names from the uploaded workbook.
- scenario_schema_json(): strict JSON schema including updates + CRUD (adds/deletes).
- parse_with_llm(prompt, dfs, default_period, provider): NL → Scenario JSON using Gemini/OpenAI.
- summarize_scenario(scenario): bullet list summary for UI.
- generate_examples_with_llm(catalog, stats, provider): creates dynamic prompt examples.

API key resolution order for Gemini:
1) st.secrets["gcp"]["gemini_api_key"]
2) st.secrets["gemini"]["api_key"]
3) os.environ["GEMINI_API_KEY"]

For OpenAI (optional):
1) st.secrets["openai"]["api_key"]
2) os.environ["OPENAI_API_KEY"]
"""

from typing import Dict, Any, List, Optional
import os
import json
import math

# --- Optional imports (don’t hard fail on missing libraries) ---
try:
    import streamlit as st  # only to read secrets if available
except Exception:
    st = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


DEFAULT_PERIOD = 2023


# =========================
# Secrets / Client helpers
# =========================

def _get_secret(path: List[str]) -> Optional[str]:
    """Safely read from st.secrets using a path like ['gcp','gemini_api_key']."""
    if st is None:
        return None
    try:
        cur = st.secrets
        for p in path:
            if p not in cur:
                return None
            cur = cur[p]
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
        return None
    except Exception:
        return None


def _get_gemini_key() -> Optional[str]:
    # Preferred: .streamlit/secrets.toml -> [gcp] gemini_api_key = "..."
    key = _get_secret(["gcp", "gemini_api_key"])
    if key:
        return key
    # Also accept: [gemini] api_key = "..."
    key = _get_secret(["gemini", "api_key"])
    if key:
        return key
    # Fallback: environment variable
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


def _get_openai_key() -> Optional[str]:
    key = _get_secret(["openai", "api_key"])
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


def _client_gemini():
    """Configure and return the Gemini module (google.generativeai) or None."""
    if genai is None:
        return None
    key = _get_gemini_key()
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        return genai
    except Exception:
        return None


def _client_openai():
    """Return an OpenAI client if key/library present, else None."""
    if OpenAI is None:
        return None
    key = _get_openai_key()
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def _choose_provider(provider: Optional[str] = None) -> str:
    """
    Decide which provider to use:
    - 'gemini' if available (default preference)
    - else 'openai' if available
    - else 'none'
    """
    if provider in ("gemini", "openai", "none"):
        return provider
    if _client_gemini() is not None:
        return "gemini"
    if _client_openai() is not None:
        return "openai"
    return "none"


# =========================
# Workbook → catalog helpers
# =========================

def _vals(df, col) -> List[str]:
    if df is None:
        return []
    if col not in df.columns:
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
    return {
        "products": products,
        "customers": customers,
        "warehouses": warehouses,
        "suppliers": suppliers,
        "modes": modes,
        "locations": locations,
    }


# =========================
# Scenario JSON Schema
# =========================

def scenario_schema_json() -> Dict[str, Any]:
    """
    Schema that covers:
    - period
    - demand_updates (pct + optional set fields)
    - warehouse_changes
    - supplier_changes
    - transport_updates
    - adds (CRUD)
    - deletes (CRUD)
    """
    return {
        "type": "object",
        "properties": {
            "period": {"type": "integer"},
            "demand_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                        "customer": {"type": "string"},
                        "location": {"type": "string"},
                        "delta_pct": {"type": "number"},
                        "set": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["product", "customer", "location"],
                    "additionalProperties": True,
                },
            },
            "warehouse_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "warehouse": {"type": "string"},
                        "field": {"type": "string"},
                        "new_value": {},
                    },
                    "required": ["warehouse", "field", "new_value"],
                    "additionalProperties": True,
                },
            },
            "supplier_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                        "supplier": {"type": "string"},
                        "location": {"type": "string"},
                        "field": {"type": "string"},
                        "new_value": {},
                    },
                    "required": ["product", "supplier", "location", "field", "new_value"],
                    "additionalProperties": True,
                },
            },
            "transport_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string"},
                        "product": {"type": "string"},
                        "from_location": {"type": "string"},
                        "to_location": {"type": "string"},
                        "period": {"type": "integer"},
                        "fields": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["mode", "product", "from_location", "to_location", "fields"],
                    "additionalProperties": True,
                },
            },
            "adds": {
                "type": "object",
                "properties": {
                    "customers": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"customer": {"type": "string"}, "location": {"type": "string"}},
                        "required": ["customer", "location"],
                        "additionalProperties": True,
                    }},
                    "customer_demands": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "string"},
                            "customer": {"type": "string"},
                            "location": {"type": "string"},
                            "period": {"type": "integer"},
                            "demand": {"type": "number"},
                            "lead_time": {"type": ["number", "null"]},
                        },
                        "required": ["product", "customer", "location", "period", "demand"],
                        "additionalProperties": True,
                    }},
                    "warehouses": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"warehouse": {"type": "string"}, "location": {"type": "string"}, "fields": {"type": "object", "additionalProperties": True}},
                        "required": ["warehouse", "location"],
                        "additionalProperties": True,
                    }},
                    "supplier_products": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"product": {"type": "string"}, "supplier": {"type": "string"}, "location": {"type": "string"}, "period": {"type": "integer"}, "fields": {"type": "object", "additionalProperties": True}},
                        "required": ["product", "supplier", "location", "period"],
                        "additionalProperties": True,
                    }},
                    "transport_lanes": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"mode": {"type": "string"}, "product": {"type": "string"}, "from_location": {"type": "string"}, "to_location": {"type": "string"}, "period": {"type": "integer"}, "fields": {"type": "object", "additionalProperties": True}},
                        "required": ["mode", "product", "from_location", "to_location", "period"],
                        "additionalProperties": True,
                    }},
                },
                "additionalProperties": False,
            },
            "deletes": {
                "type": "object",
                "properties": {
                    "customers": {"type": "array", "items": {
                        "type": "object", "properties": {"customer": {"type": "string"}}, "required": ["customer"]
                    }},
                    "customer_product_rows": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"product": {"type": "string"}, "customer": {"type": "string"}, "location": {"type": "string"}, "period": {"type": "integer"}},
                        "required": ["product", "customer", "location", "period"]
                    }},
                    "warehouses": {"type": "array", "items": {
                        "type": "object", "properties": {"warehouse": {"type": "string"}}, "required": ["warehouse"]
                    }},
                    "supplier_products": {"type": "array", "items": {
                        "type": "object", "properties": {"product": {"type": "string"}, "supplier": {"type": "string"}}, "required": ["product", "supplier"]
                    }},
                    "transport_lanes": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"mode": {"type": "string"}, "product": {"type": "string"}, "from_location": {"type": "string"}, "to_location": {"type": "string"}, "period": {"type": "integer"}},
                        "required": ["mode", "product", "from_location", "to_location", "period"]
                    }},
                },
                "additionalProperties": False,
            },
        },
        "required": [
            "period",
            "demand_updates",
            "warehouse_changes",
            "supplier_changes",
            "transport_updates",
            "adds",
            "deletes",
        ],
        "additionalProperties": False,
    }


def _default_scenario(period: int) -> Dict[str, Any]:
    return {
        "period": period,
        "demand_updates": [],
        "warehouse_changes": [],
        "supplier_changes": [],
        "transport_updates": [],
        "adds": {
            "customers": [],
            "customer_demands": [],
            "warehouses": [],
            "supplier_products": [],
            "transport_lanes": [],
        },
        "deletes": {
            "customers": [],
            "customer_product_rows": [],
            "warehouses": [],
            "supplier_products": [],
            "transport_lanes": [],
        },
    }


# =========================
# LLM JSON call wrappers
# =========================

def llm_json(instruction: str, schema: Dict[str, Any], system: Optional[str] = None, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to return JSON matching `schema`.
    provider: "gemini" | "openai" | "none" | None
    """
    prov = _choose_provider(provider)

    # --- OpenAI path (optional) ---
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
                    {"role": "user", "content": [{"type": "text", "text": instruction}]}
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "genie_schema", "schema": schema}},
            )
            txt = resp.output[0].content[0].text
            return json.loads(txt)
        except Exception:
            return None

    # --- Gemini path (primary) ---
    if prov == "gemini":
        g = _client_gemini()
        if g is None:
            return None
        try:
            # Prefer the "flash" model for speed/cost on free tier; upgrade to "pro" if you need more reasoning.
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
        except Exception as e:
            # Fallback: best-effort JSON extraction if model returns text with JSON inside
            try:
                import re
                m = re.search(r"\{.*\}", resp.text, flags=re.S)
                if m:
                    return json.loads(m.group(0))
            except Exception:
                pass
            return None

    # If no provider available
    return None


# =========================
# Parsers & Summaries
# =========================

def parse_with_llm(prompt: str, dfs, default_period: int = DEFAULT_PERIOD, provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert NL scenario prompt to strict Scenario JSON using LLM.
    Uses the workbook catalog to constrain allowed entity names.
    """
    catalog = build_catalog(dfs)
    system = (
        "Convert the user's supply-chain what-if into a strict JSON that updates Excel sheets. "
        "Return ONLY JSON matching the given schema. Use EXACT entity names from the allowed lists. "
        f"Default period is {default_period} unless the user specifies another year. "
        "You may also add or delete rows (CRUD) using the 'adds' and 'deletes' sections."
    )
    guide = {
        "allowed_entities": catalog,
        "default_period": default_period,
        "rules": [
            "Do not invent entities not in allowed_entities.",
            "Prefer existing fields: Demand, Lead Time, Maximum Capacity, Force Open/Close, Available, Cost Per UOM.",
            "Use 'adds' to create rows (customers, warehouses, supplier products, lanes, CPD).",
            "Use 'deletes' to remove rows. Include the period for deletions where applicable.",
            "If a year is missing, use default period.",
        ],
    }
    schema = scenario_schema_json()
    instruction = "Instruction:\n" + (prompt or "") + "\n\nConstraints:\n" + json.dumps(guide)

    data = llm_json(instruction, schema, system=system, provider=provider)
    if not isinstance(data, dict):
        return _default_scenario(default_period)

    # Ensure required keys exist with correct shapes
    data.setdefault("period", default_period)
    if "adds" not in data or not isinstance(data["adds"], dict):
        data["adds"] = {"customers": [], "customer_demands": [], "warehouses": [], "supplier_products": [], "transport_lanes": []}
    else:
        for k in ["customers", "customer_demands", "warehouses", "supplier_products", "transport_lanes"]:
            data["adds"].setdefault(k, [])

    if "deletes" not in data or not isinstance(data["deletes"], dict):
        data["deletes"] = {"customers": [], "customer_product_rows": [], "warehouses": [], "supplier_products": [], "transport_lanes": []}
    else:
        for k in ["customers", "customer_product_rows", "warehouses", "supplier_products", "transport_lanes"]:
            data["deletes"].setdefault(k, [])

    for k in ["demand_updates", "warehouse_changes", "supplier_changes", "transport_updates"]:
        data.setdefault(k, [])

    return data


def summarize_scenario(s: Dict[str, Any]) -> List[str]:
    """Human-readable bullets for the UI."""
    bullets = []
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
        bullets.append(f"Transport: {t.get('mode')} {t.get('from_location')}→{t.get('to_location')} ({t.get('product')}) {t.get('fields')}")

    adds = s.get("adds", {})
    for c in adds.get("customers", []):
        bullets.append(f"Add customer {c.get('customer')} at {c.get('location')}")
    for d in adds.get("customer_demands", []):
        bullets.append(f"Add demand {d.get('demand')} of {d.get('product')} at {d.get('customer')} (period {d.get('period')})")
    for w in adds.get("warehouses", []):
        bullets.append(f"Add warehouse {w.get('warehouse')} at {w.get('location')} {w.get('fields')}")
    for sp in adds.get("supplier_products", []):
        bullets.append(f"Add supplier product {sp.get('product')} at {sp.get('supplier')} (Available={sp.get('fields',{}).get('Available')})")
    for tl in adds.get("transport_lanes", []):
        bullets.append(f"Add lane {tl.get('mode')} {tl.get('from_location')}→{tl.get('to_location')} ({tl.get('product')})")

    dels = s.get("deletes", {})
    for c in dels.get("customers", []):
        bullets.append(f"Delete customer {c.get('customer')}")
    for d in dels.get("customer_product_rows", []):
        bullets.append(f"Delete demand row {d.get('product')} at {d.get('customer')} (period {d.get('period')})")
    for w in dels.get("warehouses", []):
        bullets.append(f"Delete warehouse {w.get('warehouse')}")
    for sp in dels.get("supplier_products", []):
        bullets.append(f"Delete supplier product {sp.get('product')} at {sp.get('supplier')}")
    for tl in dels.get("transport_lanes", []):
        bullets.append(f"Delete lane {tl.get('mode')} {tl.get('from_location')}→{tl.get('to_location')} ({tl.get('product')})")

    return bullets or ["No actionable changes detected."]


# =========================
# Example Generation (LLM)
# =========================

def example_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "minItems": 4,
                "maxItems": 8,
                "items": {"type": "string"},
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    }


def generate_examples_with_llm(catalog: Dict[str, List[str]], stats: Dict[str, Any], provider: Optional[str] = None) -> List[str]:
    """
    Ask the LLM to craft 4-8 concise, copy-pasteable prompts tailored to the user's file.
    Coverage includes base-run, demand changes, warehouse toggles, supplier enable, transport cost,
    plus CRUD adds/deletes.
    """
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
            "1 add-customer with demand",
            "1 add-warehouse with capacity",
            "1 delete transport lane"
        ],
        "style": ["short, direct", "no extra commentary", "ASCII arrow '->' is fine"],
        "defaults": {"period": DEFAULT_PERIOD},
        "tips": ["Do not invent entities", "Respect exact spellings/case/underscores"],
    }
    instruction = "Produce 4-8 example prompts.\nContext:\n" + json.dumps(guidance)

    data = llm_json(instruction, example_schema(), system=system, provider=provider)
    if not isinstance(data, dict):
        return []
    ex = data.get("examples") or []
    seen = set()
    out: List[str] = []
    for e in ex:
        e = str(e).strip()
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    return out[:8]
