from typing import Dict, Any, List, Optional
import os
import json

# Optional providers
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ===== Existing functions (kept) =====
def parse_with_llm(prompt: str, dfs: Dict[str, Any], default_period: int = 2023, provider: str = "gemini") -> Dict[str, Any]:
    """
    EXISTING: NL → Scenario JSON. (You already had this; keep behavior.)
    This is a placeholder that should already exist in your repo; keeping signature intact.
    """
    # If you already implemented this, keep your implementation.
    # Here, return a minimal stub if someone imports it without your file being present.
    return {"period": default_period}

def summarize_scenario(scn: Dict[str, Any]) -> List[str]:
    """EXISTING: short bullet summary for scenario JSON."""
    out = []
    if not scn:
        return ["No scenario changes parsed."]
    if "period" in scn:
        out.append(f"Period: {scn['period']}")
    if scn.get("demand_updates"):
        out.append(f"{len(scn['demand_updates'])} demand update(s).")
    if scn.get("warehouse_changes"):
        out.append(f"{len(scn['warehouse_changes'])} warehouse change(s).")
    if scn.get("supplier_changes"):
        out.append(f"{len(scn['supplier_changes'])} supplier change(s).")
    if scn.get("transport_updates"):
        out.append(f"{len(scn['transport_updates'])} transport update(s).")
    if scn.get("adds"):
        out.append("Includes additions.")
    if scn.get("deletes"):
        out.append("Includes deletions.")
    return out

# ===== NEW: Results Q&A =====
SYSTEM_QA_INSTRUCTIONS = """
You are GENIE, a supply chain network design assistant.
Answer questions using the structured context provided (KPIs, flows, throughput).
When asked to identify or rank, use the computed numbers, not assumptions.
Keep answers concise, list top items with values, and include brief reasoning.
If data is missing (e.g., no flows), explain why and suggest fixes (lanes, capacities, demand).
"""

def _gemini_client() -> Optional[Any]:
    if genai is None:
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model
    except Exception:
        return None

def _openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI()
    except Exception:
        return None

def answer_question(question: str, kpis: Dict[str, Any], diag: Dict[str, Any], provider: str = "gemini") -> str:
    """
    LLM Q&A grounded in optimizer results. Returns a concise answer string.
    """
    # Build compact, grounded context
    ctx = {
        "kpis": kpis or {},
        "lowest_throughput_warehouses": (diag or {}).get("lowest_throughput_warehouses", []),
        "binding_warehouses": (diag or {}).get("binding_warehouses", []),
        "lane_flow_top": (diag or {}).get("lane_flow_top", []),
        "warehouse_throughput": (diag or {}).get("warehouse_throughput", {}),
        "cap_by_warehouse": (diag or {}).get("cap_by_warehouse", {}),
    }
    ctx_text = json.dumps(ctx, indent=2, sort_keys=True)

    if provider == "openai":
        client = _openai_client()
        if client is None:
            return "OpenAI client not configured."
        try:
            msg = [
                {"role": "system", "content": SYSTEM_QA_INSTRUCTIONS},
                {"role": "user", "content": f"Context JSON:\n{ctx_text}\n\nQuestion: {question}"},
            ]
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msg,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI error: {e}"

    # default: gemini
    model = _gemini_client()
    if model is None:
        return "Gemini client not configured."
    try:
        prompt = f"{SYSTEM_QA_INSTRUCTIONS}\n\nContext JSON:\n{ctx_text}\n\nQuestion: {question}"
        out = model.generate_content(prompt)
        return out.text.strip() if hasattr(out, "text") and out.text else "No answer."
    except Exception as e:
        return f"Gemini error: {e}"
