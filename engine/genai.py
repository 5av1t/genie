from typing import Dict, Any, List, Optional
import os
import json
import math
import re

# Optional providers
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ===== Existing functions kept for compatibility =====
def parse_with_llm(prompt: str, dfs: Dict[str, Any], default_period: int = 2023, provider: str = "gemini") -> Dict[str, Any]:
    """
    NL → Scenario JSON. Keep your existing impl if you have one; this stub keeps the app running.
    """
    return {"period": default_period}

def summarize_scenario(scn: Dict[str, Any]) -> List[str]:
    out = []
    if not scn:
        return ["No scenario changes parsed."]
    if "period" in scn:
        out.append(f"Period: {scn['period']}")
    for k in ["demand_updates","warehouse_changes","supplier_changes","transport_updates"]:
        if scn.get(k):
            out.append(f"{len(scn[k])} {k.replace('_',' ')}")
    if scn.get("adds"):    out.append("Includes additions.")
    if scn.get("deletes"): out.append("Includes deletions.")
    return out

# ===== Results Q&A =====

SYSTEM_QA_INSTRUCTIONS = """
You are GENIE, a supply chain network design assistant.
Answer using the structured context provided (KPIs, flows, throughput, caps).
Do not invent products or entities. If the data is missing, say so briefly.
Keep answers concise, show top items with values, and one-line reasoning.
"""

def _gemini_client():
    if genai is None: return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None

def _openai_client():
    if OpenAI is None: return None
    if not os.getenv("OPENAI_API_KEY"): return None
    try:
        return OpenAI()
    except Exception:
        return None

def _fmt_top(items, limit=5):
    lines = []
    for i, row in enumerate(items[:limit], 1):
        if isinstance(row, (list, tuple)):
            lines.append(f"{i}. " + ", ".join(map(str, row)))
        elif isinstance(row, dict):
            lines.append(f"{i}. " + ", ".join(f"{k}={v}" for k, v in row.items()))
        else:
            lines.append(f"{i}. {row}")
    return "\n".join(lines) if lines else "—"

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _grounded_answer(question: str, kpis: Dict[str, Any], diag: Dict[str, Any], dfs: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Deterministic answers for common queries. Returns string if handled; else None.
    """
    q = (question or "").lower().strip()

    flows = diag.get("flows", []) if isinstance(diag, dict) else []
    wh_thr = diag.get("warehouse_throughput", {}) if isinstance(diag, dict) else {}
    lane_top = diag.get("lane_flow_top", []) if isinstance(diag, dict) else []
    binding = diag.get("binding_warehouses", []) if isinstance(diag, dict) else []
    caps = diag.get("cap_by_warehouse", {}) if isinstance(diag, dict) else {}

    # 0) No flows case
    if any(x in q for x in ["flow", "throughput", "lane", "route"]) and len(flows) == 0:
        status = (kpis or {}).get("status")
        if status in {"no_feasible_arcs","no_demand","no_positive_demand","no_transport_cost"}:
            return f"No flows available (status: {status}). Add/enable lanes, ensure demand > 0, and warehouses have capacity."
        # else keep going; maybe question is about demand/cost

    # 1) service level / total cost
    if re.search(r"(service|fill)\s*(level|rate)|serv(ed|ice)", q):
        return f"Service level: {(kpis or {}).get('service_pct', 0)}% (served {(kpis or {}).get('served', 0)} of {(kpis or {}).get('total_demand', 0)})."
    if "total cost" in q or re.search(r"\bcost\b", q) and "unit" not in q:
        tc = (kpis or {}).get("total_cost")
        return f"Total transport cost: {tc if tc is not None else 'N/A'}."

    # 2) lowest / highest throughput warehouse
    if "lowest" in q and "throughput" in q:
        if not wh_thr:
            return "No warehouse throughput computed (no flows)."
        wh_sorted = sorted(wh_thr.items(), key=lambda kv: _safe_float(kv[1]))
        top = [(w, round(_safe_float(qty),2)) for w, qty in wh_sorted[:5]]
        return "Lowest‑throughput warehouses:\n" + _fmt_top(top)
    if "highest" in q and "throughput" in q:
        if not wh_thr:
            return "No warehouse throughput computed (no flows)."
        wh_sorted = sorted(wh_thr.items(), key=lambda kv: _safe_float(kv[1]), reverse=True)
        top = [(w, round(_safe_float(qty),2)) for w, qty in wh_sorted[:5]]
        return "Highest‑throughput warehouses:\n" + _fmt_top(top)

    # 3) top lanes by flow
    if re.search(r"top\s+\d*\s*lanes", q) or ("top" in q and "lane" in q) or ("lanes" in q and "top" in q):
        if not lane_top:
            return "No lane flows available."
        topn = []
        for w, c, qv in lane_top[:10]:
            topn.append((w, c, round(_safe_float(qv), 2)))
        return "Top lanes by flow (warehouse → customer, qty):\n" + _fmt_top(topn)

    # 4) binding capacity / bottlenecks
    if "bottleneck" in q or "binding" in q or ("capacity" in q and ("bind" in q or "tight" in q)):
        if not binding:
            return "No binding warehouses detected."
        rows = []
        for b in binding:
            rows.append((b.get("warehouse"), round(_safe_float(b.get("used")),2), round(_safe_float(b.get("capacity")),2)))
        return "Binding (near‑full) warehouses (name, used, capacity):\n" + _fmt_top(rows)

    # 5) capacity of a specific warehouse
    m = re.search(r"capacity\s+of\s+([A-Za-z0-9_\-]+)", q)
    if m:
        w = m.group(1)
        if w in caps:
            return f"Capacity of {w}: {round(_safe_float(caps[w]),2)}"
        return f"No capacity entry found for {w}."

    # 6) which customer received most flow
    if ("customer" in q and ("most" in q or "highest" in q) and "flow" in q) or "top customers" in q:
        if not flows:
            return "No flows computed."
        by_cust = {}
        for f in flows:
            by_cust[f["customer"]] = by_cust.get(f["customer"], 0.0) + _safe_float(f.get("qty"), 0.0)
        topc = sorted(by_cust.items(), key=lambda kv: kv[1], reverse=True)[:5]
        topc = [(c, round(v,2)) for c, v in topc]
        return "Top customers by received qty:\n" + _fmt_top(topc)

    # Not handled
    return None

def answer_question(question: str, kpis: Dict[str, Any], diag: Dict[str, Any], provider: str = "gemini", dfs: Optional[Dict[str, Any]] = None, force_llm: bool = False) -> str:
    """
    Q&A about results. Tries grounded (no LLM) first for common queries.
    If force_llm=True (or grounded didn't handle), call LLM if configured.
    """
    # 1) try grounded first
    ans = _grounded_answer(question, kpis, diag, dfs=dfs)
    if ans is not None and not force_llm:
        return ans

    # 2) build compact context for LLM
    ctx = {
        "kpis": kpis or {},
        "flows": (diag or {}).get("flows", []),
        "lowest_throughput_warehouses": (diag or {}).get("lowest_throughput_warehouses", []),
        "binding_warehouses": (diag or {}).get("binding_warehouses", []),
        "lane_flow_top": (diag or {}).get("lane_flow_top", []),
        "warehouse_throughput": (diag or {}).get("warehouse_throughput", {}),
        "cap_by_warehouse": (diag or {}).get("cap_by_warehouse", {}),
    }
    ctx_text = json.dumps(ctx, indent=2, sort_keys=True)

    # 3) pick provider
    if provider == "openai":
        client = _openai_client()
        if client is None:
            return ans or "OpenAI not configured; also grounded patterns did not match this question."
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_QA_INSTRUCTIONS},
                    {"role": "user", "content": f"Context:\n{ctx_text}\n\nQuestion: {question}"},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return ans or f"OpenAI error: {e}"

    # default: gemini
    model = _gemini_client()
    if model is None:
        return ans or "Gemini not configured; also grounded patterns did not match this question."
    try:
        out = model.generate_content(f"{SYSTEM_QA_INSTRUCTIONS}\n\nContext:\n{ctx_text}\n\nQuestion: {question}")
        return out.text.strip() if hasattr(out, "text") and out.text else (ans or "No answer.")
    except Exception as e:
        return ans or f"Gemini error: {e}"
