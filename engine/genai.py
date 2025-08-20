from __future__ import annotations
from typing import Dict, Any, List
import os
import pandas as pd

from engine.parser import parse_rules, DEFAULT_PERIOD

def parse_with_llm(text: str, dfs: Dict[str, pd.DataFrame], default_period: int = DEFAULT_PERIOD, provider: str = "none") -> Dict[str, Any]:
    # For now, keep deterministic to avoid hallucination. If GEMINI_API_KEY present and provider=='gemini',
    # you could integrate google.generativeai here to extract fields, then post-validate against dfs entities.
    return parse_rules(text, dfs, default_period=default_period)

def _unserved(diag: Dict[str, Any], dfs: Dict[str, pd.DataFrame]) -> List[str]:
    flows = diag.get("flows", [])
    served_pairs = {(f["customer"], f["product"]) for f in flows}
    cpd = dfs.get("Customer Product Data", pd.DataFrame()).copy()
    if cpd.empty: return []
    cpd["Demand"] = pd.to_numeric(cpd.get("Demand", 0), errors="coerce").fillna(0.0)
    cpd = cpd[cpd["Demand"] > 0]
    out = []
    for _, r in cpd.iterrows():
        key = (str(r["Customer"]), str(r["Product"]))
        if key not in served_pairs:
            out.append(f"{key[0]} ({key[1]})")
    return sorted(set(out))

def answer_question(question: str, kpis: Dict[str, Any], diag: Dict[str, Any], provider: str = "none", dfs: Dict[str, pd.DataFrame] = None, model_index: Dict[str, Any] = None) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "Please ask a question about the model or solution."

    # Grounded heuristics
    if "unserved" in q or "not served" in q:
        un = _unserved(diag, dfs or {})
        if un:
            return "Unserved customer-product pairs:\n- " + "\n- ".join(un)
        return "All demanded customer-product pairs appear served in the current solution."

    if "least demand" in q or "lowest demand" in q:
        cpd = (dfs or {}).get("Customer Product Data", pd.DataFrame()).copy()
        if cpd.empty: return "I can't find demand data."
        cpd["Demand"] = pd.to_numeric(cpd.get("Demand", 0), errors="coerce").fillna(0.0)
        grp = cpd.groupby("Customer")["Demand"].sum().sort_values()
        if grp.empty:
            return "No demand rows found."
        c, d = grp.index[0], grp.iloc[0]
        return f"Lowest aggregated demand by customer: {c} = {int(d)} units."

    if "kpi" in q or "status" in q:
        return f"Status: {kpis.get('status')}, Service: {kpis.get('service_pct')}%, Total Cost: {kpis.get('total_cost')}."

    # Fallback generic
    if provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
        # Here you'd pass a compact context (kpis, top flows) to Gemini for narrative. Kept off for safety.
        pass

    return "I can answer about unserved pairs, KPIs, and simple demand summaries. Ask, e.g., 'Which customers are unserved?'"
