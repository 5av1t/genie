# --- GENIE: Supply Chain Network Designer (complete app.py) ---
# Upload Excel → (Prompt OR Manual CRUD) → Apply → Optimize → KPIs + Maps → Q&A → Export
# Sticky results across reruns; Ask GENIE under the map; Delta tables only when user actually edited.

import os
from io import BytesIO
import json
import hashlib
from typing import Dict, Any, List, Tuple
import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:
    print("This application requires Streamlit to run. Please add `streamlit` to requirements.txt and redeploy.")
    raise

try:
    import pydeck as pdk
except Exception:
    pdk = None

st.set_page_config(page_title="GENIE - Supply Chain Network Designer", layout="wide")

# --- Secrets (keep API keys out of code) ---
for block in ("openai", "gcp", "gemini"):
    if block in st.secrets:
        for k, v in st.secrets[block].items():
            if isinstance(v, str):
                os.environ[f"{block.upper()}_{k.upper()}"] = v
if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
    os.environ["GEMINI_API_KEY"] = st.secrets["gemini"]["api_key"]

# --- Import engine modules (optional GenAI; rest required) ---
missing = []
try:
    from engine.loader import load_and_validate_excel as loader
except Exception as e:
    st.error("❌ Could not import loader from engine/loader.py.\n" + str(e)); st.stop()

try:
    from engine.parser import parse_prompt as parse_rules
except ModuleNotFoundError:
    missing.append("engine/parser.py (parse_prompt)"); parse_rules = None

try:
    from engine.updater import apply_scenario
except ModuleNotFoundError:
    missing.append("engine/updater.py (apply_scenario)"); apply_scenario = None

try:
    from engine.optimizer import run_optimizer
except ModuleNotFoundError:
    missing.append("engine/optimizer.py (run_optimizer)"); run_optimizer = None

try:
    from engine.reporter import build_summary
except ModuleNotFoundError:
    missing.append("engine/reporter.py (build_summary)"); build_summary = None

try:
    from engine.geo import build_nodes, flows_to_geo, guess_map_center
except ModuleNotFoundError:
    missing.append("engine/geo.py (build_nodes, flows_to_geo, guess_map_center)")
    build_nodes = flows_to_geo = guess_map_center = None

# GenAI optional
try:
    from engine.genai import parse_with_llm, summarize_scenario, answer_question
except ModuleNotFoundError:
    parse_with_llm = summarize_scenario = answer_question = None

try:
    from engine.example_gen import examples_for_file
except ModuleNotFoundError:
    examples_for_file = None

if missing:
    st.error("❌ Missing engine modules:\n- " + "\n- ".join(missing))
    st.stop()

# --- Header / Sidebar ---
st.title("🔮 GENIE — Generative Engine for Network Intelligence & Execution")
st.caption("Upload a base case → describe a scenario or use Manual Edits → GENIE updates sheets, optimizes, and maps your network.")

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox(
        "Scenario Parser Provider",
        ["Google Gemini (free tier)", "OpenAI", "Rules only (no LLM)"],
        index=0,
        key="cfg_provider"
    )
    use_llm_parser = st.checkbox("Use GenAI parser for scenarios", value=(provider != "Rules only (no LLM)"), key="cfg_use_llm")
    show_llm_vs_rules = st.checkbox("Show LLM vs Rules (debug)", value=False, key="cfg_debug_compare")

    st.markdown("---")
    st.subheader("🧠 GENIE can:")
    st.markdown(
        "- Adjust **Demand** / **Lead Time** per Customer & Product\n"
        "- Change **Warehouse** capacity/costs; Force Open/Close\n"
        "- Enable/disable **Supplier Product** availability\n"
        "- Set **Transport Cost** per lane (Mode, From, To, Product, Period)\n"
        "- **Add / Delete** customers, warehouses, lanes, demand rows\n"
        "- Run the **base model** (no edits)\n"
        "- **Ask about results** (unserved customers, service %, lowest-demand customers, throughput, top flows)"
    )
    st.markdown("---")
    st.caption("Secrets: Streamlit Cloud → App settings → Secrets. Local dev: `.streamlit/secrets.toml` (add gemini/api_key or openai/api_key).")

# ---------------- Helpers: deltas & autoconnect ----------------
def build_delta_view(before: pd.DataFrame, after: pd.DataFrame, keys: list) -> pd.DataFrame:
    if after is None or not isinstance(after, pd.DataFrame) or after.empty:
        return pd.DataFrame(columns=keys + ["Field","Before","After","ChangeType"])
    before = before.copy() if isinstance(before, pd.DataFrame) else pd.DataFrame(columns=keys)
    after = after.copy()
    common = list(set(before.columns) & set(after.columns)) if not before.empty else list(after.columns)
    if not common: common = list(after.columns)
    keys_used = [k for k in keys if k in common]
    merged = after[common].merge(before[common], on=keys_used, how="left", suffixes=("_after","_before"), indicator=True)
    fields = [c for c in common if c not in keys_used]
    parts = []
    for col in fields:
        a, b = f"{col}_after", f"{col}_before"
        if a not in merged.columns or b not in merged.columns: continue
        diff = (merged[a] != merged[b]) | (merged[b].isna() & merged[a].notna())
        if diff.any():
            chunk = merged.loc[diff, keys_used + [b,a,"_merge"]].copy()
            chunk["Field"] = col
            chunk.rename(columns={b:"Before", a:"After"}, inplace=True)
            chunk["ChangeType"] = chunk["_merge"].map({"left_only":"New","both":"Updated","right_only":"Removed"})
            chunk.drop(columns=["_merge"], inplace=True)
            parts.append(chunk)
    if not parts: return pd.DataFrame(columns=keys_used + ["Field","Before","After","ChangeType"])
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(keys_used + ["Field"], kind="stable")

def show_delta(title, before, after, keys):
    st.markdown(f"#### Δ Changes — {title}")
    d = build_delta_view(before, after, keys)
    if d.empty: st.info(f"No visible changes detected in {title}."); return
    st.dataframe(d, use_container_width=True)
    added = d.loc[d["ChangeType"]=="New", keys].drop_duplicates().shape[0] if "ChangeType" in d.columns else 0
    updated = d.loc[d["ChangeType"]=="Updated", keys].drop_duplicates().shape[0] if "ChangeType" in d.columns else 0
    st.success(f"Applied changes: {updated} updated row(s), {added} new row(s).")

def autoconnect_lanes(dfs, period=2023, default_cost=10.0):
    """Create placeholder lanes from every warehouse Location to every customer with positive demand, if missing."""
    wh = dfs.get("Warehouse", pd.DataFrame())
    cpd = dfs.get("Customer Product Data", pd.DataFrame())
    tc  = dfs.get("Transport Cost", pd.DataFrame())
    if not isinstance(tc, pd.DataFrame) or tc.empty:
        tc = pd.DataFrame(columns=["Mode of Transport","Product","From Location","To Location","Period","UOM","Available","Retrieve Distance","Average Load Size","Cost Per UOM","Cost per Distance","Cost per Trip","Minimum Cost Per Trip"])
    wlocs = {}
    if isinstance(wh, pd.DataFrame) and not wh.empty:
        for _, r in wh.iterrows():
            w = str(r.get("Warehouse"))
            loc = str(r.get("Location")) if pd.notna(r.get("Location")) else w
            wlocs[w] = loc
    need = set()
    if isinstance(cpd, pd.DataFrame) and not cpd.empty:
        use = cpd.copy()
        if "Period" in use.columns: use = use[use["Period"] == period]
        for _, r in use.iterrows():
            cust = str(r.get("Customer")); prod = str(r.get("Product"))
            try: dem = float(r.get("Demand", 0) or 0)
            except Exception: dem = 0.0
            if dem > 0: need.add((cust, prod))
    existing = set()
    if isinstance(tc, pd.DataFrame) and not tc.empty:
        for _, r in tc.iterrows():
            try:
                if int(r.get("Period", period)) != period: continue
            except Exception: continue
            existing.add((str(r.get("From Location")), str(r.get("To Location")), str(r.get("Product"))))
    new_rows = []
    for _, from_loc in wlocs.items():
        for (to_cust, prod) in need:
            key = (str(from_loc), str(to_cust), str(prod))
            if key in existing: continue
            new_rows.append({
                "Mode of Transport":"Secondary Delivery LTL","Product":prod,"From Location":from_loc,"To Location":to_cust,"Period":period,
                "UOM":"Each","Available":1,"Retrieve Distance":0.0,"Average Load Size":1.0,"Cost Per UOM":float(default_cost),
                "Cost per Distance":0.0,"Cost per Trip":0.0,"Minimum Cost Per Trip":0.0
            })
    if new_rows:
        tc = pd.concat([tc, pd.DataFrame(new_rows)], ignore_index=True)
    dfs["Transport Cost"] = tc
    return dfs, len(new_rows)

# ---------- Grounded QA helpers (no hallucinations) ----------
def _served_by_customer(diag: Dict[str, Any]) -> Dict[str, float]:
    served = {}
    flows = (diag or {}).get("flows", [])
    if not isinstance(flows, list):
        return served
    for f in flows:
        c = str(f.get("customer"))
        q = float(f.get("qty", 0) or 0)
        served[c] = served.get(c, 0.0) + q
    return served

def _demand_by_customer(dfs: Dict[str, pd.DataFrame], period: int = 2023) -> Dict[str, float]:
    cpd = dfs.get("Customer Product Data", pd.DataFrame())
    out = {}
    if isinstance(cpd, pd.DataFrame) and not cpd.empty:
        df = cpd.copy()
        if "Period" in df.columns:
            df = df[df["Period"] == period]
        for _, r in df.iterrows():
            cust = str(r.get("Customer"))
            q = float(r.get("Demand", 0) or 0)
            out[cust] = out.get(cust, 0.0) + q
    return out

def _unserved_customers(dfs: Dict[str, pd.DataFrame], diag: Dict[str, Any], period: int = 2023) -> Tuple[int, List[Tuple[str, float]]]:
    dem = _demand_by_customer(dfs, period)
    srv = _served_by_customer(diag)
    unserved_list = []
    for c, d in dem.items():
        s = srv.get(c, 0.0)
        if d > 0 and s < d - 1e-9:
            unserved_list.append((c, d - s))
    unserved_list.sort(key=lambda x: (-x[1], x[0]))
    return len(unserved_list), unserved_list

def grounded_answer(q: str, dfs: Dict[str, pd.DataFrame], kpis: Dict[str, Any], diag: Dict[str, Any], period: int = 2023) -> str:
    qs = (q or "").strip().lower()
    # unserved customers
    if any(t in qs for t in ["unserved", "not served", "not-served", "unmet"]):
        cnt, lst = _unserved_customers(dfs, diag, period)
        preview = ", ".join([f"{c}({int(u)})" for c, u in lst[:10]]) if lst else "—"
        return f"{cnt} customer(s) have unmet demand. Top gaps: {preview}"
    # least/lowest demand
    if any(t in qs for t in ["least demand", "lowest demand", "smallest demand"]):
        dem = _demand_by_customer(dfs, period)
        if not dem: return "No demand found for the selected period."
        items = [(c, v) for c, v in dem.items() if v > 0]
        if not items: return "All customers have zero demand."
        items.sort(key=lambda x: (x[1], x[0]))
        top = ", ".join([f"{c}({int(v)})" for c, v in items[:10]])
        return f"Lowest-demand customers: {top}"
    # service level
    if "service level" in qs or ("service" in qs and "%" in qs):
        svc = kpis.get("service_pct")
        total = kpis.get("total_demand")
        served = kpis.get("served")
        if svc is None:
            if total and total > 0:
                svc = round(100.0 * float(served or 0) / float(total), 2)
            else:
                svc = 0.0
        return f"Service level: {svc}% (served {int(served or 0)} of {int(total or 0)})."
    # top flows/lanes
    if any(t in qs for t in ["top flows", "top lanes", "largest flows"]):
        flows = (diag or {}).get("flows", [])
        if not flows: return "No flows available."
        rows = sorted([(f.get("warehouse"), f.get("customer"), f.get("product"), float(f.get("qty", 0) or 0)) for f in flows],
                      key=lambda x: -x[3])[:10]
        txt = "; ".join([f"{w}→{c} {p}:{int(q)}" for w,c,p,q in rows])
        return f"Top flows: {txt}"
    # lowest throughput warehouses
    if any(t in qs for t in ["lowest throughput", "least throughput", "lowest outbound"]):
        flows = (diag or {}).get("flows", [])
        tot_by_w = {}
        for f in flows:
            tot_by_w[f.get("warehouse")] = tot_by_w.get(f.get("warehouse"), 0.0) + float(f.get("qty", 0) or 0)
        if not tot_by_w: return "No flows available to compute throughput."
        rows = sorted(tot_by_w.items(), key=lambda x: (x[1], str(x[0])))[:5]
        txt = ", ".join([f"{w}:{int(v)}" for w,v in rows])
        return f"Lowest outbound throughput warehouses: {txt}"
    return ""  # unknown → caller may escalate to LLM

def build_llm_context(dfs: Dict[str, pd.DataFrame], kpis: Dict[str, Any], diag: Dict[str, Any], period: int = 2023) -> str:
    dem = _demand_by_customer(dfs, period)
    _, unserved_list = _unserved_customers(dfs, diag, period)
    least = sorted([(c, v) for c, v in dem.items()], key=lambda x: (x[1], x[0]))[:20]
    def fmt_rows(rows): return "\n".join([f"- {c}: {int(v)}" for c, v in rows]) if rows else "- (none)"
    cpd = dfs.get("Customer Product Data", pd.DataFrame())
    prods = []
    if isinstance(cpd, pd.DataFrame) and "Product" in cpd.columns:
        prods = sorted(set(cpd["Product"].dropna().astype(str).tolist()))
    lines = [
        f"Period: {period}",
        f"KPIs: status={kpis.get('status')}, total_demand={int(kpis.get('total_demand') or 0)}, served={int(kpis.get('served') or 0)}, service_pct={kpis.get('service_pct')}",
        "Least-demand customers (up to 20):",
        fmt_rows(least),
        "Unserved customers (up to 20):",
        fmt_rows(unserved_list[:20]),
        f"Products: {', '.join(prods[:20])}" + ("" if len(prods) <= 20 else ", ..."),
    ]
    return "\n".join(lines)

# ------------- Suggested prompts -------------
STATIC_EXAMPLES = [
    "run the base model",
    "Increase demand at a specific customer by 10% and set lead time to 8",
    "Cap a warehouse Maximum Capacity at 25000; force close another warehouse",
    "Set a lane cost per uom to 9.5 for a chosen mode/from/to/product in 2023",
    "Add a new customer with demand and lead time",
    "Add a new warehouse at a city; force open; set capacity",
    "Delete a specific transport lane in 2023",
]

# Prefill buffer for prompt
user_prompt_default = st.session_state.get("user_prompt","")
if "prefill_prompt" in st.session_state:
    user_prompt_default = st.session_state.pop("prefill_prompt", user_prompt_default)

# ---------------- Results renderer (sticky across reruns) ----------------
def render_results(dfs: Dict[str, pd.DataFrame],
                   kpis: Dict[str, Any],
                   diag: Dict[str, Any],
                   period: int,
                   scenario: Dict[str, Any] = None,
                   user_prompt: str = "",
                   before_after: Dict[str, Dict[str, pd.DataFrame]] = None):
    """Render KPIs → Diagnostics → Map → Ask GENIE → Preview/Delta → Export; survives widget reruns."""
    # KPIs & Summary
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 KPIs"); st.write(kpis)
    with c2:
        st.subheader("📝 Executive Summary")
        if build_summary:
            st.markdown(build_summary(user_prompt or "", scenario or {}, kpis, diag))
        else:
            st.info("reporter.build_summary missing.")

    # Diagnostics
    st.subheader("🧪 Flow Diagnostics")
    try:
        tc = dfs.get("Transport Cost", pd.DataFrame())
        lanes_period = 0
        if isinstance(tc, pd.DataFrame) and not tc.empty:
            tcf = tc.copy()
            try: tcf = tcf[tcf["Period"] == period]
            except Exception: pass
            if "Available" in tcf.columns: tcf = tcf[tcf["Available"].fillna(1) != 0]
            if "Cost Per UOM" in tcf.columns: tcf = tcf[pd.to_numeric(tcf["Cost Per UOM"], errors="coerce").fillna(0) >= 0]
            lanes_period = len(tcf)
        arcs_considered = (diag or {}).get("num_arcs", None)
        flows = (diag or {}).get("flows", [])
        flows_count = len(flows) if isinstance(flows, list) else 0
        geocoded = 0
        arcs_df = pd.DataFrame()
        if flows_to_geo and isinstance(flows, list):
            try:
                arcs_df = flows_to_geo(flows, dfs); geocoded = len(arcs_df) if isinstance(arcs_df, pd.DataFrame) else 0
            except Exception: geocoded = 0
        st.write({"period": period, "lanes_in_transport_cost": lanes_period, "optimizer_arcs_considered": arcs_considered, "flows_returned": flows_count, "flows_geocoded_for_map": geocoded})
        if (kpis or {}).get("status") in {"no_feasible_arcs","no_demand","no_positive_demand"}:
            st.warning("No flows because the optimizer couldn't route demand. Check lanes, demand > 0, and warehouse capacity/availability.")
        elif flows_count > 0 and geocoded == 0:
            st.warning("Flows exist but none were plotted. Add a 'Locations' sheet with `Location, Latitude, Longitude` for Warehouse.Location and Customer names.")
    except Exception as e:
        st.info(f"Diagnostics unavailable: {e}")

    # Map with flows (thin + color by product + legend)
    st.subheader("🌍 Network Map (With Flows)")
    if pdk and build_nodes and flows_to_geo and guess_map_center:
        try:
            nodes_df = build_nodes(dfs)
            flow_list = (diag or {}).get("flows", [])
            arcs_df = flows_to_geo(flow_list, dfs)

            if nodes_df.empty:
                st.info("No geocoded nodes available. Add a 'Locations' sheet with Latitude/Longitude or recognized city names.")
            else:
                lat0, lon0 = guess_map_center(nodes_df)
                wh = nodes_df[nodes_df["type"]=="warehouse"]; cu = nodes_df[nodes_df["type"]=="customer"]
                layers=[]
                if not wh.empty:
                    layers.append(
                        pdk.Layer("ScatterplotLayer", data=wh, get_position='[lon, lat]', get_radius=45000,
                                  pickable=True, filled=True, get_fill_color=[30,136,229])
                    )
                if not cu.empty:
                    layers.append(
                        pdk.Layer("ScatterplotLayer", data=cu, get_position='[lon, lat]', get_radius=30000,
                                  pickable=True, filled=True, get_fill_color=[76,175,80])
                    )

                if isinstance(arcs_df, pd.DataFrame) and not arcs_df.empty:
                    arcs_df = arcs_df[arcs_df["qty"] > 1e-9].copy()
                    if not arcs_df.empty:
                        qmax = float(arcs_df["qty"].max())
                        arcs_df["width"] = 1.0 if qmax <= 0 else (1.0 + 5.0 * (arcs_df["qty"] / qmax).clip(0, 1))

                        base_palette = [
                            (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                            (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128),
                        ]
                        prod_colors = {}
                        def color_for(prod: str):
                            if prod in prod_colors: return prod_colors[prod]
                            if len(prod_colors) < len(base_palette):
                                prod_colors[prod] = base_palette[len(prod_colors)]
                            else:
                                import colorsys
                                h = abs(hash(prod)) % 360
                                r, g, b = colorsys.hsv_to_rgb(h/360.0, 0.7, 1.0)
                                prod_colors[prod] = (int(r*255), int(g*255), int(b*255))
                            return prod_colors[prod]

                        arcs_df["rgb"] = arcs_df["product"].fillna("Unknown").map(color_for)
                        arcs_df["color"] = arcs_df["rgb"].apply(lambda t: [int(t[0]), int(t[1]), int(t[2])])

                        layers.append(
                            pdk.Layer(
                                "ArcLayer",
                                data=arcs_df,
                                get_source_position='[from_lon, from_lat]',
                                get_target_position='[to_lon, to_lat]',
                                get_width='width',
                                get_source_color='color',
                                get_target_color='color',
                                pickable=True,
                            )
                        )

                deck = pdk.Deck(
                    initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=3.3),
                    layers=layers,
                    tooltip={"html":"<b>{from}</b> → <b>{to}</b><br/><i>{product}</i>: {qty}", "style":{"color":"white"}},
                    map_style="dark",
                )
                st.pydeck_chart(deck)

                # Legend
                try:
                    if isinstance(arcs_df, pd.DataFrame) and not arcs_df.empty:
                        legend_rows = (
                            arcs_df[["product", "color"]]
                            .drop_duplicates()
                            .sort_values("product")
                            .values.tolist()
                        )
                        if legend_rows:
                            html = "<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;'>"
                            for prod, col in legend_rows:
                                r, g, b = col
                                html += (
                                    "<div style='display:flex;align-items:center;gap:6px; margin:2px 6px;'>"
                                    f"<span style='display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});border-radius:2px;'></span>"
                                    f"<span style='font-size:12px;'>{prod}</span></div>"
                                )
                            html += "</div>"
                            st.markdown(html, unsafe_allow_html=True)
                except Exception:
                    pass

                if (diag or {}).get("flows", []) and (arcs_df is None or arcs_df.empty):
                    st.warning("Flows exist but none were plotted. Ensure warehouse/customer names resolve to coordinates (Locations sheet or known city names).")
        except Exception as e:
            st.warning(f"Map rendering skipped: {e}")
    else:
        st.info("pydeck not available or geo module missing; skipping map.")

    # ---- Ask GENIE (immediately under map) ----
    st.subheader("💬 Ask GENIE about results")
    qa_mode = st.radio("Answer mode", ["Grounded (no LLM)","Gemini (LLM)","OpenAI (LLM)"], index=0, horizontal=True, key="qa_mode")
    with st.form("qa_form", clear_on_submit=False):
        q = st.text_input("Ask things like: 'Which warehouse has the lowest throughput?' 'How many customers are unserved?' 'Lowest-demand customers?'", key="qa_text")
        submit_qa = st.form_submit_button("Ask")
    if submit_qa:
        # Always try grounded first for known intents
        ans = grounded_answer(q, dfs, kpis, diag, period=period)

        # If LLM selected or not recognized, escalate with bounded context
        use_llm = (qa_mode.startswith("Gemini") or qa_mode.startswith("OpenAI")) and (answer_question is not None)
        if (not ans) and use_llm:
            context = build_llm_context(dfs, kpis, diag, period=period)
            prov = "gemini" if qa_mode.startswith("Gemini") else "openai"
            q2 = f"Use ONLY this context to answer:\n{context}\n\nQuestion: {q}\nAnswer briefly and numerically when possible."
            try:
                ans = answer_question(q2, kpis, diag, provider=prov, dfs=dfs, force_llm=True)
            except Exception as e:
                ans = f"(LLM error; fallback) {grounded_answer(q, dfs, kpis, diag, period=period) or 'Sorry, I could not answer that.'}"
        if not ans:
            ans = "Sorry, I couldn't interpret that. Try asking about service %, unserved customers, lowest-demand customers, top flows, or lowest throughput warehouses."
        st.markdown(f"**Answer**: {ans}")

    # ---- Before/After preview + Delta (only when user actually edited) ----
    if before_after:
        # Quick preview
        st.subheader("📋 Before vs After (Quick Preview)")
        b = before_after.get("before", {})
        a = before_after.get("after", {})
        tabs = st.tabs(["Customers","Customer Product Data","Warehouse","Supplier Product","Transport Cost"])
        with tabs[0]:
            cA, cB = st.columns(2)
            with cA: st.markdown("**Before**"); st.dataframe(b.get("Customers", pd.DataFrame()).head(25), use_container_width=True)
            with cB: st.markdown("**After**");  st.dataframe(a.get("Customers", pd.DataFrame()).head(25), use_container_width=True)
        with tabs[1]:
            cA, cB = st.columns(2)
            with cA: st.markdown("**Before**"); st.dataframe(b.get("Customer Product Data", pd.DataFrame()).head(25), use_container_width=True)
            with cB: st.markdown("**After**");  st.dataframe(a.get("Customer Product Data", pd.DataFrame()).head(25), use_container_width=True)
        with tabs[2]:
            cA, cB = st.columns(2)
            with cA: st.markdown("**Before**"); st.dataframe(b.get("Warehouse", pd.DataFrame()).head(25), use_container_width=True)
            with cB: st.markdown("**After**");  st.dataframe(a.get("Warehouse", pd.DataFrame()).head(25), use_container_width=True)
        with tabs[3]:
            cA, cB = st.columns(2)
            with cA: st.markdown("**Before**"); st.dataframe(b.get("Supplier Product", pd.DataFrame()).head(25), use_container_width=True)
            with cB: st.markdown("**After**");  st.dataframe(a.get("Supplier Product", pd.DataFrame()).head(25), use_container_width=True)
        with tabs[4]:
            cA, cB = st.columns(2)
            with cA: st.markdown("**Before**"); st.dataframe(b.get("Transport Cost", pd.DataFrame()).head(25), use_container_width=True)
            with cB: st.markdown("**After**");  st.dataframe(a.get("Transport Cost", pd.DataFrame()).head(25), use_container_width=True)

        # Delta tables shown only if user made edits (base model → skip)
        user_change_count = before_after.get("user_change_count", 0)
        if user_change_count > 0:
            st.subheader("Δ Delta Views (Changes Only)")
            show_delta("Customers", b.get("Customers", pd.DataFrame()), a.get("Customers", pd.DataFrame()), ["Customer","Location"])
            show_delta("Customer Product Data", b.get("Customer Product Data", pd.DataFrame()), a.get("Customer Product Data", pd.DataFrame()), ["Product","Customer","Location","Period"])
            bw, aw = b.get("Warehouse", pd.DataFrame()), a.get("Warehouse", pd.DataFrame())
            if isinstance(aw, pd.DataFrame) and not aw.empty:
                wh_keys = ["Warehouse"] + (["Period"] if "Period" in aw.columns and "Period" in bw.columns else [])
                show_delta("Warehouse", bw, aw, wh_keys)
            show_delta("Supplier Product", b.get("Supplier Product", pd.DataFrame()), a.get("Supplier Product", pd.DataFrame()), ["Product","Supplier","Location","Period"])
            show_delta("Transport Cost", b.get("Transport Cost", pd.DataFrame()), a.get("Transport Cost", pd.DataFrame()), ["Mode of Transport","Product","From Location","To Location","Period"])
        else:
            st.info("No user edits detected (base model). Skipping delta tables.")

    # Export current scenario sheets
    st.subheader("💾 Export")
    try:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            for name, df in dfs.items():
                if isinstance(df, pd.DataFrame):
                    sname = str(name)[:31] or "Sheet"
                    df.to_excel(w, sheet_name=sname, index=False)
        st.download_button(
            "Download Updated Scenario Excel",
            data=buf.getvalue(),
            file_name="genie_updated_scenario.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.warning(f"Creating the Excel download failed: {e}")

# ----------------------- Main page flow -----------------------
uploaded = st.file_uploader("📤 Upload base case Excel (.xlsx)", type=["xlsx"])

if uploaded:
    # Load + validate
    try:
        dfs, report = loader(uploaded)
    except Exception as e:
        st.error(f"❌ Error reading Excel: {e}"); st.stop()

    with st.expander("✅ Sheet Validation Report"):
        try:
            for sheet, rep in report.items():
                if sheet == "_warnings": continue
                if not isinstance(rep, dict): continue
                miss = rep.get("missing_columns", [])
                if miss: st.error(f"{sheet}: Missing columns - {miss}")
                else:    st.success(f"{sheet}: OK ({rep.get('num_rows',0)} rows, {rep.get('num_columns',0)} columns)")
            if isinstance(report.get("_warnings", []), list) and report["_warnings"]:
                st.warning("General warnings:")
                for w in report["_warnings"]: st.write(f"• {w}")
        except Exception as e:
            st.warning(f"Validation display issue: {e}")

    st.success("Base case loaded successfully.")

    # Suggested prompts ABOVE the prompt box
    st.subheader("💡 Suggested prompts")
    try:
        fp = hashlib.sha1("|".join(sorted([f"{k}:{v.shape[0]}x{v.shape[1]}" for k, v in dfs.items() if isinstance(v, pd.DataFrame)])).encode()).hexdigest()
    except Exception:
        fp = "nofp"
    if "examples_cache" not in st.session_state: st.session_state["examples_cache"] = {}
    dyn = st.session_state["examples_cache"].get(fp)
    if dyn is None:
        try:
            dyn = examples_for_file(dfs, provider="gemini") if examples_for_file else []
        except Exception:
            dyn = []
        st.session_state["examples_cache"][fp] = dyn

    with st.expander("Show suggestions"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Static**")
            for ex in STATIC_EXAMPLES: st.code(ex, language="text")
            pick1 = st.selectbox("Insert static example:", ["(choose)"] + STATIC_EXAMPLES, key="pick_static")
            if st.button("Insert static"):
                if pick1 != "(choose)":
                    st.session_state["prefill_prompt"] = pick1; st.rerun()
        with c2:
            st.markdown("**From your file**")
            if dyn:
                for ex in dyn: st.code(ex, language="text")
                pick2 = st.selectbox("Insert generated example:", ["(choose)"] + dyn, key="pick_dyn")
                if st.button("Insert generated"):
                    if pick2 != "(choose)":
                        st.session_state["prefill_prompt"] = pick2; st.rerun()
            else:
                st.info("Upload contains limited metadata; static examples only.")

    # Map of nodes (on upload)
    st.subheader("🌍 Network Map (Nodes)")
    if pdk and build_nodes and guess_map_center:
        try:
            nodes = build_nodes(dfs)
            if nodes.empty:
                st.info("No geocoded nodes. Add a 'Locations' sheet with Location, Latitude, Longitude or use city-like names.")
            else:
                lat0, lon0 = guess_map_center(nodes)
                wh = nodes[nodes["type"]=="warehouse"]; cu = nodes[nodes["type"]=="customer"]
                layers=[]
                if not wh.empty:
                    layers.append(pdk.Layer("ScatterplotLayer", data=wh, get_position='[lon, lat]', get_radius=60000, pickable=True, filled=True, get_fill_color=[30,136,229]))
                if not cu.empty:
                    layers.append(pdk.Layer("ScatterplotLayer", data=cu, get_position='[lon, lat]', get_radius=40000, pickable=True, filled=True, get_fill_color=[76,175,80]))
                deck=pdk.Deck(initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=3.2), layers=layers, tooltip={"html":"<b>{name}</b><br/>{location}","style":{"color":"white"}})
                st.pydeck_chart(deck)
        except Exception as e:
            st.warning(f"Map rendering skipped: {e}")
    else:
        st.info("pydeck not available or geo module missing; skipping map.")

    st.markdown("---")

    # ---------------- Manual Edits (CRUD) ----------------
    st.subheader("🛠️ Manual Edits (point‑and‑click)")
    with st.expander("Open Manual Edits"):
        # Pull values for selects
        products  = sorted(dfs.get("Customer Product Data", pd.DataFrame()).get("Product", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist())
        customers = sorted(dfs.get("Customers", pd.DataFrame()).get("Customer", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist())
        locations_c = sorted(dfs.get("Customers", pd.DataFrame()).get("Location", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist())
        warehouses = sorted(dfs.get("Warehouse", pd.DataFrame()).get("Warehouse", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist())
        locations_w = sorted(dfs.get("Warehouse", pd.DataFrame()).get("Location", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist())
        modes      = sorted(dfs.get("Transport Cost", pd.DataFrame()).get("Mode of Transport", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist())

        tabs_crud = st.tabs(["Demand update","Add customer","Add/Update warehouse","Add/Update lane","Delete"])

        # Demand update
        with tabs_crud[0]:
            st.caption("Update demand for a given Product × Customer × (Location=Customer by default)")
            p = st.selectbox("Product", products, key="crud_dem_p") if products else st.text_input("Product (type exact)", key="crud_dem_p_txt")
            c = st.selectbox("Customer", customers, key="crud_dem_c") if customers else st.text_input("Customer (type exact)", key="crud_dem_c_txt")
            loc = st.text_input("Location (defaults to Customer)", value="", key="crud_dem_loc")
            col1, col2 = st.columns(2)
            with col1:
                delta = st.number_input("Δ% (apply percent change, optional)", value=0.0, step=1.0, format="%.2f")
            with col2:
                set_abs = st.number_input("Set absolute Demand (overrides Δ%, optional)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
            lt = st.number_input("Set Lead Time (optional)", min_value=0, value=0, step=1)
            if st.button("➕ Queue demand update"):
                loc_final = (loc.strip() or (c if isinstance(c, str) else ""))
                ed = {"product": str(p), "customer": str(c), "location": str(loc_final)}
                if set_abs > 0:
                    ed["set"] = {"Demand": float(set_abs)}
                elif abs(delta) > 0:
                    ed["delta_pct"] = float(delta)
                if lt > 0:
                    ed.setdefault("set", {})["Lead Time"] = int(lt)
                st.session_state.setdefault("manual_scenario", {"period": 2023})
                st.session_state["manual_scenario"].setdefault("demand_updates", []).append(ed)
                st.success(f"Queued: {ed}")

        # Add customer
        with tabs_crud[1]:
            st.caption("Create customer and (optionally) its demand row.")
            nc = st.text_input("New Customer name", key="crud_addcust_name")
            nloc = st.text_input("Location (defaults to same name)", key="crud_addcust_loc")
            ap = st.selectbox("Product for initial demand (optional)", ["(none)"] + products, key="crud_addcust_prod")
            ad = st.number_input("Initial Demand (optional)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
            alt = st.number_input("Lead Time (optional)", min_value=0, value=0, step=1)
            if st.button("➕ Queue customer add"):
                loc_final = nloc.strip() or nc.strip()
                scn = st.session_state.setdefault("manual_scenario", {"period": 2023})
                scn.setdefault("adds", {}).setdefault("Customers", []).append({"Customer": nc.strip(), "Location": loc_final})
                if ap != "(none)" and ad > 0:
                    scn.setdefault("adds", {}).setdefault("Customer Product Data", []).append({
                        "Product": ap, "Customer": nc.strip(), "Location": loc_final, "Period": 2023, "UOM": "Each",
                        "Demand": float(ad), "Lead Time": int(alt) if alt>0 else None
                    })
                st.success(f"Queued: Customer {nc} (Location {loc_final})")

        # Add/Update warehouse
        with tabs_crud[2]:
            st.caption("Create or modify a warehouse row.")
            mode_w = st.radio("Action", ["Add new","Update existing"], horizontal=True, key="crud_wh_mode")
            if mode_w == "Add new":
                nw = st.text_input("Warehouse name", key="crud_wh_new")
                nlocw = st.text_input("Location", key="crud_wh_new_loc")
                cap = st.number_input("Maximum Capacity", min_value=0.0, value=0.0, step=100.0)
                f_open = st.checkbox("Force Open", value=True)
                if st.button("➕ Queue warehouse add"):
                    scn = st.session_state.setdefault("manual_scenario", {"period": 2023})
                    scn.setdefault("adds", {}).setdefault("Warehouse", []).append({
                        "Warehouse": nw.strip(), "Location": nlocw.strip() or nw.strip(), "Maximum Capacity": float(cap),
                        "Force Open": 1 if f_open else 0
                    })
                    st.success(f"Queued: Warehouse {nw}")
            else:
                uw = st.selectbox("Warehouse", warehouses, key="crud_wh_upd")
                field = st.selectbox("Field", ["Maximum Capacity","Fixed Cost","Variable Cost","Force Open","Force Close"], key="crud_wh_field")
                newv = st.text_input("New value (number or 0/1 for Force)", key="crud_wh_val")
                if st.button("➕ Queue warehouse update"):
                    v = float(newv) if field in {"Maximum Capacity","Fixed Cost","Variable Cost"} else int(float(newv or 0))
                    scn = st.session_state.setdefault("manual_scenario", {"period": 2023})
                    scn.setdefault("warehouse_changes", []).append({"warehouse": uw, "field": field, "new_value": v})
                    st.success(f"Queued: {uw} {field} -> {v}")

        # Add/Update lane
        with tabs_crud[3]:
            st.caption("Create or modify a Transport Cost lane.")
            m = st.selectbox("Mode of Transport", modes if modes else ["Secondary Delivery LTL"], key="crud_tc_mode")
            fp = st.selectbox("Product (blank applies to all demand for To)", ["(blank)"] + products, key="crud_tc_prod")
            fr = st.selectbox("From (use warehouse name or its Location)", warehouses + locations_w, key="crud_tc_from")
            to = st.selectbox("To (use Customer name or Customers.Location)", customers + locations_c, key="crud_tc_to")
            cpu = st.number_input("Cost Per UOM (leave 0 to compute via distance*cost_per_distance if your file uses those)", min_value=0.0, value=10.0, step=0.5)
            avail = st.selectbox("Available", [1,0], index=0)
            if st.button("➕ Queue lane upsert"):
                prod_val = "" if fp == "(blank)" else fp
                st.session_state.setdefault("manual_scenario", {"period": 2023})
                st.session_state["manual_scenario"].setdefault("transport_updates", []).append({
                    "mode": m, "product": prod_val, "from_location": fr, "to_location": to, "period": 2023,
                    "fields": {"Cost Per UOM": float(cpu), "Available": int(avail)}
                })
                st.success(f"Queued: {m} {fr} -> {to} ({prod_val or 'ALL'})")

        # Delete
        with tabs_crud[4]:
            st.caption("Careful: deletions require explicit checkbox during Process.")
            del_kind = st.selectbox("What to delete?", ["Transport lane","Customer","Warehouse"], key="crud_del_kind")
            if del_kind == "Transport lane":
                m = st.selectbox("Mode", modes if modes else ["Secondary Delivery LTL"], key="crud_del_mode")
                fr = st.text_input("From Location (warehouse or its location)", key="crud_del_from")
                to = st.text_input("To Location (customer or its location)", key="crud_del_to")
                prod = st.text_input("Product (exact; leave blank to match all rows)", key="crud_del_prod")
                if st.button("➕ Queue delete"):
                    st.session_state.setdefault("manual_scenario", {"period": 2023})
                    st.session_state["manual_scenario"].setdefault("deletes", {}).setdefault("Transport Cost", []).append({
                        "Mode of Transport": m, "From Location": fr.strip(), "To Location": to.strip(), "Product": prod.strip(), "Period": 2023
                    })
                    st.success("Queued: delete lane")
            elif del_kind == "Customer":
                c = st.selectbox("Customer", customers, key="crud_del_cust")
                if st.button("➕ Queue delete customer"):
                    st.session_state.setdefault("manual_scenario", {"period": 2023})
                    st.session_state["manual_scenario"].setdefault("deletes", {}).setdefault("Customers", []).append({"Customer": c})
                    st.success(f"Queued: delete customer {c}")
            else:
                w = st.selectbox("Warehouse", warehouses, key="crud_del_wh")
                if st.button("➕ Queue delete warehouse"):
                    st.session_state.setdefault("manual_scenario", {"period": 2023})
                    st.session_state["manual_scenario"].setdefault("deletes", {}).setdefault("Warehouse", []).append({"Warehouse": w})
                    st.success(f"Queued: delete warehouse {w}")

        queued = st.session_state.get("manual_scenario")
        if queued:
            st.markdown("**Queued Manual Scenario (will be merged with Prompt scenario)**")
            st.json(queued)

    # ---------------- Natural-language Prompt ----------------
    st.subheader("🧠 Describe your what‑if scenario")
    user_prompt = st.text_area("Prompt (or type 'run the base model')", height=120, key="user_prompt", value=user_prompt_default)
    go = st.button("🚀 Process Scenario")

    # Optional LLM vs Rules comparison
    if show_llm_vs_rules and (user_prompt or "").strip():
        tabs = st.tabs(["LLM Parser","Rules Parser","Diff"])
        llm_scn = rules_scn = None
        with tabs[0]:
            if parse_with_llm is None or provider == "Rules only (no LLM)":
                st.error("GenAI parser disabled or not available.")
            else:
                prov = "gemini" if provider.startswith("Google") else "openai"
                try: llm_scn = parse_with_llm(user_prompt, dfs, default_period=2023, provider=prov)
                except Exception as e: llm_scn = {}; st.error(f"LLM parse error: {e}")
                if summarize_scenario: st.markdown("**Summary**"); st.markdown("\n".join(f"- {b}" for b in summarize_scenario(llm_scn)))
                with st.expander("Advanced: raw JSON"): st.json(llm_scn)
        with tabs[1]:
            if parse_rules is None:
                st.error("Rules parser not available.")
            else:
                try: rules_scn = parse_rules(user_prompt, dfs, default_period=2023)
                except Exception as e: rules_scn = {}; st.error(f"Rules parse error: {e}")
                if summarize_scenario: st.markdown("**Summary**"); st.markdown("\n".join(f"- {b}" for b in summarize_scenario(rules_scn)))
                with st.expander("Advanced: raw JSON"): st.json(rules_scn)
        with tabs[2]:
            try:
                import difflib
                a = json.dumps(llm_scn or {}, indent=2, sort_keys=True).splitlines()
                b = json.dumps(rules_scn or {}, indent=2, sort_keys=True).splitlines()
                diff = difflib.unified_diff(a, b, fromfile="LLM", tofile="Rules", lineterm="")
                st.code("\n".join(diff) or "No differences.")
            except Exception as e:
                st.info(f"Diff unavailable: {e}")

    # ---------------- Process scenario (Prompt + Manual) ----------------
    if go and (user_prompt or "").strip():
        # 1) Parse
        try:
            if user_prompt.strip().lower() in {"run the base model","run base model"}:
                prompt_scn = {"period": 2023}
                parsed_by = "base"
            elif use_llm_parser and parse_with_llm is not None and provider != "Rules only (no LLM)":
                prov = "gemini" if provider.startswith("Google") else "openai"
                prompt_scn = parse_with_llm(user_prompt, dfs, default_period=2023, provider=prov); parsed_by = provider
            else:
                prompt_scn = parse_rules(user_prompt, dfs, default_period=2023) if parse_rules else {}
                parsed_by = "rules"
        except Exception as e:
            st.error(f"❌ Parsing failed: {e}"); st.stop()

        # 2) Merge Manual Edits
        manual = st.session_state.get("manual_scenario", {})
        scenario = {}
        scenario["period"] = (manual.get("period") or prompt_scn.get("period") or 2023)
        for key in ["demand_updates","warehouse_changes","supplier_changes","transport_updates"]:
            scenario[key] = []
            if isinstance(prompt_scn.get(key), list): scenario[key].extend(prompt_scn[key])
            if isinstance(manual.get(key), list):     scenario[key].extend(manual[key])
        for key in ["adds","deletes"]:
            if prompt_scn.get(key) or manual.get(key):
                scenario[key] = {}
                if isinstance(prompt_scn.get(key), dict):
                    for k,v in prompt_scn[key].items():
                        scenario[key].setdefault(k, []).extend(v)
                if isinstance(manual.get(key), dict):
                    for k,v in manual[key].items():
                        scenario[key].setdefault(k, []).extend(v)

        st.subheader(f"Scenario to apply ({parsed_by} + manual)")
        if summarize_scenario: st.markdown("\n".join(f"- {b}" for b in summarize_scenario(scenario)))
        with st.expander("Advanced: raw JSON"): st.json(scenario)

        # 3) Deletion safety
        pending_del = sum(len(v) for v in (scenario.get("deletes") or {}).values()) if isinstance(scenario.get("deletes"), dict) else 0
        allow_delete = True
        if pending_del > 0:
            allow_delete = st.checkbox(f"🛑 Apply deletions ({pending_del} row(s))", value=False)

        # 4) Before copies
        b_cpd = dfs.get("Customer Product Data", pd.DataFrame()).copy()
        b_wh  = dfs.get("Warehouse", pd.DataFrame()).copy()
        b_sp  = dfs.get("Supplier Product", pd.DataFrame()).copy()
        b_tc  = dfs.get("Transport Cost", pd.DataFrame()).copy()
        b_cu  = dfs.get("Customers", pd.DataFrame()).copy()

        # 5) Apply
        try:
            newdfs = apply_scenario(dfs, scenario, allow_delete=allow_delete)
        except Exception as e:
            st.error(f"❌ Applying scenario failed: {e}"); st.stop()

        a_cpd = newdfs.get("Customer Product Data", pd.DataFrame()).copy()
        a_wh  = newdfs.get("Warehouse", pd.DataFrame()).copy()
        a_sp  = newdfs.get("Supplier Product", pd.DataFrame()).copy()
        a_tc  = newdfs.get("Transport Cost", pd.DataFrame()).copy()
        a_cu  = newdfs.get("Customers", pd.DataFrame()).copy()

        # 6) Auto-connect (doesn't count as "user edits" for delta visibility)
        auto = st.checkbox("🔗 Auto-create missing lanes for feasibility (demo)", value=True)
        auto_created = 0
        if auto:
            newdfs, auto_created = autoconnect_lanes(newdfs, period=scenario.get("period", 2023), default_cost=10.0)
            if auto_created > 0: st.info(f"Auto-connect created {auto_created} placeholder lane(s).")

        # 7) Optimize
        if run_optimizer is None: st.error("Optimizer module missing."); st.stop()
        kpis, diag = run_optimizer(newdfs, period=scenario.get("period", 2023))

        # Persist every piece needed for sticky results
        st.session_state["last_newdfs"]  = newdfs
        st.session_state["last_kpis"]    = kpis
        st.session_state["last_diag"]    = diag
        st.session_state["last_period"]  = scenario.get("period", 2023)
        st.session_state["last_prompt"]  = user_prompt
        st.session_state["last_scenario"]= scenario

        # Track before/after for delta (only if user made edits)
        user_change_count = 0
        for k in ["demand_updates","warehouse_changes","supplier_changes","transport_updates"]:
            user_change_count += len(scenario.get(k, []) or [])
        if isinstance(scenario.get("adds"), dict):
            user_change_count += sum(len(v) for v in scenario["adds"].values())
        if isinstance(scenario.get("deletes"), dict):
            user_change_count += sum(len(v) for v in scenario["deletes"].values())
        st.session_state["last_before_after"] = {
            "before": {
                "Customer Product Data": b_cpd, "Warehouse": b_wh, "Supplier Product": b_sp, "Transport Cost": b_tc, "Customers": b_cu
            },
            "after": {
                "Customer Product Data": a_cpd, "Warehouse": a_wh, "Supplier Product": a_sp, "Transport Cost": a_tc, "Customers": a_cu
            },
            "user_change_count": user_change_count
        }

        # Render results now
        render_results(
            dfs=newdfs, kpis=kpis, diag=diag, period=scenario.get("period", 2023),
            scenario=scenario, user_prompt=user_prompt,
            before_after=st.session_state["last_before_after"]
        )

    # --- Sticky results view (so changing Gemini/OpenAI radio won't reset the page) ---
    elif st.session_state.get("last_newdfs") is not None and st.session_state.get("last_kpis") is not None:
        render_results(
            dfs=st.session_state["last_newdfs"],
            kpis=st.session_state["last_kpis"],
            diag=st.session_state["last_diag"],
            period=st.session_state.get("last_period", 2023),
            scenario=st.session_state.get("last_scenario", {}),
            user_prompt=st.session_state.get("last_prompt", ""),
            before_after=st.session_state.get("last_before_after", None)
        )

else:
    st.info("Upload your base case to begin. Or download a sample template:")
    raw_url = "https://raw.githubusercontent.com/5av1t/genie/main/sample_base_case.xlsx"
    if os.path.exists("sample_base_case.xlsx"):
        with open("sample_base_case.xlsx","rb") as f:
            st.download_button("⬇️ Download Sample Base Case Template", data=f, file_name="sample_base_case.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.markdown(f"[⬇️ Download Sample Base Case Template]({raw_url})")
        st.caption("Tip: add `sample_base_case.xlsx` into repo root to enable in‑app download.")
    st.subheader("💡 Suggested prompts")
    with st.expander("Show suggestions"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Static**")
            for ex in STATIC_EXAMPLES: st.code(ex, language="text")
            pick = st.selectbox("Insert static example:", ["(choose)"] + STATIC_EXAMPLES, key="pick_static_nf")
            if st.button("Insert static", key="insert_static_nf"):
                if pick != "(choose)":
                    st.session_state["prefill_prompt"] = pick; st.rerun()
        with c2:
            st.markdown("**From your file**"); st.info("Upload a file to generate tailored prompts.")
    st.text_area("🧠 Describe your what‑if scenario", height=120, key="user_prompt", value=user_prompt_default)
