# app.py — GENIE: Supply Chain Network Designer
# Upload → Validate → Nodes map → Scenario (Rules/LLM) → Apply edits → Optimize
# → KPIs + Flow map → Q&A → Download

from __future__ import annotations
import os, io, json
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import requests

# ------------------------------ App Config ------------------------------
st.set_page_config(
    page_title="GENIE — Supply Chain Network Designer",
    page_icon="🧞‍♂️",
    layout="wide",
)

# ------------------------------ Secrets -> Env ------------------------------
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
if "gcp" in st.secrets and "gemini_api_key" in st.secrets["gcp"]:
    os.environ["GEMINI_API_KEY"] = st.secrets["gcp"]["gemini_api_key"]
if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
    os.environ["GEMINI_API_KEY"] = st.secrets["gemini"]["api_key"]

DEFAULT_PERIOD = 2023
missing_engine: List[str] = []

# ------------------------------ Engine Imports (defensive) ------------------------------
try:
    from engine.loader import load_and_validate_excel, build_model_index
except ModuleNotFoundError:
    load_and_validate_excel = None  # type: ignore
    build_model_index = None  # type: ignore
    missing_engine.append("engine/loader.py (load_and_validate_excel, build_model_index)")

try:
    from engine.parser import parse_rules
except ModuleNotFoundError:
    parse_rules = None  # type: ignore
    missing_engine.append("engine/parser.py (parse_rules)")

try:
    from engine.updater import apply_scenario_edits, diff_tables
except ModuleNotFoundError:
    apply_scenario_edits = None  # type: ignore
    diff_tables = None  # type: ignore
    missing_engine.append("engine/updater.py (apply_scenario_edits, diff_tables)")

try:
    from engine.optimizer import run_optimizer
except ModuleNotFoundError:
    run_optimizer = None  # type: ignore
    missing_engine.append("engine/optimizer.py (run_optimizer)")

try:
    from engine.geo import build_nodes, flows_to_geo, guess_map_center, ensure_location_row
except ModuleNotFoundError:
    build_nodes = None  # type: ignore
    flows_to_geo = None  # type: ignore
    guess_map_center = None  # type: ignore
    ensure_location_row = None  # type: ignore
    missing_engine.append("engine/geo.py (build_nodes, flows_to_geo, guess_map_center, ensure_location_row)")

try:
    from engine.genai import parse_with_llm, answer_question, suggest_location_candidates
except ModuleNotFoundError:
    parse_with_llm = None  # type: ignore
    answer_question = None  # type: ignore
    suggest_location_candidates = None  # type: ignore
    missing_engine.append("engine/genai.py (parse_with_llm, answer_question, suggest_location_candidates)")

try:
    from engine.example_gen import rules_examples, llm_examples
except ModuleNotFoundError:
    rules_examples = None  # type: ignore
    llm_examples = None  # type: ignore
    missing_engine.append("example_gen.py (rules_examples, llm_examples)")

# ------------------------------ Helpers ------------------------------
def _downloadable_excel(dfs: Dict[str, pd.DataFrame], filename: str = "scenario.xlsx") -> Tuple[bytes, str]:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in dfs.items():
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=str(name)[:31] or "Sheet1", index=False)
    bio.seek(0)
    return bio.read(), filename

def _thin_color_for_product(p: str) -> List[int]:
    palette = [
        [33, 150, 243], [76, 175, 80], [244, 67, 54], [255, 193, 7], [156, 39, 176],
        [0, 188, 212], [121, 85, 72], [63, 81, 181], [255, 87, 34], [205, 220, 57],
    ]
    if not p:
        return [120, 120, 120]
    return palette[abs(hash(p)) % len(palette)]

def _arc_layer_from_flows_geo(geo_df: pd.DataFrame):
    if geo_df is None or geo_df.empty:
        return None
    q = geo_df["qty"].astype(float).clip(lower=0.0).fillna(0.0)
    width = (1.0 + np.sqrt(q) * 0.5).tolist()  # thin lines
    colors = []
    for _, r in geo_df.iterrows():
        colors.append(_thin_color_for_product(str(r.get("product", ""))) + [180])
    return pdk.Layer(
        "ArcLayer",
        data=geo_df.assign(width=width, color=colors),
        get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"],
        get_width="width",
        get_source_color="color",
        get_target_color="color",
        pickable=True,
        auto_highlight=True,
    )

def _scatter_layers_from_nodes(nodes_df: pd.DataFrame):
    if nodes_df is None or nodes_df.empty:
        return []
    wh_df = nodes_df[nodes_df["type"] == "warehouse"]
    cu_df = nodes_df[nodes_df["type"] == "customer"]
    layers = []
    if not wh_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=wh_df,
                get_position=["lon", "lat"],
                get_radius=50000,
                get_fill_color=[142, 36, 170, 200],  # violet
                pickable=True,
            )
        )
    if not cu_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=cu_df,
                get_position=["lon", "lat"],
                get_radius=30000,
                get_fill_color=[66, 66, 66, 180],  # dark gray
                pickable=True,
            )
        )
    return layers

# ------------------------------ Title ------------------------------
st.title("🧞‍♂️ GENIE — Supply Chain Network Designer")

if missing_engine:
    st.error("Missing engine modules:\n- " + "\n- ".join(missing_engine))
    st.stop()

# ------------------------------ Session Defaults ------------------------------
ss = st.session_state
for k, v in {
    "dfs": {},
    "model_index": {},
    "scenario": {"period": DEFAULT_PERIOD},
    "updated_dfs": {},
    "kpis": {},
    "diag": {},
    "user_prompt": "",
    "qa_input": "",
    "pending_prompt": None,
}.items():
    if k not in ss:
        ss[k] = v

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.markdown("### 1) Upload Base-Case Excel")
    sample_url = "https://raw.githubusercontent.com/5av1t/genie/main/sample_base-case.xlsx"
    st.caption("Need a sample? Download the template:")

    # Robust sample download: try to fetch bytes else show link button
    try:
        r = requests.get(sample_url, timeout=10, headers={"User-Agent": "genie-app"})
        if r.ok and r.content:
            st.download_button(
                "Download Sample Base-Case",
                data=r.content,
                file_name="sample_base-case.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.link_button("Open sample in browser", sample_url)
    except Exception:
        st.link_button("Open sample in browser", sample_url)

    uploaded = st.file_uploader("Upload your workbook (.xlsx)", type=["xlsx"], key="uploader")

    st.markdown("---")
    st.markdown("**GenAI provider**")
    prov_options = ["None (grounded only)", "Google Gemini"]
    default_idx = 0
    prev_val = ss.get("prov_choice", None)
    if isinstance(prev_val, str) and prev_val in prov_options:
        default_idx = prov_options.index(prev_val)
    prov = st.radio("Choose", prov_options, index=default_idx, key="prov_choice")

    if prov == "Google Gemini":
        st.caption("Optional: set GEMINI_API_KEY for LLM parsing and Q&A.")
        gemini_existing = os.environ.get("GEMINI_API_KEY", "")
        gemini_key_in = st.text_input("GEMINI_API_KEY", type="password", value=gemini_existing, key="gemini_api_key_input")
        if gemini_key_in and gemini_key_in != gemini_existing:
            os.environ["GEMINI_API_KEY"] = gemini_key_in
            st.success("Gemini key set for this session.")
    else:
        st.caption("LLM disabled — app will use grounded/deterministic logic only.")

    st.markdown("---")
    st.markdown("**Use cases GENIE can do with GenAI**")
    st.write("- Natural language scenario edits (demand/warehouse/transport/supplier).")
    st.write("- CRUD via text (add/update/delete) with guardrails.")
    st.write("- Explanations of KPIs and flows using model context.")
    st.write("- Infeasibility helper: what’s missing and suggested fixes.")
    st.write("- Location helper: propose lat/lon and upsert into Locations.")

# ------------------------------ Load + Validate ------------------------------
if uploaded is not None:
    with st.spinner("Reading and validating workbook..."):
        dfs, validation_report = load_and_validate_excel(uploaded)
        ss["dfs"] = dfs
        ss["model_index"] = build_model_index(dfs, period=DEFAULT_PERIOD)
    st.success("Workbook loaded.")
    with st.expander("✅ Sheet Validation Report"):
        for sheet, report in validation_report.items():
            if sheet == "_warnings":
                continue
            if isinstance(report, dict) and report.get("missing_columns"):
                st.error(f"{sheet}: Missing columns - {report['missing_columns']}")
            else:
                if isinstance(report, dict):
                    st.success(f"{sheet}: OK ({report.get('num_rows',0)} rows, {report.get('num_columns',0)} cols)")
        if validation_report.get("_warnings"):
            st.warning("\n".join(validation_report["_warnings"]))

    # Nodes map immediately
    nodes_df = build_nodes(ss["dfs"])
    if nodes_df is not None and not nodes_df.empty:
        st.subheader("📍 Network Map (nodes)")
        lat, lon = guess_map_center(nodes_df)
        layers = _scatter_layers_from_nodes(nodes_df)
        st.pydeck_chart(pdk.Deck(
            map_style=None,  # avoid mapbox token issues
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=3.5),
            layers=layers,
            tooltip={"text": "{name} ({type})\n{location}"}
        ))
    else:
        st.info("No mappable nodes found. Ensure Warehouse/Customers have locations or provide a Locations sheet.")

st.markdown("---")

# ------------------------------ Scenario Editor ------------------------------
st.subheader("🧪 Scenario Editor")

# If an example was chosen in prior run, adopt it BEFORE rendering the text area
if ss.get("pending_prompt"):
    ss["user_prompt"] = ss.pop("pending_prompt", "")

left, right = st.columns([2, 1])

with left:
    st.markdown("**Insert an example from your file**")
    dyn_rules = rules_examples(ss["dfs"]) if rules_examples else []
    ex_choice = st.selectbox("Examples (dynamic)", ["(choose one)"] + dyn_rules, index=0, key="ex_choice_rules")
    if st.button("Insert example"):
        if ex_choice and ex_choice != "(choose one)":
            ss["pending_prompt"] = ex_choice
            st.experimental_rerun()

    st.markdown("**Write prompt** (natural language)")
    user_prompt_val = ss.get("user_prompt", "")
    st.text_area(
        "Describe your what-if...",
        value=user_prompt_val,
        height=120,
        key="user_prompt",
        label_visibility="collapsed",
    )

with right:
    st.markdown("**Parser**")
    parse_mode = st.radio("Use", ["Rules (deterministic)", "LLM (Gemini)"], index=0, key="parse_mode_choice")
    run_base = st.checkbox("Run base model (no edits)", value=False, key="run_base_flag")
    do_process = st.button("Process Scenario & Optimize", type="primary")

# ------------------------------ Process + Optimize ------------------------------
if do_process:
    if not ss["dfs"]:
        st.error("Please upload a workbook first.")
    else:
        dfs = ss["dfs"]
        scenario = {"period": DEFAULT_PERIOD, "demand_updates": [], "warehouse_changes": [], "supplier_changes": [], "transport_updates": []}
        text = ss.get("user_prompt", "") or ""

        if not run_base and text.strip():
            if parse_mode == "Rules (deterministic)":
                scenario = parse_rules(text, dfs, default_period=DEFAULT_PERIOD) if parse_rules else scenario
            else:
                provider = "gemini" if ss.get("prov_choice") == "Google Gemini" else "none"
                scenario_llm = parse_with_llm(text, dfs, default_period=DEFAULT_PERIOD, provider=provider) if parse_with_llm else {}
                rules_first = parse_rules(text, dfs, default_period=DEFAULT_PERIOD) if parse_rules else {}
                def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
                    out = {"period": a.get("period", DEFAULT_PERIOD)}
                    for k in ["demand_updates","warehouse_changes","supplier_changes","transport_updates"]:
                        out[k] = (a.get(k, []) or []) + (b.get(k, []) or [])
                    return out
                scenario = _merge(rules_first or {"period": DEFAULT_PERIOD}, scenario_llm or {"period": DEFAULT_PERIOD})
        else:
            scenario = {"period": DEFAULT_PERIOD, "demand_updates": [], "warehouse_changes": [], "supplier_changes": [], "transport_updates": []}

        ss["scenario"] = scenario

        # Apply edits (or keep base)
        before = ss["dfs"]
        if run_base:
            updated = before
        else:
            updated = apply_scenario_edits(before, scenario, default_period=DEFAULT_PERIOD) if apply_scenario_edits else before
        ss["updated_dfs"] = updated

        # Deltas only if any change
        deltas = diff_tables(before, updated) if diff_tables else {}
        any_changes = any(isinstance(dfchg, pd.DataFrame) and not dfchg.empty for dfchg in (deltas or {}).values())

        if any_changes:
            st.subheader("🔁 Changes (Before → After)")
            tab1, tab2, tab3, tab4 = st.tabs(["Customer Product Data", "Warehouse", "Supplier Product", "Transport Cost"])
            with tab1:
                dfchg = deltas.get("Customer Product Data", pd.DataFrame())
                if not dfchg.empty: st.dataframe(dfchg, use_container_width=True)
            with tab2:
                dfchg = deltas.get("Warehouse", pd.DataFrame())
                if not dfchg.empty: st.dataframe(dfchg, use_container_width=True)
            with tab3:
                dfchg = deltas.get("Supplier Product", pd.DataFrame())
                if not dfchg.empty: st.dataframe(dfchg, use_container_width=True)
            with tab4:
                dfchg = deltas.get("Transport Cost", pd.DataFrame())
                if not dfchg.empty: st.dataframe(dfchg, use_container_width=True)
        else:
            st.info("No changes detected (showing base model).")

        # Optimize
        with st.spinner("Solving quick MILP..."):
            kpis, diag = run_optimizer(updated, period=scenario.get("period", DEFAULT_PERIOD)) if run_optimizer else ({}, {})
        ss["kpis"] = kpis
        ss["diag"] = diag

        # KPIs
        st.subheader("📊 KPIs")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Status", kpis.get("status", "n/a"))
        c2.metric("Total Cost", kpis.get("total_cost", "n/a"))
        c3.metric("Total Demand", kpis.get("total_demand", 0))
        c4.metric("Served", kpis.get("served", 0))
        c5.metric("Service %", kpis.get("service_pct", 0))

        # Solved Map (flows)
        flows = diag.get("flows", [])
        st.subheader("🗺️ Solved Network (flows)")
        if flows:
            geo = flows_to_geo(flows, updated) if flows_to_geo else pd.DataFrame()
            if geo is not None and not geo.empty:
                nodes_df2 = build_nodes(updated)
                lat, lon = guess_map_center(nodes_df2)
                layers2 = _scatter_layers_from_nodes(nodes_df2)
                arc_layer = _arc_layer_from_flows_geo(geo)
                if arc_layer is not None:
                    layers2.append(arc_layer)
                st.pydeck_chart(pdk.Deck(
                    map_style=None,  # avoid token
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=3.5),
                    layers=layers2,
                    tooltip={"text": "{product}: {qty}\n{from} → {to}"}
                ))
            else:
                st.info("Flows solved, but could not geocode locations. Check the 'Locations' sheet.")
        else:
            st.warning("No flows found. Use the Infeasibility Helper to auto-connect lanes or review data.")

        # Q&A (right under solved map)
        st.subheader("💬 Ask GENIE (about this model/solution)")
        colq1, colq2 = st.columns([3, 1])
        with colq1:
            qa_val = ss.get("qa_input", "")
            qtext = st.text_input("Ask a question (e.g., 'Which customers are unserved?')",
                                  value=qa_val, key="qa_input")
        with colq2:
            ask_btn = st.button("Ask", key="qa_btn")
        if ask_btn:
            provider = "gemini" if ss.get("prov_choice") == "Google Gemini" else "none"
            ans = answer_question(qtext, kpis, diag, provider=provider, dfs=updated, model_index=ss.get("model_index", {})) if answer_question else "Q&A module not available."
            st.info(ans)

        # Download updated scenario
        st.markdown("---")
        data_bytes, fname = _downloadable_excel(ss["updated_dfs"], filename="genie_scenario.xlsx")
        st.download_button("⬇️ Download Scenario Excel", data=data_bytes, file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------ Infeasibility / Location Tools ------------------------------
st.markdown("---")
st.subheader("🧩 Infeasibility Helper & Location Tools")

help_col1, help_col2 = st.columns([1,1])

with help_col1:
    st.markdown("**Suggest location coordinates**")
    loc_query = st.text_input("Location name to geocode (uses Locations sheet first)", key="loc_query")
    allow_online = st.checkbox("Allow online geocoding (Nominatim)", value=False, key="allow_online_geo")
    if st.button("Suggest Coordinates"):
        if not ss["dfs"]:
            st.error("Upload a workbook first.")
        else:
            cands = suggest_location_candidates(loc_query, ss["dfs"], provider="none", allow_online=allow_online) if suggest_location_candidates else []
            if cands:
                st.success("Candidates:")
                st.json(cands)
            else:
                st.warning("No candidates found. Try a different name.")

with help_col2:
    st.markdown("**Upsert into Locations**")
    up_loc = st.text_input("Location")
    up_country = st.text_input("Country/Descriptor")
    up_lat = st.number_input("Latitude", value=0.0, step=0.0001, format="%.6f")
    up_lon = st.number_input("Longitude", value=0.0, step=0.0001, format="%.6f")
    if st.button("Add/Update Location"):
        if not ss["dfs"]:
            st.error("Upload a workbook first.")
        elif ensure_location_row is None:
            st.error("geo.ensure_location_row not available.")
        else:
            newdfs = ensure_location_row(ss["dfs"], up_loc, up_country, up_lat, up_lon)
            ss["dfs"] = newdfs
            ss["model_index"] = build_model_index(newdfs, period=DEFAULT_PERIOD)
            st.success(f"Location '{up_loc}' upserted.")
            nodes_df = build_nodes(ss["dfs"])
            if nodes_df is not None and not nodes_df.empty:
                lat, lon = guess_map_center(nodes_df)
                layers = _scatter_layers_from_nodes(nodes_df)
                st.pydeck_chart(pdk.Deck(
                    map_style=None,
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=3.5),
                    layers=layers
                ))

# ------------------------------ Footer ------------------------------
st.markdown("---")
st.caption("GENIE — network design assistant. Upload → Prompt → Optimize → Explain.")
