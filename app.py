# app.py — GENIE: Supply Chain Network Designer (hackathon-ready, stable UI)
# - Upload base-case Excel -> validate
# - Show nodes map immediately (warehouses/customers)
# - Prompt -> parse scenario (Rules or Gemini), apply edits, show Before/After if changed
# - Optimize -> KPIs + solved flow map (thin, product-colored arcs)
# - Q&A (grounded + optional Gemini) directly below solved map
# - Helper: Template download, Auto-Connect tips, and Provider selection (robust)

from __future__ import annotations
import os
import io
import json
import time
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# ------------------------------ Engine Imports (defensive) ------------------------------

missing_engine: List[str] = []

# loader
try:
    from engine.loader import load_and_validate_excel, build_model_index, DEFAULT_PERIOD as LDR_DEFAULT_PERIOD
except ModuleNotFoundError:
    load_and_validate_excel = None  # type: ignore
    build_model_index = None  # type: ignore
    LDR_DEFAULT_PERIOD = 2023
    missing_engine.append("engine/loader.py (load_and_validate_excel, build_model_index)")

# parser (rules)
try:
    from engine.parser import parse_rules, DEFAULT_PERIOD as PAR_DEFAULT_PERIOD
except ModuleNotFoundError:
    parse_rules = None  # type: ignore
    PAR_DEFAULT_PERIOD = 2023
    missing_engine.append("engine/parser.py (parse_rules)")

# updater
try:
    from engine.updater import apply_scenario_edits, diff_tables
except ModuleNotFoundError:
    apply_scenario_edits = None  # type: ignore
    diff_tables = None  # type: ignore
    missing_engine.append("engine/updater.py (apply_scenario_edits, diff_tables)")

# optimizer
try:
    from engine.optimizer import run_optimizer, DEFAULT_PERIOD as OPT_DEFAULT_PERIOD
except ModuleNotFoundError:
    run_optimizer = None  # type: ignore
    OPT_DEFAULT_PERIOD = 2023
    missing_engine.append("engine/optimizer.py (run_optimizer)")

# geo
try:
    from engine.geo import build_nodes, flows_to_geo, guess_map_center, ensure_location_row
except ModuleNotFoundError:
    build_nodes = None  # type: ignore
    flows_to_geo = None  # type: ignore
    guess_map_center = None  # type: ignore
    ensure_location_row = None  # type: ignore
    missing_engine.append("engine/geo.py (build_nodes, flows_to_geo, guess_map_center, ensure_location_row)")

# genai
try:
    from engine.genai import parse_with_llm, answer_question, suggest_location_candidates
except ModuleNotFoundError:
    parse_with_llm = None  # type: ignore
    answer_question = None  # type: ignore
    suggest_location_candidates = None  # type: ignore
    missing_engine.append("engine/genai.py (parse_with_llm, answer_question, suggest_location_candidates)")

# ------------------------------ App Config ------------------------------

st.set_page_config(
    page_title="GENIE — Supply Chain Network Designer",
    page_icon="🧞‍♂️",
    layout="wide",
)

# Apply secrets into env (if present)
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
if "gcp" in st.secrets and "gemini_api_key" in st.secrets["gcp"]:
    os.environ["GEMINI_API_KEY"] = st.secrets["gcp"]["gemini_api_key"]
if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
    os.environ["GEMINI_API_KEY"] = st.secrets["gemini"]["api_key"]

DEFAULT_PERIOD = 2023

# ------------------------------ Helpers ------------------------------

def _fmt_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def _downloadable_excel(dfs: Dict[str, pd.DataFrame], filename: str = "scenario.xlsx") -> Tuple[bytes, str]:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in dfs.items():
            if isinstance(df, pd.DataFrame):
                # preserve headers/order
                df.to_excel(writer, sheet_name=str(name)[:31] or "Sheet1", index=False)
    bio.seek(0)
    return bio.read(), filename

def _thin_color_for_product(p: str) -> List[int]:
    # Stable palette for up to ~10 products, thin and high-contrast
    palette = [
        [33, 150, 243],   # Blue
        [76, 175, 80],    # Green
        [244, 67, 54],    # Red
        [255, 193, 7],    # Amber
        [156, 39, 176],   # Purple
        [0, 188, 212],    # Cyan
        [121, 85, 72],    # Brown
        [63, 81, 181],    # Indigo
        [255, 87, 34],    # Deep Orange
        [205, 220, 57],   # Lime
    ]
    if not p:
        return [120, 120, 120]
    idx = abs(hash(p)) % len(palette)
    return palette[idx]

def _arc_layer_from_flows_geo(geo_df: pd.DataFrame):
    """Build pydeck ArcLayer with thin width scaled by sqrt(qty)."""
    if geo_df is None or geo_df.empty:
        return None
    # Compute width ~ 1 + sqrt(qty) * 0.5 (thin lines)
    q = geo_df["qty"].astype(float).clip(lower=0.0).fillna(0.0)
    width = (1.0 + np.sqrt(q) * 0.5).tolist()
    colors = []
    for _, r in geo_df.iterrows():
        col = _thin_color_for_product(str(r.get("product", "")))
        colors.append(col + [180])  # add alpha
    layer = pdk.Layer(
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
    return layer

def _scatter_layers_from_nodes(nodes_df: pd.DataFrame):
    if nodes_df is None or nodes_df.empty:
        return []
    # Warehouses: violet; Customers: dark gray
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
                get_fill_color=[142, 36, 170, 200],
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
                get_fill_color=[66, 66, 66, 180],
                pickable=True,
            )
        )
    return layers

def _safe_get(df_dict: Dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    df = df_dict.get(name)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

# ------------------------------ Sidebar ------------------------------

st.title("🧞‍♂️ GENIE — Supply Chain Network Designer")

with st.sidebar:
    st.markdown("### 1) Upload Base-Case Excel")
    sample_url = "https://raw.githubusercontent.com/5av1t/genie/main/sample_base-case.xlsx"
    st.caption("Need a sample? Download the template:")
    st.download_button(
        "Download Sample Base-Case",
        data=None,
        file_name="sample_base-case.xlsx",
        disabled=True,
        help=f"Get it from: {sample_url}\n(Use raw.githubusercontent.com for direct download.)",
    )
    st.write(f"[Open sample URL]({sample_url})")

    uploaded = st.file_uploader("Upload your workbook (.xlsx)", type=["xlsx"], key="uploader")

    st.markdown("---")
    st.markdown("**GenAI provider**")

    prov_options = ["None (grounded only)", "Google Gemini"]
    default_idx = 0
    prev_val = st.session_state.get("prov_choice", None)
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
    st.write("- CRUD via text (add/update/delete rows with guardrails).")
    st.write("- Explanations of KPIs and flows using model context.")
    st.write("- Infeasibility helper: what’s missing and suggested fixes.")
    st.write("- Location helper: propose lat/lon and upsert into Locations.")

# ------------------------------ Main Panels ------------------------------

if missing_engine:
    st.error("Missing engine modules:\n- " + "\n- ".join(missing_engine))
    st.stop()

# Session defaults
if "dfs" not in st.session_state:
    st.session_state["dfs"] = {}
if "model_index" not in st.session_state:
    st.session_state["model_index"] = {}
if "scenario" not in st.session_state:
    st.session_state["scenario"] = {"period": DEFAULT_PERIOD}
if "updated_dfs" not in st.session_state:
    st.session_state["updated_dfs"] = {}
if "kpis" not in st.session_state:
    st.session_state["kpis"] = {}
if "diag" not in st.session_state:
    st.session_state["diag"] = {}
if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = ""
if "qa_input" not in st.session_state:
    st.session_state["qa_input"] = ""

# 1) Load + Validate
if uploaded is not None:
    with st.spinner("Reading and validating workbook..."):
        dfs, validation_report = load_and_validate_excel(uploaded)
        st.session_state["dfs"] = dfs
        st.session_state["model_index"] = build_model_index(dfs, period=DEFAULT_PERIOD)

    st.success("Workbook loaded.")
    # Validation summary
    with st.expander("✅ Sheet Validation Report"):
        for sheet, report in validation_report.items():
            if sheet == "_warnings":
                continue
            if isinstance(report, dict) and report.get("missing_columns"):
                st.error(f"{sheet}: Missing columns - {report['missing_columns']}")
            else:
                # Count rows/cols if possible
                if isinstance(report, dict):
                    st.success(f"{sheet}: OK ({report.get('num_rows',0)} rows, {report.get('num_columns',0)} cols)")
        if validation_report.get("_warnings"):
            st.warning("\n".join(validation_report["_warnings"]))

    # 2) Show Nodes Map right away
    nodes_df = build_nodes(st.session_state["dfs"])
    if nodes_df is not None and not nodes_df.empty:
        st.subheader("📍 Network Map (nodes)")
        lat, lon = guess_map_center(nodes_df)
        layers = _scatter_layers_from_nodes(nodes_df)
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=3.5)
        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9",
                                 initial_view_state=view_state,
                                 layers=layers,
                                 tooltip={"text": "{name} ({type})\n{location}"}))
    else:
        st.info("No mappable nodes found. Ensure 'Warehouse' and 'Customers' sheets have known locations or provide 'Locations' sheet with coordinates.")

    st.markdown("---")

# 2) Scenario Entry (Rules or LLM) + Process + Optimize
st.subheader("🧪 Scenario Editor")

colA, colB = st.columns([2, 1])

with colA:
    st.markdown("**Write prompt** (natural language)")
    # Keep a stable key and do not mutate it later in code
    st.session_state["user_prompt"] = st.text_area(
        "Describe your what-if (e.g., “Increase Mono-Crystalline demand at Abu Dhabi by 10% and set Lead Time to 8”):",
        value=st.session_state.get("user_prompt", ""),
        height=120,
        key="user_prompt",
        label_visibility="collapsed"
    )

    st.caption("Examples:")
    st.code("Increase Mono-Crystalline demand at Abu Dhabi by 10% and set Lead Time to 8", language="text")
    st.code("Cap Bucharest_CDC Maximum Capacity at 25000; force close Berlin_LDC", language="text")
    st.code("Enable Thin-Film at Antalya_FG", language="text")
    st.code("Set Secondary Delivery LTL lane Paris_CDC → Aalborg for Poly-Crystalline to cost per uom = 9.5", language="text")

with colB:
    st.markdown("**Parser**")
    parse_mode = st.radio("Use", ["Rules (deterministic)", "LLM (Gemini)"], index=0, key="parse_mode_choice")
    run_base = st.checkbox("Run base model (no edits)", value=False, key="run_base_flag")
    do_process = st.button("Process Scenario & Optimize", type="primary")

# Handle click
if do_process:
    if not st.session_state["dfs"]:
        st.error("Please upload a workbook first.")
    else:
        dfs = st.session_state["dfs"]
        # Build scenario JSON
        scenario = {"period": DEFAULT_PERIOD, "demand_updates": [], "warehouse_changes": [], "supplier_changes": [], "transport_updates": []}
        text = st.session_state.get("user_prompt", "") or ""

        if not run_base and text.strip():
            if parse_mode == "Rules (deterministic)":
                scenario = parse_rules(text, dfs, default_period=DEFAULT_PERIOD) if parse_rules else scenario
            else:
                # LLM path
                prov_selected = st.session_state.get("prov_choice", "None (grounded only)")
                provider = "gemini" if prov_selected == "Google Gemini" else "none"
                scenario_llm = parse_with_llm(text, dfs, default_period=DEFAULT_PERIOD, provider=provider) if parse_with_llm else {}
                # Merge with rules (optional safety): rules pass first to normalize
                rules_first = parse_rules(text, dfs, default_period=DEFAULT_PERIOD) if parse_rules else {}
                # Simple merge: prefer LLM fields if present
                def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
                    out = {"period": a.get("period", DEFAULT_PERIOD)}
                    for k in ["demand_updates","warehouse_changes","supplier_changes","transport_updates"]:
                        out[k] = (a.get(k, []) or []) + (b.get(k, []) or [])
                    return out
                scenario = _merge(rules_first or {"period": DEFAULT_PERIOD}, scenario_llm or {"period": DEFAULT_PERIOD})
        else:
            scenario = {"period": DEFAULT_PERIOD, "demand_updates": [], "warehouse_changes": [], "supplier_changes": [], "transport_updates": []}

        st.session_state["scenario"] = scenario

        # Apply edits or keep base
        before = st.session_state["dfs"]
        if run_base:
            updated = before
        else:
            updated = apply_scenario_edits(before, scenario, default_period=DEFAULT_PERIOD) if apply_scenario_edits else before
        st.session_state["updated_dfs"] = updated

        # Deltas (only show if any changes)
        deltas = diff_tables(before, updated) if diff_tables else {}
        any_changes = False
        for _, dfchg in (deltas or {}).items():
            if isinstance(dfchg, pd.DataFrame) and not dfchg.empty:
                any_changes = True
                break

        if any_changes:
            st.subheader("🔁 Changes (Before → After)")
            tab1, tab2, tab3, tab4 = st.tabs(["Customer Product Data", "Warehouse", "Supplier Product", "Transport Cost"])
            with tab1:
                dfchg = deltas.get("Customer Product Data", pd.DataFrame())
                if not dfchg.empty:
                    st.dataframe(dfchg, use_container_width=True)
            with tab2:
                dfchg = deltas.get("Warehouse", pd.DataFrame())
                if not dfchg.empty:
                    st.dataframe(dfchg, use_container_width=True)
            with tab3:
                dfchg = deltas.get("Supplier Product", pd.DataFrame())
                if not dfchg.empty:
                    st.dataframe(dfchg, use_container_width=True)
            with tab4:
                dfchg = deltas.get("Transport Cost", pd.DataFrame())
                if not dfchg.empty:
                    st.dataframe(dfchg, use_container_width=True)
        else:
            st.info("No changes detected (showing base model).")

        # Optimize
        with st.spinner("Solving quick MILP..."):
            kpis, diag = run_optimizer(updated, period=scenario.get("period", DEFAULT_PERIOD)) if run_optimizer else ({}, {})
        st.session_state["kpis"] = kpis
        st.session_state["diag"] = diag

        # KPIs
        st.subheader("📊 KPIs")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Status", kpis.get("status", "n/a"))
        col2.metric("Total Cost", kpis.get("total_cost", "n/a"))
        col3.metric("Total Demand", kpis.get("total_demand", 0))
        col4.metric("Served", kpis.get("served", 0))
        col5.metric("Service %", kpis.get("service_pct", 0))

        # Solved Map (flows)
        flows = diag.get("flows", [])
        if flows:
            st.subheader("🗺️ Solved Network (flows)")
            geo = flows_to_geo(flows, updated) if flows_to_geo else pd.DataFrame()
            if geo is not None and not geo.empty:
                nodes_df = build_nodes(updated)
                lat, lon = guess_map_center(nodes_df)
                layers = _scatter_layers_from_nodes(nodes_df)
                arc_layer = _arc_layer_from_flows_geo(geo)
                if arc_layer is not None:
                    layers.append(arc_layer)
                st.pydeck_chart(pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=3.5),
                    layers=layers,
                    tooltip={"text": "{product}: {qty}\n{from} → {to}"}
                ))
            else:
                st.info("Flows solved, but could not geocode locations. Please check the 'Locations' sheet.")
        else:
            st.warning("No flows found. Use the Infeasibility Helper to auto-connect lanes or review data.")

        # Q&A directly under solved map
        st.subheader("💬 Ask GENIE (about this model/solution)")
        colq1, colq2 = st.columns([3, 1])
        with colq1:
            st.session_state["qa_input"] = st.text_input("Ask a question (e.g., 'Which customers are unserved?')",
                                                         value=st.session_state.get("qa_input",""),
                                                         key="qa_input")
        with colq2:
            ask_btn = st.button("Ask", key="qa_btn")
        if ask_btn:
            prov_selected = st.session_state.get("prov_choice", "None (grounded only)")
            provider = "gemini" if prov_selected == "Google Gemini" else "none"
            q = st.session_state.get("qa_input","")
            ans = answer_question(q, kpis, diag, provider=provider, dfs=updated, model_index=st.session_state.get("model_index", {})) if answer_question else "Q&A module not available."
            st.info(ans)

        # Download updated scenario
        st.markdown("---")
        data_bytes, fname = _downloadable_excel(st.session_state["updated_dfs"], filename="genie_scenario.xlsx")
        st.download_button("⬇️ Download Scenario Excel", data=data_bytes, file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# 3) Infeasibility / Auto-connect Helper (manual trigger)
st.markdown("---")
st.subheader("🧩 Infeasibility Helper & Location Tools")

help_col1, help_col2 = st.columns([1,1])

with help_col1:
    st.markdown("**Suggest location coordinates**")
    loc_query = st.text_input("Location name to geocode (uses Locations sheet first)", key="loc_query")
    allow_online = st.checkbox("Allow online geocoding (Nominatim)", value=False, key="allow_online_geo")
    if st.button("Suggest Coordinates"):
        if not st.session_state["dfs"]:
            st.error("Upload a workbook first.")
        else:
            cands = suggest_location_candidates(loc_query, st.session_state["dfs"], provider="none", allow_online=allow_online) if suggest_location_candidates else []
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
        if not st.session_state["dfs"]:
            st.error("Upload a workbook first.")
        elif ensure_location_row is None:
            st.error("geo.ensure_location_row not available.")
        else:
            newdfs = ensure_location_row(st.session_state["dfs"], up_loc, up_country, up_lat, up_lon)
            st.session_state["dfs"] = newdfs
            st.session_state["model_index"] = build_model_index(newdfs, period=DEFAULT_PERIOD)
            st.success(f"Location '{up_loc}' upserted.")
            # Show nodes again
            nodes_df = build_nodes(st.session_state["dfs"])
            if nodes_df is not None and not nodes_df.empty:
                lat, lon = guess_map_center(nodes_df)
                layers = _scatter_layers_from_nodes(nodes_df)
                st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9",
                                         initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=3.5),
                                         layers=layers))

# Footer
st.markdown("---")
st.caption("GENIE — hackathon-ready network design assistant. Upload → Prompt → Optimize → Explain.")
