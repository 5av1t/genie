# app.py
# GENIE — Supply Chain Network Designer (Streamlit App)
# - Upload base-case Excel
# - Validate sheets
# - Visualize warehouses/customers on a map
# - Turn NL prompts into sheet edits (Rules or Gemini)
# - Run quick MILP optimizer and draw flows
# - Ask GENIE (Q&A) grounded on model data
# - Deterministic location helper (no LLM coordinates)

from __future__ import annotations
import os
import io
import json
import math
import time
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# ----------------------------- Imports from Engine -----------------------------
missing_engine: List[str] = []

# loader
try:
    from engine.loader import load_and_validate_excel, build_model_index
except ModuleNotFoundError:
    load_and_validate_excel = None  # type: ignore
    build_model_index = None  # type: ignore
    missing_engine.append("engine/loader.py")

# parser (rules)
try:
    from engine.parser import parse_rules
except ModuleNotFoundError:
    parse_rules = None  # type: ignore
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
    from engine.optimizer import run_optimizer
except ModuleNotFoundError:
    run_optimizer = None  # type: ignore
    missing_engine.append("engine/optimizer.py (run_optimizer)")

# geo
try:
    from engine.geo import build_nodes, flows_to_geo, guess_map_center, ensure_location_row
except ModuleNotFoundError:
    build_nodes = None  # type: ignore
    flows_to_geo = None  # type: ignore
    guess_map_center = None  # type: ignore
    ensure_location_row = None  # type: ignore
    missing_engine.append("engine/geo.py (build_nodes, flows_to_geo, guess_map_center)")

# reporter (optional)
try:
    from engine.reporter import build_summary
except ModuleNotFoundError:
    build_summary = None  # type: ignore
    # optional; no missing message

# genai
try:
    from engine.genai import parse_with_llm, answer_question, suggest_location_candidates
except ModuleNotFoundError:
    parse_with_llm = None  # type: ignore
    answer_question = None  # type: ignore
    suggest_location_candidates = None  # type: ignore
    missing_engine.append("engine/genai.py (parse_with_llm, answer_question, suggest_location_candidates)")

# For configuring Gemini if user enters a key at runtime
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

# ----------------------------- Constants & Helpers -----------------------------

DEFAULT_PERIOD = 2023

RAW_TEMPLATE_URL = st.secrets.get(
    "RAW_TEMPLATE_URL",
    "https://raw.githubusercontent.com/5av1t/genie/main/sample_base-case.xlsx"
)

PALETTE = [
    [52, 152, 219],  # blue
    [46, 204, 113],  # green
    [231, 76, 60],   # red
    [155, 89, 182],  # purple
    [241, 196, 15],  # yellow
    [230, 126, 34],  # orange
    [26, 188, 156],  # teal
    [149, 165, 166], # gray
    [52, 73, 94],    # dark
    [39, 174, 96],   # dark green
]

def _product_color_map(products: List[str]) -> Dict[str, List[int]]:
    mp: Dict[str, List[int]] = {}
    for i, p in enumerate(products):
        mp[p] = PALETTE[i % len(PALETTE)]
    if not mp:
        mp["(default)"] = [52, 152, 219]
    return mp

def _prep_node_layers(nodes_df: pd.DataFrame) -> List[pdk.Layer]:
    """Scatterplot for warehouses and customers."""
    layers = []
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return layers
    # warehouses
    wh = nodes_df[nodes_df["type"] == "warehouse"]
    if not wh.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=wh,
            get_position=["lon", "lat"],
            get_radius=20000,
            radius_min_pixels=4,
            radius_max_pixels=10,
            get_fill_color=[255, 140, 0],  # orange
            pickable=True,
        ))
    # customers
    cu = nodes_df[nodes_df["type"] == "customer"]
    if not cu.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=cu,
            get_position=["lon", "lat"],
            get_radius=15000,
            radius_min_pixels=3,
            radius_max_pixels=8,
            get_fill_color=[0, 122, 255],  # blue
            pickable=True,
        ))
    return layers

def _prep_flow_layer(arcs_df: pd.DataFrame, color_by_product=True) -> Optional[pdk.Layer]:
    """Thin line flows; color by product; width ~ qty."""
    if not isinstance(arcs_df, pd.DataFrame) or arcs_df.empty:
        return None

    df = arcs_df.copy()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    q = df["qty"].clip(lower=0)
    if q.max() <= 0:
        width = np.full(len(df), 1.0)
    else:
        q95 = np.quantile(q, 0.95)
        q95 = q95 if q95 > 0 else q.max()
        width = 0.6 + 3.4 * (q / (q95 + 1e-9))  # 0.6..4.0 px
    df["width"] = width

    if color_by_product:
        prods = sorted([str(x) for x in df["product"].dropna().unique().tolist()])
        cmap = _product_color_map(prods)
        df["color"] = df["product"].apply(lambda p: cmap.get(str(p), [52, 152, 219]))
    else:
        df["color"] = [52, 152, 219]  # single color

    return pdk.Layer(
        "LineLayer",
        data=df,
        get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"],
        get_width="width",
        get_color="color",
        pickable=True,
        auto_highlight=True,
    )

def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _download_bytesio_from_url(url: str) -> Optional[bytes]:
    try:
        import requests
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
    return None

def _has_diffs(diffs: Dict[str, pd.DataFrame]) -> bool:
    if not isinstance(diffs, dict):
        return False
    for v in diffs.values():
        if isinstance(v, pd.DataFrame) and not v.empty:
            return True
    return False

def _scenario_bullets(scn: Dict[str, Any]) -> List[str]:
    bullets = []
    if not scn:
        return ["No edits (base model)."]
    p = scn.get("period", DEFAULT_PERIOD)
    if scn.get("demand_updates"): bullets.append(f"Demand updates: {len(scn['demand_updates'])} row(s) in {p}.")
    if scn.get("warehouse_changes"): bullets.append(f"Warehouse changes: {len(scn['warehouse_changes'])}.")
    if scn.get("supplier_changes"): bullets.append(f"Supplier changes: {len(scn['supplier_changes'])}.")
    if scn.get("transport_updates"): bullets.append(f"Transport updates: {len(scn['transport_updates'])}.")
    if scn.get("adds"): bullets.append(f"Adds: {sum(len(v) for v in scn['adds'].values())} row(s).")
    if scn.get("deletes"): bullets.append(f"Deletes: {sum(len(v) for v in scn['deletes'].values())} row(s).")
    return bullets or [f"No edits for period {p}."]

# ----------------------------- Streamlit Page Config -----------------------------

st.set_page_config(page_title="GENIE — Supply Chain Network Designer", layout="wide", page_icon="🧞")

# Initialize session defaults BEFORE widgets
for k, v in {
    "uploaded_name": None,
    "dfs": None,
    "report": None,
    "model_index": None,
    "last_updated_dfs": None,
    "last_scenario": {},
    "kpis": {},
    "diag": {},
    "flows_geo": pd.DataFrame(),
    "user_prompt": "",
    "llm_provider": "None",
    "gemini_key_set": False,
    "qa_answer": "",
    "color_by_product": True,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------- Sidebar -----------------------------

st.sidebar.title("🧞 GENIE")
st.sidebar.caption("Hackathon-ready GenAI network designer")

# Template download
st.sidebar.markdown("**Get a template**")
c1, c2 = st.sidebar.columns([1,1])
with c1:
    if st.button("Download template"):
        raw = _download_bytesio_from_url(RAW_TEMPLATE_URL)
        if raw:
            st.download_button("Save Excel", data=raw, file_name="sample_base-case.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.sidebar.warning("Unable to fetch template. Verify RAW_TEMPLATE_URL.")

with c2:
    st.sidebar.markdown("[View repo](https://github.com/5av1t/genie)")

# Provider selection (does NOT clear state)
st.sidebar.markdown("---")
st.sidebar.markdown("**GenAI provider**")
prov = st.sidebar.radio("Choose", ["None (grounded only)", "Google Gemini", "OpenAI"], index={"None (grounded only)":0,"Google Gemini":1,"OpenAI":2}[st.session_state["llm_provider"] if st.session_state["llm_provider"] in ("None (grounded only)","Google Gemini","OpenAI") else "None (grounded only)"], key="llm_provider")

# API key field (password), updates env and config in-place
if prov == "Google Gemini":
    st.sidebar.write("Set **GEMINI_API_KEY** (stored only for this session):")
    gem_key = st.sidebar.text_input("GEMINI_API_KEY", type="password")
    if gem_key and genai is not None:
        os.environ["GEMINI_API_KEY"] = gem_key
        try:
            genai.configure(api_key=gem_key)
            st.session_state["gemini_key_set"] = True
            st.sidebar.success("Gemini configured.")
        except Exception as e:
            st.sidebar.error(f"Gemini config failed: {e}")
elif prov == "OpenAI":
    st.sidebar.write("Set **OPENAI_API_KEY** in your Streamlit secrets or environment.")
    # openai lib picks key from env; we don't reconfigure here.

# Use cases
st.sidebar.markdown("---")
st.sidebar.subheader("What can GENIE do?")
st.sidebar.markdown(
"""
- Diagnose infeasibility and propose **concrete fixes**.
- Replace/relocate a warehouse and **clone lanes**.
- Reach a **service target** (e.g., 98%) with minimal edits.
- Add/enable **suppliers/lanes** and re-run.
- Explain **unserved customers** with reasons.
"""
)

# ----------------------------- Main Layout -----------------------------

st.title("🧞 GENIE — Supply Chain Network Designer")
st.caption("Upload your base-case Excel, create what‑if scenarios in plain English, optimize, and visualize flows.")

if missing_engine:
    with st.expander("⚠️ Missing engine modules"):
        for m in missing_engine:
            st.warning(m)

# Upload
uploaded = st.file_uploader("Upload your base-case Excel (.xlsx)", type=["xlsx"], key="uploader")
if uploaded:
    st.session_state["uploaded_name"] = uploaded.name

# Load + Validate
if uploaded and load_and_validate_excel is not None:
    try:
        dfs, report = load_and_validate_excel(uploaded)
        st.session_state["dfs"] = dfs
        st.session_state["report"] = report
        if build_model_index is not None:
            st.session_state["model_index"] = build_model_index(dfs, period=DEFAULT_PERIOD)
        st.success(f"Loaded: {st.session_state['uploaded_name']}")
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")

# Validation summary
if st.session_state["report"]:
    with st.expander("✅ Sheet Validation Report", expanded=True):
        rep = st.session_state["report"]
        if isinstance(rep, dict):
            for sheet, info in rep.items():
                if sheet == "_warnings":
                    continue
                if isinstance(info, dict):
                    miss = info.get("missing_columns", [])
                    rows = info.get("num_rows", 0)
                    cols = info.get("num_columns", 0)
                    if miss:
                        st.error(f"{sheet}: Missing columns - {miss}")
                    else:
                        st.success(f"{sheet}: OK ({rows} rows, {cols} columns)")
            # warnings
            warns = rep.get("_warnings", [])
            if warns:
                st.info("Warnings:")
                for w in warns:
                    st.write(f"• {w}")
        else:
            st.write(rep)

# ----------------------------- Base Map (after upload) -----------------------------

if st.session_state["dfs"] and build_nodes is not None and guess_map_center is not None:
    nodes = build_nodes(st.session_state["dfs"])
    if nodes is not None and not nodes.empty:
        st.subheader("Network Map — Nodes")
        center_lat, center_lon = guess_map_center(nodes, default=(30.0, 15.0))
        view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3.7)
        layers = _prep_node_layers(nodes)
        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view, layers=layers, tooltip={"text": "{name}\n{type}\n{location}"}))
    else:
        st.info("No nodes could be resolved. Add 'Locations' with Latitude/Longitude or use the Location helper below.")

# ----------------------------- Scenario Input (Rules + LLM side-by-side) -----------------------------

if st.session_state["dfs"]:
    st.markdown("---")
    st.header("Scenario Builder")

    # Dynamic examples based on current data (lightweight)
    dyn_examples: List[str] = []
    idx = st.session_state.get("model_index") or {}
    products = idx.get("products", [])
    warehouses = sorted((st.session_state["dfs"].get("Warehouse") or pd.DataFrame()).get("Warehouse", pd.Series([])).astype(str).tolist()) if isinstance(st.session_state["dfs"].get("Warehouse"), pd.DataFrame) else []
    customers = sorted((st.session_state["dfs"].get("Customers") or pd.DataFrame()).get("Customer", pd.Series([])).astype(str).tolist()) if isinstance(st.session_state["dfs"].get("Customers"), pd.DataFrame) else []
    if products and customers:
        dyn_examples.append(f"Increase {products[0]} demand at {customers[0]} by 10% and set Lead Time to 8 days")
    if warehouses:
        dyn_examples.append(f"Cap {warehouses[0]} Maximum Capacity at 25000; force close {warehouses[-1]}")
    if products and warehouses:
        dyn_examples.append(f"Enable {products[-1]} at Antalya_FG")
        if customers:
            dyn_examples.append(f"Set Secondary Delivery LTL lane {warehouses[0]} → {customers[0]} for {products[0]} to Cost Per UOM = 9.5")

    col_rules, col_llm = st.columns(2)
    # RULES PARSER
    with col_rules:
        st.subheader("Rules Parser (deterministic)")
        st.caption("Parses specific intents exactly—safe for production.")
        prompt_rules = st.text_area("Write scenario (rules):", value=st.session_state.get("user_prompt") or "", key="user_prompt", height=120, placeholder="e.g., Increase Mono-Crystalline demand at Abu Dhabi by 10%")
        if dyn_examples:
            ex_choice = st.selectbox("Insert example (from file):", ["(choose one)"] + dyn_examples, key="ex_rules")
            if st.button("Insert example (rules)"):
                if ex_choice != "(choose one)":
                    st.session_state["user_prompt"] = ex_choice
                    st.experimental_rerun()
        scn_rules: Dict[str, Any] = {}
        if parse_rules is not None and st.button("Parse (Rules)"):
            try:
                scn_rules = parse_rules(prompt_rules, st.session_state["dfs"], default_period=DEFAULT_PERIOD)
                st.session_state["last_scenario_rules"] = scn_rules
                st.success("Parsed (rules).")
            except Exception as e:
                st.error(f"Rules parser failed: {e}")
        if st.session_state.get("last_scenario_rules"):
            st.json(st.session_state["last_scenario_rules"])

    # LLM PARSER
    with col_llm:
        st.subheader("LLM Parser (Gemini/OpenAI)")
        st.caption("Optional. Uses GenAI to interpret free text.")
        prompt_llm = st.text_area("Write scenario (LLM):", value=st.session_state.get("user_prompt_llm") or "", key="user_prompt_llm", height=120, placeholder="Natural language what‑if")
        if dyn_examples:
            ex_choice2 = st.selectbox("Insert example (LLM):", ["(choose one)"] + dyn_examples, key="ex_llm")
            if st.button("Insert example (LLM)"):
                if ex_choice2 != "(choose one)":
                    st.session_state["user_prompt_llm"] = ex_choice2
                    st.experimental_rerun()
        scn_llm: Dict[str, Any] = {}
        if parse_with_llm is not None and st.button("Parse (LLM)"):
            try:
                provider = "gemini" if prov == "Google Gemini" else ("openai" if prov == "OpenAI" else "none")
                scn_llm = parse_with_llm(prompt_llm, st.session_state["dfs"], default_period=DEFAULT_PERIOD, provider=provider)
                st.session_state["last_scenario_llm"] = scn_llm
                st.success("Parsed (LLM).")
            except Exception as e:
                st.error(f"LLM parser failed: {e}")
        if st.session_state.get("last_scenario_llm"):
            st.json(st.session_state["last_scenario_llm"])

    # Choose scenario to run
    st.markdown("### Scenario to run")
    mode = st.radio("Choose edits", ["Base (no edits)", "Use Rules result", "Use LLM result"], horizontal=True, key="scenario_mode")
    if mode == "Use Rules result":
        scenario = st.session_state.get("last_scenario_rules") or {}
    elif mode == "Use LLM result":
        scenario = st.session_state.get("last_scenario_llm") or {}
    else:
        scenario = {}

    # ----------------------------- Process Scenario -----------------------------

    if st.button("🚀 Process Scenario & Optimize"):
        if apply_scenario_edits is None or run_optimizer is None:
            st.error("Updater or Optimizer not available.")
        else:
            try:
                before = st.session_state["dfs"].copy()
                # Capture 'before' copies for diffs
                before_cpd = _safe_df(before.get("Customer Product Data")).copy()
                before_wh  = _safe_df(before.get("Warehouse")).copy()
                before_sp  = _safe_df(before.get("Supplier Product")).copy()
                before_tc  = _safe_df(before.get("Transport Cost")).copy()

                # Apply scenario (or pass empty)
                updated = apply_scenario_edits(before, scenario or {}, default_period=DEFAULT_PERIOD)

                # Diffs only if changes actually occurred
                diffs = diff_tables(
                    before={"Customer Product Data": before_cpd, "Warehouse": before_wh, "Supplier Product": before_sp, "Transport Cost": before_tc},
                    after={"Customer Product Data": _safe_df(updated.get("Customer Product Data")), "Warehouse": _safe_df(updated.get("Warehouse")), "Supplier Product": _safe_df(updated.get("Supplier Product")), "Transport Cost": _safe_df(updated.get("Transport Cost"))},
                ) if diff_tables is not None else {}

                st.session_state["last_updated_dfs"] = updated
                st.session_state["last_scenario"] = scenario

                # Optimization
                kpis, diag = run_optimizer(updated, period=scenario.get("period", DEFAULT_PERIOD))
                st.session_state["kpis"] = kpis
                st.session_state["diag"] = diag

                # Flows → geo
                if flows_to_geo is not None:
                    geo_flows = flows_to_geo(diag.get("flows", []), updated)
                    st.session_state["flows_geo"] = geo_flows
                else:
                    st.session_state["flows_geo"] = pd.DataFrame()

                st.success("Scenario applied and optimized.")
                # Show diffs only if there are changes
                if _has_diffs(diffs):
                    with st.expander("🔁 Deltas written to sheets (only if changed)", expanded=False):
                        for name, df in diffs.items():
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                st.markdown(f"**{name}**")
                                st.dataframe(df, use_container_width=True)
                else:
                    st.info("No changes were applied to the sheets (base model).")

            except Exception as e:
                st.error(f"Processing failed: {e}")

# ----------------------------- Results: KPIs + Map + Ask GENIE -----------------------------

if st.session_state["kpis"]:
    st.markdown("---")
    st.header("Results")

    col1, col2, col3, col4 = st.columns(4)
    k = st.session_state["kpis"]
    with col1:
        st.metric("Status", k.get("status", "n/a"))
    with col2:
        st.metric("Service %", k.get("service_pct", 0))
    with col3:
        st.metric("Total Demand", int(k.get("total_demand", 0) or 0))
    with col4:
        served = int(k.get("served", 0) or 0)
        st.metric("Served Units", served)

    # Flow map (after optimization)
    if st.session_state["last_updated_dfs"] is not None and build_nodes is not None:
        nodes_after = build_nodes(st.session_state["last_updated_dfs"])
    else:
        nodes_after = None

    st.subheader("Solved Map — Nodes & Flows")
    color_by = st.checkbox("Color flows by product", value=st.session_state.get("color_by_product", True), key="color_by_product")
    layers = []
    if isinstance(nodes_after, pd.DataFrame) and not nodes_after.empty:
        layers.extend(_prep_node_layers(nodes_after))
    arcs = st.session_state.get("flows_geo")
    flow_layer = _prep_flow_layer(arcs, color_by_product=color_by) if isinstance(arcs, pd.DataFrame) else None
    if flow_layer:
        layers.append(flow_layer)
    # Center on nodes if possible
    if isinstance(nodes_after, pd.DataFrame) and not nodes_after.empty and guess_map_center is not None:
        lat, lon = guess_map_center(nodes_after, default=(30.0, 15.0))
    else:
        lat, lon = (30.0, 15.0)
    view = pdk.ViewState(latitude=lat, longitude=lon, zoom=3.7)
    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view, layers=layers))

    # -------------------- Ask GENIE (Q&A) just BELOW the solved map --------------------
    st.subheader("Ask GENIE")
    st.caption("Grounded Q&A on your data. Try: *Why is Abu Dhabi unserved?*, *Which countries have suppliers?*, *Why no supplier in India?*")

    q = st.text_input("Your question:", key="qa_input", value=st.session_state.get("qa_input_val", ""))
    ask_col1, ask_col2 = st.columns([1,3])
    with ask_col1:
        ask_btn = st.button("Ask")
    with ask_col2:
        st.caption(f"Provider: {prov}")

    if ask_btn and answer_question is not None:
        try:
            provider = "gemini" if prov == "Google Gemini" else ("openai" if prov == "OpenAI" else "none")
            ans = answer_question(
                q,
                st.session_state.get("kpis") or {},
                st.session_state.get("diag") or {},
                provider=provider,
                dfs=st.session_state.get("last_updated_dfs") or st.session_state.get("dfs"),
                model_index=st.session_state.get("model_index"),
                force_llm=(provider in ("gemini", "openai")),
            )
            st.session_state["qa_answer"] = ans
        except Exception as e:
            st.session_state["qa_answer"] = f"Q&A failed: {e}"

    if st.session_state.get("qa_answer"):
        st.markdown("**Answer:**")
        st.write(st.session_state["qa_answer"])

# ----------------------------- Manual Edits / CRUD (incl. Location helper) -----------------------------

if st.session_state["dfs"]:
    st.markdown("---")
    st.header("Manual Edits (Advanced)")
    st.caption("Make targeted changes directly. Use with care.")

    tabs = st.tabs(["Add/Update Locations", "Add Row", "Delete Row"])

    # Locations Helper — deterministic (Gemini parses text only, geocode is deterministic)
    with tabs[0]:
        st.markdown("**📍 Location helper (deterministic geocoding)**")
        loc_query = st.text_input("City/Location text (e.g., 'Prague_LDC, Czechia')", key="loc_query")
        use_online = st.checkbox("Allow online geocoding (Nominatim)", value=False, help="Enable also by setting GENIE_GEOCODE_ONLINE=true in environment.")
        if st.button("Resolve coordinates"):
            if suggest_location_candidates is None:
                st.warning("engine.genai.suggest_location_candidates not available.")
            else:
                provider = "gemini" if prov == "Google Gemini" else ("openai" if prov == "OpenAI" else "none")
                cands = suggest_location_candidates(loc_query, st.session_state["dfs"], provider=provider, allow_online=use_online)
                if not cands:
                    st.warning("No deterministic match. Add to 'Locations' manually or enable online geocoding.")
                else:
                    best = cands[0]
                    st.success(f"Found: {best['location']} ({best['country']}) → lat={best['lat']:.4f}, lon={best['lon']:.4f}")
                    if st.button("Add/Update in Locations"):
                        if ensure_location_row is None:
                            st.warning("engine.geo.ensure_location_row not available.")
                        else:
                            newdfs = ensure_location_row(st.session_state["dfs"], best["location"], best["country"], best["lat"], best["lon"])
                            st.session_state["dfs"] = newdfs
                            # refresh model index
                            if build_model_index is not None:
                                st.session_state["model_index"] = build_model_index(newdfs, period=DEFAULT_PERIOD)
                            st.success("Locations sheet updated.")

    # Add Row UI
    with tabs[1]:
        st.markdown("**Add a row to a sheet**")
        sheet = st.selectbox("Sheet", ["Customers", "Warehouse", "Supplier Product", "Customer Product Data", "Transport Cost", "Locations"])
        st.caption("Fill columns as needed; leave blank for optional fields.")
        # Build blank row from existing columns (if present), else a common set:
        cols_guess = list((_safe_df(st.session_state["dfs"].get(sheet))).columns) or {
            "Customers": ["Customer", "Location"],
            "Warehouse": ["Warehouse", "Location", "Maximum Capacity", "Force Open", "Force Close"],
            "Supplier Product": ["Product", "Supplier", "Location", "Period", "Available"],
            "Customer Product Data": ["Product", "Customer", "Location", "Period", "Demand", "Lead Time", "UOM", "Variable Cost"],
            "Transport Cost": ["Mode of Transport", "Product", "From Location", "To Location", "Period", "Available", "Cost Per UOM"],
            "Locations": ["Location", "Country", "Latitude", "Longitude"],
        }[sheet]
        # inputs
        new_row: Dict[str, Any] = {}
        for c in cols_guess:
            val = st.text_input(c, key=f"add_{sheet}_{c}")
            new_row[c] = val
        if st.button("➕ Add row"):
            df = _safe_df(st.session_state["dfs"].get(sheet)).copy()
            if df.empty:
                df = pd.DataFrame(columns=cols_guess)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state["dfs"][sheet] = df
            st.success(f"Row added to {sheet}. Remember to re-run scenario & optimize.")

    # Delete Row UI (very simple key-based match)
    with tabs[2]:
        st.markdown("**Delete row(s) by simple filter**")
        sheet_d = st.selectbox("Sheet", ["Customers", "Warehouse", "Supplier Product", "Customer Product Data", "Transport Cost", "Locations"], key="del_sheet")
        key_col = st.text_input("Filter column (exact)", key="del_col")
        key_val = st.text_input("Value equals", key="del_val")
        if st.button("🗑️ Delete"):
            df = _safe_df(st.session_state["dfs"].get(sheet_d)).copy()
            if df.empty or key_col not in df.columns:
                st.warning("Sheet empty or column not found.")
            else:
                before_n = len(df)
                df = df[df[key_col].astype(str) != key_val]
                st.session_state["dfs"][sheet_d] = df
                st.success(f"Deleted {before_n - len(df)} row(s) from {sheet_d}. Re-run to apply.")

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.caption("© GENIE — Supply Chain Network Designer. This app does not send your workbook to an LLM. LLMs are used only for text understanding; all numbers and coordinates are computed deterministically.")
