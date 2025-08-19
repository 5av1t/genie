# --- GENIE: Streamlit App (Full + GenAI vs Rules, Auto-Connect, Early Map, Base Model Run) ---
# Parse → Apply → (Auto-Connect Lanes) → Optimize → KPIs & Summary → Map → Delta Views → Download

import os
import sys
from io import BytesIO
import pandas as pd

# Streamlit import (fail clearly if missing)
try:
    import streamlit as st
except ModuleNotFoundError:
    print("This app requires Streamlit. Add `streamlit` to requirements.txt and redeploy.")
    raise

# Optional mapping lib (pydeck ships with Streamlit, but guard anyway)
try:
    import pydeck as pdk
except Exception:
    pdk = None

st.set_page_config(page_title="GENIE - Supply Chain Network Designer", layout="wide")

# Inject OPENAI_API_KEY from Streamlit secrets (for GitHub→Streamlit Cloud)
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

# ========= Loader import (two styles supported) =========
loader = None
try:
    # Preferred: validator-enabled loader
    from engine.loader import load_and_validate_excel as loader
except Exception:
    try:
        from engine.loader import load_excel as _load_excel

        def loader(file):
            dfs = _load_excel(file)
            # fabricate a simple validation report if your loader doesn't validate
            report = {
                name: {
                    "missing_columns": [],
                    "num_rows": df.shape[0],
                    "num_columns": df.shape[1],
                }
                for name, df in dfs.items()
            }
            return dfs, report
    except Exception as e:
        st.error(
            "❌ Could not import a loader from engine/loader.py.\n"
            "Make sure one of these exists:\n"
            "  - load_and_validate_excel(file)\n"
            "  - load_excel(file)\n"
            f"\nImport error: {e}"
        )
        st.stop()

# ========= Engine imports =========
missing_engine = []
try:
    from engine.parser import parse_prompt as parse_rules
except ModuleNotFoundError:
    missing_engine.append("engine/parser.py (parse_prompt)")
    parse_rules = None

try:
    from engine.updater import apply_scenario
except ModuleNotFoundError:
    missing_engine.append("engine/updater.py (apply_scenario)")
try:
    from engine.optimizer import run_optimizer
except ModuleNotFoundError:
    missing_engine.append("engine/optimizer.py (run_optimizer)")
try:
    from engine.reporter import build_summary
except ModuleNotFoundError:
    missing_engine.append("engine/reporter.py (build_summary)")
try:
    from engine.geo import build_nodes, flows_to_geo, guess_map_center
except ModuleNotFoundError:
    missing_engine.append("engine/geo.py (build_nodes, flows_to_geo)")

# GenAI (optional)
try:
    from engine.genai import parse_with_llm, summarize_scenario
except ModuleNotFoundError:
    parse_with_llm = None
    summarize_scenario = None

if missing_engine:
    st.error("❌ Missing engine modules:\n- " + "\n- ".join(missing_engine))
    st.stop()

# ========= Header & help =========
st.title("🔮 GENIE - Generative Engine for Network Intelligence & Execution")
st.markdown(
    """
Upload your base case Excel, type a what‑if scenario, and GENIE will:
1) parse it into **Scenario JSON** (LLM or rules),  
2) **apply network updates** to your data,  
3) (optional) **auto‑connect missing lanes** for feasibility,  
4) run a **quick optimization**,  
5) show **KPIs + executive summary**,  
6) draw an **interactive network map** (warehouses, customers, flows), and  
7) provide **Delta Views** and an **updated Excel** to download.
"""
)

EXAMPLES = [
    "run the base model",
    "Increase Mono-Crystalline demand at Abu Dhabi by 10% and set lead time to 8",
    "Cap Bucharest_CDC Maximum Capacity at 25000; force close Berlin_LDC",
    "Enable Thin-Film at Antalya_FG",
    "Set Secondary Delivery LTL lane Paris_CDC → Aalborg for Poly-Crystalline to cost per uom = 9.5",
]

with st.expander("💡 Prompt examples"):
    for ex in EXAMPLES:
        st.code(ex, language="text")
    st.divider()
    selected = st.selectbox("Insert an example into the prompt box:", ["(choose one)"] + EXAMPLES, key="example_select")
    if st.button("Insert example"):
        if selected != "(choose one)":
            st.session_state["user_prompt"] = selected
            st.success("Inserted example into the prompt box below.")

with st.expander("📘 How to use"):
    st.markdown(
        """
1. **Upload** your base case Excel (.xlsx)  
2. **Type** a prompt, or pick an example above and click **Insert example**  
3. Click **🚀 Process Scenario** to see parsed scenario (bullets), applied edits, KPIs, map, deltas, and download  
4. Use **Auto‑connect** to generate placeholder lanes for demo feasibility  
5. Use **Parse with LLM & Rules (side‑by‑side)** to compare parsers
"""
    )

# ========= Inputs =========
uploaded_file = st.file_uploader("📤 Upload base case Excel (.xlsx)", type=["xlsx"])
user_prompt = st.text_area("🧠 Describe your what‑if scenario", height=120, key="user_prompt")

use_llm = st.checkbox(
    "🤖 Use GenAI parser (schema‑validated)",
    value=True,
    help="Use the LLM to parse your prompt into scenario JSON. Falls back to rules parser if unavailable."
)
side_by_side = st.checkbox(
    "🧪 Parse with LLM & Rules (side‑by‑side)",
    value=False,
    help="Debug mode: parse the prompt with both parsers and compare outputs before applying."
)

process = st.button("🚀 Process Scenario")

# ========= Helpers: Delta Views & Auto-connect =========
def build_delta_view(before: pd.DataFrame, after: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    if after is None or after.empty:
        return pd.DataFrame(columns=key_cols + ["Field", "Before", "After", "ChangeType"])
    before = before.copy() if isinstance(before, pd.DataFrame) else pd.DataFrame(columns=key_cols)
    after = after.copy()
    common_cols = list(set(before.columns) & set(after.columns)) if not before.empty else list(after.columns)
    if not common_cols:
        common_cols = list(after.columns)
    key_cols_used = [c for c in key_cols if c in common_cols]
    merged = after[common_cols].merge(
        before[common_cols],
        on=key_cols_used,
        how="left",
        suffixes=("_after", "_before"),
        indicator=True,
    )
    compare_fields = [c for c in common_cols if c not in key_cols_used]
    changes = []
    for col in compare_fields:
        a = f"{col}_after"; b = f"{col}_before"
        if a not in merged.columns or b not in merged.columns:
            continue
        diff_mask = (merged[a] != merged[b]) | (merged[b].isna() & merged[a].notna())
        if diff_mask.any():
            diff_rows = merged.loc[diff_mask, key_cols_used + [b, a, "_merge"]].copy()
            diff_rows["Field"] = col
            diff_rows.rename(columns={b: "Before", a: "After"}, inplace=True)
            diff_rows["ChangeType"] = diff_rows["_merge"].map({"left_only": "New", "both": "Updated", "right_only": "Removed"})
            diff_rows.drop(columns=["_merge"], inplace=True)
            changes.append(diff_rows)
    if not changes:
        return pd.DataFrame(columns=key_cols_used + ["Field", "Before", "After", "ChangeType"])
    out = pd.concat(changes, ignore_index=True)
    sort_cols = key_cols_used + ["Field"]
    out = out.sort_values(sort_cols, kind="stable")
    return out

def show_delta_block(title: str, before: pd.DataFrame, after: pd.DataFrame, key_cols: list):
    st.markdown(f"#### Δ Changes — {title}")
    delta_df = build_delta_view(before, after, key_cols)
    if not delta_df.empty:
        st.dataframe(delta_df, use_container_width=True)
        added_rows = delta_df.loc[delta_df["ChangeType"] == "New", key_cols].drop_duplicates().shape[0] if "ChangeType" in delta_df.columns else 0
        updated_rows = delta_df.loc[delta_df["ChangeType"] == "Updated", key_cols].drop_duplicates().shape[0] if "ChangeType" in delta_df.columns else 0
        st.success(f"Applied changes: {updated_rows} updated row(s), {added_rows} new row(s).")
    else:
        st.info(f"No visible changes detected in {title}.")

def autoconnect_lanes(dfs, period=2023, default_cost=10.0):
    wh = dfs.get("Warehouse", pd.DataFrame())
    cpd = dfs.get("Customer Product Data", pd.DataFrame())
    tc  = dfs.get("Transport Cost", pd.DataFrame())
    if not isinstance(tc, pd.DataFrame) or tc.empty:
        tc = pd.DataFrame(columns=[
            "Mode of Transport","Product","From Location","To Location","Period",
            "UOM","Available","Retrieve Distance","Average Load Size",
            "Cost Per UOM","Cost per Distance","Cost per Trip","Minimum Cost Per Trip",
        ])
    # warehouse -> location
    wlocs = {}
    if isinstance(wh, pd.DataFrame) and not wh.empty:
        for _, r in wh.iterrows():
            w = str(r.get("Warehouse"))
            loc = str(r.get("Location")) if pd.notna(r.get("Location")) else w
            wlocs[w] = loc
    # demanded pairs (customer, product)
    need = set()
    if isinstance(cpd, pd.DataFrame) and not cpd.empty:
        use = cpd.copy()
        if "Period" in use.columns:
            use = use[use["Period"] == period]
        for _, r in use.iterrows():
            cust = str(r.get("Customer")); prod = str(r.get("Product"))
            dem = float(r.get("Demand", 0) or 0)
            if dem > 0:
                need.add((cust, prod))
    # existing arcs (for period)
    existing = set()
    if isinstance(tc, pd.DataFrame) and not tc.empty:
        for _, r in tc.iterrows():
            try:
                if int(r.get("Period", period)) != period: continue
            except Exception:
                continue
            existing.add((str(r.get("From Location")), str(r.get("To Location")), str(r.get("Product"))))
    # create missing
    new_rows = []
    for _, from_loc in wlocs.items():
        for (to_cust, prod) in need:
            key = (str(from_loc), str(to_cust), str(prod))
            if key in existing: continue
            new_rows.append({
                "Mode of Transport": "Secondary Delivery LTL",
                "Product": prod,
                "From Location": from_loc,
                "To Location": to_cust,
                "Period": period,
                "UOM": "Each",
                "Available": 1,
                "Retrieve Distance": 0.0,
                "Average Load Size": 1.0,
                "Cost Per UOM": float(default_cost),
                "Cost per Distance": 0.0,
                "Cost per Trip": 0.0,
                "Minimum Cost Per Trip": 0.0,
            })
    if new_rows:
        tc = pd.concat([tc, pd.DataFrame(new_rows)], ignore_index=True)
    dfs["Transport Cost"] = tc
    return dfs, len(new_rows)

# ========= Main flow =========
if uploaded_file:
    # ---- Load & validate ----
    try:
        dataframes, validation_report = loader(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading Excel: {e}")
        st.stop()

    with st.expander("✅ Sheet Validation Report"):
    # First: per-sheet reports (dict entries)
        for sheet, rep in validation_report.items():
            if sheet == "_warnings" or not isinstance(rep, dict):
                continue  # skip warnings list or any non-dict
                missing = rep.get("missing_columns", [])
            if missing:
                st.error(f"{sheet}: Missing columns - {missing}")
            else:
                st.success(f"{sheet}: OK ({rep.get('num_rows', 0)} rows, {rep.get('num_columns', 0)} columns)")

    # Then: global warnings list
    warns = validation_report.get("_warnings", [])
    if isinstance(warns, list) and warns:
        st.warning("General warnings detected:")
        for w in warns:
            st.write(f"• {w}")

    st.success("Base case loaded successfully.")

    # ---- Early map (nodes only) ----
    st.subheader("🌍 Network Map (Nodes)")
    if pdk:
        try:
            nodes_df = build_nodes(dataframes)
            if nodes_df.empty:
                st.info("No geocoded nodes available yet. Add a 'Locations' sheet with Latitude/Longitude or use recognized city names.")
            else:
                lat0, lon0 = guess_map_center(nodes_df)
                wh_nodes = nodes_df[nodes_df["type"] == "warehouse"]
                cu_nodes = nodes_df[nodes_df["type"] == "customer"]
                layers = []
                if not wh_nodes.empty:
                    layers.append(pdk.Layer(
                        "ScatterplotLayer",
                        data=wh_nodes,
                        get_position='[lon, lat]',
                        get_radius=60000,
                        pickable=True,
                        filled=True,
                        get_fill_color=[30, 136, 229],  # blue
                    ))
                if not cu_nodes.empty:
                    layers.append(pdk.Layer(
                        "ScatterplotLayer",
                        data=cu_nodes,
                        get_position='[lon, lat]',
                        get_radius=40000,
                        pickable=True,
                        filled=True,
                        get_fill_color=[76, 175, 80],  # green
                    ))
                deck = pdk.Deck(
                    initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=3.2, pitch=0),
                    layers=layers,
                    tooltip={"html": "<b>{name}</b><br/>{location}", "style": {"color": "white"}},
                )
                st.pydeck_chart(deck)
        except Exception as e:
            st.warning(f"Map rendering skipped: {e}")
    else:
        st.info("pydeck not available in this environment, skipping map.")

    # ---- Controls ----
    colP1, colP2 = st.columns([1,1], vertical_alignment="center")
    with colP1:
        st.caption("Process with the selected parser below")
        go_clicked = process
    with colP2:
        both_clicked = st.button("🧪 Parse with LLM & Rules (side‑by‑side)")

    # ---- Side-by-side parser comparison (optional) ----
    if both_clicked and (st.session_state.get("user_prompt") or "").strip():
        tabs = st.tabs(["LLM Parser", "Rules Parser", "Diff"])
        llm_scenario = None
        rules_scenario = None
        with tabs[0]:
            if parse_with_llm is None:
                st.error("GenAI parser not available (engine/genai.py missing or OPENAI_API_KEY not set).")
            else:
                llm_scenario = parse_with_llm(st.session_state["user_prompt"], dataframes, default_period=2023)
                if summarize_scenario:
                    bullets = summarize_scenario(llm_scenario)
                    st.markdown("**Summary**")
                    st.markdown("\n".join([f"- {b}" for b in bullets]))
                with st.expander("Advanced: raw JSON"):
                    st.json(llm_scenario)
        with tabs[1]:
            if parse_rules is None:
                st.error("Rules parser not available.")
            else:
                rules_scenario = parse_rules(st.session_state["user_prompt"], dataframes, default_period=2023)
                if summarize_scenario:
                    bullets = summarize_scenario(rules_scenario)
                    st.markdown("**Summary**")
                    st.markdown("\n".join([f"- {b}" for b in bullets]))
                with st.expander("Advanced: raw JSON"):
                    st.json(rules_scenario)
        with tabs[2]:
            st.markdown("**Simple textual diff** (non-blocking):")
            try:
                import json, difflib
                a = json.dumps(llm_scenario or {}, indent=2, sort_keys=True).splitlines()
                b = json.dumps(rules_scenario or {}, indent=2, sort_keys=True).splitlines()
                diff = difflib.unified_diff(a, b, fromfile="LLM", tofile="Rules", lineterm="")
                st.code("\n".join(diff) or "No differences.")
            except Exception as e:
                st.info(f"Diff unavailable: {e}")

    # ---- Main Process flow ----
    if go_clicked and (st.session_state.get("user_prompt") or "").strip():
        # 1) Parse NL → Scenario JSON (LLM or rules)
        try:
            if use_llm and parse_with_llm is not None:
                scenario = parse_with_llm(st.session_state["user_prompt"], dataframes, default_period=2023)
                parsed_by = "GenAI"
            else:
                scenario = parse_rules(st.session_state["user_prompt"], dataframes, default_period=2023) if parse_rules else {}
                parsed_by = "rules"
        except Exception as e:
            st.error(f"❌ Parsing failed: {e}")
            st.stop()

        st.subheader(f"Scenario ({parsed_by})")
        if summarize_scenario:
            bullets = summarize_scenario(scenario)
            st.markdown("\n".join([f"- {b}" for b in bullets]))
        with st.expander("Advanced: show raw JSON"):
            st.json(scenario)

        # Special case: "run the base model" → do not apply edits; just run optimizer
        run_base = str(st.session_state["user_prompt"]).strip().lower() in {"run the base model", "run base model"}
        dfs_to_use = dataframes if run_base else None

        # 2) Apply scenario across sheets (unless running base)
        before_cpd = dataframes.get("Customer Product Data", pd.DataFrame()).copy()
        before_wh  = dataframes.get("Warehouse", pd.DataFrame()).copy()
        before_sp  = dataframes.get("Supplier Product", pd.DataFrame()).copy()
        before_tc  = dataframes.get("Transport Cost", pd.DataFrame()).copy()

        if not run_base:
            try:
                updated = apply_scenario(dataframes, scenario)
            except Exception as e:
                st.error(f"❌ Applying scenario failed: {e}")
                st.stop()
        else:
            updated = dataframes  # no changes

        after_cpd = updated.get("Customer Product Data", pd.DataFrame()).copy()
        after_wh  = updated.get("Warehouse", pd.DataFrame()).copy()
        after_sp  = updated.get("Supplier Product", pd.DataFrame()).copy()
        after_tc  = updated.get("Transport Cost", pd.DataFrame()).copy()

        # 2.5) Side-by-side previews (Before vs After) — still useful for base model (identical)
        st.subheader("📋 Before vs After (Quick Preview)")
        tabs = st.tabs(["Customer Product Data", "Warehouse", "Supplier Product", "Transport Cost"])
        with tabs[0]:
            colA, colB = st.columns(2)
            with colA: st.markdown("**Before**"); st.dataframe(before_cpd.head(25), use_container_width=True)
            with colB: st.markdown("**After**");  st.dataframe(after_cpd.head(25), use_container_width=True)
        with tabs[1]:
            colA, colB = st.columns(2)
            with colA: st.markdown("**Before**"); st.dataframe(before_wh.head(25), use_container_width=True)
            with colB: st.markdown("**After**");  st.dataframe(after_wh.head(25), use_container_width=True)
        with tabs[2]:
            colA, colB = st.columns(2)
            with colA: st.markdown("**Before**"); st.dataframe(before_sp.head(25), use_container_width=True)
            with colB: st.markdown("**After**");  st.dataframe(after_sp.head(25), use_container_width=True)
        with tabs[3]:
            colA, colB = st.columns(2)
            with colA: st.markdown("**Before**"); st.dataframe(before_tc.head(25), use_container_width=True)
            with colB: st.markdown("**After**");  st.dataframe(after_tc.head(25), use_container_width=True)

        # 2.6) Optional: auto-connect lanes to guarantee feasibility (helpful for demos)
        auto = st.checkbox(
            "🔗 Auto-create missing lanes for feasibility (demo)",
            value=True,
            help="Creates placeholder lanes Warehouse.Location → Customer with Cost Per UOM = 10.0 for the chosen period."
        )
        if auto:
            updated, created = autoconnect_lanes(updated, period=scenario.get("period", 2023), default_cost=10.0)
            if created > 0:
                st.info(f"Auto-connect created {created} placeholder lane(s) at Cost Per UOM = 10.0")

        # 3) Optimization (quick MILP) → KPIs + flows
        kpis, diag = run_optimizer(updated, period=scenario.get("period", 2023))

        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("📊 KPIs")
            st.write(kpis)
        with col2:
            st.subheader("📝 Executive Summary")
            st.markdown(build_summary(st.session_state["user_prompt"], scenario, kpis, diag))

        # Helper if infeasible / empty arcs
        status = (kpis or {}).get("status")
        if status in {"no_feasible_arcs", "no_demand", "no_positive_demand"}:
            st.warning(
                "⚠️ The model couldn't route flows.\n\n"
                "- Ensure **Transport Cost** has lanes for the period, finite **Cost Per UOM**, and From/To locations match Warehouse.Location and Customer.\n"
                "- Check **Warehouse** capacity: `Maximum Capacity > 0`, `Force Close = 0`, `Available (Warehouse) = 1`.\n"
                "- Enable **Auto-create missing lanes** for demo feasibility."
            )

        # 4) Network Map (with flows if any)
        st.subheader("🌍 Network Map (With Flows)")
        if pdk:
            try:
                nodes_df = build_nodes(updated)
                flows = (diag or {}).get("flows", [])
                arcs_df = flows_to_geo(flows, updated)
                if nodes_df.empty:
                    st.info("No geocoded nodes available. Add a 'Locations' sheet with Latitude/Longitude or use recognized city names.")
                else:
                    lat0, lon0 = guess_map_center(nodes_df)
                    wh_nodes = nodes_df[nodes_df["type"] == "warehouse"]
                    cu_nodes = nodes_df[nodes_df["type"] == "customer"]
                    layers = []
                    if not wh_nodes.empty:
                        layers.append(pdk.Layer(
                            "ScatterplotLayer",
                            data=wh_nodes,
                            get_position='[lon, lat]',
                            get_radius=60000,
                            pickable=True,
                            filled=True,
                            get_fill_color=[30, 136, 229],  # blue
                        ))
                    if not cu_nodes.empty:
                        layers.append(pdk.Layer(
                            "ScatterplotLayer",
                            data=cu_nodes,
                            get_position='[lon, lat]',
                            get_radius=40000,
                            pickable=True,
                            filled=True,
                            get_fill_color=[76, 175, 80],  # green
                        ))
                    if isinstance(arcs_df, pd.DataFrame) and not arcs_df.empty:
                        arcs_df = arcs_df.assign(width=(arcs_df["qty"].clip(lower=1) ** 0.5))
                        layers.append(pdk.Layer(
                            "ArcLayer",
                            data=arcs_df,
                            get_source_position='[from_lon, from_lat]',
                            get_target_position='[to_lon, to_lat]',
                            get_width='width',
                            get_source_color=[255, 140, 0],
                            get_target_color=[255, 64, 64],
                            pickable=True,
                        ))
                    deck = pdk.Deck(
                        initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=3.2, pitch=0),
                        layers=layers,
                        tooltip={"html": "<b>{name}</b><br/>{location}", "style": {"color": "white"}},
                    )
                    st.pydeck_chart(deck)
            except Exception as e:
                st.warning(f"Map rendering skipped: {e}")
        else:
            st.info("pydeck not available in this environment, skipping map.")

        # 5) Delta Views
        st.subheader("Δ Delta Views (Changes Only)")
        show_delta_block(
            "Customer Product Data",
            before_cpd, after_cpd,
            ["Product", "Customer", "Location", "Period"],
        )
        if isinstance(after_wh, pd.DataFrame) and not after_wh.empty:
            wh_keys = ["Warehouse"] + (["Period"] if "Period" in after_wh.columns and "Period" in before_wh.columns else [])
            show_delta_block("Warehouse", before_wh, after_wh, wh_keys)
        if isinstance(after_sp, pd.DataFrame) and not after_sp.empty:
            show_delta_block(
                "Supplier Product",
                before_sp, after_sp,
                ["Product", "Supplier", "Location", "Period"],
            )
        if isinstance(after_tc, pd.DataFrame) and not after_tc.empty:
            show_delta_block(
                "Transport Cost",
                before_tc, after_tc,
                ["Mode of Transport", "Product", "From Location", "To Location", "Period"],
            )

        # 6) Download updated workbook
        st.subheader("💾 Export")
        try:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                for name, df in updated.items():
                    if isinstance(df, pd.DataFrame):
                        sheet_name = str(name)[:31] if name else "Sheet"
                        df.to_excel(w, sheet_name=sheet_name, index=False)
            st.download_button(
                label="Download Updated Scenario Excel",
                data=buf.getvalue(),
                file_name="genie_updated_scenario.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.warning(f"Creating the Excel download failed: {e}")
else:
    # Sample download (prefer bundling the file; fallback to raw GitHub URL)
    raw_url = "https://raw.githubusercontent.com/5av1t/genie/main/sample_base_case.xlsx"
    if os.path.exists("sample_base_case.xlsx"):
        with open("sample_base_case.xlsx", "rb") as f:
            st.download_button(
                label="⬇️ Download Sample Base Case Template",
                data=f,
                file_name="sample_base_case.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.markdown(f"[⬇️ Download Sample Base Case Template]({raw_url})")
        st.info("Tip: add `sample_base_case.xlsx` to the repo root to enable the in‑app download button.")
