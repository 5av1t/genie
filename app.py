# --- GENIE: Streamlit App (Complete: Parse → Apply → Optimize → Map → Delta Views) ---

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

# --------- Safe loader import (supports two styles) ----------
# We support either:
#   from engine.loader import load_and_validate_excel
# OR
#   from engine.loader import load_excel            (simple loader)
loader = None
try:
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

# --------- Engine imports ----------
missing_engine = []
try:
    from engine.parser import parse_prompt
except ModuleNotFoundError:
    missing_engine.append("engine/parser.py (parse_prompt)")
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
    from engine.geo import build_nodes, flows_to_geo
except ModuleNotFoundError:
    missing_engine.append("engine/geo.py (build_nodes, flows_to_geo)")

if missing_engine:
    st.error("❌ Missing engine modules:\n- " + "\n- ".join(missing_engine))
    st.stop()

# --------- Optional env health ----------
with st.expander("🔧 Environment health"):
    st.write(
        {
            "python_version": sys.version.split()[0],
            "streamlit_version": st.__version__,
            "pandas_version": pd.__version__,
            "pydeck_available": bool(pdk),
        }
    )

# --------- UI ---------
st.title("🔮 GENIE - Generative Engine for Network Intelligence & Execution")
st.markdown(
    """
Upload your base case Excel, type a what‑if scenario, and GENIE will:
1) parse it into **Scenario JSON**,  
2) **apply network updates** to your data,  
3) run a **quick optimization** to route flows,  
4) show **KPIs + executive summary**,  
5) draw an **interactive network map** (warehouses, customers, flows), and  
6) provide **Delta Views** and an **updated Excel** to download.
"""
)

# 💡 Prompt examples (copyable + insertable)
EXAMPLES = [
    "Increase Mono-Crystalline demand at Abu Dhabi by 10% and set lead time to 8",
    "Cap Bucharest_CDC Maximum Capacity at 25000; force close Berlin_LDC",
    "Enable Thin-Film at Antalya_FG",
    "Set Secondary Delivery LTL lane Paris_CDC → Aalborg for Poly-Crystalline to cost per uom = 9.5",
]

with st.expander("💡 Prompt examples"):
    for ex in EXAMPLES:
        st.code(ex, language="text")
    st.divider()
    selected = st.selectbox("Insert an example into the prompt box:", ["(choose one)"] + EXAMPLES)
    if st.button("Insert example"):
        if selected != "(choose one)":
            st.session_state["user_prompt"] = selected
            st.success("Inserted example into the prompt box below.")

with st.expander("📘 How to use"):
    st.markdown(
        """
1. **Upload** your base case Excel (.xlsx)  
2. **Type** a prompt, or pick an example above and click **Insert example**  
3. Click **🚀 Process Scenario** to see the parsed JSON, applied edits, KPIs, map, and deltas  
4. Download the updated workbook
"""
    )

uploaded_file = st.file_uploader("📤 Upload base case Excel (.xlsx)", type=["xlsx"])
user_prompt = st.text_area("🧠 Describe your what‑if scenario", height=120, key="user_prompt")

# ---------- Helpers: Delta Views ----------
def build_delta_view(before: pd.DataFrame, after: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    """
    Returns a tall delta table showing only changed/new rows:
    key_cols... | Field | Before | After | ChangeType
    """
    if after is None or after.empty:
        return pd.DataFrame(columns=key_cols + ["Field", "Before", "After", "ChangeType"])

    before = before.copy() if isinstance(before, pd.DataFrame) else pd.DataFrame(columns=key_cols)
    after = after.copy()

    # Align common cols
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

# ---------- Main flow ----------
if uploaded_file:
    # ---- Load & validate ----
    try:
        dataframes, validation_report = loader(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading Excel: {e}")
        st.stop()

    with st.expander("✅ Sheet Validation Report"):
        for sheet, report in validation_report.items():
            if report.get("missing_columns"):
                st.error(f"{sheet}: Missing columns - {report['missing_columns']}")
            else:
                st.success(f"{sheet}: OK ({report['num_rows']} rows, {report['num_columns']} columns)")

    st.success("Base case loaded successfully.")

    # ---- Process button ----
    process = st.button("🚀 Process Scenario")

    if process and (st.session_state.get("user_prompt") or "").strip():
        # 1) Parse NL → Scenario JSON
        try:
            scenario = parse_prompt(st.session_state["user_prompt"], dataframes, default_period=2023)
        except Exception as e:
            st.error(f"❌ Parsing failed: {e}")
            st.stop()

        st.subheader("Parsed Scenario JSON")
        st.json(scenario)

        # 2) Apply scenario across sheets
        before_cpd = dataframes.get("Customer Product Data", pd.DataFrame()).copy()
        before_wh  = dataframes.get("Warehouse", pd.DataFrame()).copy()
        before_sp  = dataframes.get("Supplier Product", pd.DataFrame()).copy()
        before_tc  = dataframes.get("Transport Cost", pd.DataFrame()).copy()

        try:
            updated = apply_scenario(dataframes, scenario)
        except Exception as e:
            st.error(f"❌ Applying scenario failed: {e}")
            st.stop()

        after_cpd = updated.get("Customer Product Data", pd.DataFrame()).copy()
        after_wh  = updated.get("Warehouse", pd.DataFrame()).copy()
        after_sp  = updated.get("Supplier Product", pd.DataFrame()).copy()
        after_tc  = updated.get("Transport Cost", pd.DataFrame()).copy()

        # 3) Optimization (quick MILP) → KPIs + flows
        kpis, diag = run_optimizer(updated, period=scenario.get("period", 2023))

        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("KPIs")
            st.write(kpis)
        with col2:
            st.subheader("Executive Summary")
            st.markdown(build_summary(st.session_state["user_prompt"], scenario, kpis, diag))

        # 4) Network Map
        st.subheader("Network Map")
        if pdk:
            try:
                nodes_df = build_nodes(updated)
                flows = diag.get("flows", [])
                arcs_df = flows_to_geo(flows, updated)

                if nodes_df.empty:
                    st.info("No geocoded nodes available. Add a 'Locations' sheet with Latitude/Longitude or use recognized city names.")
                else:
                    center_lat = nodes_df["lat"].mean()
                    center_lon = nodes_df["lon"].mean()

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
                    if not arcs_df.empty:
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
                        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3.2, pitch=0),
                        layers=layers,
                        tooltip={"html": "<b>{name}</b><br/>{location}", "style": {"color": "white"}},
                    )
                    st.pydeck_chart(deck)
            except Exception as e:
                st.warning(f"Map rendering skipped: {e}")
        else:
            st.info("pydeck not available in this environment, skipping map.")

        # 5) Delta Views
        st.subheader("Delta Views")
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

        # 6) Download the updated workbook
        try:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                for name, df in updated.items():
                    if isinstance(df, pd.DataFrame):
                        sheet_name = str(name)[:31] if name else "Sheet"
                        df.to_excel(w, sheet_name=sheet_name, index=False)
            st.download_button(
                label="💾 Download Updated Scenario Excel",
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
