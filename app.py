# --- GENIE: Streamlit App (Gemini/OpenAI + Rules, CRUD flows, Auto-Connect, Maps, Optimizer) ---

import os
from io import BytesIO
import json
import hashlib
import pandas as pd

# Streamlit import
try:
    import streamlit as st
except ModuleNotFoundError:
    print("This application requires Streamlit to run. Please add `streamlit` to requirements.txt.")
    raise

# Optional mapping lib
try:
    import pydeck as pdk
except Exception:
    pdk = None

st.set_page_config(page_title="GENIE - Supply Chain Network Designer", layout="wide")

# ========= Read secrets → env for providers (optional convenience) =========
# NOTE: we never display any secret in the UI.
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
if "gcp" in st.secrets and "gemini_api_key" in st.secrets["gcp"]:
    os.environ["GEMINI_API_KEY"] = st.secrets["gcp"]["gemini_api_key"]
if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
    os.environ["GEMINI_API_KEY"] = st.secrets["gemini"]["api_key"]

# ========= Imports from engine =========
missing_engine = []

try:
    from engine.loader import load_and_validate_excel as loader
except Exception as e:
    st.error("❌ Could not import loader from engine/loader.py.\nMake sure `load_and_validate_excel(file_like)` exists.\n\n" + str(e))
    st.stop()

try:
    from engine.parser import parse_prompt as parse_rules
except ModuleNotFoundError:
    missing_engine.append("engine/parser.py (parse_prompt)")
    parse_rules = None

try:
    from engine.updater import apply_scenario
except ModuleNotFoundError:
    missing_engine.append("engine/updater.py (apply_scenario)")
    apply_scenario = None

try:
    from engine.optimizer import run_optimizer
except ModuleNotFoundError:
    missing_engine.append("engine/optimizer.py (run_optimizer)")
    run_optimizer = None

try:
    from engine.reporter import build_summary
except ModuleNotFoundError:
    missing_engine.append("engine/reporter.py (build_summary)")
    build_summary = None

try:
    from engine.geo import build_nodes, flows_to_geo, guess_map_center
except ModuleNotFoundError:
    missing_engine.append("engine/geo.py (build_nodes, flows_to_geo, guess_map_center)")
    build_nodes = flows_to_geo = guess_map_center = None

try:
    from engine.genai import parse_with_llm, summarize_scenario, answer_question
except ModuleNotFoundError:
    parse_with_llm = None
    summarize_scenario = None
    answer_question = None

try:
    from engine.example_gen import examples_for_file
except ModuleNotFoundError:
    examples_for_file = None

if missing_engine:
    st.error("❌ Missing engine modules:\n- " + "\n- ".join(missing_engine))
    st.stop()

# ========= Header =========
st.title("🔮 GENIE — Generative Engine for Network Intelligence & Execution")
st.caption("Upload a base case → describe a scenario → GENIE updates sheets, optimizes, and maps your network.")

# ========= Sidebar =========
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox(
        "Model Provider",
        ["Google Gemini (free tier)", "OpenAI", "Rules only (no LLM)"],
        index=0,
        help="Gemini free tier is recommended for demos. Switch to OpenAI if you have billing. Rules-only disables LLM parsing."
    )
    use_llm_parser = st.checkbox("Use GenAI parser", value=(provider != "Rules only (no LLM)"))
    show_llm_vs_rules = st.checkbox("Show LLM vs Rules (debug tabs)", value=False)
    st.markdown("---")
    st.subheader("🧠 What can GENIE do?")
    st.caption("Natural language actions supported by the GenAI parser:")
    st.markdown(
        "- Adjust customer **Demand** (and **Lead Time**)\n"
        "- Toggle warehouse **Force Open/Close**; change **Maximum Capacity**, costs\n"
        "- **Enable/disable supplier** availability for a product\n"
        "- Set **Transport Cost** per lane (Mode, From, To, Product, Period)\n"
        "- **Add** new customers/warehouses/supplier products/lanes/demand rows\n"
        "- **Delete** customers/warehouses/supplier products/lanes/demand rows\n"
        "- \"**Run the base model**\" (no edits, just optimize + map)\n"
        "- **Ask about results** (e.g., “lowest throughput warehouse”, “top lanes by flow”)"
    )
    st.markdown("---")
    st.caption("Configure secrets in Streamlit Cloud → App settings → Secrets. For local dev, use `.streamlit/secrets.toml`.")

# ========= Inputs (upload) =========
uploaded_file = st.file_uploader("📤 Upload base case Excel (.xlsx)", type=["xlsx"])

# ========= Helpers =========
def build_delta_view(before: pd.DataFrame, after: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    if after is None or not isinstance(after, pd.DataFrame) or after.empty:
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
        added = delta_df.loc[delta_df["ChangeType"] == "New", key_cols].drop_duplicates().shape[0] if "ChangeType" in delta_df.columns else 0
        updated = delta_df.loc[delta_df["ChangeType"] == "Updated", key_cols].drop_duplicates().shape[0] if "ChangeType" in delta_df.columns else 0
        st.success(f"Applied changes: {updated} updated row(s), {added} new row(s).")
    else:
        st.info(f"No visible changes detected in {title}.")

def autoconnect_lanes(dfs, period=2023, default_cost=10.0):
    wh = dfs.get("Warehouse", pd.DataFrame())
    cpd = dfs.get("Customer Product Data", pd.DataFrame())
    tc  = dfs.get("Transport Cost", pd.DataFrame())
    if not isinstance(tc, pd.DataFrame) or tc.empty:
        tc = pd.DataFrame(columns=[
            "Mode of Transport","Product","From Location","To Location","Period",
            "UOM","Available","Retrieve Distance","Average Load Size","Cost Per UOM","Cost per Distance","Cost per Trip","Minimum Cost Per Trip",
        ])
    wlocs = {}
    if isinstance(wh, pd.DataFrame) and not wh.empty:
        for _, r in wh.iterrows():
            w = str(r.get("Warehouse"))
            loc = str(r.get("Location")) if pd.notna(r.get("Location")) else w
            wlocs[w] = loc
    need = set()
    if isinstance(cpd, pd.DataFrame) and not cpd.empty:
        use = cpd.copy()
        if "Period" in use.columns:
            use = use[use["Period"] == period]
        for _, r in use.iterrows():
            cust = str(r.get("Customer")); prod = str(r.get("Product"))
            try:
                dem = float(r.get("Demand", 0) or 0)
            except Exception:
                dem = 0.0
            if dem > 0:
                need.add((cust, prod))
    existing = set()
    if isinstance(tc, pd.DataFrame) and not tc.empty:
        for _, r in tc.iterrows():
            try:
                if int(r.get("Period", period)) != period: 
                    continue
            except Exception:
                continue
            existing.add((str(r.get("From Location")), str(r.get("To Location")), str(r.get("Product"))))
    new_rows = []
    for _, from_loc in wlocs.items():
        for (to_cust, prod) in need:
            key = (str(from_loc), str(to_cust), str(prod))
            if key in existing: 
                continue
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

# ========= Static examples =========
STATIC_EXAMPLES = [
    "run the base model",
    "Increase Mono-Crystalline demand at Abu Dhabi by 10% and set lead time to 8",
    "Cap Bucharest_CDC Maximum Capacity at 25000; force close Berlin_LDC",
    "Enable Thin-Film at Antalya_FG",
    "Set Secondary Delivery LTL lane Paris_CDC -> Aalborg for Poly-Crystalline to cost per uom = 9.5",
    "Add customer Muscat; demand 8000 of Mono-Crystalline; lead time 9",
    "Add warehouse Prague_CDC at Prague; Maximum Capacity 30000; force open",
    "Delete Secondary Delivery LTL lane Paris_CDC -> Aalborg for Poly-Crystalline in 2023",
]

# Prefill buffer
user_prompt_value = st.session_state.get("user_prompt", "")
if "prefill_prompt" in st.session_state:
    user_prompt_value = st.session_state.pop("prefill_prompt", user_prompt_value)

# ========= Uploaded flow =========
if uploaded_file:
    try:
        dataframes, validation_report = loader(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading Excel: {e}")
        st.stop()

    with st.expander("✅ Sheet Validation Report"):
        try:
            for sheet, rep in validation_report.items():
                if sheet == "_warnings":
                    continue
                if not isinstance(rep, dict):
                    continue
                missing = rep.get("missing_columns", [])
                if missing:
                    st.error(f"{sheet}: Missing columns - {missing}")
                else:
                    st.success(f"{sheet}: OK ({rep.get('num_rows', 0)} rows, {rep.get('num_columns', 0)} columns)")
            warns = validation_report.get("_warnings", [])
            if isinstance(warns, list) and warns:
                st.warning("General warnings:")
                for w in warns:
                    st.write(f"• {w}")
        except Exception as e:
            st.warning(f"Validation report display issue: {e}")

    st.success("Base case loaded successfully.")

    # ---- Suggested prompts (above prompt) ----
    st.subheader("💡 Suggested prompts")
    dyn_examples = []
    try:
        fp_hasher = hashlib.sha1()
        fp_hasher.update("|".join(sorted([f"{k}:{v.shape[0]}x{v.shape[1]}" for k, v in dataframes.items() if isinstance(v, pd.DataFrame)])).encode())
        fingerprint = fp_hasher.hexdigest()
    except Exception:
        fingerprint = "nofp"

    if "examples_cache" not in st.session_state:
        st.session_state["examples_cache"] = {}
    dyn_examples = st.session_state["examples_cache"].get(fingerprint, None)
    if dyn_examples is None:
        dyn_examples = []
        try:
            if examples_for_file is not None:
                prov = "gemini" if provider.startswith("Google") else ("openai" if provider == "OpenAI" else "none")
                dyn_examples = examples_for_file(dataframes, provider=prov) or []
        except Exception:
            dyn_examples = []
        st.session_state["examples_cache"][fingerprint] = dyn_examples

    with st.expander("Show suggestions"):
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Static**")
            for ex in STATIC_EXAMPLES:
                st.code(ex, language="text")
            chosen_static = st.selectbox("Insert static example:", ["(choose one)"] + STATIC_EXAMPLES, key="static_select")
            if st.button("Insert static"):
                if chosen_static != "(choose one)":
                    st.session_state["prefill_prompt"] = chosen_static
                    st.rerun()
        with colB:
            st.markdown("**From your file**")
            if dyn_examples:
                for ex in dyn_examples:
                    st.code(ex, language="text")
                chosen_dyn = st.selectbox("Insert generated example:", ["(choose one)"] + dyn_examples, key="dyn_select")
                if st.button("Insert generated"):
                    if chosen_dyn != "(choose one)":
                        st.session_state["prefill_prompt"] = chosen_dyn
                        st.rerun()
            else:
                st.info("No generated prompts available right now (fallback to static).")

    # ---- Map (nodes only) ----
    st.subheader("🌍 Network Map (Nodes)")
    if pdk and build_nodes and guess_map_center:
        try:
            nodes_df = build_nodes(dataframes)
            if nodes_df.empty:
                st.info("No geocoded nodes yet. Ensure a 'Locations' sheet with Latitude/Longitude or recognized city names.")
            else:
                lat0, lon0 = guess_map_center(nodes_df)
                wh_nodes = nodes_df[nodes_df["type"] == "warehouse"]
                cu_nodes = nodes_df[nodes_df["type"] == "customer"]
                layers = []
                if not wh_nodes.empty:
                    layers.append(pdk.Layer("ScatterplotLayer", data=wh_nodes, get_position='[lon, lat]', get_radius=60000, pickable=True, filled=True, get_fill_color=[30,136,229]))
                if not cu_nodes.empty:
                    layers.append(pdk.Layer("ScatterplotLayer", data=cu_nodes, get_position='[lon, lat]', get_radius=40000, pickable=True, filled=True, get_fill_color=[76,175,80]))
                deck = pdk.Deck(initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=3.2, pitch=0), layers=layers, tooltip={"html": "<b>{name}</b><br/>{location}", "style": {"color": "white"}})
                st.pydeck_chart(deck)
        except Exception as e:
            st.warning(f"Map rendering skipped: {e}")
    else:
        st.info("pydeck not available or geo module missing; skipping map.")

    st.markdown("---")
    # ---- Prompt input ----
    user_prompt = st.text_area("🧠 Describe your what‑if scenario", height=120, key="user_prompt", value=user_prompt_value)
    process_btn = st.button("🚀 Process Scenario")

    # ---- Optional: LLM vs Rules ----
    if show_llm_vs_rules and (user_prompt or "").strip():
        tabs = st.tabs(["LLM Parser", "Rules Parser", "Diff"])
        llm_scenario = None
        rules_scenario = None
        with tabs[0]:
            if parse_with_llm is None or provider == "Rules only (no LLM)":
                st.error("GenAI parser disabled or not available.")
            else:
                prov = "gemini" if provider.startswith("Google") else "openai"
                try:
                    llm_scenario = parse_with_llm(user_prompt, dataframes, default_period=2023, provider=prov)
                except Exception as e:
                    llm_scenario = {}
                    st.error(f"LLM parse error: {e}")
                if summarize_scenario:
                    st.markdown("**Summary**"); st.markdown("\n".join([f"- {b}" for b in summarize_scenario(llm_scenario)]))
                with st.expander("Advanced: raw JSON"): st.json(llm_scenario)
        with tabs[1]:
            if parse_rules is None:
                st.error("Rules parser not available.")
            else:
                try:
                    rules_scenario = parse_rules(user_prompt, dataframes, default_period=2023)
                except Exception as e:
                    rules_scenario = {}
                    st.error(f"Rules parse error: {e}")
                if summarize_scenario:
                    st.markdown("**Summary**"); st.markdown("\n".join([f"- {b}" for b in summarize_scenario(rules_scenario)]))
                with st.expander("Advanced: raw JSON"): st.json(rules_scenario)
        with tabs[2]:
            st.markdown("**Simple textual diff**")
            try:
                import difflib
                a = json.dumps(llm_scenario or {}, indent=2, sort_keys=True).splitlines()
                b = json.dumps(rules_scenario or {}, indent=2, sort_keys=True).splitlines()
                diff = difflib.unified_diff(a, b, fromfile="LLM", tofile="Rules", lineterm="")
                st.code("\n".join(diff) or "No differences.")
            except Exception as e:
                st.info(f"Diff unavailable: {e}")

    # ---- Main Process ----
    if process_btn and (user_prompt or "").strip():
        try:
            if (provider != "Rules only (no LLM)") and (parse_with_llm is not None):
                prov = "gemini" if provider.startswith("Google") else "openai"
                scenario = parse_with_llm(user_prompt, dataframes, default_period=2023, provider=prov)
                parsed_by = provider
            else:
                scenario = parse_rules(user_prompt, dataframes, default_period=2023) if parse_rules else {}
                parsed_by = "rules"
        except Exception as e:
            st.error(f"❌ Parsing failed: {e}")
            st.stop()

        st.subheader(f"Scenario ({parsed_by})")
        if summarize_scenario:
            st.markdown("\n".join([f"- {b}" for b in summarize_scenario(scenario)]))
        with st.expander("Advanced: show raw JSON"): st.json(scenario)

        run_base = str(user_prompt).strip().lower() in {"run the base model", "run base model"}

        pending_deletes = 0
        if isinstance(scenario.get("deletes"), dict):
            pending_deletes = sum(len(v) for v in scenario["deletes"].values() if isinstance(v, list))
        allow_delete = True
        if pending_deletes > 0:
            allow_delete = st.checkbox(f"🛑 Apply deletions ({pending_deletes} row(s))", value=False)

        before_cpd = dataframes.get("Customer Product Data", pd.DataFrame()).copy()
        before_wh  = dataframes.get("Warehouse", pd.DataFrame()).copy()
        before_sp  = dataframes.get("Supplier Product", pd.DataFrame()).copy()
        before_tc  = dataframes.get("Transport Cost", pd.DataFrame()).copy()
        before_cu  = dataframes.get("Customers", pd.DataFrame()).copy()

        if not run_base:
            try:
                updated = apply_scenario(dataframes, scenario, allow_delete=allow_delete)
            except Exception as e:
                st.error(f"❌ Applying scenario failed: {e}")
                st.stop()
        else:
            updated = dataframes

        after_cpd = updated.get("Customer Product Data", pd.DataFrame()).copy()
        after_wh  = updated.get("Warehouse", pd.DataFrame()).copy()
        after_sp  = updated.get("Supplier Product", pd.DataFrame()).copy()
        after_tc  = updated.get("Transport Cost", pd.DataFrame()).copy()
        after_cu  = updated.get("Customers", pd.DataFrame()).copy()

        auto = st.checkbox("🔗 Auto-create missing lanes for feasibility (demo)", value=True)
        if auto:
            updated, created = autoconnect_lanes(updated, period=scenario.get("period", 2023), default_cost=10.0)
            if created > 0:
                st.info(f"Auto-connect created {created} placeholder lane(s).")

        if run_optimizer is None:
            st.error("Optimizer module missing."); st.stop()
        kpis, diag = run_optimizer(updated, period=scenario.get("period", 2023))

        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("📊 KPIs"); st.write(kpis)
        with col2:
            st.subheader("📝 Executive Summary")
            if build_summary: st.markdown(build_summary(user_prompt, scenario, kpis, diag))
            else: st.info("reporter.build_summary missing.")

        # --- Flow Diagnostics ---
        st.subheader("🧪 Flow Diagnostics")
        try:
            period = scenario.get("period", 2023)
            tc = updated.get("Transport Cost", pd.DataFrame())
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
                    arcs_df = flows_to_geo(flows, updated); geocoded = len(arcs_df) if isinstance(arcs_df, pd.DataFrame) else 0
                except Exception: geocoded = 0
            st.write({"period": period, "lanes_in_transport_cost": lanes_period, "optimizer_arcs_considered": arcs_considered, "flows_returned": flows_count, "flows_geocoded_for_map": geocoded})
            if (kpis or {}).get("status") in {"no_feasible_arcs", "no_demand", "no_positive_demand"}:
                st.warning("No flows because the optimizer couldn't route demand. Check lanes, demand > 0, and warehouse capacity/availability.")
            elif flows_count > 0 and geocoded == 0:
                st.warning("Flows exist but none were plotted. Add a 'Locations' sheet with `Location, Latitude, Longitude` for Warehouse.Location and Customer names.")
        except Exception as e:
            st.info(f"Diagnostics unavailable: {e}")

        # ---- Map with flows ----
        st.subheader("🌍 Network Map (With Flows)")
        if pdk and build_nodes and flows_to_geo and guess_map_center:
            try:
                nodes_df = build_nodes(updated)
                flows = (diag or {}).get("flows", [])
                arcs_df = flows_to_geo(flows, updated)
                if nodes_df.empty:
                    st.info("No geocoded nodes available. Add a 'Locations' sheet with Latitude/Longitude or recognized city names.")
                else:
                    lat0, lon0 = guess_map_center(nodes_df)
                    wh_nodes = nodes_df[nodes_df["type"] == "warehouse"]
                    cu_nodes = nodes_df[nodes_df["type"] == "customer"]
                    layers = []
                    if not wh_nodes.empty:
                        layers.append(pdk.Layer("ScatterplotLayer", data=wh_nodes, get_position='[lon, lat]', get_radius=60000, pickable=True, filled=True, get_fill_color=[30,136,229]))
                    if not cu_nodes.empty:
                        layers.append(pdk.Layer("ScatterplotLayer", data=cu_nodes, get_position='[lon, lat]', get_radius=40000, pickable=True, filled=True, get_fill_color=[76,175,80]))
                    if isinstance(arcs_df, pd.DataFrame) and not arcs_df.empty:
                        arcs_df = arcs_df.assign(width=(arcs_df["qty"].clip(lower=1) ** 0.5))
                        layers.append(pdk.Layer("ArcLayer", data=arcs_df, get_source_position='[from_lon, from_lat]', get_target_position='[to_lon, to_lat]', get_width='width', get_source_color=[255,140,0], get_target_color=[255,64,64], pickable=True))
                    deck = pdk.Deck(initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=3.2, pitch=0), layers=layers, tooltip={"html": "<b>{name}</b><br/>{location}", "style": {"color": "white"}})
                    st.pydeck_chart(deck)
            except Exception as e:
                st.warning(f"Map rendering skipped: {e}")
        else:
            st.info("pydeck not available or geo module missing; skipping map.")

        # ---- Results Q&A (NEW) ----
        st.subheader("💬 Ask GENIE about results")
        qa_q = st.text_input("Ask a question about the KPIs/flows (e.g., 'Which warehouse has the lowest throughput?')", key="qa_q")
        if st.button("Ask"):
            if not qa_q.strip():
                st.info("Type a question first.")
            else:
                if answer_question is None:
                    st.error("GenAI Q&A not available (engine.genai.answer_question missing).")
                else:
                    prov = "gemini" if provider.startswith("Google") else "openai"
                    answer = answer_question(qa_q.strip(), kpis, diag, provider=prov)
                    st.markdown(f"**Answer**: {answer}")

        # ---- Quick preview & Deltas ----
        st.subheader("📋 Before vs After (Quick Preview)")
        tabs = st.tabs(["Customers", "Customer Product Data", "Warehouse", "Supplier Product", "Transport Cost"])
        with tabs[0]:
            colA, colB = st.columns(2)
            with colA: st.markdown("**Before**"); st.dataframe(before_cu.head(25), use_container_width=True)
            with colB: st.markdown("**After**");  st.dataframe(after_cu.head(25), use_container_width=True)
        with tabs[1]:
            colA, colB = st.columns(2)
            with colA: st.markdown("**Before**"); st.dataframe(before_cpd.head(25), use_container_width=True)
            with colB: st.markdown("**After**");  st.dataframe(after_cpd.head(25), use_container_width=True)
        with tabs[2]:
            colA, colB = st.columns(2)
            with colA: st.markdown("**Before**"); st.dataframe(before_wh.head(25), use_container_width=True)
            with colB: st.markdown("**After**");  st.dataframe(after_wh.head(25), use_container_width=True)
        with tabs[3]:
            colA, colB = st.columns(2)
            with colA: st.markdown("**Before**"); st.dataframe(before_sp.head(25), use_container_width=True)
            with colB: st.markdown("**After**");  st.dataframe(after_sp.head(25), use_container_width=True)
        with tabs[4]:
            colA, colB = st.columns(2)
            with colA: st.markdown("**Before**"); st.dataframe(before_tc.head(25), use_container_width=True)
            with colB: st.markdown("**After**");  st.dataframe(after_tc.head(25), use_container_width=True)

        st.subheader("Δ Delta Views (Changes Only)")
        show_delta_block("Customers", before_cu, after_cu, ["Customer", "Location"])
        show_delta_block("Customer Product Data", before_cpd, after_cpd, ["Product", "Customer", "Location", "Period"])
        if isinstance(after_wh, pd.DataFrame) and not after_wh.empty:
            wh_keys = ["Warehouse"] + (["Period"] if "Period" in after_wh.columns and "Period" in before_wh.columns else [])
            show_delta_block("Warehouse", before_wh, after_wh, wh_keys)
        if isinstance(after_sp, pd.DataFrame) and not after_sp.empty:
            show_delta_block("Supplier Product", before_sp, after_sp, ["Product", "Supplier", "Location", "Period"])
        if isinstance(after_tc, pd.DataFrame) and not after_tc.empty:
            show_delta_block("Transport Cost", before_tc, after_tc, ["Mode of Transport", "Product", "From Location", "To Location", "Period"])

        # ---- Export ----
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

# ========= No-file flow =========
else:
    st.info("Upload your base case to begin. Or download a sample template:")
    raw_url = "https://raw.githubusercontent.com/5av1t/genie/main/sample_base_case.xlsx"
    if os.path.exists("sample_base_case.xlsx"):
        with open("sample_base_case.xlsx", "rb") as f:
            st.download_button("⬇️ Download Sample Base Case Template", data=f, file_name="sample_base_case.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.markdown(f"[⬇️ Download Sample Base Case Template]({raw_url})")
        st.caption("Tip: Add `sample_base_case.xlsx` to the repo root to show a direct in‑app download button.")

    st.subheader("💡 Suggested prompts")
    with st.expander("Show suggestions"):
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Static**")
            for ex in STATIC_EXAMPLES:
                st.code(ex, language="text")
            chosen_static = st.selectbox("Insert static example:", ["(choose one)"] + STATIC_EXAMPLES, key="static_select_no_file")
            if st.button("Insert static", key="insert_static_no_file"):
                if chosen_static != "(choose one)":
                    st.session_state["prefill_prompt"] = chosen_static; st.rerun()
        with colB:
            st.markdown("**From your file**"); st.info("Upload a file to generate tailored prompts here.")
    user_prompt = st.text_area("🧠 Describe your what‑if scenario", height=120, key="user_prompt", value=user_prompt_value)
    st.caption("Upload a file to process scenarios.")
