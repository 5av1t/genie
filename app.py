import os
import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# -----------------------------------------------------------------------------
# API Keys setup (from st.secrets if available)
# -----------------------------------------------------------------------------
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
if "gcp" in st.secrets and "gemini_api_key" in st.secrets["gcp"]:
    os.environ["GEMINI_API_KEY"] = st.secrets["gcp"]["gemini_api_key"]
if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
    os.environ["GEMINI_API_KEY"] = st.secrets["gemini"]["api_key"]

# -----------------------------------------------------------------------------
# Engine imports (loader, parser, updater, optimizer)
# -----------------------------------------------------------------------------
missing_engine = []

try:
    from engine.loader import load_and_validate_excel, build_model_index
except ModuleNotFoundError:
    load_and_validate_excel = None
    build_model_index = None
    missing_engine.append("engine/loader.py")

try:
    from engine.parser import parse_rules
except ModuleNotFoundError:
    parse_rules = None
    missing_engine.append("engine/parser.py")

try:
    from engine.updater import apply_scenario_edits, diff_tables
except ModuleNotFoundError:
    apply_scenario_edits = None
    diff_tables = None
    missing_engine.append("engine/updater.py")

try:
    from engine.optimizer import solve_network
except ModuleNotFoundError:
    solve_network = None
    missing_engine.append("engine/optimizer.py")

# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------
if "base_data" not in st.session_state:
    st.session_state.base_data = None
if "scenario_data" not in st.session_state:
    st.session_state.scenario_data = None
if "solution" not in st.session_state:
    st.session_state.solution = None
if "model_index" not in st.session_state:
    st.session_state.model_index = None
if "genai_provider" not in st.session_state:
    st.session_state.genai_provider = "None (grounded only)"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------------------------------------------------------
# App Layout
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="GENIE - Network Optimizer")

st.title("🧞 GENIE: Generative Engine for Network Intelligence & Execution")

if missing_engine:
    st.error(f"Missing engine modules: {', '.join(missing_engine)}. Please add them.")
    st.stop()

# -----------------------------------------------------------------------------
# Upload Section
# -----------------------------------------------------------------------------
st.markdown("### 1) Upload Base-Case Excel")
sample_url = "https://raw.githubusercontent.com/5av1t/genie/main/sample_base-case.xlsx"
st.caption("Need a sample? Download the template:")

# Robust download button with fallback
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

if uploaded:
    try:
        base_data = load_and_validate_excel(uploaded)
        st.session_state.base_data = base_data
        st.session_state.scenario_data = base_data.copy()
        st.session_state.model_index = build_model_index(base_data)
        st.success("✅ Base-case data loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to parse Excel: {e}")
        st.stop()

# -----------------------------------------------------------------------------
# Scenario Editor
# -----------------------------------------------------------------------------
st.markdown("### 2) Scenario Editor")

if st.session_state.base_data is not None:
    with st.expander("Modify scenario data", expanded=False):
        st.write("You can manually edit demand, capacities, or costs below:")

        sheet_names = list(st.session_state.scenario_data.keys())
        selected_sheet = st.selectbox("Select sheet to edit", sheet_names)

        df = st.session_state.scenario_data[selected_sheet]
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        st.session_state.scenario_data[selected_sheet] = edited_df

        if st.button("Apply Scenario Edits"):
            if apply_scenario_edits:
                diffs = diff_tables(st.session_state.base_data, st.session_state.scenario_data)
                if diffs:
                    st.write("### Scenario Changes (Δ)")
                    for name, delta in diffs.items():
                        st.write(f"**{name}**")
                        st.dataframe(delta)
                else:
                    st.info("No changes detected vs. base case.")
            else:
                st.warning("Updater module not available, skipping diff.")

# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------
st.markdown("### 3) Solve Optimization")

if st.session_state.scenario_data is not None:
    if st.button("Run Optimizer"):
        if solve_network:
            try:
                sol = solve_network(st.session_state.scenario_data)
                st.session_state.solution = sol
                st.success("✅ Optimization complete.")

                st.write("**Solution summary:**")
                st.json(sol)
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")
        else:
            st.error("Optimizer not available.")

# -----------------------------------------------------------------------------
# GenAI Q&A
# -----------------------------------------------------------------------------
st.markdown("### 4) Ask GENIE (LLM-powered Q&A)")

provider = st.radio("Choose Provider", ["None (grounded only)", "Google Gemini", "OpenAI GPT"], 
                    index=["None (grounded only)", "Google Gemini", "OpenAI GPT"].index(st.session_state.genai_provider),
                    key="provider_choice")

st.session_state.genai_provider = provider

user_q = st.text_input("Ask a question about the model/solution:")

if user_q:
    # Append to history
    st.session_state.chat_history.append({"role": "user", "content": user_q})

    # Here we would call the provider (Gemini or GPT) with context from model_index/solution
    # For now, return a stub
    if provider == "None (grounded only)":
        answer = "Grounded-only mode: I can summarize based on the uploaded data but won’t call an external LLM."
    elif provider == "Google Gemini":
        if st.session_state.model_index is None:
            answer = "Gemini selected, but no model index available. Please upload data first."
        else:
            answer = "Gemini would answer here using model index + solution context."
    elif provider == "OpenAI GPT":
        if st.session_state.model_index is None:
            answer = "GPT selected, but no model index available. Please upload data first."
        else:
            answer = "GPT would answer here using model index + solution context."
    else:
        answer = "Invalid provider selection."

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Show chat history
if st.session_state.chat_history:
    st.markdown("#### Chat History")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**GENIE:** {msg['content']}")
