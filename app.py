# --- GENIE: Streamlit App (MVP with Parsing) ---

# Robust import for Streamlit
try:
    import streamlit as st
except ModuleNotFoundError:
    print("This app requires Streamlit. Add `streamlit` to requirements.txt and redeploy.")
    raise

import sys
from engine.parser import parse_prompt
import pandas as pd

# Internal modules
try:
    from engine.loader import load_and_validate_excel
    from engine.parser import parse_prompt  # <-- NEW: parser import
except ModuleNotFoundError as e:
    if e.name.startswith("engine"):
        msg = (
            "\n❌ Import error: missing engine module.\n"
            "Ensure your repo structure is:\n\n"
            "genie/\n"
            "├─ app.py\n"
            "├─ engine/\n"
            "│  ├─ __init__.py\n"
            "│  ├─ loader.py\n"
            "│  └─ parser.py\n"
            "├─ requirements.txt\n"
            "└─ README.md\n"
        )
        try:
            st.write(msg)
        except Exception:
            print(msg)
        raise
    else:
        raise

# -------------------------------- UI --------------------------------

st.set_page_config(page_title="GENIE - Supply Chain Network Designer", layout="wide")

# Env health (helps debugging on Streamlit Cloud)
with st.expander("🔧 Environment health"):
    st.write({
        "python_version": sys.version.split()[0],
        "streamlit_version": st.__version__,
        "pandas_version": pd.__version__,
    })

# Title + intro
st.title("🔮 GENIE - Generative Engine for Network Intelligence & Execution")
st.markdown(
    """
Welcome to **GENIE** — your GenAI assistant for supply chain network design.

Upload your base case Excel file, describe a scenario in natural language (e.g., _"Increase demand at Abu Dhabi by 10%"_), and GENIE will parse your intent into a structured **Scenario JSON**.  
*(Applying updates and optimization will be added in the next steps.)*
"""
)

st.info(
    """
**Example Prompt:**  
_“Increase Mono-Crystalline demand at Abu Dhabi by 10% and set Lead Time to 8 days.”_
"""
)

# How-to
with st.expander("📘 How to use GENIE"):
    st.markdown(
        """
1. **Upload** your base case Excel file (template provided below).
2. **Enter** your 'what-if' scenario in plain English.
3. Click **🚀 Process Scenario** to see the parsed **Scenario JSON**.
4. Next steps will apply updates & run optimization automatically.
"""
    )

# Inputs
uploaded_file = st.file_uploader("📤 Upload your base case Excel (.xlsx)", type=["xlsx"])
user_prompt = st.text_area("🧠 Describe your what-if scenario", height=120)

# When file is uploaded, load & validate
if uploaded_file:
    try:
        dataframes, validation_report = load_and_validate_excel(uploaded_file)
    except ValueError as ve:
        st.error(f"Error reading Excel: {ve}")
        st.stop()

    with st.expander("✅ Sheet Validation Report"):
        for sheet, report in validation_report.items():
            if report.get("missing_columns"):
                st.error(f"{sheet}: Missing columns - {report['missing_columns']}")
            else:
                st.success(f"{sheet}: OK ({report['num_rows']} rows, {report['num_columns']} columns)")

    st.success("Base case loaded successfully.")

    # 🚀 Process button (NOW ACTIVE)
    process = st.button("🚀 Process Scenario")
if process and user_prompt.strip():
    scenario = parse_prompt(user_prompt, dataframes, default_period=2023)
    st.subheader("Parsed Scenario JSON")
    st.json(scenario)
        # (Step 2 will apply these changes to DataFrames; Step 3 will optimize)
else:
    # Provide a real download if the sample file exists in the repo
    try:
        with open("sample_base_case.xlsx", "rb") as f:
            st.download_button(
                label="⬇️ Download Sample Base Case Template",
                data=f,
                file_name="sample_base_case.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except FileNotFoundError:
        st.warning(
            "Sample base case not found in the repo. Upload your own Excel or add `sample_base_case.xlsx` to the repo root."
        )
