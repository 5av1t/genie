try:
    import streamlit as st
    import pandas as pd
    from engine.loader import load_and_validate_excel

    st.set_page_config(page_title="GENIE - Supply Chain Network Designer", layout="wide")

    # ✨ Title + Intro
    st.title("🔮 GENIE - Generative Engine for Network Intelligence & Execution")

    st.markdown("""
    Welcome to **GENIE** — your GenAI assistant for supply chain network design.

    Upload your base case Excel file, describe a scenario in natural language (e.g., _"Increase demand at Abu Dhabi by 10%"_), and let GENIE update the Excel, run an optimization, and return a smart executive summary.
    """)

    st.info("""
    **Example Prompt:**
    > _"Increase Mono-Crystalline demand at Abu Dhabi by 10% and set Lead Time to 8 days."_
    """)

    # Step-by-step guide
    with st.expander("📘 How to use GENIE"):
        st.markdown("""
        1. **Upload your base case Excel file** (template provided below)
        2. **Enter your 'what-if' scenario** in plain English
        3. **Click Process Scenario** to update the model
        4. **Review results**: KPIs, executive summary, download updated Excel
        """)

    # Upload
    uploaded_file = st.file_uploader("📤 Upload your base case Excel (.xlsx)", type=["xlsx"])
    user_prompt = st.text_area("🧠 Describe your what-if scenario", height=100)

    if uploaded_file:
        # Load and validate
        dataframes, validation_report = load_and_validate_excel(uploaded_file)

        with st.expander("✅ Sheet Validation Report"):
            for sheet, report in validation_report.items():
                if report["missing_columns"]:
                    st.error(f"{sheet}: Missing columns - {report['missing_columns']}")
                else:
                    st.success(f"{sheet}: OK ({report['num_rows']} rows, {report['num_columns']} columns)")

        st.success("Base case loaded successfully.")

        if user_prompt:
            st.button("🚀 Process Scenario (Coming Next)", disabled=True)
            st.warning("Scenario processing and optimization will be added in the next step.")

    else:
        st.markdown("""
        👉 Or [Download Sample Base Case Template](https://example.com/sample_base_case.xlsx) to try it out.
        """)

except ModuleNotFoundError as e:
    print("This application requires Streamlit to run. Please install it using `pip install streamlit`.")
    print(f"Missing module: {e.name}")
