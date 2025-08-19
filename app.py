# --- GENIE: Streamlit App (MVP with Parsing) ---

import sys
import pandas as pd

# Streamlit import (fail clearly if missing)
try:
    import streamlit as st
except ModuleNotFoundError:
    print("This app requires Streamlit. Add `streamlit` to requirements.txt and redeploy.")
    raise

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
            report = {name: {"missing_columns": [], "num_rows": df.shape[0], "num_columns": df.shape[1]}
                      for name, df in dfs.items()}
            return dfs, report
    except Exception as e:
        st.error("❌ Could not import a loader from engine/loader.py.\n"
                 "Make sure one of these exists:\n"
                 "  - load_and_validate_excel(file)\n"
                 "  - load_excel(file)\n"
                 f"\nImport error: {e}")
        st.stop()

# --------- Parser import ----------
try:
    from engine.parser import parse_prompt
except ModuleNotFoundError as e:
    st.error("❌ Missing `engine/parser.py`.\n"
             "Create `engine/parser.py` with a `parse_prompt(prompt, dfs, default_period=2023)` function.")
    st.stop()

# --------- Optional env health ----------
with st.expander("🔧 Environment health"):
    st.write({
        "python_version": sys.version.split()[0],
        "streamlit_version": st.__version__,
        "pandas_version": pd.__version__,
    })

# --------- UI ---------
st.title("🔮 GENIE - Generative Engine for Network Intelligence & Execution")
st.markdown(
    """
Upload your base case Excel, type a what‑if scenario, and GENIE will parse it into **Scenario JSON**.
*(Applying updates & optimization comes next.)*
"""
)

with st.expander("📘 How to use"):
    st.markdown(
        """
1. **Upload** your base case Excel (.xlsx)
2. **Type** a prompt, e.g.  
   `Increase Mono-Crystalline demand at Abu Dhabi by 10% and set lead time to 8`
3. Click **🚀 Process Scenario** to see the parsed JSON
"""
    )

uploaded_file = st.file_uploader("📤 Upload base case Excel (.xlsx)", type=["xlsx"])
user_prompt = st.text_area("🧠 Describe your what‑if scenario", height=120)

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

    # ---- Process button (define it BEFORE using it) ----
    process = st.button("🚀 Process Scenario")

    if process and user_prompt.strip():
        try:
            scenario = parse_prompt(user_prompt, dataframes, default_period=2023)
        except Exception as e:
            st.error(f"❌ Parsing failed: {e}")
            st.stop()

        st.subheader("Parsed Scenario JSON")
        st.json(scenario)
else:
    # Optional sample download
    import io, requests, streamlit as st

raw_url = "https://raw.githubusercontent.com/5av1t/genie/main/sample_base_case.xlsx"
try:
    r = requests.get(raw_url, timeout=10)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    st.download_button(
        "⬇️ Download Sample Base Case Template",
        buf.getvalue(),
        file_name="sample_base_case.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
except Exception as e:
    st.warning(f"Template download failed from GitHub raw: {e}")
    st.markdown(f"Try opening it directly: [{raw_url}]({raw_url})")
    except FileNotFoundError:
        st.info("Add a `sample_base_case.xlsx` to the repo root to enable a sample download.")
