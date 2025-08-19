# --- GENIE: Streamlit App (MVP with Parsing + Prompt Examples) ---

import os
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

# --------- Parser import ----------
try:
    from engine.parser import parse_prompt
except ModuleNotFoundError:
    st.error(
        "❌ Missing `engine/parser.py`.\n"
        "Create `engine/parser.py` with a `parse_prompt(prompt, dfs, default_period=2023)` function."
    )
    st.stop()

# --------- Optional env health ----------
with st.expander("🔧 Environment health"):
    st.write(
        {
            "python_version": sys.version.split()[0],
            "streamlit_version": st.__version__,
            "pandas_version": pd.__version__,
        }
    )

# --------- UI ---------
st.title("🔮 GENIE - Generative Engine for Network Intelligence & Execution")
st.markdown(
    """
Upload your base case Excel, type a what‑if scenario, and GENIE will parse it into **Scenario JSON**.
*(Applying updates & optimization comes next.)*
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
    # Small helper: insert one into the text area
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
3. Click **🚀 Process Scenario** to see the parsed JSON
"""
    )

uploaded_file = st.file_uploader("📤 Upload base case Excel (.xlsx)", type=["xlsx"])
# use a key so we can programmatically set it
user_prompt = st.text_area("🧠 Describe your what‑if scenario", height=120, key="user_prompt")

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

    if process and (st.session_state.get("user_prompt") or "").strip():
        try:
            scenario = parse_prompt(st.session_state["user_prompt"], dataframes, default_period=2023)
        except Exception as e:
            st.error(f"❌ Parsing failed: {e}")
            st.stop()

        st.subheader("Parsed Scenario JSON")
        st.json(scenario)
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
        # Fallback: direct link to raw GitHub (do not proxy through custom domains)
        st.markdown(f"[⬇️ Download Sample Base Case Template]({raw_url})")
        st.info("Tip: add `sample_base_case.xlsx` to the repo root to enable the in‑app download button.")
