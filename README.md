# 🔮 GENIE — Generative Engine for Network Intelligence & Execution

GENIE is a GenAI-powered assistant for **supply chain network design**. It reads a structured Excel base case, turns plain‑English “what‑if” prompts into consistent **Excel updates + an optimization run + an executive summary**.

> **MVP scope (this commit):** Upload your base case Excel and get an **instant sheet validation report**. Scenario parsing and optimization will be added next.

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

---

## 📁 Project Structure

```
.
├── app.py                    # Streamlit UI (upload, validate, how-to)
├── engine/
│   ├── __init__.py
│   └── loader.py            # Excel ingestion + light fixes + validation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 How to Use

1. Prepare your **base case Excel** in the provided template structure.
2. In the app, **upload** the file.
3. Review the **validation report** (missing columns auto‑fixed when sensible).
4. Next versions will enable natural‑language scenario edits and an optimization run.

### Example “what‑if” prompt (coming soon)

> *Increase Mono‑Crystalline demand at Abu Dhabi by 10% and set Lead Time to 8 days.*

---



