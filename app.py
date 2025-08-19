import streamlit as st
import pandas as pd
import json
import traceback

# Local imports
from engine import loader, parser, updater, optimizer, geo

st.set_page_config(page_title="GENIE - Supply Chain Network Design", layout="wide")

st.title("🧠 GENIE: Generative Engine for Network Intelligence & Execution")

# Sidebar
st.sidebar.header("Instructions")
st.sidebar.write("Upload a base-case Excel file, enter a natural language scenario prompt, and process it.")

st.sidebar.markdown("### Example Prompts")
st.sidebar.code("""
Increase Mono-Crystalline demand at Abu Dhabi by 10% and set lead time to 8

Cap Bucharest_CDC Maximum Capacity at 25000; force close Berlin_LDC

Enable Thin-Film at Antalya_FG

Set Secondary Delivery LTL lane Paris_CDC → Aalborg for Poly-Crystalline to cost per uom = 9.5
""")

# Upload Excel
uploaded_file = st.file_uploader("📤 Upload Base Network Excel", type=["xlsx"])

scenario_prompt = st.text_area("✍️ Scenario Prompt", height=120)

process_btn = st.button("🚀 Process Scenario")

if uploaded_file:
    try:
        dfs = loader.load_excel(uploaded_file)
        st.success("Base-case data loaded successfully ✅")
        st.write("Available Sheets:", list(dfs.keys()))

        if process_btn:
            try:
                # Parse + Update
                scenario = parser.parse_prompt(scenario_prompt)
                st.subheader("Parsed Scenario JSON")
                st.json(scenario)

                updated = updater.apply_updates(dfs, scenario)

                # Show diffs
                st.subheader("🔍 Updated Data Snapshots")
                after_cpd = updated.get("Customer Product Data")
                after_wh = updated.get("Warehouse")
                after_tc = updated.get("Transport Cost")

                if after_cpd is not None:
                    st.markdown("**Customer Product Data (after updates):**")
                    st.dataframe(after_cpd.head(10))

                if after_wh is not None:
                    st.markdown("**Warehouse (after updates):**")
                    st.dataframe(after_wh.head(10))

                if after_tc is not None:
                    st.markdown("**Transport Cost (after updates):**")
                    st.dataframe(after_tc.head(10))

                # Optimization
                st.subheader("📊 Optimization Results")
                try:
                    kpis, diag = optimizer.run_optimizer(updated, period=scenario.get("period", 2023))

                    if kpis.get("status") == "no_feasible_arcs":
                        st.error("❌ No feasible arcs found! The network may be disconnected.")
                        st.info("💡 Tip: Use auto-connect helper — ensure each customer has at least one warehouse/product lane defined in Transport Cost.")
                    else:
                        st.json(kpis)
                        st.write("Diagnostics:", diag)

                        # Map
                        st.subheader("🌍 Network Visualization")
                        flows = diag.get("flows", [])
                        if flows:
                            nodes = geo.build_nodes(updated)
                            arcs = geo.flows_to_geo(flows, updated)
                            if not arcs.empty:
                                import folium
                                from streamlit_folium import st_folium

                                m = folium.Map(location=[48.0, 14.0], zoom_start=4)
                                # Warehouses
                                for _, r in nodes[nodes["type"] == "warehouse"].iterrows():
                                    folium.Marker(
                                        [r["lat"], r["lon"]],
                                        popup=f"Warehouse: {r['name']}",
                                        icon=folium.Icon(color="blue", icon="building")
                                    ).add_to(m)
                                # Customers
                                for _, r in nodes[nodes["type"] == "customer"].iterrows():
                                    folium.Marker(
                                        [r["lat"], r["lon"]],
                                        popup=f"Customer: {r['name']}",
                                        icon=folium.Icon(color="green", icon="user")
                                    ).add_to(m)
                                # Arcs
                                for _, r in arcs.iterrows():
                                    folium.PolyLine(
                                        [(r["from_lat"], r["from_lon"]), (r["to_lat"], r["to_lon"])],
                                        tooltip=f"{r['from']} → {r['to']} ({r['qty']})",
                                        color="red", weight=2
                                    ).add_to(m)

                                st_folium(m, width=800, height=500)
                            else:
                                st.warning("⚠️ No valid arcs to plot on map.")
                        else:
                            st.warning("⚠️ No flows returned by optimizer.")

                except Exception as e:
                    st.error(f"Optimization error: {e}")
                    st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Failed to load Excel: {e}")
        st.code(traceback.format_exc())
else:
    st.info("⬆️ Please upload a base-case Excel file to begin.")
