import pandas as pd

REQUIRED_SHEETS = {
    "Products": ["Product", "Description", "IsSKU", "IsUnion", "IsIntersection"],
    "Supplier Product": ["Product", "Supplier", "Location", "Step", "Period", "UOM", "Available"],
    "Supplier": ["Supplier", "Location"],
    "Warehouse": [
        "Warehouse", "Location", "Step", "Period", "Available (Warehouse)",
        "Minimum Capacity", "Maximum Capacity", "Fixed Cost", "Variable Cost", "Force Open", "Force Close"
    ],
    "Warehouse Product": ["Product", "Warehouse", "Location", "Step", "Period"],  # 'Available' added later
    "Customer Product Data": ["Product", "Customer", "Location", "Period", "UOM", "Demand", "Lead Time", "Variable Cost"],
    "Customers": ["Customer"],  # 'Location' added later
    "Mode of Transport": ["Mode of Transport", "Speed", "Wiggle Factor (%)"],
    "Transport Cost": [
        "Mode of Transport", "Product", "From Location", "To Location", "Period", "UOM", "Available",
        "Retrieve Distance", "Average Load Size", "Cost Per UOM", "Cost per Distance", "Cost per Trip", "Minimum Cost Per Trip"
    ],
    "Periods": ["Start Date", "End Date"]  # 'Year' added later
}

def load_and_validate_excel(file_path):
    xls = pd.ExcelFile(file_path)
    dataframes = {}
    validation_report = {}

    for sheet, required_cols in REQUIRED_SHEETS.items():
        df = xls.parse(sheet)

        # Add missing columns if needed and sensible
        if sheet == "Warehouse Product" and "Available" not in df.columns:
            df["Available"] = 1
        if sheet == "Customers" and "Location" not in df.columns:
            df["Location"] = df["Customer"]
        if sheet == "Periods" and "Year" not in df.columns and "Start Date" in df.columns:
            df["Year"] = pd.to_datetime(df["Start Date"]).dt.year

        # Validation
        missing_cols = [col for col in required_cols if col not in df.columns]
        validation_report[sheet] = {
            "missing_columns": missing_cols,
            "num_rows": df.shape[0],
            "num_columns": df.shape[1]
        }

        dataframes[sheet] = df

    return dataframes, validation_report
