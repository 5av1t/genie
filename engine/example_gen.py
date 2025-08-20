from typing import Dict, List, Any, Optional
import random

def _vals(df, col) -> List[str]:
    if df is None or col not in df.columns:
        return []
    return [str(x) for x in df[col].dropna().tolist()]

def _build_stats(dfs) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    cpd = dfs.get("Customer Product Data")
    stats["has_lead_time"] = bool(cpd is not None and "Lead Time" in cpd.columns)
    # simple top demands (names only)
    if cpd is not None and not cpd.empty:
        try:
            c = cpd.copy()
            c["Demand"] = c["Demand"].fillna(0)
            g = c.groupby(["Product","Customer"], as_index=False)["Demand"].sum().sort_values("Demand", ascending=False)
            stats["top_demands"] = g.head(5)[["Product","Customer"]].to_dict("records")
        except Exception:
            stats["top_demands"] = []
    else:
        stats["top_demands"] = []
    return stats

def _fallback_examples(catalog: Dict[str, List[str]], stats: Dict[str, Any]) -> List[str]:
    products = catalog.get("products", [])
    customers = catalog.get("customers", [])
    warehouses= catalog.get("warehouses", [])
    suppliers = catalog.get("suppliers", [])
    modes     = catalog.get("modes", [])
    examples = ["run the base model"]
    if products and customers:
        examples.append(f"Increase {products[0]} demand at {customers[0]} by 10%" + (" and set lead time to 8" if stats.get("has_lead_time") else ""))
    if warehouses:
        if len(warehouses) >= 2:
            examples.append(f"Cap {warehouses[0]} Maximum Capacity at 25000; force close {warehouses[1]}")
        else:
            examples.append(f"Cap {warehouses[0]} Maximum Capacity at 25000")
    if products and suppliers:
        p2 = products[1] if len(products) > 1 else products[0]
        examples.append(f"Enable {p2} at {suppliers[0]}")
    if modes and warehouses and customers and products:
        examples.append(f"Set {modes[0]} lane {warehouses[0]} -> {customers[0]} for {products[0]} to cost per uom = 9.5")
    return examples

def _validate_examples(examples: List[str], catalog: Dict[str, List[str]]) -> List[str]:
    """Keep examples that reference only known entities; allow 'run the base model'."""
    if not examples:
        return []
    known = {
        *catalog.get("products", []),
        *catalog.get("customers", []),
        *catalog.get("warehouses", []),
        *catalog.get("suppliers", []),
        *catalog.get("modes", []),
        *catalog.get("locations", []),
    }
    out = []
    for e in examples:
        if e.strip().lower() in {"run the base model", "run base model"}:
            out.append(e)
            continue
        # naive filter: all capitalized tokens that are in known sets pass; keep permissive
        out.append(e)
    # dedupe
    seen = set()
    deduped = []
    for e in out:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    return deduped

def examples_for_file(dfs, provider: str = "gemini") -> List[str]:
    # Build catalog (reuse from genai)
    try:
        from engine.genai import build_catalog, generate_examples_with_llm
    except Exception:
        build_catalog = None
        generate_examples_with_llm = None

    products  = _vals(dfs.get("Products"), "Product")
    customers = _vals(dfs.get("Customers"), "Customer") or _vals(dfs.get("Customer Product Data"), "Customer")
    warehouses= _vals(dfs.get("Warehouse"), "Warehouse")
    suppliers = _vals(dfs.get("Supplier Product"), "Supplier")
    modes     = _vals(dfs.get("Mode of Transport"), "Mode of Transport")
    locations = list(set(_vals(dfs.get("Warehouse"), "Location")) |
                    set(_vals(dfs.get("Customers"), "Location")) |
                    set(_vals(dfs.get("Customer Product Data"), "Location")) |
                    set(customers) | set(warehouses) | set(suppliers))
    catalog = {"products": products, "customers": customers, "warehouses": warehouses, "suppliers": suppliers, "modes": modes, "locations": locations}

    stats = _build_stats(dfs)

    # Try LLM first (unless provider is 'none')
    ex = []
    if provider in ("gemini", "openai") and generate_examples_with_llm is not None:
        try:
            ex = generate_examples_with_llm(catalog, stats, provider=provider)
        except Exception:
            ex = []

    # Fallback if needed
    if not ex:
        ex = _fallback_examples(catalog, stats)

    return _validate_examples(ex, catalog)
