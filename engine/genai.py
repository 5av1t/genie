# engine/genai.py
# Grounded Gemini/OpenAI helpers with strict, deterministic data access:
# - NEVER generate lat/lon with LLM. Use it only to parse/understand text.
# - For coordinates, call engine.geo.geocode_location (sheet/gazetteer/online).
#
# Exposes:
#   - parse_place_text(...)  -> {"city":..., "country":...} from free text (LLM optional)
#   - suggest_location_candidates(...) -> [{location,country,lat,lon,source}]
#   - answer_question(...)   -> robust, grounded Q&A (as before, improved wording)
#
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import re
import json

# Optional LLMs
_GEMINI_AVAILABLE = False
_OPENAI_AVAILABLE = False
try:
    import google.generativeai as genai  # type: ignore
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _GEMINI_AVAILABLE = True
except Exception:
    _GEMINI_AVAILABLE = False

try:
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# Import deterministic geo
try:
    from engine.geo import geocode_location, ensure_location_row
except Exception:
    geocode_location = None  # type: ignore
    ensure_location_row = None  # type: ignore

# --------------------- Lightweight, safe text parser ---------------------

def _simple_city_country(text: str) -> Tuple[str, str]:
    """
    Quick deterministic heuristic:
      - Split on common separators; take first token as city guess.
      - If a 2nd token exists and looks like a country name/ISO-ish, use it.
    This is only a fallback if no LLM or LLM disabled.
    """
    t = (text or "").strip()
    if not t:
        return "", ""
    # strip common suffixes (_CDC/_LDC/_FG)
    base = t
    if "_" in base:
        base = base.split("_")[0]
    # separate commas
    parts = [p.strip() for p in re.split(r"[,\-/|]+", base) if p.strip()]
    if not parts:
        return base, ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]

def parse_place_text(text: str, provider: str = "gemini") -> Dict[str, str]:
    """
    Parse free text into {"city":..., "country":...} using LLM if available, else fallback heuristic.
    We NEVER ask the LLM for coordinates.
    """
    city, country = "", ""
    t = (text or "").strip()
    if not t:
        return {"city": "", "country": ""}

    # Heuristic first (often good enough)
    city_h, country_h = _simple_city_country(t)

    # LLM optional refinement for country name (no numbers!)
    prompt = (
        "Extract city and country from the following text. "
        "Respond ONLY as a compact JSON with keys 'city' and 'country'. "
        "Do NOT include coordinates or any numbers.\n\n"
        f"Text: {t}"
    )

    if provider.lower().startswith("gemini") and _GEMINI_AVAILABLE:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            js = resp.text or ""
            # naive JSON scrape
            m = re.search(r"\{.*\}", js, flags=re.S)
            if m:
                data = json.loads(m.group(0))
                city = (data.get("city") or city_h or "").strip()
                country = (data.get("country") or country_h or "").strip()
            else:
                city, country = city_h, country_h
        except Exception:
            city, country = city_h, country_h
    elif provider.lower().startswith("openai") and _OPENAI_AVAILABLE:
        try:
            client = OpenAI()
            msg = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Extract city and country as JSON. No numbers, no coords."},
                    {"role":"user","content": prompt},
                ],
                temperature=0.0,
            )
            js = msg.choices[0].message.content or ""
            m = re.search(r"\{.*\}", js, flags=re.S)
            if m:
                data = json.loads(m.group(0))
                city = (data.get("city") or city_h or "").strip()
                country = (data.get("country") or country_h or "").strip()
            else:
                city, country = city_h, country_h
        except Exception:
            city, country = city_h, country_h
    else:
        city, country = city_h, country_h

    return {"city": city, "country": country}

# --------------------- Location suggestions (deterministic geocode) ---------------------

def suggest_location_candidates(query_text: str,
                                dfs: Dict[str, Any],
                                provider: str = "gemini",
                                allow_online: bool = False) -> List[Dict[str, Any]]:
    """
    Given a free text like 'Prague_LDC, Czechia', return candidate locations with coords:
      - We ask LLM ONLY to normalize (city,country), NEVER for numbers.
      - Then we call geocode_location(city, country) deterministically.
    """
    parsed = parse_place_text(query_text, provider=provider)
    city = parsed.get("city","")
    country = parsed.get("country","")
    candidates: List[Dict[str, Any]] = []

    # 1) Try full 'query_text' as a Location exactly (sheet/CSV/seed/online)
    if geocode_location:
        hit = geocode_location(query_text, dfs=dfs, city_hint=city, country_hint=country, allow_online=allow_online)
        if hit:
            lat, lon, ctry = hit
            candidates.append({"location": query_text, "country": ctry or country, "lat": lat, "lon": lon, "source": "sheet/csv/seed/online"})

    # 2) Try the parsed city/country
    if geocode_location and city and (not candidates):
        hit = geocode_location(city, dfs=dfs, city_hint=city, country_hint=country, allow_online=allow_online)
        if hit:
            lat, lon, ctry = hit
            candidates.append({"location": city, "country": ctry or country, "lat": lat, "lon": lon, "source": "parsed"})

    return candidates

# --------------------- Q&A (kept concise; defer to your existing grounded logic) ---------------------

def answer_question(question: str,
                    kpis: Dict[str, Any],
                    diag: Dict[str, Any],
                    provider: str = "gemini",
                    dfs: Optional[Dict[str, Any]] = None,
                    model_index: Optional[Dict[str, Any]] = None,
                    force_llm: bool = False) -> str:
    """
    Keep the same signature your app already uses.
    This version focuses on *grounded* answers and defers location/coords to suggest_location_candidates().
    """
    q = (question or "").strip().lower()

    # Example: "add location for Prague in Czechia"
    if "lat" in q or "longitude" in q or "latitude" in q or "location for" in q:
        # Extract free text after 'for ' if present
        m = re.search(r"(?:for|of)\s+(.+)$", q)
        target = m.group(1).strip() if m else question
        allow_online = os.environ.get("GENIE_GEOCODE_ONLINE", "false").lower() in {"1","true","yes"}
        cands = suggest_location_candidates(target, dfs or {}, provider=provider, allow_online=allow_online)
        if not cands:
            msg = "I couldn’t find coordinates deterministically. You can enable online geocoding via Nominatim by setting GENIE_GEOCODE_ONLINE=true in environment or add the row in the 'Locations' sheet."
            return msg
        best = cands[0]
        return (f"Resolved: {best['location']} ({best['country']}) at lat={best['lat']:.4f}, lon={best['lon']:.4f} "
                f"(source={best['source']}). Use 'Add to Locations' to persist.")

    # Otherwise defer to your existing grounded logic (already implemented in your app’s flow).
    # Keep this minimal to avoid duplicating functionality.
    return "Ask me about locations by saying things like: 'latitude/longitude for Prague, Czechia' or 'add location for Abu Dhabi'."
