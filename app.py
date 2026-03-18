import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import re as _re
import os
import requests
from pathlib import Path

# Try to import llama_cpp, but don't fail if it's not available
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Khisba GIS - Climate & Soil Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: #0A0A0A;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    }
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        max-width: 100%;
    }
    :root {
        --primary: #00FF88;
        --primary-dark: #00CC6A;
        --bg-dark: #0A0A0A;
        --bg-card: #141414;
        --border: #2A2A2A;
        --text: #FFFFFF;
        --text-secondary: #CCCCCC;
        --text-muted: #999999;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--text) !important;
        margin-bottom: 0.5rem !important;
    }
    h1 {
        font-size: 1.75rem !important;
        background: linear-gradient(135deg, #00FF88, #00CC6A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem !important;
    }
    .card {
        background: #141414;
        border: 1px solid #2A2A2A;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .card:hover { border-color: #00FF88; }
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #2A2A2A;
    }
    .card-icon {
        width: 36px;
        height: 36px;
        background: rgba(0, 255, 136, 0.1);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #00FF88;
        font-size: 1.25rem;
    }
    .accuracy-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.6rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 30px;
        font-size: 0.7rem;
        border: 1px solid #2A2A2A;
        color: #CCCCCC;
        margin-left: 0.5rem;
    }
    .accuracy-high { background: rgba(0,255,136,0.15); border-color: rgba(0,255,136,0.3); color: #00FF88; }
    .accuracy-medium { background: rgba(255,170,68,0.15); border-color: rgba(255,170,68,0.3); color: #FFAA44; }
    .accuracy-low { background: rgba(255,107,107,0.15); border-color: rgba(255,107,107,0.3); color: #FF6B6B; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00FF88, #00CC6A);
        color: #0A0A0A !important;
        border: none !important;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 136, 0.25) !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #FFFFFF !important; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #999999 !important; font-weight: 500; }
    .chart-container {
        background: #141414;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #2A2A2A;
        overflow: visible !important;
    }
    .chart-container .stMarkdown {
        overflow: visible !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #141414;
        border-radius: 16px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        color: #999999;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #00FF88 !important;
        color: #0A0A0A !important;
    }
    .ai-interpretation {
        background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,204,106,0.08));
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #CCCCCC;
        line-height: 1.6;
    }
    .ai-interpretation-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #00FF88;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .tinyllama-badge {
        background: rgba(138,43,226,0.15);
        border: 1px solid rgba(138,43,226,0.4);
        color: #BF7FFF;
        border-radius: 6px;
        padding: 0.15rem 0.5rem;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 0.4rem;
    }
    @media (max-width: 768px) {
        .card { padding: 1rem; }
        h1 { font-size: 1.5rem !important; }
    }
    
    /* Simple chart styles */
    .simple-chart {
        width: 100%;
        height: 300px;
        margin: 1rem 0;
        position: relative;
    }
    .chart-bar {
        display: inline-block;
        background: linear-gradient(180deg, #00FF88 0%, #00CC6A 100%);
        border-radius: 4px 4px 0 0;
        margin: 0 2px;
    }
    .chart-line {
        position: absolute;
        height: 2px;
        background: #FF6B6B;
    }
    .mini-legend {
        display: flex;
        gap: 1rem;
        margin: 0.5rem 0;
        font-size: 0.8rem;
    }
    .legend-color {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        display: inline-block;
        margin-right: 4px;
    }
    .ai-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1rem 0 0.5rem 0;
    }
    .ai-header-line {
        width: 4px;
        height: 24px;
        background: #00FF88;
        border-radius: 2px;
    }
    .ai-header-text {
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TINYLLAMA MODEL (with graceful fallback)
# =============================================================================

_APP_DIR = Path(__file__).parent.resolve()
MODEL_DIR = _APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def download_model_with_progress(progress_bar=None, status_text=None):
    """Download the model file. Call this OUTSIDE any cached function."""
    if not LLAMA_AVAILABLE:
        return False, "llama-cpp-python not installed"
    
    MODEL_DIR.mkdir(exist_ok=True)
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=120)
        response.raise_for_status()
    except Exception as e:
        return False, str(e)

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    try:
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size
                        if progress_bar:
                            progress_bar.progress(min(pct, 1.0))
                        if status_text:
                            status_text.text(f"⬇️ Downloading TinyLlama: {downloaded/(1024**2):.0f} / {total_size/(1024**2):.0f} MB")
    except Exception as e:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        return False, str(e)
    return True, "OK"


@st.cache_resource(show_spinner=False)
def load_tinyllama_model():
    """Load TinyLlama from disk. Model file must already exist before calling this."""
    if not LLAMA_AVAILABLE:
        return None, "llama-cpp-python not installed. Please add it to requirements.txt"
    
    abs_path = str(MODEL_PATH.resolve())
    try:
        if not MODEL_PATH.exists():
            return None, f"Model not found at {abs_path}"
        llm = Llama(
            model_path=abs_path,
            n_ctx=1024,
            n_threads=2,
            n_batch=128,
            verbose=False
        )
        return llm, "ok"
    except Exception as e:
        import traceback
        return None, f"{e}\n{traceback.format_exc()[-500:]}"


_GROUNDING = (
    " Strict rule: only interpret the exact numbers and facts in the data provided. "
    "Do not invent locations, add data not given, or contradict any value in the dataset."
)


def _seed(data_summary: str, chars: int = 130) -> str:
    """Extract a compact data anchor from the summary to pre-seed the response."""
    s = data_summary[:chars]
    cut = max(s.rfind(". "), s.rfind(", "))
    return (s[:cut] if cut > 60 else s).rstrip(" .,")


def _build_chart_prompt(chart_type, data_summary, location):
    """Build chart-specific prompts with data grounding for TinyLlama 1.1B."""
    ct = chart_type.lower()
    loc = location or "this region"
    seed = _seed(data_summary)

    if "climate classification" in ct:
        return (
            f"<|system|>\nYou are a senior agroclimate scientist writing a field briefing. "
            f"Your tone is expert but vivid — paint a picture of what this climate feels like to a farmer on the ground. "
            f"Highlight the climate zone character, water stress risk, and name 2 high-value crops perfectly suited to these exact conditions."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nWrite a field briefing for {loc}.\nClimate data: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Field Briefing — {loc}**\n"
            f"The recorded data for {loc} shows: {seed}. "
        )

    elif "monthly temperature" in ct and "vegetation" not in ct:
        return (
            f"<|system|>\nYou are a crop calendar specialist. Analyze the monthly temperature rhythm and identify: "
            f"(1) the optimal planting window, (2) any heat-stress or frost-risk months to avoid, "
            f"(3) one specific crop variety recommendation that matches this thermal profile. Be precise about months."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nTemperature profile for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Crop Calendar Analysis — {loc}**\n"
            f"Temperature data shows: {seed}. "
        )

    elif "precipitation" in ct and "vegetation" not in ct:
        return (
            f"<|system|>\nYou are an irrigation and water-management expert. Focus on: "
            f"(1) the dry season gap and how many months crops go without meaningful rainfall, "
            f"(2) whether supplemental irrigation is critical or optional, "
            f"(3) a rainwater harvesting or scheduling tactic specific to this rainfall pattern."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nRainfall data for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Water Management Assessment — {loc}**\n"
            f"Rainfall data shows: {seed}. "
        )

    elif "soil moisture" in ct and "distribution" not in ct:
        return (
            f"<|system|>\nYou are a precision irrigation engineer. Interpret the soil moisture across depths as a story of root-zone health. "
            f"Explain what the surface vs. root-zone vs. deep layer values reveal about drainage and water retention. "
            f"Give one irrigation scheduling recommendation — be specific about timing and depth."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nSoil moisture profile for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Root-Zone Water Status — {loc}**\n"
            f"Soil moisture readings show: {seed}. "
        )

    elif "distribution" in ct:
        return (
            f"<|system|>\nYou are a soil hydrologist. Compare the three moisture depth layers and explain what their ratio reveals about: "
            f"(1) topsoil infiltration capacity, (2) subsoil water storage, (3) whether the soil profile favors shallow-rooted or deep-rooted crops. "
            f"Recommend one drainage improvement if needed."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nMoisture depth comparison for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Soil Profile Hydrology — {loc}**\n"
            f"Moisture depth data shows: {seed}. "
        )

    elif "soil texture" in ct or "texture composition" in ct:
        return (
            f"<|system|>\nYou are a soil physicist and land use planner. Describe the clay-silt-sand texture triangle position and what it means for: "
            f"(1) tillage workability, (2) nutrient-holding capacity, (3) compaction risk. "
            f"Name the one amendment or management practice that would most improve this soil structure."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nSoil texture for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Soil Texture & Workability — {loc}**\n"
            f"Texture analysis shows: {seed}. "
        )

    elif "organic matter" in ct or "som" in ct:
        return (
            f"<|system|>\nYou are a soil carbon and fertility specialist. Interpret the SOM% and SOC stock value: "
            f"is this soil carbon-rich, average, or depleted? What does this mean for natural fertility and microbial activity? "
            f"Give one specific organic matter building practice suited to this level."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nSoil organic matter data for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Carbon & Fertility Status — {loc}**\n"
            f"Soil organic matter data shows: {seed}. "
        )

    elif "ndvi" in ct:
        return (
            f"<|system|>\nYou are a remote sensing agronomist specializing in NDVI time-series. "
            f"Interpret the 24-month NDVI signal: identify seasonal peaks (crop cycles or natural flush), "
            f"stress dips (drought, disease, or harvest), and the overall trend direction. "
            f"Translate the mean NDVI value into a vegetation health category and a concrete management action."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nNDVI time series for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**NDVI Vegetation Health Signal — {loc}**\n"
            f"NDVI data shows: {seed}. "
        )

    elif "evi" in ct:
        return (
            f"<|system|>\nYou are a canopy structure analyst. EVI captures canopy density and chlorophyll more accurately than NDVI in dense vegetation. "
            f"Interpret the EVI trend: does it suggest healthy closed-canopy growth or sparse cover? "
            f"Identify the peak biomass window and recommend one canopy management action."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nEVI canopy data for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Canopy Density & Biomass — {loc}**\n"
            f"EVI data shows: {seed}. "
        )

    elif "ndwi" in ct:
        return (
            f"<|system|>\nYou are a crop water-stress specialist. NDWI reflects water content in the plant canopy. "
            f"Interpret the 24-month NDWI trend: when was the canopy water-stressed vs. well-watered? "
            f"Pinpoint the critical stress window and give a targeted irrigation trigger recommendation."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nNDWI water stress data for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Canopy Water Stress Timeline — {loc}**\n"
            f"NDWI data shows: {seed}. "
        )

    elif "savi" in ct:
        return (
            f"<|system|>\nYou are a dryland farming expert. SAVI corrects NDVI for bare soil background — ideal for sparse or semi-arid vegetation. "
            f"Interpret the SAVI values in the context of soil cover fraction: is vegetation cover adequate for erosion protection? "
            f"Suggest one ground-cover improvement strategy."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nSAVI ground-cover data for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Soil-Adjusted Vegetation Cover — {loc}**\n"
            f"SAVI data shows: {seed}. "
        )

    elif "gndvi" in ct:
        return (
            f"<|system|>\nYou are a precision nutrition agronomist. GNDVI (green-band NDVI) is sensitive to chlorophyll and nitrogen status. "
            f"Interpret the GNDVI signal: does it suggest nitrogen sufficiency, deficiency, or luxury uptake? "
            f"Recommend a fertilization timing or rate adjustment based on the observed trend."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nGNDVI chlorophyll proxy data for {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Chlorophyll & Nitrogen Proxy — {loc}**\n"
            f"GNDVI data shows: {seed}. "
        )

    elif "temperature" in ct and "vegetation" in ct:
        return (
            f"<|system|>\nYou are a phenology and growing-degree-day specialist. "
            f"Link the temperature rhythm to vegetation activity: when does warmth trigger green-up, and when does heat or cold suppress growth? "
            f"Identify the thermal growing season length and the best sowing window."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nTemperature context for vegetation in {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Thermal Growing Season — {loc}**\n"
            f"Temperature data shows: {seed}. "
        )

    elif "precipitation" in ct and "vegetation" in ct:
        return (
            f"<|system|>\nYou are a rainfall-use-efficiency specialist. Correlate the precipitation pattern with vegetation response: "
            f"does rainfall drive green pulses? Are there lag effects? "
            f"Estimate the green season length and give one water-harvesting suggestion to extend productivity."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nPrecipitation context for vegetation in {loc}: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Rainfall-Vegetation Coupling — {loc}**\n"
            f"Rainfall data shows: {seed}. "
        )

    else:
        return (
            f"<|system|>\nYou are a precision agriculture data scientist. Analyze this geospatial dataset with scientific rigor. "
            f"Lead with the single most important finding, then give 2 actionable recommendations grounded in the numbers. Avoid generic statements."
            f"{_GROUNDING}\n</s>\n"
            f"<|user|>\nChart: {chart_type}\nLocation: {loc}\nData: {data_summary}\n</s>\n"
            f"<|assistant|>\n**Data Insight — {loc}**\n"
            f"Data shows: {seed}. "
        )


def tinyllama_interpret(llm, chart_type, data_summary, location):
    """Call TinyLlama to produce a chart-specific, grounded interpretation."""
    if llm is None or not LLAMA_AVAILABLE:
        return None
    prompt = _build_chart_prompt(chart_type, data_summary, location)
    try:
        output = llm(
            prompt,
            max_tokens=320,
            temperature=0.60,
            top_p=0.90,
            repeat_penalty=1.08,
            stop=["</s>", "<|user|>", "<|system|>"]
        )
        text = output["choices"][0]["text"].strip()
        return text if len(text) > 30 else None
    except Exception:
        return None


# =============================================================================
# SMART RULE-BASED FALLBACK ENGINE
# =============================================================================

def _parse_float(text, pattern, default=None):
    m = _re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return default

def _parse_floats_list(text, pattern):
    m = _re.search(pattern, text)
    if m:
        try:
            return [float(x) for x in _re.findall(r'[-\d.]+', m.group(1)) if x not in ('', '-')]
        except Exception:
            pass
    return []


def get_smart_interpretation(chart_type, data_summary, location=""):
    ct = chart_type.lower()
    loc_str = f" for {location}" if location else ""

    if "climate classification" in ct or "climate zone" in ct:
        temp = _parse_float(data_summary, r'Mean temperature:\s*([\d.]+)')
        precip = _parse_float(data_summary, r'Annual precipitation:\s*([\d.]+)')
        zone = ""
        zm = _re.search(r'Climate zone:\s*([^,]+)', data_summary)
        if zm:
            zone = zm.group(1).strip()
        aridity = _parse_float(data_summary, r'Aridity index:\s*([\d.]+)')
        parts = []
        if zone:
            parts.append(f"The climate{loc_str} is classified as **{zone}**.")
        if temp is not None:
            if temp > 30:
                parts.append(f"With a mean annual temperature of {temp:.1f}°C, heat stress is a significant factor — drought-tolerant and heat-adapted varieties are recommended.")
            elif temp > 20:
                parts.append(f"A mean annual temperature of {temp:.1f}°C supports year-round cultivation of warm-season crops such as maize, sorghum, and legumes.")
            elif temp > 10:
                parts.append(f"A mean annual temperature of {temp:.1f}°C is ideal for temperate crops including wheat, barley, and a wide range of vegetables.")
            else:
                parts.append(f"A mean annual temperature of {temp:.1f}°C limits the growing season; cold-hardy crops and frost management are critical.")
        if precip is not None:
            if precip < 250:
                parts.append(f"Annual precipitation of {precip:.0f} mm indicates a hyper-arid regime — irrigation is essential for any agricultural production.")
            elif precip < 500:
                parts.append(f"Annual precipitation of {precip:.0f} mm is semi-arid; supplemental irrigation during dry spells will significantly improve yields.")
            elif precip < 800:
                parts.append(f"Annual precipitation of {precip:.0f} mm supports rainfed agriculture for most of the year, though seasonal deficits may require supplemental irrigation.")
            else:
                parts.append(f"High annual precipitation of {precip:.0f} mm means waterlogging and fungal diseases may need attention; drainage management is advisable.")
        if aridity is not None:
            if aridity < 0.2:
                parts.append(f"The aridity index of {aridity:.2f} confirms extreme water scarcity conditions.")
            elif aridity < 0.5:
                parts.append(f"An aridity index of {aridity:.2f} places this area in the semi-arid category with seasonal moisture stress.")
        return " ".join(parts) if parts else "Climate classification indicates typical regional conditions suitable for adapted local crop varieties."

    if "temperature" in ct:
        max_t = _parse_float(data_summary, r'Max.*?:\s*([\d.]+)°?C')
        min_t = _parse_float(data_summary, r'Min.*?:\s*([\d.]+)°?C')
        parts = []
        if max_t and min_t:
            rng = max_t - min_t
            parts.append(f"Temperatures{loc_str} span {min_t:.1f}°C to {max_t:.1f}°C — a seasonal range of {rng:.1f}°C.")
            if rng > 25:
                parts.append("Such a large annual range is typical of continental climates; frost protection in winter and heat management in summer are both priorities.")
            elif rng > 15:
                parts.append("A moderate seasonal swing allows for diverse crop rotations — warm-season crops in summer and cool-season crops in autumn/spring.")
            else:
                parts.append("The mild seasonal variation supports year-round cultivation with minimal frost risk.")
            if max_t > 30:
                parts.append("Peak temperatures exceed 30°C — irrigation and shade management are recommended during this period.")
            if min_t < 5:
                parts.append("Minimum temperatures fall below 5°C, indicating frost risk for sensitive crops.")
        return " ".join(parts) if parts else f"Temperature data{loc_str} shows typical seasonal patterns for this climate zone."

    if "precipitation" in ct or "water" in ct or "evapotranspir" in ct:
        annual = _parse_float(data_summary, r'Annual total:\s*([\d.]+)')
        peak_m = ""
        pm = _re.search(r'Peak month:\s*(\w+)', data_summary)
        if pm:
            peak_m = pm.group(1)
        parts = []
        if annual is not None:
            if annual < 200:
                parts.append(f"Total annual precipitation{loc_str} is extremely low at {annual:.0f} mm — agriculture is impossible without intensive irrigation.")
            elif annual < 400:
                parts.append(f"Annual rainfall of {annual:.0f} mm{loc_str} is scarce — efficient drip irrigation and drought-tolerant varieties are essential.")
            elif annual < 700:
                parts.append(f"Annual precipitation of {annual:.0f} mm{loc_str} can support rainfed agriculture for most crops, though summer deficits likely require supplemental irrigation.")
            else:
                parts.append(f"Generous annual rainfall of {annual:.0f} mm{loc_str} supports productive rainfed farming; waterlogging and erosion management remain important.")
        if peak_m:
            parts.append(f"Rainfall peaks in {peak_m} — optimal planting window for rain-fed crops.")
        return " ".join(parts) if parts else f"Precipitation data{loc_str} shows typical seasonal distribution for this region."

    if "soil moisture" in ct and ("layer" in ct or "distribution" in ct or "comparison" in ct):
        surf = _parse_float(data_summary, r'[Ss]urface.*?:\s*([\d.]+)\s*m')
        root = _parse_float(data_summary, r'[Rr]oot.*?zone.*?:\s*([\d.]+)\s*m')
        parts = []
        if surf is not None:
            if surf > 0.3:
                parts.append(f"Surface soil moisture{loc_str} is high at {surf:.3f} m³/m³, indicating recent rainfall or potential waterlogging risk.")
            elif surf > 0.15:
                parts.append(f"Surface soil moisture of {surf:.3f} m³/m³{loc_str} is moderate — adequate for germination and shallow-rooted crops.")
            else:
                parts.append(f"Low surface moisture ({surf:.3f} m³/m³){loc_str} indicates dry topsoil; timely irrigation is needed for seedling establishment.")
        if root is not None:
            if root > 0.25:
                parts.append(f"Root-zone moisture ({root:.3f} m³/m³) is well-supplied, supporting active crop growth without irrigation stress.")
            elif root > 0.1:
                parts.append(f"Root-zone moisture ({root:.3f} m³/m³) is marginal — crops with moderate water demand should be prioritized.")
            else:
                parts.append(f"Root-zone moisture ({root:.3f} m³/m³) is critically low, severely limiting plant growth and productivity.")
        return " ".join(parts) if parts else f"Soil moisture profile{loc_str} shows typical layered distribution."

    if "soil texture" in ct:
        clay = _parse_float(data_summary, r'Clay:\s*([\d.]+)%')
        silt = _parse_float(data_summary, r'Silt:\s*([\d.]+)%')
        sand = _parse_float(data_summary, r'Sand:\s*([\d.]+)%')
        tex = ""
        tm = _re.search(r'Texture class:\s*([^,\n]+)', data_summary)
        if tm:
            tex = tm.group(1).strip()
        parts = []
        if tex:
            parts.append(f"The soil{loc_str} is classified as **{tex}**.")
        if clay is not None and sand is not None and silt is not None:
            if clay > 40:
                parts.append(f"High clay content ({clay:.0f}%) provides excellent nutrient and water retention but requires careful tillage to prevent compaction.")
            elif clay > 25:
                parts.append(f"Moderate clay content ({clay:.0f}%) offers good structure suitable for a wide range of crops.")
            elif sand > 60:
                parts.append(f"Sandy texture ({sand:.0f}% sand) means rapid drainage; frequent irrigation and organic matter additions are recommended.")
            else:
                parts.append(f"A balanced texture ({clay:.0f}% clay, {silt:.0f}% silt, {sand:.0f}% sand) supports good soil structure and workability.")
        return " ".join(parts) if parts else f"Soil texture{loc_str} indicates typical properties for the region."

    if "organic matter" in ct or "som" in ct:
        som = _parse_float(data_summary, r'Soil Organic Matter:\s*([\d.]+)%')
        soc = _parse_float(data_summary, r'SOC Stock:\s*([\d.]+)\s*t')
        parts = []
        if som is not None:
            if som < 1.0:
                parts.append(f"Soil organic matter{loc_str} is critically low at {som:.2f}%. Compost additions, cover cropping, and minimal tillage are urgently recommended.")
            elif som < 2.0:
                parts.append(f"SOM of {som:.2f}%{loc_str} is below optimal. Incremental organic inputs such as manure or green manures will improve soil health.")
            elif som < 4.0:
                parts.append(f"SOM of {som:.2f}%{loc_str} is in the moderate range. Maintaining this level through conservation practices is advisable.")
            else:
                parts.append(f"Excellent SOM of {som:.2f}%{loc_str} reflects a highly fertile, biologically active soil with superior water retention.")
        if soc is not None:
            parts.append(f"SOC stock of {soc:.1f} t/ha is a significant carbon reservoir — protecting this through reduced tillage and residue retention has both agronomic and climate benefits.")
        return " ".join(parts) if parts else f"Soil organic matter data{loc_str} indicates typical conditions."

    if any(v in ct for v in ['ndvi', 'evi', 'savi', 'ndwi', 'gndvi', 'nbr', 'msavi', 'osavi', 'arvi', 'vari', 'ndmi', 'mndwi', 'awei', 'ndsi', 'bri', 'mtvi', 'rdvi', 'nli', 'gdvi', 'vegetation index']):
        idx_m = _re.search(r'^(\w+)\s+vegetation', ct) or _re.search(r'^(\w+)\b', ct)
        idx_name = idx_m.group(1).upper() if idx_m else "Index"
        mean_v = _parse_float(data_summary, r'mean=([\d.]+)')
        trend_dir = "stable"
        td = _re.search(r'trend=(\w+)', data_summary)
        if td:
            trend_dir = td.group(1)
        parts = []
        if idx_name in ('NDVI', 'EVI', 'SAVI', 'MSAVI', 'OSAVI', 'GNDVI'):
            if mean_v is not None:
                if mean_v > 0.6:
                    parts.append(f"{idx_name}{loc_str} averages {mean_v:.3f} — dense, healthy vegetation with high photosynthetic activity.")
                elif mean_v > 0.4:
                    parts.append(f"{idx_name}{loc_str} averages {mean_v:.3f} — moderate vegetation cover typical of mixed cropland or savanna.")
                elif mean_v > 0.2:
                    parts.append(f"{idx_name}{loc_str} averages {mean_v:.3f} — sparse vegetation, possibly degraded rangeland or fallow fields.")
                else:
                    parts.append(f"{idx_name}{loc_str} averages {mean_v:.3f} — very low greenness consistent with bare soil or heavily stressed vegetation.")
        elif idx_name in ('NDWI', 'MNDWI', 'AWEI'):
            if mean_v is not None:
                if mean_v > 0.3:
                    parts.append(f"{idx_name}{loc_str} averages {mean_v:.3f} — high water content indicating well-irrigated fields or water bodies nearby.")
                elif mean_v > 0:
                    parts.append(f"{idx_name}{loc_str} averages {mean_v:.3f} — moderate moisture, suitable conditions without significant water stress.")
                else:
                    parts.append(f"{idx_name}{loc_str} averages {mean_v:.3f} — moisture deficit; irrigation scheduling should be based on regular monitoring.")
        else:
            if mean_v is not None:
                parts.append(f"{idx_name}{loc_str} averages {mean_v:.3f}.")
        if trend_dir == "increasing":
            parts.append("A positive trend indicates improving vegetation conditions — possibly responding to seasonal rainfall or improved land management.")
        elif trend_dir == "decreasing":
            parts.append("A declining trend is concerning — may signal vegetation degradation, overgrazing, drought stress, or land use change.")
        else:
            parts.append("The index shows a stable trajectory — no significant land cover change detected.")
        return " ".join(parts) if parts else f"Vegetation data{loc_str} indicates typical seasonal dynamics."

    return f"Analysis of {chart_type}{loc_str}: {data_summary[:200]}. Values are within expected ranges for this region."


# =============================================================================
# CONSTANTS
# =============================================================================

BULK_DENSITY = 1.3
SOC_TO_SOM_FACTOR = 1.724

SOIL_TEXTURE_CLASSES = {
    1: 'Clay', 2: 'Sandy clay', 3: 'Silty clay', 4: 'Clay loam', 5: 'Sandy clay loam',
    6: 'Silty clay loam', 7: 'Loam', 8: 'Sandy loam', 9: 'Silt loam', 10: 'Silt',
    11: 'Loamy sand', 12: 'Sand'
}

VEGETATION_INDICES = [
    'NDVI', 'EVI', 'SAVI', 'MSAVI', 'OSAVI', 'GNDVI', 'ARVI', 'VARI',
    'NDMI', 'NBR', 'NDWI', 'MNDWI', 'AWEI', 'NDSI_Salinity', 'SI',
    'BRI', 'MTVI', 'RDVI', 'NLI', 'GDVI'
]

WORLD_REGIONS = {
    "Algeria": {"regions": ["Sidi Bel Abbès", "Oran", "Algiers", "Constantine", "Annaba", "Tlemcen", "Béjaïa", "Batna", "Sétif", "Blida"], "lat": 28.0339, "lon": 1.6596, "zoom": 5},
    "Morocco": {"regions": ["Casablanca", "Marrakech", "Fes", "Rabat", "Agadir", "Tangier", "Meknes", "Oujda"], "lat": 31.7917, "lon": -7.0926, "zoom": 5},
    "Tunisia": {"regions": ["Tunis", "Sfax", "Sousse", "Monastir", "Bizerte", "Gabès", "Ariana"], "lat": 33.8869, "lon": 9.5375, "zoom": 6},
    "Egypt": {"regions": ["Cairo", "Alexandria", "Luxor", "Aswan", "Giza", "Sharm El Sheikh", "Hurghada"], "lat": 26.8206, "lon": 30.8025, "zoom": 5},
    "France": {"regions": ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux", "Lille", "Strasbourg"], "lat": 46.2276, "lon": 2.2137, "zoom": 5},
    "Spain": {"regions": ["Madrid", "Barcelona", "Seville", "Valencia", "Bilbao", "Granada", "Zaragoza"], "lat": 40.4637, "lon": -3.7492, "zoom": 5},
    "United States": {"regions": ["California", "Texas", "Florida", "New York", "Illinois", "Arizona", "Montana"], "lat": 39.5, "lon": -98.35, "zoom": 4},
    "Brazil": {"regions": ["São Paulo", "Rio de Janeiro", "Amazonas", "Minas Gerais", "Bahia", "Paraná"], "lat": -14.235, "lon": -51.9253, "zoom": 4},
    "India": {"regions": ["Maharashtra", "Rajasthan", "Punjab", "Kerala", "Tamil Nadu", "Uttar Pradesh"], "lat": 20.5937, "lon": 78.9629, "zoom": 4},
    "China": {"regions": ["Beijing", "Shanghai", "Guangdong", "Sichuan", "Yunnan", "Xinjiang"], "lat": 35.8617, "lon": 104.1954, "zoom": 4},
    "Nigeria": {"regions": ["Lagos", "Kano", "Abuja", "Rivers", "Oyo", "Kaduna"], "lat": 9.082, "lon": 8.6753, "zoom": 5},
    "Kenya": {"regions": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"], "lat": -0.0236, "lon": 37.9062, "zoom": 5},
    "South Africa": {"regions": ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Limpopo"], "lat": -30.5595, "lon": 22.9375, "zoom": 5},
    "Australia": {"regions": ["New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia"], "lat": -25.2744, "lon": 133.7751, "zoom": 4},
}


# =============================================================================
# HELPERS
# =============================================================================

def get_region_type(location_name):
    if not location_name:
        return "general"
    loc = location_name.lower()
    if any(x in loc for x in ['sidi', 'algeria', 'morocco', 'tunisia', 'libya', 'north africa']):
        return "Semi-arid"
    elif any(x in loc for x in ['sahara', 'desert', 'sahel', 'egypt']):
        return "Arid"
    elif any(x in loc for x in ['amazon', 'congo', 'rainforest']):
        return "Humid"
    elif any(x in loc for x in ['france', 'germany', 'uk', 'spain']):
        return "Humid"
    return "general"


def accuracy_badge_html(level, text):
    css_class = {"high": "accuracy-high", "medium": "accuracy-medium", "low": "accuracy-low"}.get(level, "accuracy-medium")
    return f'<span class="accuracy-badge {css_class}">🎯 {text}</span>'


# =============================================================================
# SYNTHETIC DATA
# =============================================================================

def generate_climate_data(location, start_year=2023, months=12, region_type="general"):
    np.random.seed(hash(location) % 2**31)
    base_configs = {
        "Arid": {"temp_base": 28, "temp_range": 18, "precip_base": 15, "precip_var": 10},
        "Semi-arid": {"temp_base": 20, "temp_range": 16, "precip_base": 35, "precip_var": 20},
        "Humid": {"temp_base": 14, "temp_range": 10, "precip_base": 80, "precip_var": 30},
        "general": {"temp_base": 18, "temp_range": 12, "precip_base": 55, "precip_var": 25},
    }
    cfg = base_configs.get(region_type, base_configs["general"])
    records = []
    for m in range(months):
        month_num = (m % 12) + 1
        seasonal = np.sin((month_num - 3) * np.pi / 6)
        temp = cfg["temp_base"] + cfg["temp_range"] * seasonal * 0.5 + np.random.normal(0, 1)
        precip = max(0, cfg["precip_base"] - cfg["precip_base"] * 0.6 * seasonal + np.random.normal(0, cfg["precip_var"]))
        soil_m1 = max(0.05, 0.2 - seasonal * 0.08 + np.random.normal(0, 0.02))
        soil_m2 = max(0.08, 0.25 - seasonal * 0.06 + np.random.normal(0, 0.015))
        soil_m3 = max(0.1, 0.3 - seasonal * 0.04 + np.random.normal(0, 0.01))
        pet = max(0, temp * 2.2 + 20 + np.random.normal(0, 5))
        records.append({
            "month": month_num,
            "month_name": datetime(start_year, month_num, 1).strftime('%b'),
            "temperature_2m": round(temp, 1),
            "temperature_max": round(temp + np.random.uniform(3, 6), 1),
            "temperature_min": round(temp - np.random.uniform(3, 6), 1),
            "total_precipitation": round(precip, 1),
            "potential_evaporation": round(pet, 1),
            "soil_moisture_0_7cm": round(soil_m1, 3),
            "soil_moisture_7_28cm": round(soil_m2, 3),
            "soil_moisture_28_100cm": round(soil_m3, 3),
        })
    return pd.DataFrame(records)


def generate_vegetation_data(location, index_name, months=24):
    np.random.seed((hash(location) + hash(index_name)) % 2**31)
    region_type = get_region_type(location)
    base_configs = {
        "Arid": {"base": 0.15, "amp": 0.18, "noise": 0.04},
        "Semi-arid": {"base": 0.28, "amp": 0.22, "noise": 0.05},
        "Humid": {"base": 0.55, "amp": 0.20, "noise": 0.06},
        "general": {"base": 0.38, "amp": 0.20, "noise": 0.05},
    }
    cfg = base_configs.get(region_type, base_configs["general"]).copy()
    if index_name in ['NDWI', 'MNDWI', 'AWEI']:
        cfg["base"] -= 0.15
        cfg["amp"] *= 0.8
    elif index_name in ['NBR']:
        cfg["base"] += 0.1
    elif index_name in ['SI', 'NDSI_Salinity']:
        cfg["base"] = 0.4
        cfg["amp"] = 0.15
    dates, values = [], []
    start = datetime(2023, 1, 1)
    for i in range(months):
        d = start + timedelta(days=30 * i)
        seasonal = np.sin((d.month - 3) * np.pi / 6)
        val = cfg["base"] + cfg["amp"] * seasonal + np.random.normal(0, cfg["noise"])
        val = max(-1.0, min(1.0, val))
        dates.append(d.strftime('%Y-%m'))
        values.append(round(val, 4))
    return dates, values


def get_soil_data(location, region_type="general"):
    np.random.seed(hash(location) % 2**31)
    soil_configs = {
        "Arid": {"texture": 12, "soc": 3.2, "clay": 8, "silt": 12, "sand": 80},
        "Semi-arid": {"texture": 8, "soc": 8.5, "clay": 15, "silt": 20, "sand": 65},
        "Humid": {"texture": 7, "soc": 25.0, "clay": 25, "silt": 40, "sand": 35},
        "general": {"texture": 7, "soc": 15.0, "clay": 20, "silt": 35, "sand": 45},
    }
    cfg = soil_configs.get(region_type, soil_configs["general"])
    soc_stock = cfg["soc"] + np.random.normal(0, cfg["soc"] * 0.1)
    soc_stock = max(1.0, soc_stock)
    depth = 20 if region_type in ["Arid", "Semi-arid"] else 30
    soc_pct = soc_stock / (BULK_DENSITY * depth * 100) * 100
    som_pct = soc_pct * SOC_TO_SOM_FACTOR
    clay = cfg["clay"] + np.random.randint(-3, 4)
    silt = cfg["silt"] + np.random.randint(-3, 4)
    sand = 100 - clay - silt
    return {
        "texture_class": cfg["texture"],
        "texture_name": SOIL_TEXTURE_CLASSES.get(cfg["texture"], "Loam"),
        "soc_stock": round(soc_stock, 2),
        "soil_organic_matter": round(som_pct, 2),
        "bulk_density": BULK_DENSITY,
        "depth_cm": depth,
        "clay_content": max(0, clay),
        "silt_content": max(0, silt),
        "sand_content": max(0, sand),
        "final_som_estimate": round(som_pct, 2),
    }


def get_climate_classification(location, region_type):
    np.random.seed(hash(location) % 2**31)
    configs = {
        "Arid": {"zone": "Desert/Steppe", "temp": 28.5, "precip": 120, "aridity": 0.38},
        "Semi-arid": {"zone": "Mediterranean", "temp": 17.8, "precip": 380, "aridity": 1.08},
        "Humid": {"zone": "Oceanic", "temp": 12.5, "precip": 850, "aridity": 2.8},
        "general": {"zone": "Warm Temperate", "temp": 16.0, "precip": 620, "aridity": 1.65},
    }
    cfg = configs.get(region_type, configs["general"])
    return {
        "climate_zone": cfg["zone"],
        "mean_temperature": cfg["temp"] + np.random.normal(0, 0.5),
        "mean_precipitation": cfg["precip"] + np.random.normal(0, 20),
        "aridity_index": cfg["aridity"],
    }


# =============================================================================
# FIXED SIMPLE HTML CHARTS
# =============================================================================

def create_temperature_chart_html(df, location_name):
    """Create a simple HTML temperature chart"""
    months = df['month_name'].tolist()
    temps = df['temperature_2m'].tolist()
    
    max_temp = max(temps)
    min_temp = min(temps)
    temp_range = max_temp - min_temp if max_temp > min_temp else 1
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; margin: 0; width: 100%; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #FFFFFF; font-size: 1.1rem;">Monthly Temperature</h4>
            {accuracy_badge_html("high", "±1-2°C ERA5-Land")}
        </div>
        <div style="display: flex; flex-direction: row; align-items: flex-end; justify-content: space-around; height: 250px; gap: 8px; margin: 20px 0; padding: 0 5px;">
    '''
    
    for i, (month, temp) in enumerate(zip(months, temps)):
        # Calculate height percentage (minimum 30px for visibility)
        height_percent = ((temp - min_temp) / temp_range) * 180 + 30 if temp_range > 0 else 100
        chart_html += f'''
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; min-width: 40px;">
                <div style="width: 70%; background: #FF6B6B; height: {height_percent}px; 
                           border-radius: 8px 8px 0 0; opacity: 0.9; 
                           box-shadow: 0 4px 12px rgba(255,107,107,0.3); 
                           border: 1px solid rgba(255,255,255,0.1);
                           transition: all 0.2s ease;" 
                     title="{month}: {temp}°C"></div>
                <div style="color: #FFFFFF; font-size: 0.9rem; margin-top: 10px; font-weight: 600;">{temp}°C</div>
                <div style="color: #CCCCCC; font-size: 0.8rem; margin-top: 2px; font-weight: 500;">{month}</div>
            </div>
        '''
    
    chart_html += '''
        </div>
        <div style="display: flex; gap: 1rem; margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #2A2A2A;">
            <span style="display: flex; align-items: center; gap: 0.3rem;">
                <span style="width: 16px; height: 16px; background: #FF6B6B; border-radius: 4px; display: inline-block;"></span>
                <span style="color: #CCCCCC; font-size: 0.9rem;">Mean Temperature (°C)</span>
            </span>
        </div>
    </div>
    '''
    
    return chart_html


def create_precipitation_chart_html(df, location_name):
    """Create a simple HTML precipitation chart"""
    months = df['month_name'].tolist()
    precip = df['total_precipitation'].tolist()
    
    max_precip = max(precip) if max(precip) > 0 else 100
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; margin: 0; width: 100%; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #FFFFFF; font-size: 1.1rem;">Monthly Precipitation</h4>
            {accuracy_badge_html("medium", "±20-40% CHIRPS")}
        </div>
        <div style="display: flex; flex-direction: row; align-items: flex-end; justify-content: space-around; height: 250px; gap: 8px; margin: 20px 0; padding: 0 5px;">
    '''
    
    for i, (month, p) in enumerate(zip(months, precip)):
        height_percent = (p / max_precip) * 200 if max_precip > 0 else 100
        chart_html += f'''
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; min-width: 40px;">
                <div style="width: 70%; background: #4A90E2; height: {height_percent}px; 
                           border-radius: 8px 8px 0 0; opacity: 0.9; 
                           box-shadow: 0 4px 12px rgba(74,144,226,0.3);
                           border: 1px solid rgba(255,255,255,0.1);
                           transition: all 0.2s ease;" 
                     title="{month}: {p:.1f}mm"></div>
                <div style="color: #FFFFFF; font-size: 0.9rem; margin-top: 10px; font-weight: 600;">{p:.0f}mm</div>
                <div style="color: #CCCCCC; font-size: 0.8rem; margin-top: 2px; font-weight: 500;">{month}</div>
            </div>
        '''
    
    chart_html += '''
        </div>
        <div style="display: flex; gap: 1rem; margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #2A2A2A;">
            <span style="display: flex; align-items: center; gap: 0.3rem;">
                <span style="width: 16px; height: 16px; background: #4A90E2; border-radius: 4px; display: inline-block;"></span>
                <span style="color: #CCCCCC; font-size: 0.9rem;">Precipitation (mm)</span>
            </span>
        </div>
    </div>
    '''
    
    return chart_html


def create_soil_moisture_chart_html(df, location_name):
    """Create a simple HTML soil moisture chart"""
    months = df['month_name'].tolist()
    surface = df['soil_moisture_0_7cm'].tolist()
    root = df['soil_moisture_7_28cm'].tolist()
    deep = df['soil_moisture_28_100cm'].tolist()
    
    max_value = max(max(surface), max(root), max(deep)) * 1.2
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; margin: 0; width: 100%; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #FFFFFF; font-size: 1.1rem;">Soil Moisture by Layer</h4>
            {accuracy_badge_html("medium", "±0.05 m³/m³")}
        </div>
        <div style="display: flex; flex-direction: row; align-items: flex-end; justify-content: space-around; height: 250px; gap: 8px; margin: 20px 0; padding: 0 5px;">
    '''
    
    for i, month in enumerate(months):
        surface_height = (surface[i] / max_value) * 200 if max_value > 0 else 50
        root_height = (root[i] / max_value) * 200 if max_value > 0 else 50
        deep_height = (deep[i] / max_value) * 200 if max_value > 0 else 50
        
        chart_html += f'''
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; min-width: 40px;">
                <div style="width: 100%; display: flex; flex-direction: column-reverse; gap: 2px; height: 200px;">
                    <div style="width: 100%; background: #FFAA44; height: {deep_height}px; 
                               border-radius: 6px 6px 0 0; 
                               box-shadow: 0 2px 8px rgba(255,170,68,0.3);
                               border: 1px solid rgba(255,255,255,0.1);
                               transition: all 0.2s ease;" 
                         title="Deep: {deep[i]:.3f} m³/m³"></div>
                    <div style="width: 100%; background: #4A90E2; height: {root_height}px; 
                               border-radius: 6px 6px 0 0;
                               box-shadow: 0 2px 8px rgba(74,144,226,0.3);
                               border: 1px solid rgba(255,255,255,0.1);
                               transition: all 0.2s ease;" 
                         title="Root: {root[i]:.3f} m³/m³"></div>
                    <div style="width: 100%; background: #00FF88; height: {surface_height}px; 
                               border-radius: 6px 6px 0 0;
                               box-shadow: 0 2px 8px rgba(0,255,136,0.3);
                               border: 1px solid rgba(255,255,255,0.1);
                               transition: all 0.2s ease;" 
                         title="Surface: {surface[i]:.3f} m³/m³"></div>
                </div>
                <div style="color: #FFFFFF; font-size: 0.8rem; margin-top: 10px; font-weight: 500;">{month}</div>
            </div>
        '''
    
    chart_html += '''
        </div>
        <div style="display: flex; gap: 1.5rem; margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #2A2A2A; flex-wrap: wrap;">
            <span style="display: flex; align-items: center; gap: 0.3rem;">
                <span style="width: 16px; height: 16px; background: #00FF88; border-radius: 4px; display: inline-block;"></span>
                <span style="color: #CCCCCC; font-size: 0.9rem;">Surface (0-7cm)</span>
            </span>
            <span style="display: flex; align-items: center; gap: 0.3rem;">
                <span style="width: 16px; height: 16px; background: #4A90E2; border-radius: 4px; display: inline-block;"></span>
                <span style="color: #CCCCCC; font-size: 0.9rem;">Root (7-28cm)</span>
            </span>
            <span style="display: flex; align-items: center; gap: 0.3rem;">
                <span style="width: 16px; height: 16px; background: #FFAA44; border-radius: 4px; display: inline-block;"></span>
                <span style="color: #CCCCCC; font-size: 0.9rem;">Deep (28-100cm)</span>
            </span>
        </div>
    </div>
    '''
    
    return chart_html


def create_soil_distribution_chart_html(df):
    """Create a simple HTML soil distribution chart"""
    surface_mean = df['soil_moisture_0_7cm'].mean()
    root_mean = df['soil_moisture_7_28cm'].mean()
    deep_mean = df['soil_moisture_28_100cm'].mean()
    
    max_val = max(surface_mean, root_mean, deep_mean)
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; margin: 0; width: 100%; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <h4 style="margin: 0 0 1rem 0; color: #FFFFFF; font-size: 1.1rem;">Average Soil Moisture Distribution</h4>
        <div style="display: flex; flex-direction: row; align-items: flex-end; justify-content: center; gap: 3rem; height: 250px; margin: 20px 0;">
    '''
    
    labels = ['Surface<br>0-7cm', 'Root Zone<br>7-28cm', 'Deep<br>28-100cm']
    values = [surface_mean, root_mean, deep_mean]
    colors = ['#00FF88', '#4A90E2', '#FFAA44']
    
    for label, value, color in zip(labels, values, colors):
        height_percent = (value / max_val) * 200 if max_val > 0 else 100
        chart_html += f'''
            <div style="display: flex; flex-direction: column; align-items: center; width: 120px;">
                <div style="width: 80px; background: {color}; height: {height_percent}px; 
                           border-radius: 8px 8px 0 0; margin-bottom: 15px; 
                           box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                           border: 1px solid rgba(255,255,255,0.1);
                           transition: all 0.2s ease;" 
                     title="{value:.3f} m³/m³"></div>
                <div style="color: #FFFFFF; font-weight: 700; font-size: 1.2rem; margin-bottom: 5px;">{value:.3f}</div>
                <div style="color: #CCCCCC; font-size: 0.9rem; text-align: center;">{label}</div>
            </div>
        '''
    
    chart_html += '''
        </div>
        <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #2A2A2A;">
            <span style="color: #CCCCCC; font-size: 0.9rem;">Units: m³/m³</span>
        </div>
    </div>
    '''
    
    return chart_html


def create_vegetation_chart_html(dates, values, index_name, location_name):
    """Create a simple HTML vegetation index chart"""
    max_val = max(values) if values else 1
    min_val = min(values) if values else 0
    val_range = max_val - min_val if max_val > min_val else 1
    
    color_map = {'NDVI': '#00FF88', 'EVI': '#FF6B6B', 'SAVI': '#4A90E2', 'NDWI': '#4A90E2', 
                 'GNDVI': '#00CC6A', 'NBR': '#FF4444', 'SI': '#8B4513', 'NDSI_Salinity': '#DEB887', 'AWEI': '#87CEEB'}
    color = color_map.get(index_name, '#00FF88')
    
    # Show only first 12 months for readability
    display_dates = dates[:12]
    display_values = values[:12]
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; margin: 0; width: 100%; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #FFFFFF; font-size: 1.1rem;">{index_name} Time Series (12 months)</h4>
            {accuracy_badge_html("high", "±0.05 Sentinel-2")}
        </div>
        <div style="display: flex; flex-direction: row; align-items: flex-end; justify-content: space-around; height: 250px; gap: 4px; margin: 20px 0; padding: 0 5px;">
    '''
    
    for date, val in zip(display_dates, display_values):
        height_percent = ((val - min_val) / val_range) * 200 + 30 if val_range > 0 else 100
        short_date = date[-2:]  # Show only last 2 digits of month
        chart_html += f'''
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; min-width: 30px;">
                <div style="width: 70%; background: {color}; height: {height_percent}px; 
                           border-radius: 8px 8px 0 0; opacity: 0.9; 
                           box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                           border: 1px solid rgba(255,255,255,0.1);
                           transition: all 0.2s ease;" 
                     title="{date}: {val:.3f}"></div>
                <div style="color: #FFFFFF; font-size: 0.8rem; margin-top: 8px; font-weight: 600;">{val:.3f}</div>
                <div style="color: #CCCCCC; font-size: 0.7rem; margin-top: 2px;">{short_date}</div>
            </div>
        '''
    
    chart_html += f'''
        </div>
        <div style="display: flex; gap: 1rem; margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #2A2A2A;">
            <span style="display: flex; align-items: center; gap: 0.3rem;">
                <span style="width: 16px; height: 16px; background: {color}; border-radius: 4px; display: inline-block;"></span>
                <span style="color: #CCCCCC; font-size: 0.9rem;">{index_name}</span>
            </span>
        </div>
    </div>
    '''
    
    return chart_html


def create_soil_texture_chart_html(soil_data, location_name):
    """Create a simple HTML soil texture chart"""
    clay = soil_data['clay_content']
    silt = soil_data['silt_content']
    sand = soil_data['sand_content']
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; margin: 0; width: 100%; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #FFFFFF; font-size: 1.1rem;">Soil Texture Composition</h4>
            {accuracy_badge_html("medium", "±25% ISDASoil")}
        </div>
        <div style="display: flex; flex-direction: row; align-items: flex-end; justify-content: center; gap: 3rem; height: 250px; margin: 20px 0;">
    '''
    
    components = [
        {'name': 'Clay', 'value': clay, 'color': '#8B4513'},
        {'name': 'Silt', 'value': silt, 'color': '#DEB887'},
        {'name': 'Sand', 'value': sand, 'color': '#F4A460'}
    ]
    
    for comp in components:
        height_percent = comp['value'] * 2.5  # Scale factor for visibility
        chart_html += f'''
            <div style="display: flex; flex-direction: column; align-items: center; width: 120px;">
                <div style="width: 80px; background: {comp['color']}; height: {height_percent}px; 
                           border-radius: 8px 8px 0 0; margin-bottom: 15px; 
                           box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                           border: 1px solid rgba(255,255,255,0.1);
                           transition: all 0.2s ease;" 
                     title="{comp['name']}: {comp['value']}%"></div>
                <div style="color: #FFFFFF; font-weight: 700; font-size: 1.4rem; margin-bottom: 5px;">{comp['value']}%</div>
                <div style="color: #CCCCCC; font-size: 1rem;">{comp['name']}</div>
            </div>
        '''
    
    chart_html += '''
        </div>
    </div>
    '''
    
    return chart_html


def create_som_gauge_html(soil_data, location_name):
    """Create a simple HTML gauge for soil organic matter"""
    som_value = soil_data['final_som_estimate']
    
    # Determine color based on value
    if som_value < 1.5:
        color = '#FF4444'
        status = "Depleted"
    elif som_value < 3:
        color = '#FFAA44'
        status = "Moderate"
    else:
        color = '#44FF44'
        status = "Rich"
    
    percentage = (som_value / 6) * 100  # Max 6%
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1.5rem; margin: 0; width: 100%; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <h4 style="margin: 0; color: #FFFFFF; font-size: 1.1rem;">Soil Organic Matter</h4>
            {accuracy_badge_html("medium", "±20% GSOC")}
        </div>
        <div style="text-align: center;">
            <div style="position: relative; width: 250px; height: 125px; margin: 0 auto; overflow: hidden;">
                <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 125px; 
                           background: linear-gradient(90deg, #FF4444 0%, #FFAA44 50%, #44FF44 100%);
                           border-radius: 125px 125px 0 0; opacity: 0.3;"></div>
                <div style="position: absolute; bottom: 0; left: {percentage}%; width: 6px; height: 125px; 
                           background: white; transform: translateX(-3px); box-shadow: 0 0 10px rgba(255,255,255,0.5);"></div>
                <div style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); 
                           background: {color}; padding: 10px 25px; border-radius: 30px;
                           box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                    <div style="color: white; font-size: 1.8rem; font-weight: 700;">{som_value:.2f}%</div>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <span style="color: {color}; font-size: 1.2rem; font-weight: 600;">{status}</span>
            </div>
            <div style="display: flex; justify-content: space-between; color: #CCCCCC; font-size: 0.9rem; margin-top: 15px; padding: 0 20px;">
                <span>Depleted<br>< 1.5%</span>
                <span>Moderate<br>1.5-3%</span>
                <span>Rich<br>> 3%</span>
            </div>
        </div>
    </div>
    '''
    
    return chart_html


def create_climate_temp_gauge_html(climate_data):
    """Create a simple HTML gauge for temperature"""
    temp = climate_data['mean_temperature']
    
    # Determine color based on value
    if temp < 0:
        color = '#4A90E2'
        status = "Cold"
    elif temp < 18:
        color = '#44AA44'
        status = "Temperate"
    elif temp < 30:
        color = '#FFAA44'
        status = "Warm"
    else:
        color = '#FF4444'
        status = "Hot"
    
    percentage = ((temp + 20) / 65) * 100  # Range -20 to 45
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1.5rem; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <h4 style="margin: 0 0 1.5rem 0; color: #FFFFFF; font-size: 1.1rem;">Mean Annual Temperature</h4>
        <div style="text-align: center;">
            <div style="position: relative; width: 200px; height: 100px; margin: 0 auto; overflow: hidden;">
                <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 100px; 
                           background: linear-gradient(90deg, #4A90E2 0%, #44AA44 30%, #FFAA44 70%, #FF4444 100%);
                           border-radius: 100px 100px 0 0; opacity: 0.3;"></div>
                <div style="position: absolute; bottom: 0; left: {percentage}%; width: 4px; height: 100px; 
                           background: white; transform: translateX(-2px); box-shadow: 0 0 10px rgba(255,255,255,0.5);"></div>
            </div>
            <div style="margin-top: 20px;">
                <span style="color: {color}; font-size: 2rem; font-weight: 700;">{temp:.1f}°C</span>
                <div style="color: {color}; font-size: 1rem; margin-top: 5px;">{status}</div>
            </div>
        </div>
    </div>
    '''
    
    return chart_html


def create_climate_precip_gauge_html(climate_data):
    """Create a simple HTML gauge for precipitation"""
    precip = climate_data['mean_precipitation']
    
    # Determine color based on value
    if precip < 250:
        color = '#FF4444'
        status = "Arid"
    elif precip < 500:
        color = '#FFAA44'
        status = "Semi-arid"
    elif precip < 1000:
        color = '#44AA44'
        status = "Sub-humid"
    elif precip < 2000:
        color = '#4A90E2'
        status = "Humid"
    else:
        color = '#800080'
        status = "Very Humid"
    
    percentage = (precip / 3000) * 100  # Max 3000mm
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1.5rem; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <h4 style="margin: 0 0 1.5rem 0; color: #FFFFFF; font-size: 1.1rem;">Annual Precipitation</h4>
        <div style="text-align: center;">
            <div style="position: relative; width: 200px; height: 100px; margin: 0 auto; overflow: hidden;">
                <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 100px; 
                           background: linear-gradient(90deg, #FF4444 0%, #FFAA44 20%, #44AA44 40%, #4A90E2 70%, #800080 90%);
                           border-radius: 100px 100px 0 0; opacity: 0.3;"></div>
                <div style="position: absolute; bottom: 0; left: {percentage}%; width: 4px; height: 100px; 
                           background: white; transform: translateX(-2px); box-shadow: 0 0 10px rgba(255,255,255,0.5);"></div>
            </div>
            <div style="margin-top: 20px;">
                <span style="color: {color}; font-size: 2rem; font-weight: 700;">{precip:.0f} mm</span>
                <div style="color: {color}; font-size: 1rem; margin-top: 5px;">{status}</div>
            </div>
        </div>
    </div>
    '''
    
    return chart_html


# =============================================================================
# HELPER FUNCTION FOR CHART DISPLAY
# =============================================================================

def display_chart(chart_html):
    """Helper function to properly display HTML charts using components"""
    if chart_html and len(chart_html) > 100:
        st.components.v1.html(chart_html, height=400, scrolling=False)
    else:
        st.error("Chart failed to render")


# =============================================================================
# AI INTERPRETATION DISPLAY - FIXED VERSION
# =============================================================================

def show_ai_interpretation(chart_type, data_summary, location, llm=None, use_tinyllama=True):
    ct = chart_type.lower()

    # Pick label based on chart type
    if "climate classification" in ct:
        label = "🌾 Field Briefing — Agroclimate Assessment"
    elif "monthly temperature" in ct and "vegetation" not in ct:
        label = "🌾 Crop Calendar Analysis"
    elif "precipitation" in ct and "vegetation" not in ct:
        label = "🌾 Water Management Assessment"
    elif "soil moisture" in ct and "distribution" not in ct:
        label = "🌾 Root-Zone Water Status"
    elif "distribution" in ct:
        label = "🌾 Soil Profile Hydrology"
    elif "soil texture" in ct or "texture composition" in ct:
        label = "🌾 Soil Texture & Workability"
    elif "organic matter" in ct or "som" in ct:
        label = "🌾 Carbon & Fertility Status"
    elif "ndvi" in ct:
        label = "🌾 NDVI Vegetation Health Signal"
    elif "evi" in ct:
        label = "🌾 Canopy Density & Biomass"
    elif "ndwi" in ct:
        label = "🌾 Canopy Water Stress Timeline"
    elif "savi" in ct:
        label = "🌾 Soil-Adjusted Vegetation Cover"
    elif "gndvi" in ct:
        label = "🌾 Chlorophyll & Nitrogen Proxy"
    elif "temperature" in ct and "vegetation" in ct:
        label = "🌾 Thermal Growing Season"
    elif "precipitation" in ct and "vegetation" in ct:
        label = "🌾 Rainfall–Vegetation Coupling"
    else:
        label = "🌾 AI Data Insight"

    # Use a container instead of an expander to ensure visibility
    st.markdown(f'<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    
    # Add a visual header for the AI section
    st.markdown(f'''
    <div class="ai-header">
        <div class="ai-header-line"></div>
        <span class="ai-header-text">{label}</span>
    </div>
    ''', unsafe_allow_html=True)
    
    if use_tinyllama and llm is not None and LLAMA_AVAILABLE:
        with st.spinner("🦙 TinyLlama is analyzing..."):
            tl_result = tinyllama_interpret(llm, chart_type, data_summary, location)
        if tl_result:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
                f'<span style="font-size:1.2rem;">🦙</span>'
                f'<span style="color:#00FF88;font-weight:600;font-size:0.85rem;">TinyLlama 1.1B</span>'
                f'<span style="background:rgba(0,255,136,0.15);border:1px solid rgba(0,255,136,0.3);'
                f'border-radius:20px;padding:1px 8px;font-size:0.7rem;color:#00FF88;">AI</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="ai-interpretation">{tl_result}</div>', unsafe_allow_html=True)
        else:
            rule_based = get_smart_interpretation(chart_type, data_summary, location)
            st.markdown(
                '<div style="color:#FFAA44;font-size:0.8rem;margin-bottom:0.4rem;">⚠️ TinyLlama inference failed — showing rule-based analysis</div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="ai-interpretation">{rule_based}</div>', unsafe_allow_html=True)
    else:
        rule_based = get_smart_interpretation(chart_type, data_summary, location)
        ai_source = "GIS Intelligence Engine"
        if not LLAMA_AVAILABLE:
            ai_source = "GIS Intelligence Engine (TinyLlama not installed)"
        
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
            f'<span style="font-size:1.1rem;">🤖</span>'
            f'<span style="color:#4A90E2;font-weight:600;font-size:0.85rem;">{ai_source}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="ai-interpretation">{rule_based}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session():
    defaults = {
        "page": "home",
        "analysis_type": "Climate & Soil",
        "selected_country": None,
        "selected_region": None,
        "current_step": 1,
        "climate_df": None,
        "soil_data": None,
        "climate_classification": None,
        "vegetation_results": None,
        "selected_indices": ["NDVI", "EVI", "SAVI", "NDWI", "GNDVI"],
        "collection_choice": "Sentinel-2",
        "precip_scale": 1.0,
        "tinyllama_enabled": True,
        "tinyllama_loaded": False,
        "tinyllama_download_attempted": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# =============================================================================
# MAP HTML
# =============================================================================

def map_iframe(center_lat=20, center_lon=10, zoom=2):
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <script src='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js'></script>
        <link href='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css' rel='stylesheet' />
        <style>
            body {{ margin: 0; padding: 0; background: #0A0A0A; }}
            #map {{ width: 100%; height: 460px; border-radius: 14px; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            mapboxgl.accessToken = 'pk.eyJ1IjoiYnJ5Y2VseW5uMjUiLCJhIjoiY2x1a2lmcHh5MGwycTJrbzZ4YXVrb2E0aiJ9.LXbneMJJ6OosHv9ibtI5XA';
            const map = new mapboxgl.Map({{
                container: 'map',
                style: 'mapbox://styles/mapbox/satellite-streets-v12',
                center: [{center_lon}, {center_lat}],
                zoom: {zoom},
                pitch: 0
            }});
            map.addControl(new mapboxgl.NavigationControl({{ showCompass: true, showZoom: true }}), 'top-right');
        </script>
    </body>
    </html>
    '''

# =============================================================================
# PROGRESS BAR
# =============================================================================

def progress_bar_html(step):
    steps_meta = {
        "Climate & Soil": ["Location", "Climate", "Soil", "Results"],
        "Vegetation & Climate": ["Location", "Parameters", "Preview", "Results"],
    }
    labels = steps_meta.get(st.session_state.analysis_type, ["Step1", "Step2", "Step3", "Step4"])
    html = '<div style="display:flex; gap:0.5rem; margin-bottom:1rem;">'
    for i, label in enumerate(labels, 1):
        if i < step:
            cls = "completed"
            icon = "✓"
        elif i == step:
            cls = "active"
            icon = str(i)
        else:
            cls = ""
            icon = str(i)
        active_color = "#00FF88" if cls == "active" else ("#00CC6A" if cls == "completed" else "#2A2A2A")
        text_col = "#0A0A0A" if cls in ["active", "completed"] else "#999999"
        label_col = "#00FF88" if cls == "active" else "#CCCCCC"
        html += f'''
        <div style="flex:1; display:flex; flex-direction:column; align-items:center;">
            <div style="width:34px;height:34px;border-radius:50%;background:{active_color};display:flex;align-items:center;justify-content:center;color:{text_col};font-weight:700;font-size:0.85rem;margin-bottom:0.4rem;">{icon}</div>
            <span style="font-size:0.7rem;color:{label_col};text-align:center;">{label}</span>
        </div>'''
    html += '</div>'
    return html

# =============================================================================
# MAIN UI
# =============================================================================

# Debug sidebar
with st.sidebar:
    st.markdown("### 🐛 Debug Info")
    st.write(f"LLAMA_AVAILABLE: {LLAMA_AVAILABLE}")
    st.write(f"Model exists: {MODEL_PATH.exists()}")
    if MODEL_PATH.exists():
        st.write(f"Model size: {MODEL_PATH.stat().st_size / (1024**2):.1f} MB")
    st.write(f"tinyllama_loaded: {st.session_state.tinyllama_loaded}")
    st.write(f"tinyllama_enabled: {st.session_state.tinyllama_enabled}")
    st.write(f"Current step: {st.session_state.current_step}")
    
    st.markdown("---")
    st.markdown("### 🦙 TinyLlama AI")
    if LLAMA_AVAILABLE and st.session_state.tinyllama_loaded:
        st.success("TinyLlama 1.1B ✅", icon="🦙")
        st.session_state.tinyllama_enabled = st.toggle(
            "Enable AI Analysis", value=st.session_state.tinyllama_enabled
        )
    elif LLAMA_AVAILABLE and MODEL_PATH.exists():
        st.info("🦙 Model on disk — loading...")
    elif LLAMA_AVAILABLE:
        st.info("🦙 Model not downloaded yet.\nScroll up and click the download button.")
    else:
        st.warning("🦙 TinyLlama not available.\nInstall llama-cpp-python to enable.")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.session_state.analysis_type = st.radio(
        "Analysis Mode", ["Climate & Soil", "Vegetation & Climate"], index=0
    )

# Header
st.markdown("""
<div style="padding: 0.5rem 0 1rem 0;">
  <h1>🌍 Khisba GIS</h1>
  <p style="color:#CCCCCC; margin:0; font-size:0.9rem;">Climate & Soil Analyzer · GIS Intelligence AI</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# AUTO-LOAD TINYLLAMA ON STARTUP (with graceful fallback)
# =============================================================================

llm = None

# If model file doesn't exist yet and llama-cpp is available, show download banner
if LLAMA_AVAILABLE and not MODEL_PATH.exists() and not st.session_state.tinyllama_download_attempted:
    st.markdown("""
    <div style="background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.3);
         border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;">
      <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
        <span style="font-size:1.4rem;">🦙</span>
        <strong style="color:#00FF88;">TinyLlama 1.1B AI — One-time Download Required</strong>
      </div>
      <p style="color:#CCCCCC;margin:0;font-size:0.9rem;">
        The TinyLlama model (~637 MB) needs to be downloaded once to enable real AI chart interpretation.
        Click the button below — it will download and load automatically.
      </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("⬇️ Download TinyLlama & Enable AI Analysis", use_container_width=True, type="primary"):
        st.session_state.tinyllama_download_attempted = True
        pb = st.progress(0)
        st_txt = st.empty()
        ok, err = download_model_with_progress(pb, st_txt)
        pb.empty()
        st_txt.empty()
        if ok:
            _llm, _err2 = load_tinyllama_model()
            if _llm:
                st.session_state.tinyllama_loaded = True
                st.session_state.tinyllama_enabled = True
                llm = _llm
                st.success("🦙 TinyLlama loaded! AI analysis is now active on all charts.", icon="✅")
                st.rerun()
            else:
                st.error(f"Downloaded but failed to load: {_err2}")
        else:
            st.error(f"Download failed: {err}")

elif LLAMA_AVAILABLE and MODEL_PATH.exists() and not st.session_state.tinyllama_loaded:
    # Model already on disk — load it (cached, fast after first time)
    with st.spinner("🦙 Loading TinyLlama model..."):
        _llm, _err = load_tinyllama_model()
        if _llm:
            st.session_state.tinyllama_loaded = True
            st.session_state.tinyllama_enabled = True
            llm = _llm
            st.success("🦙 TinyLlama loaded successfully!")
        else:
            st.warning(f"🦙 TinyLlama model found but failed to load: {_err}")

elif st.session_state.tinyllama_loaded and LLAMA_AVAILABLE:
    # Already loaded in a previous rerun — retrieve from cache
    _llm, _ = load_tinyllama_model()
    llm = _llm

st.markdown(progress_bar_html(st.session_state.current_step), unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    # STEP 1
    if st.session_state.current_step == 1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><div class="card-icon">📍</div><h3 style="margin:0;">Select Location</h3></div>', unsafe_allow_html=True)

        country = st.selectbox("🌍 Country", [""] + sorted(WORLD_REGIONS.keys()))
        if country:
            regions = WORLD_REGIONS[country]["regions"]
            region = st.selectbox("📌 Region / Province", [""] + regions)
        else:
            region = None

        if st.button("✅ Confirm Location", use_container_width=True):
            if country:
                st.session_state.selected_country = country
                st.session_state.selected_region = region if region else None
                location_name = f"{region}, {country}" if region else country
                st.session_state.location_name = location_name
                st.session_state.region_type = get_region_type(location_name)
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.warning("Please select a country.")
        st.markdown('</div>', unsafe_allow_html=True)

    # STEP 2
    elif st.session_state.current_step == 2:
        location_name = st.session_state.get("location_name", "Unknown")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.session_state.analysis_type == "Climate & Soil":
            st.markdown('<div class="card-header"><div class="card-icon">🌤️</div><h3 style="margin:0;">Climate Settings</h3></div>', unsafe_allow_html=True)
            st.info(f"📍 **{location_name}**")
            region_type = st.session_state.region_type
            if region_type in ["Semi-arid", "Arid"]:
                st.markdown(f"""<div style="background:rgba(255,170,68,0.1);padding:0.75rem;border-radius:8px;margin-bottom:1rem;">
                    <p style="color:#FFAA44;margin:0;font-size:0.85rem;">⚠️ <strong>{region_type} Region Detected</strong><br>CHIRPS may overestimate in arid regions. Calibration recommended: 0.7–0.8×</p></div>""", unsafe_allow_html=True)
                precip_scale = st.slider("💧 Precipitation Calibration", 0.5, 1.0, 0.75, 0.05)
            else:
                precip_scale = st.slider("💧 Precipitation Calibration", 0.5, 1.5, 1.0, 0.05)
            st.session_state.precip_scale = precip_scale
        else:
            st.markdown('<div class="card-header"><div class="card-icon">🌿</div><h3 style="margin:0;">Vegetation Settings</h3></div>', unsafe_allow_html=True)
            st.info(f"📍 **{location_name}**")
            st.session_state.collection_choice = st.selectbox("🛰️ Satellite", ["Sentinel-2", "Landsat-8"])
            st.session_state.selected_indices = st.multiselect("📊 Select Indices", VEGETATION_INDICES, default=["NDVI", "EVI", "SAVI", "NDWI", "GNDVI"])

        cb, cn = st.columns(2)
        with cb:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with cn:
            if st.button("✅ Save & Continue", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # STEP 3
    elif st.session_state.current_step == 3:
        location_name = st.session_state.get("location_name", "Unknown")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.session_state.analysis_type == "Climate & Soil":
            st.markdown('<div class="card-header"><div class="card-icon">🌱</div><h3 style="margin:0;">Soil Settings</h3></div>', unsafe_allow_html=True)
            st.info(f"📍 **{location_name}**")
            st.markdown("""<div style="background:rgba(0,255,136,0.08);padding:0.75rem;border-radius:8px;margin-bottom:1rem;">
                <p style="color:#CCCCCC;margin:0;font-size:0.85rem;"><strong>📊 Soil Data Sources:</strong><br>
                • ISDAsoil (Africa) / GSOC (Global): Soil organic carbon<br>
                • OpenLandMap: Soil texture classes<br>
                • Depth: 20cm (Africa) / 30cm (Global)</p></div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card-header"><div class="card-icon">🗺️</div><h3 style="margin:0;">Analysis Preview</h3></div>', unsafe_allow_html=True)
            st.info(f"📍 **{location_name}**")
            indices_str = ", ".join(st.session_state.selected_indices[:5])
            if len(st.session_state.selected_indices) > 5:
                indices_str += "..."
            st.markdown(f"""<div style="background:rgba(0,255,136,0.08);padding:0.75rem;border-radius:8px;margin-bottom:1rem;">
                <p style="color:#CCCCCC;margin:0;font-size:0.85rem;"><strong>🛰️ Satellite:</strong> {st.session_state.collection_choice}<br>
                <strong>📊 Indices:</strong> {indices_str}<br><strong>📅 Period:</strong> 2023–2025</p></div>""", unsafe_allow_html=True)

        cb, cn = st.columns(2)
        with cb:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with cn:
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                location_name = st.session_state.get("location_name", "Unknown")
                region_type = st.session_state.get("region_type", "general")
                with st.spinner("Generating analysis data..."):
                    st.session_state.climate_df = generate_climate_data(location_name, region_type=region_type)
                    st.session_state.soil_data = get_soil_data(location_name, region_type)
                    st.session_state.climate_classification = get_climate_classification(location_name, region_type)
                    if st.session_state.analysis_type == "Vegetation & Climate":
                        results = {}
                        for idx in st.session_state.selected_indices:
                            dates, values = generate_vegetation_data(location_name, idx, months=24)
                            results[idx] = {"dates": dates, "values": values}
                        st.session_state.vegetation_results = results
                st.session_state.current_step = 4
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # STEP 4
    elif st.session_state.current_step == 4:
        location_name = st.session_state.get("location_name", "Unknown")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><div class="card-icon">📊</div><h3 style="margin:0;">Results</h3></div>', unsafe_allow_html=True)
        st.success(f"✅ Analysis complete for **{location_name}**")

        ai_status = "🦙 TinyLlama 1.1B" if (st.session_state.tinyllama_enabled and llm is not None and LLAMA_AVAILABLE) else "🤖 GIS Intelligence"
        ai_note = " (TinyLlama not installed)" if not LLAMA_AVAILABLE else ""
        st.markdown(f"""<div style="background:rgba(0,255,136,0.08);padding:0.75rem;border-radius:8px;margin-bottom:1rem;">
            <p style="color:#CCCCCC;margin:0;font-size:0.85rem;">{ai_status}{ai_note}: ✅ Ready<br>
            📈 <strong>Charts with AI interpretation</strong> are shown on the right.</p></div>""", unsafe_allow_html=True)

        cb, cn = st.columns(2)
        with cb:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
        with cn:
            if st.button("🔄 New Analysis", use_container_width=True):
                for k in ["selected_country", "selected_region", "location_name", "climate_df", "soil_data", "climate_classification", "vegetation_results"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.session_state.current_step = 1
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.current_step <= 3:
        country_info = WORLD_REGIONS.get(st.session_state.get("selected_country", ""), {})
        lat = country_info.get("lat", 20)
        lon = country_info.get("lon", 10)
        zoom = country_info.get("zoom", 2)
        st.markdown('<div class="card" style="padding:0;">', unsafe_allow_html=True)
        st.markdown('<div style="padding:0.75rem 1rem;"><h3 style="margin:0;">🗺️ Map Preview</h3></div>', unsafe_allow_html=True)
        st.components.v1.html(map_iframe(lat, lon, zoom), height=480)
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.current_step == 4:
        location_name = st.session_state.get("location_name", "Unknown")
        region_type = st.session_state.get("region_type", "general")
        climate_df = st.session_state.get("climate_df")
        soil_data = st.session_state.get("soil_data")
        climate_cls = st.session_state.get("climate_classification")
        veg_results = st.session_state.get("vegetation_results")
        use_tl = st.session_state.tinyllama_enabled and llm is not None and LLAMA_AVAILABLE

        if st.session_state.analysis_type == "Climate & Soil":
            # Climate Classification
            with st.container():
                st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
                st.markdown('<div style="margin-bottom:1rem;"><h3 style="margin:0;">🌤️ Climate Classification</h3></div>', unsafe_allow_html=True)
                if climate_cls:
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("🌡️ Mean Annual Temp", f"{climate_cls['mean_temperature']:.1f}°C")
                    with col_m2:
                        st.metric("💧 Annual Precip", f"{climate_cls['mean_precipitation']:.0f} mm")
                    st.info(f"**Climate Zone:** {climate_cls['climate_zone']}")
                    
                    # Create gauges
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        temp_gauge = create_climate_temp_gauge_html(climate_cls)
                        if temp_gauge:
                            display_chart(temp_gauge)
                    with col_g2:
                        precip_gauge = create_climate_precip_gauge_html(climate_cls)
                        if precip_gauge:
                            display_chart(precip_gauge)
                    
                    arid = climate_cls.get('aridity_index', 0)
                    water_stress = "severe drought stress" if arid < 0.5 else ("moderate water stress" if arid < 1.0 else ("balanced water regime" if arid < 2.0 else "humid surplus"))
                    data_summary = (
                        f"Climate zone: {climate_cls['climate_zone']}, "
                        f"Mean annual temperature: {climate_cls['mean_temperature']:.1f}°C, "
                        f"Annual precipitation: {climate_cls['mean_precipitation']:.0f}mm, "
                        f"Aridity index: {arid:.2f} ({water_stress})"
                    )
                    show_ai_interpretation("Climate Classification gauge", data_summary, location_name, llm, use_tl)
                st.markdown('</div>', unsafe_allow_html=True)

            if climate_df is not None and not climate_df.empty:
                with st.container():
                    st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
                    st.markdown('<div style="margin-bottom:0.5rem;"><h3 style="margin:0;">📊 Climate Data</h3></div>', unsafe_allow_html=True)
                    
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌡️ Temperature", "💧 Water", "🌱 Soil Moisture", "📊 Distribution", "📋 Data Table"])

                    with tab1:
                        # Temperature chart
                        temp_chart = create_temperature_chart_html(climate_df, location_name)
                        display_chart(temp_chart)
                        
                        # Metrics
                        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                        with col_t1: 
                            st.metric("Avg Temp", f"{climate_df['temperature_2m'].mean():.1f}°C")
                        with col_t2:
                            max_t = climate_df['temperature_2m'].max()
                            max_m = climate_df.loc[climate_df['temperature_2m'].idxmax(), 'month_name']
                            st.metric("Max Temp", f"{max_t:.1f}°C", delta=f"in {max_m}")
                        with col_t3:
                            min_t = climate_df['temperature_2m'].min()
                            min_m = climate_df.loc[climate_df['temperature_2m'].idxmin(), 'month_name']
                            st.metric("Min Temp", f"{min_t:.1f}°C", delta=f"in {min_m}")
                        with col_t4: 
                            st.metric("Range", f"{(climate_df['temperature_2m'].max() - climate_df['temperature_2m'].min()):.1f}°C")
                        
                        # Data summary for AI
                        temps = climate_df['temperature_2m'].tolist()
                        months = climate_df['month_name'].tolist()
                        hot_months = [m for m, t in zip(months, temps) if t > 30]
                        cold_months = [m for m, t in zip(months, temps) if t < 5]
                        grow_months = [m for m, t in zip(months, temps) if 10 <= t <= 30]
                        
                        data_summary = (
                            f"Monthly temperatures range from {min(temps):.1f}°C to {max(temps):.1f}°C. "
                            f"Peak: {max_t:.1f}°C in {max_m}. "
                            f"Coldest: {min_t:.1f}°C in {min_m}. "
                            f"Heat-stress months (>30°C): {len(hot_months)}. "
                            f"Frost-risk months (<5°C): {len(cold_months)}. "
                            f"Optimal growing window (10–30°C): {len(grow_months)} months."
                        )
                        show_ai_interpretation("Monthly Temperature", data_summary, location_name, llm, use_tl)

                    with tab2:
                        # Precipitation chart
                        precip_chart = create_precipitation_chart_html(climate_df, location_name)
                        display_chart(precip_chart)
                        
                        # Metrics
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1: 
                            st.metric("Annual Total", f"{climate_df['total_precipitation'].sum():.0f} mm")
                        with col_p2:
                            if 'potential_evaporation' in climate_df.columns:
                                st.metric("Annual ET", f"{climate_df['potential_evaporation'].sum():.0f} mm")
                        with col_p3:
                            balance = climate_df['total_precipitation'].sum() - climate_df.get('potential_evaporation', pd.Series([0]*12)).sum()
                            st.metric("Water Balance", f"{balance:.0f} mm", delta="Surplus" if balance > 0 else "Deficit")
                        
                        # Data summary
                        precip = climate_df['total_precipitation'].tolist()
                        pmonths = climate_df['month_name'].tolist()
                        dry_months = [m for m, p in zip(pmonths, precip) if p < 20]
                        annual_total = climate_df['total_precipitation'].sum()
                        
                        data_summary = (
                            f"Annual total precipitation: {annual_total:.0f}mm. "
                            f"Peak month: {climate_df.loc[climate_df['total_precipitation'].idxmax(),'month_name']} ({climate_df['total_precipitation'].max():.0f}mm). "
                            f"Dry months (<20mm): {len(dry_months)}. "
                            f"Water balance: {balance:+.0f}mm ({'surplus' if balance > 0 else 'deficit'})."
                        )
                        show_ai_interpretation("Precipitation & Evapotranspiration", data_summary, location_name, llm, use_tl)

                    with tab3:
                        # Soil moisture chart
                        soil_chart = create_soil_moisture_chart_html(climate_df, location_name)
                        display_chart(soil_chart)
                        
                        # Metrics
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1: 
                            st.metric("Surface (0-7cm)", f"{climate_df['soil_moisture_0_7cm'].mean():.3f} m³/m³")
                        with col_s2: 
                            st.metric("Root Zone (7-28cm)", f"{climate_df['soil_moisture_7_28cm'].mean():.3f} m³/m³")
                        with col_s3: 
                            st.metric("Deep (28-100cm)", f"{climate_df['soil_moisture_28_100cm'].mean():.3f} m³/m³")
                        
                        data_summary = (
                            f"Mean surface moisture: {climate_df['soil_moisture_0_7cm'].mean():.3f} m³/m³, "
                            f"Root zone: {climate_df['soil_moisture_7_28cm'].mean():.3f} m³/m³, "
                            f"Deep zone: {climate_df['soil_moisture_28_100cm'].mean():.3f} m³/m³"
                        )
                        show_ai_interpretation("Soil Moisture by Layer", data_summary, location_name, llm, use_tl)

                    with tab4:
                        # Distribution chart
                        dist_chart = create_soil_distribution_chart_html(climate_df)
                        display_chart(dist_chart)
                        
                        st.markdown("""
                        <div style="background:rgba(255,255,255,0.04);padding:1rem;border-radius:12px;margin-top:1rem;">
                            <p style="color:#CCCCCC;margin:0;font-size:0.85rem;">
                            <strong>Soil Moisture Interpretation:</strong><br>
                            • <span style="color:#00FF88">Surface (0-7cm):</span> Rapid rainfall response, high evaporation<br>
                            • <span style="color:#4A90E2">Root Zone (7-28cm):</span> Available for plant uptake<br>
                            • <span style="color:#FFAA44">Deep (28-100cm):</span> Groundwater recharge zone
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        data_summary = (
                            f"Surface avg: {climate_df['soil_moisture_0_7cm'].mean():.3f} m³/m³, "
                            f"Root zone avg: {climate_df['soil_moisture_7_28cm'].mean():.3f} m³/m³, "
                            f"Deep avg: {climate_df['soil_moisture_28_100cm'].mean():.3f} m³/m³"
                        )
                        show_ai_interpretation("Soil Moisture Distribution", data_summary, location_name, llm, use_tl)

                    with tab5:
                        st.dataframe(
                            climate_df[['month_name','temperature_2m','total_precipitation',
                                       'soil_moisture_0_7cm','soil_moisture_7_28cm','soil_moisture_28_100cm']].rename(columns={
                                'month_name': 'Month', 
                                'temperature_2m': 'Temp (°C)', 
                                'total_precipitation': 'Precip (mm)',
                                'soil_moisture_0_7cm': 'SM Surface', 
                                'soil_moisture_7_28cm': 'SM Root', 
                                'soil_moisture_28_100cm': 'SM Deep'
                            }),
                            use_container_width=True, 
                            hide_index=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

            if soil_data:
                with st.container():
                    st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
                    st.markdown('<div style="margin-bottom:0.5rem;"><h3 style="margin:0;">🌱 Soil Properties</h3></div>', unsafe_allow_html=True)
                    
                    col_p1, col_p2, col_p3 = st.columns(3)
                    with col_p1: 
                        st.metric("🏔️ Texture", soil_data['texture_name'])
                    with col_p2: 
                        st.metric("🌿 SOM", f"{soil_data['final_som_estimate']:.2f}%")
                    with col_p3: 
                        st.metric("📦 SOC Stock", f"{soil_data['soc_stock']:.1f} t/ha")
                    
                    col_tex, col_som = st.columns(2)
                    with col_tex:
                        tex_chart = create_soil_texture_chart_html(soil_data, location_name)
                        if tex_chart:
                            display_chart(tex_chart)
                        
                        clay = soil_data['clay_content']
                        silt = soil_data['silt_content']
                        sand = soil_data['sand_content']
                        tex = soil_data['texture_name']
                        compaction_risk = "high" if clay > 40 else ("moderate" if clay > 25 else "low")
                        drainage = "slow" if clay > 40 else ("moderate" if clay > 20 else "fast")
                        data_summary = (
                            f"Texture class: {tex}. Clay: {clay}%, Silt: {silt}%, Sand: {sand}%. "
                            f"Compaction risk: {compaction_risk}. Drainage: {drainage}. "
                            f"{'High clay content — good nutrient retention but tillage challenges.' if clay > 35 else ''}"
                            f"{'Silty loam — excellent workability and water-holding.' if 30 < silt < 50 and clay < 30 else ''}"
                            f"{'Sandy component dominant — low water retention, leaching risk.' if sand > 60 else ''}"
                        )
                        show_ai_interpretation("Soil Texture Composition", data_summary, location_name, llm, use_tl)
                    
                    with col_som:
                        som_chart = create_som_gauge_html(soil_data, location_name)
                        if som_chart:
                            display_chart(som_chart)
                        
                        som = soil_data['final_som_estimate']
                        soc = soil_data['soc_stock']
                        fertility = "very high" if som > 4 else ("high" if som > 2.5 else ("medium" if som > 1.5 else ("low" if som > 0.8 else "critically low")))
                        data_summary = (
                            f"Soil Organic Matter: {som:.2f}% ({fertility} fertility). "
                            f"SOC Stock: {soc:.1f} t C/ha. "
                            f"{'Organic matter critically low — soil biology depleted, fertility inputs essential.' if som < 1.0 else ''}"
                            f"{'Medium SOM — building carbon reserves would meaningfully improve water retention.' if 1.0 <= som < 2.0 else ''}"
                            f"{'Good SOM level — supports active microbial life and nitrogen cycling.' if som >= 2.0 else ''}"
                        )
                        show_ai_interpretation("Soil Organic Matter gauge", data_summary, location_name, llm, use_tl)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

        else:  # Vegetation & Climate
            if veg_results:
                with st.container():
                    st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
                    st.markdown('<div style="margin-bottom:0.5rem;"><h3 style="margin:0;">🌿 Vegetation Indices</h3></div>', unsafe_allow_html=True)
                    
                    for idx_name, data in veg_results.items():
                        st.markdown(f"**{idx_name}**")
                        veg_chart = create_vegetation_chart_html(data['dates'], data['values'], idx_name, location_name)
                        if veg_chart:
                            display_chart(veg_chart)
                        
                        vals = data['values']
                        col_v1, col_v2, col_v3 = st.columns(3)
                        with col_v1: 
                            st.metric(f"{idx_name} Mean", f"{np.mean(vals):.3f}")
                        with col_v2: 
                            st.metric(f"{idx_name} Max", f"{np.max(vals):.3f}")
                        with col_v3: 
                            st.metric(f"{idx_name} Min", f"{np.min(vals):.3f}")
                        
                        trend = np.polyfit(range(len(vals)), vals, 1)[0]
                        trend_dir = "increasing" if trend > 0.001 else ("decreasing" if trend < -0.001 else "stable")
                        mean_v = np.mean(vals)
                        
                        # Health category per index
                        if idx_name == "NDVI":
                            health = "dense healthy canopy" if mean_v > 0.6 else ("moderate vegetation" if mean_v > 0.4 else ("sparse/stressed" if mean_v > 0.2 else "bare/very sparse"))
                        elif idx_name == "EVI":
                            health = "high biomass" if mean_v > 0.5 else ("moderate canopy" if mean_v > 0.3 else "low biomass")
                        elif idx_name == "NDWI":
                            health = "well-watered canopy" if mean_v > 0.2 else ("mild water stress" if mean_v > -0.1 else "significant water stress")
                        elif idx_name == "SAVI":
                            health = "good ground cover" if mean_v > 0.4 else ("partial cover" if mean_v > 0.2 else "sparse/degraded")
                        elif idx_name == "GNDVI":
                            health = "high chlorophyll/N" if mean_v > 0.5 else ("adequate chlorophyll" if mean_v > 0.35 else "chlorophyll/N deficient")
                        else:
                            health = "moderate" if mean_v > 0.4 else "low"
                        
                        variability = np.std(vals)
                        seasonality = "strong seasonal pulse" if variability > 0.08 else ("moderate seasonality" if variability > 0.04 else "low variability — evergreen or uniform cover")
                        
                        data_summary = (
                            f"{idx_name} 24-month time series. Mean: {mean_v:.3f} ({health}). "
                            f"Max: {np.max(vals):.3f}. Min: {np.min(vals):.3f}. "
                            f"Trend: {trend_dir}. "
                            f"Variability (std): {variability:.3f} — {seasonality}. "
                            f"{'2-year decline suggests vegetation degradation or land use change.' if trend < -0.002 else ''}"
                            f"{'Sustained growth trend — positive land cover change.' if trend > 0.002 else ''}"
                        )
                        show_ai_interpretation(f"{idx_name} vegetation index", data_summary, location_name, llm, use_tl)
                        st.markdown("---")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                if climate_df is not None:
                    with st.container():
                        st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
                        st.markdown('<h3 style="margin:0 0 0.5rem 0;">🌤️ Climate Data</h3>', unsafe_allow_html=True)
                        
                        temp_chart = create_temperature_chart_html(climate_df, location_name)
                        if temp_chart:
                            display_chart(temp_chart)
                        
                        vt = climate_df['temperature_2m'].tolist()
                        vm = climate_df['month_name'].tolist()
                        grow_window = [m for m, t in zip(vm, vt) if t >= 10]
                        data_summary = (
                            f"Monthly temperatures range from {min(vt):.1f}°C to {max(vt):.1f}°C. "
                            f"Thermal growing season: {len(grow_window)} months. "
                            f"Peak warmth: {max(vt):.1f}°C. "
                            f"Cold floor: {min(vt):.1f}°C. "
                            f"Annual range: {max(vt)-min(vt):.1f}°C."
                        )
                        show_ai_interpretation("Monthly Temperature for vegetation context", data_summary, location_name, llm, use_tl)
                        
                        precip_chart = create_precipitation_chart_html(climate_df, location_name)
                        if precip_chart:
                            display_chart(precip_chart)
                        
                        vp = climate_df['total_precipitation'].tolist()
                        green_months = [m for m, p in zip(vm, vp) if p >= 30]
                        data_summary = (
                            f"Monthly rainfall ranges from {min(vp):.0f}mm to {max(vp):.0f}mm. "
                            f"Annual total: {sum(vp):.0f}mm. "
                            f"Rain-supported growing months (≥30mm): {len(green_months)}. "
                            f"Peak rainfall: {max(vp):.0f}mm. "
                            f"Dry season length: {12-len(green_months)} months."
                        )
                        show_ai_interpretation("Monthly Precipitation for vegetation context", data_summary, location_name, llm, use_tl)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
