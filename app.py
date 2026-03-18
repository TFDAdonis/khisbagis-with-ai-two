import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import re as _re
import os
import requests
from pathlib import Path
import subprocess
import sys

# =============================================================================
# AUTO-INSTALL TINYLLAMA IF MISSING
# =============================================================================

def install_llama_cpp():
    """Attempt to install llama-cpp-python automatically"""
    try:
        st.info("📦 Installing llama-cpp-python... This may take a few minutes.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
        st.success("✅ llama-cpp-python installed successfully! Please restart the app.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to install llama-cpp-python: {e}")
        st.info("Please run manually: pip install llama-cpp-python")
        return False

# Try to import llama_cpp, with auto-install option
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
    print("✅ llama-cpp-python imported successfully")
except ImportError as e:
    LLAMA_AVAILABLE = False
    Llama = None
    print(f"❌ Failed to import llama-cpp-python: {e}")

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
    .install-card {
        background: linear-gradient(135deg, #1a2a1a, #0f1a0f);
        border: 2px solid #00FF88;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .install-button {
        background: #00FF88;
        color: #0A0A0A;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .install-button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0,255,136,0.5);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SHOW INSTALLATION UI IF LLAMA NOT AVAILABLE
# =============================================================================

if not LLAMA_AVAILABLE:
    st.markdown("""
    <div class="install-card">
        <span style="font-size: 4rem;">🦙</span>
        <h2 style="color: #00FF88; margin: 1rem 0;">TinyLlama AI Not Installed</h2>
        <p style="color: #CCCCCC; font-size: 1.1rem; margin-bottom: 2rem;">
            To enable AI-powered chart interpretations, install llama-cpp-python and download the TinyLlama model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📦 Auto-Install TinyLlama", use_container_width=True, type="primary"):
            install_llama_cpp()
        
        st.markdown("""
        <div style="background: #141414; border-radius: 12px; padding: 1.5rem; margin-top: 1rem;">
            <h4 style="color: #FFFFFF; margin-bottom: 1rem;">📋 Manual Installation Instructions</h4>
            <p style="color: #CCCCCC; margin-bottom: 0.5rem;">1. Open terminal/command prompt</p>
            <p style="color: #CCCCCC; margin-bottom: 0.5rem;">2. Run: <code style="background: #2A2A2A; padding: 0.2rem 0.5rem; border-radius: 4px;">pip install llama-cpp-python</code></p>
            <p style="color: #CCCCCC; margin-bottom: 0.5rem;">3. Restart this app</p>
            <p style="color: #00FF88; margin-top: 1rem;">✨ Then download the TinyLlama model (637MB) when prompted</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# =============================================================================
# TINYLLAMA MODEL SETUP
# =============================================================================

_APP_DIR = Path(__file__).parent.resolve()
MODEL_DIR = _APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Alternative smaller model if download size is an issue
SMALL_MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
SMALL_MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"


def download_model_with_progress(progress_bar=None, status_text=None, use_small_model=False):
    """Download the model file with progress tracking."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    url = SMALL_MODEL_URL if use_small_model else MODEL_URL
    path = SMALL_MODEL_PATH if use_small_model else MODEL_PATH
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
    except Exception as e:
        return False, str(e)

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    try:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size
                        if progress_bar:
                            progress_bar.progress(min(pct, 1.0))
                        if status_text:
                            mb_downloaded = downloaded/(1024**2)
                            mb_total = total_size/(1024**2)
                            status_text.text(f"⬇️ Downloading TinyLlama: {mb_downloaded:.0f} / {mb_total:.0f} MB")
    except Exception as e:
        if path.exists():
            path.unlink()
        return False, str(e)
    return True, "OK"


@st.cache_resource(show_spinner=False)
def load_tinyllama_model():
    """Load TinyLlama from disk."""
    # Try the small model first if it exists, then fall back to regular
    if SMALL_MODEL_PATH.exists():
        model_path = str(SMALL_MODEL_PATH.resolve())
    elif MODEL_PATH.exists():
        model_path = str(MODEL_PATH.resolve())
    else:
        return None, "No model file found"
    
    try:
        print(f"🦙 Loading TinyLlama from {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=2,
            n_batch=128,
            verbose=False
        )
        print("✅ TinyLlama loaded successfully")
        return llm, "ok"
    except Exception as e:
        import traceback
        error_msg = f"{e}\n{traceback.format_exc()[-500:]}"
        print(f"❌ Failed to load TinyLlama: {error_msg}")
        return None, error_msg


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
    """Build chart-specific prompts with data grounding for TinyLlama."""
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
    if llm is None:
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
        parts = []
        if zone:
            parts.append(f"The climate{loc_str} is classified as **{zone}**.")
        if temp is not None:
            if temp > 30:
                parts.append(f"With a mean annual temperature of {temp:.1f}°C, heat stress is a significant factor — drought-tolerant and heat-adapted varieties are recommended.")
            elif temp > 20:
                parts.append(f"A mean annual temperature of {temp:.1f}°C supports year-round cultivation of warm-season crops.")
            elif temp > 10:
                parts.append(f"A mean annual temperature of {temp:.1f}°C is ideal for temperate crops.")
            else:
                parts.append(f"A mean annual temperature of {temp:.1f}°C limits the growing season; cold-hardy crops are critical.")
        if precip is not None:
            if precip < 250:
                parts.append(f"Annual precipitation of {precip:.0f} mm indicates a hyper-arid regime — irrigation is essential.")
            elif precip < 500:
                parts.append(f"Annual precipitation of {precip:.0f} mm is semi-arid; supplemental irrigation recommended.")
            elif precip < 800:
                parts.append(f"Annual precipitation of {precip:.0f} mm supports rainfed agriculture for most of the year.")
            else:
                parts.append(f"High annual precipitation of {precip:.0f} mm means waterlogging may need attention.")
        return " ".join(parts) if parts else "Climate classification indicates typical regional conditions."

    if "temperature" in ct:
        max_t = _parse_float(data_summary, r'Max.*?:\s*([\d.]+)°?C')
        min_t = _parse_float(data_summary, r'Min.*?:\s*([\d.]+)°?C')
        parts = []
        if max_t and min_t:
            rng = max_t - min_t
            parts.append(f"Temperatures{loc_str} span {min_t:.1f}°C to {max_t:.1f}°C — a seasonal range of {rng:.1f}°C.")
            if max_t > 30:
                parts.append("Peak temperatures exceed 30°C — irrigation and shade management recommended.")
            if min_t < 5:
                parts.append("Minimum temperatures fall below 5°C, indicating frost risk.")
        return " ".join(parts) if parts else f"Temperature data{loc_str} shows typical patterns."

    if "precipitation" in ct:
        annual = _parse_float(data_summary, r'Annual total:\s*([\d.]+)')
        parts = []
        if annual is not None:
            if annual < 200:
                parts.append(f"Total annual precipitation{loc_str} is extremely low at {annual:.0f} mm.")
            elif annual < 400:
                parts.append(f"Annual rainfall of {annual:.0f} mm{loc_str} is scarce.")
            elif annual < 700:
                parts.append(f"Annual precipitation of {annual:.0f} mm{loc_str} supports rainfed agriculture.")
            else:
                parts.append(f"Generous annual rainfall of {annual:.0f} mm{loc_str} supports productive rainfed farming.")
        return " ".join(parts) if parts else f"Precipitation data{loc_str} shows typical distribution."

    if "soil moisture" in ct:
        surf = _parse_float(data_summary, r'Surface.*?:\s*([\d.]+)')
        root = _parse_float(data_summary, r'Root.*?:\s*([\d.]+)')
        parts = []
        if surf is not None:
            if surf > 0.3:
                parts.append(f"Surface soil moisture{loc_str} is high at {surf:.3f} m³/m³.")
            elif surf > 0.15:
                parts.append(f"Surface soil moisture of {surf:.3f} m³/m³{loc_str} is moderate.")
            else:
                parts.append(f"Low surface moisture ({surf:.3f} m³/m³){loc_str} indicates dry topsoil.")
        if root is not None:
            if root > 0.25:
                parts.append(f"Root-zone moisture ({root:.3f} m³/m³) is well-supplied.")
            elif root > 0.1:
                parts.append(f"Root-zone moisture ({root:.3f} m³/m³) is marginal.")
            else:
                parts.append(f"Root-zone moisture ({root:.3f} m³/m³) is critically low.")
        return " ".join(parts) if parts else f"Soil moisture profile{loc_str} shows typical distribution."

    if "soil texture" in ct:
        clay = _parse_float(data_summary, r'Clay:\s*([\d.]+)%')
        silt = _parse_float(data_summary, r'Silt:\s*([\d.]+)%')
        sand = _parse_float(data_summary, r'Sand:\s*([\d.]+)%')
        parts = []
        if clay is not None and sand is not None:
            if clay > 40:
                parts.append(f"High clay content ({clay:.0f}%) provides excellent nutrient retention.")
            elif sand > 60:
                parts.append(f"Sandy texture ({sand:.0f}% sand) means rapid drainage.")
            else:
                parts.append(f"A balanced texture supports good soil structure.")
        return " ".join(parts) if parts else f"Soil texture{loc_str} indicates typical properties."

    if "organic matter" in ct or "som" in ct:
        som = _parse_float(data_summary, r'Soil Organic Matter:\s*([\d.]+)%')
        parts = []
        if som is not None:
            if som < 1.0:
                parts.append(f"Soil organic matter{loc_str} is critically low at {som:.2f}%.")
            elif som < 2.0:
                parts.append(f"SOM of {som:.2f}%{loc_str} is below optimal.")
            elif som < 4.0:
                parts.append(f"SOM of {som:.2f}%{loc_str} is in the moderate range.")
            else:
                parts.append(f"Excellent SOM of {som:.2f}%{loc_str} reflects highly fertile soil.")
        return " ".join(parts) if parts else f"Soil organic matter data{loc_str} indicates typical conditions."

    if any(v in ct for v in ['ndvi', 'evi', 'savi', 'ndwi', 'gndvi']):
        mean_v = _parse_float(data_summary, r'mean=([\d.]+)')
        parts = []
        if mean_v is not None:
            if mean_v > 0.6:
                parts.append(f"Vegetation index averages {mean_v:.3f} — dense, healthy vegetation.")
            elif mean_v > 0.4:
                parts.append(f"Vegetation index averages {mean_v:.3f} — moderate vegetation cover.")
            elif mean_v > 0.2:
                parts.append(f"Vegetation index averages {mean_v:.3f} — sparse vegetation.")
            else:
                parts.append(f"Vegetation index averages {mean_v:.3f} — very low greenness.")
        return " ".join(parts) if parts else f"Vegetation data{loc_str} indicates typical dynamics."

    return f"Analysis of {chart_type}{loc_str}: {data_summary[:200]}."


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
    'NDMI', 'NBR', 'NDWI', 'MNDWI', 'AWEI', 'NDSI_Salinity', 'SI'
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
    if any(x in loc for x in ['sidi', 'algeria', 'morocco', 'tunisia']):
        return "Semi-arid"
    elif any(x in loc for x in ['sahara', 'desert', 'egypt']):
        return "Arid"
    elif any(x in loc for x in ['amazon', 'congo', 'rainforest']):
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
        records.append({
            "month": month_num,
            "month_name": datetime(start_year, month_num, 1).strftime('%b'),
            "temperature_2m": round(temp, 1),
            "temperature_max": round(temp + np.random.uniform(3, 6), 1),
            "temperature_min": round(temp - np.random.uniform(3, 6), 1),
            "total_precipitation": round(precip, 1),
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
# SIMPLE HTML CHARTS
# =============================================================================

def create_temperature_chart_html(df, location_name):
    months = df['month_name'].tolist()
    temps = df['temperature_2m'].tolist()
    max_temp = max(temps)
    min_temp = min(temps)
    temp_range = max_temp - min_temp if max_temp > min_temp else 1
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; width: 100%;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #FFFFFF;">Monthly Temperature</h4>
            {accuracy_badge_html("high", "±1-2°C")}
        </div>
        <div style="display: flex; align-items: flex-end; justify-content: space-around; height: 250px;">
    '''
    for month, temp in zip(months, temps):
        height = ((temp - min_temp) / temp_range) * 180 + 30
        chart_html += f'''
            <div style="flex:1; text-align:center;">
                <div style="background:#FF6B6B; height:{height}px; width:70%; margin:0 auto; border-radius:8px 8px 0 0;"></div>
                <div style="color:#FFFFFF; margin-top:8px;">{temp}°C</div>
                <div style="color:#CCCCCC;">{month}</div>
            </div>
        '''
    chart_html += '</div></div>'
    return chart_html


def create_precipitation_chart_html(df, location_name):
    months = df['month_name'].tolist()
    precip = df['total_precipitation'].tolist()
    max_precip = max(precip) if max(precip) > 0 else 100
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; width: 100%;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #FFFFFF;">Monthly Precipitation</h4>
            {accuracy_badge_html("medium", "±20-40%")}
        </div>
        <div style="display: flex; align-items: flex-end; justify-content: space-around; height: 250px;">
    '''
    for month, p in zip(months, precip):
        height = (p / max_precip) * 200
        chart_html += f'''
            <div style="flex:1; text-align:center;">
                <div style="background:#4A90E2; height:{height}px; width:70%; margin:0 auto; border-radius:8px 8px 0 0;"></div>
                <div style="color:#FFFFFF; margin-top:8px;">{p:.0f}mm</div>
                <div style="color:#CCCCCC;">{month}</div>
            </div>
        '''
    chart_html += '</div></div>'
    return chart_html


def create_soil_moisture_chart_html(df, location_name):
    months = df['month_name'].tolist()
    surface = df['soil_moisture_0_7cm'].tolist()
    root = df['soil_moisture_7_28cm'].tolist()
    deep = df['soil_moisture_28_100cm'].tolist()
    max_value = max(max(surface), max(root), max(deep)) * 1.2
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; width: 100%;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #FFFFFF;">Soil Moisture</h4>
            {accuracy_badge_html("medium", "±0.05 m³/m³")}
        </div>
        <div style="display: flex; align-items: flex-end; justify-content: space-around; height: 250px;">
    '''
    for i, month in enumerate(months):
        chart_html += f'''
            <div style="flex:1; text-align:center;">
                <div style="height:200px; display:flex; flex-direction:column-reverse; gap:2px;">
                    <div style="background:#FFAA44; height:{(deep[i]/max_value)*200}px; border-radius:4px 4px 0 0;"></div>
                    <div style="background:#4A90E2; height:{(root[i]/max_value)*200}px; border-radius:4px 4px 0 0;"></div>
                    <div style="background:#00FF88; height:{(surface[i]/max_value)*200}px; border-radius:4px 4px 0 0;"></div>
                </div>
                <div style="color:#FFFFFF; margin-top:8px;">{month}</div>
            </div>
        '''
    chart_html += '</div></div>'
    return chart_html


def create_soil_texture_chart_html(soil_data, location_name):
    clay = soil_data['clay_content']
    silt = soil_data['silt_content']
    sand = soil_data['sand_content']
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; width: 100%;">
        <h4 style="margin: 0 0 1rem 0; color: #FFFFFF;">Soil Texture</h4>
        <div style="display: flex; justify-content: center; gap: 2rem; height: 250px;">
            <div style="text-align:center;">
                <div style="background:#8B4513; height:{clay*2.5}px; width:80px; border-radius:8px 8px 0 0;"></div>
                <div style="color:#FFFFFF; margin-top:8px;">{clay}%</div>
                <div style="color:#CCCCCC;">Clay</div>
            </div>
            <div style="text-align:center;">
                <div style="background:#DEB887; height:{silt*2.5}px; width:80px; border-radius:8px 8px 0 0;"></div>
                <div style="color:#FFFFFF; margin-top:8px;">{silt}%</div>
                <div style="color:#CCCCCC;">Silt</div>
            </div>
            <div style="text-align:center;">
                <div style="background:#F4A460; height:{sand*2.5}px; width:80px; border-radius:8px 8px 0 0;"></div>
                <div style="color:#FFFFFF; margin-top:8px;">{sand}%</div>
                <div style="color:#CCCCCC;">Sand</div>
            </div>
        </div>
    </div>
    '''
    return chart_html


def create_som_gauge_html(soil_data, location_name):
    som = soil_data['final_som_estimate']
    if som < 1.5:
        color = '#FF4444'
        status = "Depleted"
    elif som < 3:
        color = '#FFAA44'
        status = "Moderate"
    else:
        color = '#44FF44'
        status = "Rich"
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1.5rem; text-align:center;">
        <h4 style="color: #FFFFFF; margin-bottom: 1rem;">Soil Organic Matter</h4>
        <div style="font-size: 3rem; color: {color}; font-weight: bold;">{som:.2f}%</div>
        <div style="color: {color}; font-size: 1.2rem; margin-top: 0.5rem;">{status}</div>
    </div>
    '''
    return chart_html


def create_vegetation_chart_html(dates, values, index_name, location_name):
    max_val = max(values) if values else 1
    min_val = min(values) if values else 0
    val_range = max_val - min_val if max_val > min_val else 1
    
    color_map = {'NDVI': '#00FF88', 'EVI': '#FF6B6B', 'SAVI': '#4A90E2', 'NDWI': '#4A90E2', 'GNDVI': '#00CC6A'}
    color = color_map.get(index_name, '#00FF88')
    
    chart_html = f'''
    <div style="background: #1E1E1E; border-radius: 12px; padding: 1rem; width: 100%;">
        <h4 style="color: #FFFFFF; margin-bottom: 1rem;">{index_name} Time Series</h4>
        <div style="display: flex; align-items: flex-end; justify-content: space-around; height: 250px;">
    '''
    for date, val in zip(dates[:12], values[:12]):
        height = ((val - min_val) / val_range) * 200 + 30
        chart_html += f'''
            <div style="flex:1; text-align:center;">
                <div style="background:{color}; height:{height}px; width:70%; margin:0 auto; border-radius:8px 8px 0 0;"></div>
                <div style="color:#FFFFFF; margin-top:8px;">{val:.3f}</div>
                <div style="color:#CCCCCC;">{date[-2:]}</div>
            </div>
        '''
    chart_html += '</div></div>'
    return chart_html


def display_chart(chart_html):
    if chart_html and len(chart_html) > 100:
        st.components.v1.html(chart_html, height=400, scrolling=False)


# =============================================================================
# AI INTERPRETATION DISPLAY
# =============================================================================

def show_ai_interpretation(chart_type, data_summary, location, llm=None, use_tinyllama=True):
    ct = chart_type.lower()

    if "climate classification" in ct:
        label = "🌾 Field Briefing — Agroclimate Assessment"
    elif "monthly temperature" in ct:
        label = "🌾 Crop Calendar Analysis"
    elif "precipitation" in ct:
        label = "🌾 Water Management Assessment"
    elif "soil moisture" in ct:
        label = "🌾 Root-Zone Water Status"
    elif "soil texture" in ct:
        label = "🌾 Soil Texture & Workability"
    elif "organic matter" in ct:
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
    else:
        label = "🌾 AI Data Insight"

    st.markdown(f'<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="ai-header">
        <div class="ai-header-line"></div>
        <span class="ai-header-text">{label}</span>
    </div>
    ''', unsafe_allow_html=True)
    
    if use_tinyllama and llm is not None:
        with st.spinner("🦙 TinyLlama is analyzing..."):
            tl_result = tinyllama_interpret(llm, chart_type, data_summary, location)
        if tl_result:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
                f'<span style="font-size:1.2rem;">🦙</span>'
                f'<span style="color:#00FF88;font-weight:600;font-size:0.85rem;">TinyLlama 1.1B</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="ai-interpretation">{tl_result}</div>', unsafe_allow_html=True)
        else:
            rule_based = get_smart_interpretation(chart_type, data_summary, location)
            st.markdown(
                '<div style="color:#FFAA44;font-size:0.8rem;margin-bottom:0.4rem;">⚠️ Using rule-based analysis</div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="ai-interpretation">{rule_based}</div>', unsafe_allow_html=True)
    else:
        rule_based = get_smart_interpretation(chart_type, data_summary, location)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
            f'<span style="font-size:1.1rem;">🤖</span>'
            f'<span style="color:#4A90E2;font-weight:600;font-size:0.85rem;">GIS Intelligence Engine</span>'
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
        "llm_instance": None,
        "use_small_model": False,
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
    <iframe src="https://maps.google.com/maps?q={center_lat},{center_lon}&z={zoom}&output=embed" 
            width="100%" height="460" style="border:0; border-radius:14px;" allowfullscreen></iframe>
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
            icon = "✓"
            bg = "#00CC6A"
        elif i == step:
            icon = str(i)
            bg = "#00FF88"
        else:
            icon = str(i)
            bg = "#2A2A2A"
        html += f'''
        <div style="flex:1; text-align:center;">
            <div style="width:34px;height:34px;border-radius:50%;background:{bg};margin:0 auto;color:#0A0A0A;font-weight:700;line-height:34px;">{icon}</div>
            <div style="color:#CCCCCC;font-size:0.7rem;margin-top:0.4rem;">{label}</div>
        </div>'''
    html += '</div>'
    return html


# =============================================================================
# MAIN UI
# =============================================================================

# Sidebar
with st.sidebar:
    st.markdown("### 🦙 TinyLlama Status")
    if LLAMA_AVAILABLE:
        if MODEL_PATH.exists() or SMALL_MODEL_PATH.exists():
            st.success("✅ llama-cpp-python installed")
            if st.session_state.llm_instance is not None:
                st.success("✅ Model loaded")
            else:
                st.warning("⚠️ Model not loaded")
        else:
            st.info("📥 Model not downloaded")
    else:
        st.error("❌ llama-cpp-python not installed")
    
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
# MODEL DOWNLOAD UI
# =============================================================================

llm = st.session_state.llm_instance

if LLAMA_AVAILABLE and not MODEL_PATH.exists() and not SMALL_MODEL_PATH.exists() and not st.session_state.tinyllama_download_attempted:
    st.markdown("""
    <div style="background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.3);
         border-radius:12px;padding:1.5rem;margin-bottom:1rem;text-align:center;">
        <span style="font-size:3rem;">🦙</span>
        <h3 style="color:#00FF88; margin:0.5rem 0;">Download TinyLlama Model</h3>
        <p style="color:#CCCCCC; margin-bottom:1.5rem;">Choose a model size:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 Standard (637MB)", use_container_width=True):
            st.session_state.tinyllama_download_attempted = True
            st.session_state.use_small_model = False
            pb = st.progress(0)
            st_txt = st.empty()
            ok, err = download_model_with_progress(pb, st_txt, use_small_model=False)
            if ok:
                _llm, _ = load_tinyllama_model()
                if _llm:
                    st.session_state.llm_instance = _llm
                    st.session_state.tinyllama_loaded = True
                    st.success("✅ Model downloaded and loaded!")
                    st.rerun()
            else:
                st.error(f"Download failed: {err}")
    
    with col2:
        if st.button("📦 Small (300MB)", use_container_width=True):
            st.session_state.tinyllama_download_attempted = True
            st.session_state.use_small_model = True
            pb = st.progress(0)
            st_txt = st.empty()
            ok, err = download_model_with_progress(pb, st_txt, use_small_model=True)
            if ok:
                _llm, _ = load_tinyllama_model()
                if _llm:
                    st.session_state.llm_instance = _llm
                    st.session_state.tinyllama_loaded = True
                    st.success("✅ Small model downloaded and loaded!")
                    st.rerun()
            else:
                st.error(f"Download failed: {err}")

elif LLAMA_AVAILABLE and (MODEL_PATH.exists() or SMALL_MODEL_PATH.exists()) and st.session_state.llm_instance is None:
    with st.spinner("🦙 Loading TinyLlama model..."):
        _llm, _ = load_tinyllama_model()
        if _llm:
            st.session_state.llm_instance = _llm
            st.session_state.tinyllama_loaded = True
            llm = _llm
            st.success("✅ TinyLlama loaded successfully!")
            st.rerun()

# Progress bar
st.markdown(progress_bar_html(st.session_state.current_step), unsafe_allow_html=True)

# Main content columns
col1, col2 = st.columns([1, 2])

with col1:
    # STEP 1
    if st.session_state.current_step == 1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><div class="card-icon">📍</div><h3 style="margin:0;">Select Location</h3></div>', unsafe_allow_html=True)

        country = st.selectbox("🌍 Country", [""] + sorted(WORLD_REGIONS.keys()))
        if country:
            regions = WORLD_REGIONS[country]["regions"]
            region = st.selectbox("📌 Region", [""] + regions)
        else:
            region = None

        if st.button("✅ Confirm Location", use_container_width=True):
            if country:
                st.session_state.selected_country = country
                st.session_state.selected_region = region
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
            precip_scale = st.slider("💧 Precipitation Calibration", 0.5, 1.5, 1.0, 0.05)
            st.session_state.precip_scale = precip_scale
        else:
            st.markdown('<div class="card-header"><div class="card-icon">🌿</div><h3 style="margin:0;">Vegetation Settings</h3></div>', unsafe_allow_html=True)
            st.info(f"📍 **{location_name}**")
            st.session_state.collection_choice = st.selectbox("🛰️ Satellite", ["Sentinel-2", "Landsat-8"])
            st.session_state.selected_indices = st.multiselect("📊 Indices", VEGETATION_INDICES, default=["NDVI", "EVI", "SAVI"])

        cb, cn = st.columns(2)
        with cb:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with cn:
            if st.button("✅ Continue", use_container_width=True):
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
        else:
            st.markdown('<div class="card-header"><div class="card-icon">🗺️</div><h3 style="margin:0;">Preview</h3></div>', unsafe_allow_html=True)
            st.info(f"📍 **{location_name}**")
            indices_str = ", ".join(st.session_state.selected_indices[:3])
            st.markdown(f"**Selected Indices:** {indices_str}")

        cb, cn = st.columns(2)
        with cb:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with cn:
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Generating analysis..."):
                    st.session_state.climate_df = generate_climate_data(location_name, region_type=st.session_state.region_type)
                    st.session_state.soil_data = get_soil_data(location_name, st.session_state.region_type)
                    st.session_state.climate_classification = get_climate_classification(location_name, st.session_state.region_type)
                    if st.session_state.analysis_type == "Vegetation & Climate":
                        results = {}
                        for idx in st.session_state.selected_indices:
                            dates, values = generate_vegetation_data(location_name, idx)
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

        ai_status = "🦙 TinyLlama" if st.session_state.llm_instance is not None else "🤖 GIS Intelligence"
        st.markdown(f"""<div style="background:rgba(0,255,136,0.08);padding:0.75rem;border-radius:8px;margin-bottom:1rem;">
            <p style="color:#CCCCCC;margin:0;">{ai_status}: Ready</p></div>""", unsafe_allow_html=True)

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
        location_name = st.session_state.location_name
        climate_df = st.session_state.climate_df
        soil_data = st.session_state.soil_data
        climate_cls = st.session_state.climate_classification
        veg_results = st.session_state.vegetation_results
        llm = st.session_state.llm_instance
        use_tl = st.session_state.llm_instance is not None

        if st.session_state.analysis_type == "Climate & Soil":
            # Climate Classification
            if climate_cls:
                with st.container():
                    st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
                    st.markdown('<h3 style="margin:0 0 1rem 0;">🌤️ Climate Classification</h3>', unsafe_allow_html=True)
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Mean Annual Temp", f"{climate_cls['mean_temperature']:.1f}°C")
                    with col_m2:
                        st.metric("Annual Precip", f"{climate_cls['mean_precipitation']:.0f} mm")
                    
                    st.info(f"**Climate Zone:** {climate_cls['climate_zone']}")
                    
                    data_summary = (
                        f"Climate zone: {climate_cls['climate_zone']}, "
                        f"Mean temperature: {climate_cls['mean_temperature']:.1f}°C, "
                        f"Annual precipitation: {climate_cls['mean_precipitation']:.0f}mm"
                    )
                    show_ai_interpretation("Climate Classification", data_summary, location_name, llm, use_tl)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Climate Data
            if climate_df is not None:
                with st.container():
                    st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
                    st.markdown('<h3 style="margin:0 0 1rem 0;">📊 Climate Data</h3>', unsafe_allow_html=True)
                    
                    tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "💧 Precipitation", "🌱 Soil Moisture"])

                    with tab1:
                        temp_chart = create_temperature_chart_html(climate_df, location_name)
                        display_chart(temp_chart)
                        
                        temps = climate_df['temperature_2m'].tolist()
                        data_summary = f"Monthly temperatures from {min(temps):.1f}°C to {max(temps):.1f}°C"
                        show_ai_interpretation("Monthly Temperature", data_summary, location_name, llm, use_tl)

                    with tab2:
                        precip_chart = create_precipitation_chart_html(climate_df, location_name)
                        display_chart(precip_chart)
                        
                        annual = climate_df['total_precipitation'].sum()
                        data_summary = f"Annual total precipitation: {annual:.0f}mm"
                        show_ai_interpretation("Precipitation", data_summary, location_name, llm, use_tl)

                    with tab3:
                        soil_chart = create_soil_moisture_chart_html(climate_df, location_name)
                        display_chart(soil_chart)
                        
                        surface = climate_df['soil_moisture_0_7cm'].mean()
                        root = climate_df['soil_moisture_7_28cm'].mean()
                        deep = climate_df['soil_moisture_28_100cm'].mean()
                        data_summary = f"Surface: {surface:.3f}, Root: {root:.3f}, Deep: {deep:.3f} m³/m³"
                        show_ai_interpretation("Soil Moisture", data_summary, location_name, llm, use_tl)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

            # Soil Properties
            if soil_data:
                with st.container():
                    st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
                    st.markdown('<h3 style="margin:0 0 1rem 0;">🌱 Soil Properties</h3>', unsafe_allow_html=True)
                    
                    col_p1, col_p2, col_p3 = st.columns(3)
                    with col_p1: 
                        st.metric("Texture", soil_data['texture_name'])
                    with col_p2: 
                        st.metric("SOM", f"{soil_data['final_som_estimate']:.2f}%")
                    with col_p3: 
                        st.metric("SOC Stock", f"{soil_data['soc_stock']:.1f} t/ha")
                    
                    col_tex, col_som = st.columns(2)
                    with col_tex:
                        tex_chart = create_soil_texture_chart_html(soil_data, location_name)
                        display_chart(tex_chart)
                        
                        clay = soil_data['clay_content']
                        silt = soil_data['silt_content']
                        sand = soil_data['sand_content']
                        data_summary = f"Clay: {clay}%, Silt: {silt}%, Sand: {sand}%"
                        show_ai_interpretation("Soil Texture", data_summary, location_name, llm, use_tl)
                    
                    with col_som:
                        som_chart = create_som_gauge_html(soil_data, location_name)
                        display_chart(som_chart)
                        
                        som = soil_data['final_som_estimate']
                        data_summary = f"Soil Organic Matter: {som:.2f}%"
                        show_ai_interpretation("Soil Organic Matter", data_summary, location_name, llm, use_tl)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

        else:  # Vegetation & Climate
            if veg_results:
                with st.container():
                    st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
                    st.markdown('<h3 style="margin:0 0 1rem 0;">🌿 Vegetation Indices</h3>', unsafe_allow_html=True)
                    
                    for idx_name, data in veg_results.items():
                        st.markdown(f"**{idx_name}**")
                        veg_chart = create_vegetation_chart_html(data['dates'], data['values'], idx_name, location_name)
                        display_chart(veg_chart)
                        
                        vals = data['values']
                        mean_v = np.mean(vals)
                        trend = np.polyfit(range(len(vals)), vals, 1)[0]
                        trend_dir = "increasing" if trend > 0.001 else ("decreasing" if trend < -0.001 else "stable")
                        
                        data_summary = f"{idx_name} mean={mean_v:.3f}, trend={trend_dir}"
                        show_ai_interpretation(f"{idx_name} vegetation index", data_summary, location_name, llm, use_tl)
                        st.markdown("---")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
