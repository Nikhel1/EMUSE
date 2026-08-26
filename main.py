# ================================
# 🔹 QUERY NORMALIZATION + LLM SETUP
# ================================

import os
import ast
import re
import json
import urllib.request

import fnmatch
import getpass
import glob
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings


import boto3

import gdown
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
import astropy.units as u
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

warnings.simplefilter("ignore", category=AstropyWarning)

EMU_S3_BUCKET = os.environ.get("EMU_S3_BUCKET", "emu-data-bucket-2026")
EMU_S3_IMAGES_PREFIX = os.environ.get("EMU_S3_IMAGES_PREFIX", "images").strip("/") + "/"
EMU_IMAGES_DIR = os.environ.get("EMU_IMAGES_DIR", "").strip()
CUTOUT_SIZE_ARCMIN = 30.0          # default radio cutout = 30 arcmin
OPTICAL_CUTOUT_SIZE_ARCMIN = 15.0  # optical/IR cutout max size = 15 arcmin
PIXEL_SIZE_ARCSEC = 2.0
CUTOUT_TTL_SEC = int(os.environ.get("EMU_CUTOUT_TTL_SEC", str(60 * 60)))  # 1 hour
TILE_CACHE_TTL_SEC = int(os.environ.get("EMU_TILE_TTL_SEC", str(6 * 60 * 60)))  # 6 hours


# ---- SYNONYM MAP (expand this over time) ----
SYNONYM_MAP = {
    "bt": "bent-tailed radio galaxy",
    "bt galaxy": "bent-tailed radio galaxy",
    "bent tail": "bent-tailed radio galaxy",
    "wat": "wide-angle tail radio galaxy",
    "nat": "narrow-angle tail radio galaxy",
    "fr1": "FR-I radio galaxy",
    "fr-1": "FR-I radio galaxy",
    "fri": "FR-I radio galaxy",
    "fr2": "FR-II radio galaxy",
    "fr-2": "FR-II radio galaxy",
    "frii": "FR-II radio galaxy",
    "xrg": "X-shaped radio galaxy",
    "x-shaped": "X-shaped radio galaxy",
    "compact": "compact radio galaxy",
}

def normalize_query(query: str) -> str:
    # Lowercase, remove punctuation, and collapse spacing for robust matching.
    q = re.sub(r"[^a-z0-9\s\-]", " ", query.lower())
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"^(a|an|the)\s+", "", q)

    # Match longer keys first so specific phrases win over short tokens.
    for key in sorted(SYNONYM_MAP, key=len, reverse=True):
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, q):
            return SYNONYM_MAP[key]
    return q if q else query

# ================================
# 🤖 🔹 GEMINI API INTEGRATION (GOOGLE GENERATIVE AI)
# ================================
# 🔹 GEMINI QUERY EXPANSION
# ================================

try:
    import google.generativeai as genai
    GEMINI_IMPORT_OK = True
except Exception:
    genai = None
    GEMINI_IMPORT_OK = False

def gemini_generate_content(prompt: str, api_key: str):
    resolved_key = (api_key or "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not resolved_key:
        raise RuntimeError("No Gemini API key found. Provide it in the sidebar or set GOOGLE_API_KEY.")
    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={resolved_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ]
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates in Gemini response: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts or "text" not in parts[0]:
            raise RuntimeError(f"Missing text in Gemini response: {data}")
        return parts[0]["text"]
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}") from e

def expand_query_gemini(user_query: str, api_key: str):
    """
    Expand user query into multiple descriptive phrases using Gemini.
    Falls back safely if API fails.
    """
    prompt = f"""
You are an expert in radio astronomy.
Expand the following query into 3–5 short descriptive search phrases
for radio galaxy morphology.
Rules:
- Expand abbreviations (FR2 → FR-II radio galaxy, BT → bent-tailed galaxy)
- Include morphology (jets, lobes, tails)
- Keep phrases concise
- Do NOT explain anything
Return ONLY a Python list of strings.
Query: "{user_query}"
"""
    try:
        response = gemini_generate_content(prompt, api_key=api_key)
        text = str(response).strip()

        # Gemini may return markdown/code-fenced content; extract list robustly.
        cleaned = text.replace("```python", "```").replace("```json", "```").strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()

        parsed = None
        try:
            parsed = ast.literal_eval(cleaned)
        except Exception:
            # Try to recover by slicing only the first bracketed list.
            lb = cleaned.find("[")
            rb = cleaned.rfind("]")
            if lb != -1 and rb != -1 and rb > lb:
                snippet = cleaned[lb:rb + 1]
                parsed = ast.literal_eval(snippet)

        if isinstance(parsed, list):
            expanded = [str(x).strip() for x in parsed if str(x).strip()]
            if expanded:
                return expanded[:5], True, "Gemini expansion used successfully."
    except Exception as e:
        return [user_query], False, f"Gemini failed: {e}"
    return [user_query], False, f"Gemini returned an unexpected response format: {text[:200]}"

# ⚡ 🔹 CACHE (IMPORTANT for free-tier limits)
@st.cache_data(show_spinner=False)
def cached_expand_query(query, api_key):
    return expand_query_gemini(query, api_key)

# 🧠 🔹 QUERY PIPELINE (CORE FUNCTION)
def build_text_query(
    search_for,
    tokenizer,
    model,
    use_gemini_llm=True,
    force_gemini=False,
    gemini_api_key="",
):
    """
    Full pipeline:
    user input → normalization → LLM expansion → embedding
    """
    # ---- Step 1: normalize ----
    normalized = normalize_query(search_for)
    normalized = f"An image of {normalized}"

    # ---- Step 2: decide if LLM needed ----
    use_llm = True
    if len(search_for) < 4:
        use_llm = False

    # ---- Step 3: expand ----
    llm_status = "Gemini expansion disabled."
    llm_used = False
    if use_llm and use_gemini_llm:
        expanded, llm_used, llm_status = cached_expand_query(normalized, gemini_api_key)
        if not llm_used:
            llm_status = (
                "Gemini API key not provided or wrong. "
                "Using SYNONYM_MAP fallback for query interpretation. "
                "You can get a free Gemini API key from https://aistudio.google.com/ "
                "for better text-query interpretations."
            )
        if force_gemini and not llm_used:
            raise RuntimeError(f"Force Gemini is enabled but expansion failed. {llm_status}")
    else:
        expanded = [normalized]
        if len(search_for) < 4:
            llm_status = "Gemini skipped because query is too short."

    def _clean_phrase(s: str) -> str:
        s = re.sub(r"\s+", " ", str(s).strip())
        s = re.sub(r"\bradio galaxy\s+radio galaxy\b", "radio galaxy", s, flags=re.IGNORECASE)
        return s

    def _append_unique(dst, item):
        cleaned = _clean_phrase(item)
        if not cleaned:
            return
        if cleaned.lower() not in {x.lower() for x in dst}:
            dst.append(cleaned)

    # ---- Step 4: include normalized first, then useful variants ----
    merged = []
    _append_unique(merged, normalized)
    _append_unique(merged, search_for)
    for q in expanded:
        _append_unique(merged, q)

    expanded = merged[:6]  # keep small and deterministic

    return expanded, llm_used, llm_status

# -------------- MODIFIED Gemini Table Assistant (ENTER key to ask) --------------

def render_gemini_table_assistant(table_df, api_key):
    """Interactive Gemini Q&A on the current result table with memory; submit with Enter key."""
    if table_df is None or table_df.empty:
        return

    table_signature = f"{len(table_df)}::{','.join(table_df.columns)}::{table_df.head(5).to_csv(index=False)}"
    if st.session_state.get("table_chat_signature") != table_signature:
        st.session_state.table_chat_signature = table_signature
        st.session_state.table_chat_messages = []
        st.session_state.table_assistant_open = False

    if not st.session_state.get("table_assistant_open", False):
        return

    st.markdown("### Ask Gemini about this table")
    st.caption(
        "Ask about any source (RA/Dec), host details, trends, or follow-up targets. "
        "Conversation context is remembered."
    )

    chat_messages = st.session_state.get("table_chat_messages", [])
    if chat_messages:
        for msg in chat_messages[-10:]:
            role_label = "You" if msg["role"] == "user" else "Gemini"
            st.markdown(f"**{role_label}:** {msg['content']}")

    # Helper: get or set a counter to force widget refresh
    if "table_chat_input_refresh_counter" not in st.session_state:
        st.session_state.table_chat_input_refresh_counter = 0

    clear_btn = st.button("Clear table chat", key="clear_table_chat_btn")
    if clear_btn:
        st.session_state.table_chat_messages = []
        st.session_state.table_chat_input_refresh_counter += 1
        st.rerun()

    chat_input_key = f"table_chat_user_input_{st.session_state.table_chat_input_refresh_counter}"
    # New: st.text_input with on_change triggers Gemini query. Submit with Enter.
    user_question = st.text_input(
        "Ask a question about this table",
        key=chat_input_key,
        placeholder="e.g. Discuss source at RA 12.34567 Dec -45.67890 and likely host.",
        label_visibility="visible",
    )

    # We want to "ask Gemini" whenever user enters a non-empty question and submits with Enter.
    if user_question is not None and user_question.strip() != "":
        # Avoid firing on every rerun: only when st.text_input is "fresh" (i.e. changed)
        # We'll use a cache of last question to avoid asking Gemini multiple times per rerun
        last_asked_key = f"table_last_asked_{chat_input_key}"
        if st.session_state.get(last_asked_key, None) != user_question:
            # Save to session that we are processing this question
            st.session_state[last_asked_key] = user_question

            question = user_question.strip()
            table_context = table_df.head(200).to_csv(index=False)
            history_context = "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in chat_messages[-8:]]
            )
            prompt = (
                "You are an astronomy assistant helping users interpret an EMU similar-sources table.\n"
                "Use only the provided table context and chat context.\n"
                "If asked for details not present in the table, say what is missing clearly.\n\n"
                f"TABLE COLUMNS: {list(table_df.columns)}\n"
                f"TABLE ROW COUNT: {len(table_df)}\n"
                f"TABLE DATA (CSV, first up to 200 rows):\n{table_context}\n\n"
                f"CHAT HISTORY:\n{history_context}\n\n"
                f"USER QUESTION:\n{question}\n"
            )

            with st.spinner("Gemini is analyzing the table..."):
                try:
                    answer = gemini_generate_content(prompt, api_key)
                    answer_text = str(answer).strip()
                    st.session_state.table_chat_messages = chat_messages + [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer_text},
                    ]
                    # Force rerun with updated messages AND clear text_input.
                    st.session_state.table_chat_input_refresh_counter += 1
                    st.rerun()
                except Exception:
                    st.warning(
                        "Gemini API key not provided or wrong. Using table view only. "
                        "Get a free Gemini API key from https://aistudio.google.com/."
                    )
                    # Clean input field after warning (to let user try again)
                    st.session_state.table_chat_input_refresh_counter += 1
                    st.rerun()
    # End MODIFIED Gemini Table Assistant

# Set page configuration
st.set_page_config(
    page_title="EMUSE - Evolutionary Map of the Universe Search Engine",
    page_icon="🔭",
    layout="wide"
)

# Custom CSS to improve the app's appearance
st.markdown("""
    <style>
    .stApp {
        max-width: auto;
        margin: 0 auto;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: dark;
        color: white;
        font-weight: bold;
    }
    .stSlider>div>div>div>div {
        background-color: dark;
    }
    /* Custom horizontal separator between Gemini Table Assistant and cutout section */
    .styled-divider {
        border-top: 2px solid #5D6D7E;
        margin: 3rem 0 2rem 0;
        width: 100%;
        opacity: 0.6;
    }
    </style>
    """, unsafe_allow_html=True)

# Display EMU logo
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("emu.png", use_container_width=True)

st.markdown("""
            <div style='text-align: center;'>
                <h1 style='color: #2E4053; margin-bottom: 0; font-size: 3em; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
                    EMUSE
                </h1>
                <h2 style='color: #566573; font-size: 1.5em; margin-top: 0; font-weight: 400;'>
                    Evolutionary Map of the Universe Search Engine
                </h2>
                <div style='text-align: center; margin: 0; line-height: 1.6; color: #34495E; font-size: 1.1em;'>
                    Welcome to EMUSE – a powerful search tool for the <a href="https://emu-survey.org/" target="_blank">EMU Survey</a> conducted with the 
                    <a href="https://www.csiro.au/en/about/facilities-collections/ATNF/ASKAP-radio-telescope" target="_blank">ASKAP telescope</a>.
                    The app leverages advanced AI tools to match your queries with objects in the EMU Survey database.
                    Find similar radio objects by using either text descriptions or uploading reference images.
                    <br><br>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Load the model and data
@st.cache_resource
def load_model_and_data():
    #model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir='./clip_pretrained/')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
    
    #model_url =  f'https://drive.google.com/uc?id=1e1O-5774mkoGYZYC1gsXiGqDeu7KtOGs'
    model_url =  f'https://drive.google.com/uc?id=1k0MNw1hyBDejxOovKwhQCPRmJil13ut5'
    model_file = 'epoch_99.pt'
    gdown.download(model_url, model_file, quiet=False)
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    feature_url =  f'https://drive.google.com/uc?id=1ihgHSS043G60ozg6v32rYUJJFx1uqs_H' # First Year, ~160 tiles
    #feature_url =  f'https://drive.google.com/uc?id=11l-iVak_8QnycuePIvPwXbDBUP_ILP_Y' # Observations till June 2025
    feature_file = 'all_sbid_image_features.pt'
    gdown.download(feature_url, feature_file, quiet=False)
    all_image_features = torch.load(feature_file)

    #idx_url =  f'https://drive.google.com/uc?id=1o-JWXmfUN1F6VMO6Lq-5U69qLDpyEMQ-'
    #idx_file = 'allidx_sbid_ra_dec.pkl'
    idx_url =  f'https://drive.google.com/uc?id=14fwWW3KkkRfhAyaBVQeEKszx2iGLTCJc'  # First Year, ~160 tiles 
    #idx_url =  f'https://drive.google.com/uc?id=1rI1RzKDMMKrOyeE_7BaCNthYrYgYoRf8'  # Observations till June 2025
    idx_file = 'allidx_sbid_ra_dec_flux_catwise.pkl'
    gdown.download(idx_url, idx_file, quiet=False)
    idx_dict = pd.read_pickle(idx_url)
    return model, preprocess, tokenizer, all_image_features, idx_dict

model, preprocess, tokenizer, all_image_features, idx_dict = load_model_and_data()

# Input options
st.sidebar.header("Search Options")
input_option = st.sidebar.radio("Choose input type:", ("Image", "Text"))

# Common parameters
remove_galactic = st.sidebar.checkbox("Remove galactic sources", value=True)
above_prob_of = st.sidebar.slider("Minimum probability", 0.0, 1.0, 0.9, 0.01)
_n_filtered = st.session_state.get("n_filtered_sources", 50000)
_top_n_max = max(1, _n_filtered)
_top_n_default = min(200, _top_n_max)
top_n = st.sidebar.slider("Number of top results to display", 1, _top_n_max, _top_n_default)
use_gemini_llm = False
gemini_api_key = ""
if "gemini_api_key_saved" not in st.session_state:
    st.session_state.gemini_api_key_saved = ""
st.sidebar.markdown("---")
st.sidebar.subheader("Gemini Settings")
if input_option == "Text":
    use_gemini_llm = st.sidebar.checkbox("Use Gemini", value=True)
#else:
#    st.sidebar.caption("Image search does not require Gemini.")
#    st.sidebar.caption("Gemini is used for table Q&A assistant only.")
gemini_api_key_input = st.sidebar.text_input(
    "Gemini API key",
    value=st.session_state.get("gemini_api_key_saved", ""),
    type="password",
    key="gemini_api_key_input",
    help="Needed for Gemini-powered interpretation and table assistant. Get a free key at https://aistudio.google.com/",
)
if gemini_api_key_input.strip():
    st.session_state.gemini_api_key_saved = gemini_api_key_input.strip()
gemini_api_key = st.session_state.get("gemini_api_key_saved", "")
if gemini_api_key:
    st.sidebar.caption("Gemini API key saved for this session.")

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ How to Use EMUSE"):
    st.markdown("""
    ### Search Methods

    #### 🔤 Text Search
    - Select **Text** from the input type options
    - Enter a description of the object (e.g. *"A bent tailed radio galaxy"*)
    - Optionally enable **Gemini** to expand and interpret your query
    - Review the interpreted queries, tick the ones to use, or enter custom ones
    - Click **Search**

    #### 🖼 Image Search
    - Select **Image** from the input type options
    - Upload a reference image (`.jpg`, `.jpeg`, or `.png`)
    - Click **Search** to find visually similar sources

    ---
    ### Search Parameters

    **Remove Galactic Sources**
    - Filters out objects within 10° of the galactic plane
    - Recommended for most extragalactic searches

    **Minimum Probability**
    - Confidence threshold for matches (0.0 – 1.0)
    - Higher = fewer but more precise results

    **Number of Top Results**
    - Slider max updates automatically after each search to match the number of sources found above the chosen probability
    - Default: 200

    ---
    ### Results Table

    After a search, three action buttons appear below the table:

    - **⬇ Download Table** — saves the visible results as a CSV
    - **🖼 Generate Cutout** — opens the cutout workflow for a selected source
    - **🤖 Gemini Assistant** — chat with Gemini about the result table

    ---
    ### Cutout Generation

    - If EMU tile data is accessible (local or S3), cutouts are generated directly
    - If not, enter your **CASDA OPAL credentials** to fetch cutouts via CASDA
    - Provide SBID, RA, Dec, and cutout size, then click **Create this cutout**
    - After the radio cutout is generated, a centred **⬇ Download Radio FITS** button appears

    #### Optical / IR Overlay
    - Fetch Legacy Survey (optical) or unWISE (IR) data to overlay on the radio cutout
    - Once fetched, three download buttons appear: **Radio FITS**, **Optical FITS**, **IR FITS**
    """)

st.sidebar.markdown(
    """
    <div style='text-align: center;'>
        <p style='color: #34495E; font-size: 0.9em; margin-top: 20px;'>
            &copy; Nikhel Gupta | CSIRO
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Persist search results/cutout state across reruns
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "cutout_fig_path" not in st.session_state:
    st.session_state.cutout_fig_path = None
if "show_search_results" not in st.session_state:
    st.session_state.show_search_results = False
if "cutout_size_arcmin" not in st.session_state:
    st.session_state.cutout_size_arcmin = CUTOUT_SIZE_ARCMIN
if "cutout_tile_cache" not in st.session_state:
    # Caches the raw tile data array + WCS for a source so size changes don't re-stream S3.
    # Shape: {"sbid": str, "ra": float, "dec": float, "data": np.ndarray, "wcs": WCS}
    st.session_state.cutout_tile_cache = None
# --- Multiwavelength / optical overlay state ---
if "optical_hdu" not in st.session_state:
    st.session_state.optical_hdu = None        # reprojected 3-band optical HDU
if "optical_fits_bytes" not in st.session_state:
    st.session_state.optical_fits_bytes = None  # raw bytes for download
if "optical_fits_filename" not in st.session_state:
    st.session_state.optical_fits_filename = None
if "optical_layer" not in st.session_state:
    st.session_state.optical_layer = None      # "ls-dr11" | "unwise-neo7" | None
if "ir_hdu" not in st.session_state:
    st.session_state.ir_hdu = None             # unWISE HDU (if separately fetched)
if "ir_fits_bytes" not in st.session_state:
    st.session_state.ir_fits_bytes = None
if "ir_fits_filename" not in st.session_state:
    st.session_state.ir_fits_filename = None
if "optical_fits_path" not in st.session_state:
    st.session_state.optical_fits_path = None
if "ir_fits_path" not in st.session_state:
    st.session_state.ir_fits_path = None
if "multiwave_layer_choice" not in st.session_state:
    st.session_state.multiwave_layer_choice = "Radio only"
if "multiwave_radio_pct_lo" not in st.session_state:
    st.session_state.multiwave_radio_pct_lo = 95.0
if "multiwave_radio_pct_hi" not in st.session_state:
    st.session_state.multiwave_radio_pct_hi = 99.9
if "multiwave_optical_ra" not in st.session_state:
    # Track which (ra, dec, size, layer) the optical cache corresponds to
    st.session_state.multiwave_optical_ra = None
if "multiwave_optical_dec" not in st.session_state:
    st.session_state.multiwave_optical_dec = None
if "multiwave_optical_size" not in st.session_state:
    st.session_state.multiwave_optical_size = None
if "fetch_optical_ir_requested" not in st.session_state:
    st.session_state.fetch_optical_ir_requested = False
if "optical_ir_size_arcmin" not in st.session_state:
    st.session_state.optical_ir_size_arcmin = OPTICAL_CUTOUT_SIZE_ARCMIN
if "display_zoom_arcmin" not in st.session_state:
    st.session_state.display_zoom_arcmin = CUTOUT_SIZE_ARCMIN

sb_ra_dec = None
filtered_probs = None
df_cleaned = None

def run_text_similarity_search(query_list):
    # Anchor-free similarity: use only selected/edited user queries.
    effective_queries = [q for q in query_list if str(q).strip()]
    if not effective_queries:
        effective_queries = ["radio galaxy"]

    text_token = tokenizer(effective_queries)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_token)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_scores = (100.0 * all_image_features @ text_features.T)

    text_scores_np = text_scores.numpy()
    if text_scores_np.ndim == 1:
        text_scores_np = text_scores_np.reshape(-1, 1)

    # Normalize each query column independently first, then combine with max.
    # This makes adding more queries monotonic w.r.t. thresholding.
    col_min = text_scores_np.min(axis=0, keepdims=True)
    col_max = text_scores_np.max(axis=0, keepdims=True)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    text_probs_by_query = (text_scores_np - col_min) / denom
    target_probs = text_probs_by_query.max(axis=1)

    idx_above_prob = np.where(target_probs > above_prob_of)[0]
    idx_above_prob_sorted = idx_above_prob[
        np.argsort(target_probs[idx_above_prob].flatten())[::-1]
    ]

    # Enforce uniqueness by sky position (RA, Dec), keeping highest-probability entry.
    sb_ra_dec_local = []
    filtered_probs_local = []
    seen_ra_dec = set()
    for idx in idx_above_prob_sorted:
        sb_entry = idx_dict.get(idx, "Key not found")
        try:
            sb_parts = sb_entry.split("_")
            ra = float(sb_parts[1])
            dec = float(sb_parts[2])
            ra_dec_key = (round(ra, 7), round(dec, 7))
        except Exception:
            # Fallback: if format is unexpected, treat full entry as key.
            ra_dec_key = (str(sb_entry),)

        if ra_dec_key in seen_ra_dec:
            continue
        seen_ra_dec.add(ra_dec_key)
        sb_ra_dec_local.append(sb_entry)
        filtered_probs_local.append(float(target_probs[idx]))

    filtered_probs_local = np.array(filtered_probs_local)
    return sb_ra_dec_local, filtered_probs_local

def reset_interpreted_query_widget_state():
    keys_to_remove = [
        k for k in list(st.session_state.keys())
        if k.startswith("query_checkbox_") or k.startswith("query_text_")
    ]
    for k in keys_to_remove:
        del st.session_state[k]

def prepare_cutout_preview(data_plot):
    """Create a high-contrast preview image for Streamlit display."""
    arr = np.nan_to_num(np.array(data_plot, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    # Robust contrast stretch to reveal faint background features.
    p_low, p_high = np.percentile(arr, [5, 99.5])
    if p_high <= p_low:
        p_low, p_high = arr.min(), arr.max()
    if p_high <= p_low:
        return np.zeros_like(arr, dtype=float)
    arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)
    return arr


def get_downloads_root():
    root = os.path.join(os.getcwd(), "Downloads")
    os.makedirs(root, exist_ok=True)
    return root




def _streamlit_runtime_session_id():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is not None:
            return ctx.session_id
    except Exception:
        pass
    return None


def _active_streamlit_session_ids():
    """Return live Streamlit websocket session ids, or None if unavailable."""
    try:
        from streamlit.runtime import get_instance
        runtime = get_instance()
        mgr = getattr(runtime, "_session_mgr", None)
        if mgr is None:
            return None
        if hasattr(mgr, "list_active_sessions"):
            sessions = mgr.list_active_sessions()
        elif hasattr(mgr, "list_sessions"):
            sessions = mgr.list_sessions()
        else:
            return None
        ids = set()
        for item in sessions:
            sid = getattr(item, "session_id", None)
            session_obj = getattr(item, "session", None)
            if sid is None and session_obj is not None:
                sid = getattr(session_obj, "id", None)
            if sid:
                ids.add(sid)
        return ids
    except Exception:
        return None


def get_session_cutout_dir():
    if "emuse_session_id" not in st.session_state:
        st.session_state.emuse_session_id = uuid.uuid4().hex
    path = os.path.join(get_downloads_root(), "sessions", st.session_state.emuse_session_id)
    os.makedirs(path, exist_ok=True)
    owner = _streamlit_runtime_session_id()
    if owner:
        try:
            with open(os.path.join(path, ".owner"), "w", encoding="utf-8") as f:
                f.write(owner)
        except OSError:
            pass
    return path


def _path_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _session_owner_id(session_dir):
    owner_path = os.path.join(session_dir, ".owner")
    try:
        with open(owner_path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def cleanup_expired_downloads(cutout_ttl=CUTOUT_TTL_SEC, tile_ttl=TILE_CACHE_TTL_SEC):
    """Remove stale session cutouts and old cached tiles to limit disk use.

    Full EMU tiles stay in a shared cache (default 6h) because S3 transfer
    dominates cost. Small per-session FITS cutouts are removed after TTL or
    when their Streamlit session is gone (refresh / closed tab).
    """
    root = get_downloads_root()
    now = time.time()
    active_runtime = _active_streamlit_session_ids()
    sessions_root = os.path.join(root, "sessions")
    if os.path.isdir(sessions_root):
        for name in os.listdir(sessions_root):
            path = os.path.join(sessions_root, name)
            if not os.path.isdir(path):
                continue
            expired = (now - _path_mtime(path)) > cutout_ttl
            owner = _session_owner_id(path)
            orphaned = (
                active_runtime is not None
                and owner is not None
                and owner not in active_runtime
            )
            if expired or orphaned:
                shutil.rmtree(path, ignore_errors=True)
    tile_dir = os.path.join(root, "emu_tiles")
    if os.path.isdir(tile_dir):
        for name in os.listdir(tile_dir):
            path = os.path.join(tile_dir, name)
            if not os.path.isfile(path):
                continue
            if name.startswith(".s3_fits_index"):
                continue
            if (now - _path_mtime(path)) > tile_ttl:
                try:
                    os.remove(path)
                except OSError:
                    pass
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        if name.startswith("cutout_") or (now - _path_mtime(path)) > cutout_ttl:
            try:
                os.remove(path)
            except OSError:
                pass


def clear_optical_state():
    """Delete optical/IR FITS files from the session folder and clear all related session state."""
    # Delete files from disk first, while the paths are still in session state
    for path_key in ("optical_fits_path", "ir_fits_path"):
        _path = st.session_state.get(path_key)
        if _path and os.path.isfile(_path):
            try:
                os.remove(_path)
            except OSError:
                pass
    for key in [
        "optical_hdu", "optical_fits_bytes", "optical_fits_filename", "optical_fits_path",
        "optical_layer",
        "ir_hdu", "ir_fits_bytes", "ir_fits_filename", "ir_fits_path",
        "multiwave_optical_ra", "multiwave_optical_dec", "multiwave_optical_size",
        "fetch_optical_ir_requested",
    ]:
        st.session_state.pop(key, None)
    st.session_state.optical_hdu = None
    st.session_state.optical_fits_bytes = None
    st.session_state.optical_fits_filename = None
    st.session_state.optical_fits_path = None
    st.session_state.optical_layer = None
    st.session_state.ir_hdu = None
    st.session_state.ir_fits_bytes = None
    st.session_state.ir_fits_filename = None
    st.session_state.ir_fits_path = None
    st.session_state.multiwave_optical_ra = None
    st.session_state.multiwave_optical_dec = None
    st.session_state.multiwave_optical_size = None
    st.session_state.fetch_optical_ir_requested = False


def clear_session_cutout_state_and_files():
    """Clear this browser session's cutout files and related UI state."""
    try:
        if "emuse_session_id" in st.session_state:
            session_dir = os.path.join(
                get_downloads_root(), "sessions", st.session_state.emuse_session_id
            )
            if os.path.isdir(session_dir):
                shutil.rmtree(session_dir, ignore_errors=True)
    except Exception:
        pass
    for key in [
        "cutout_fig_path",
        "cutout_files",
        "cutout_previews",
        "cutout_meta",
        "cutout_downloads_dir",
        "cutout_generated_at",
        "pending_single_cutout",
        "cutout_selector_ready",
        "cutout_flow_active",
        "pending_emu_cutout_generation",
        "show_credential_fields",
        "casda_ready",
        "cutout_source_choice",
        "cutout_tile_cache",
        "cutout_source_choice_prev",
        "cutout_edit_sbid",
        "cutout_edit_ra",
        "cutout_edit_dec",
    ]:
        st.session_state.pop(key, None)
    st.session_state.cutout_previews = []
    st.session_state.cutout_meta = []
    st.session_state.cutout_files = []
    clear_optical_state()


def bootstrap_download_cleanup():
    """Run once per browser session (including refresh) and apply TTLs."""
    if "emuse_download_bootstrapped" in st.session_state:
        if time.time() - st.session_state.get("emuse_last_ttl_cleanup", 0) > 60:
            cleanup_expired_downloads()
            st.session_state.emuse_last_ttl_cleanup = time.time()
        generated_at = st.session_state.get("cutout_generated_at")
        if generated_at and (time.time() - generated_at) > CUTOUT_TTL_SEC:
            clear_session_cutout_state_and_files()
        return
    cleanup_expired_downloads()
    st.session_state.emuse_download_bootstrapped = True
    st.session_state.emuse_last_ttl_cleanup = time.time()
    if "emuse_session_id" not in st.session_state:
        st.session_state.emuse_session_id = uuid.uuid4().hex


def probe_emu_images_source():
    """Check whether EMU tiles are available locally or via S3."""
    if EMU_IMAGES_DIR and os.path.isdir(EMU_IMAGES_DIR):
        fits_files = glob.glob(os.path.join(EMU_IMAGES_DIR, "*.fits"))
        if fits_files:
            return {
                "available": True,
                "mode": "local",
                "images_dir": EMU_IMAGES_DIR,
                "message": (
                    f"S3 is available via local images directory "
                    f"{EMU_IMAGES_DIR} ({len(fits_files)} FITS files)."
                ),
                "needs_login": False,
            }

    try:
        s3 = boto3.client("s3")
        s3.head_bucket(Bucket=EMU_S3_BUCKET)
        resp = s3.list_objects_v2(
            Bucket=EMU_S3_BUCKET,
            Prefix=EMU_S3_IMAGES_PREFIX,
            MaxKeys=1,
        )
        if "Contents" not in resp:
            return {
                "available": False,
                "mode": None,
                "reason": "empty_prefix",
                "needs_login": False,
                "message": (
                    f"S3 bucket is not accessible "
                    f"(bucket {EMU_S3_BUCKET} reachable but {EMU_S3_IMAGES_PREFIX} has no objects)."
                ),
            }
        return {
            "available": True,
            "mode": "s3",
            "bucket": EMU_S3_BUCKET,
            "prefix": EMU_S3_IMAGES_PREFIX,
            "needs_login": False,
            "message": f"S3 is available: s3://{EMU_S3_BUCKET}/{EMU_S3_IMAGES_PREFIX}",
        }
    except NoCredentialsError as e:
        return {
            "available": False,
            "mode": None,
            "reason": "no_credentials",
            "needs_login": True,
            "message": f"S3 bucket is not accessible ({e}).",
        }
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        needs_login = code in {
            "ExpiredToken",
            "InvalidClientTokenId",
            "UnrecognizedClientException",
            "AuthFailure",
            "InvalidAccessKeyId",
            "RequestExpired",
        }
        return {
            "available": False,
            "mode": None,
            "reason": code or "client_error",
            "needs_login": needs_login or (not aws_is_logged_in()),
            "message": f"S3 bucket is not accessible ({e}).",
        }
    except BotoCoreError as e:
        return {
            "available": False,
            "mode": None,
            "reason": "botocore",
            "needs_login": not aws_is_logged_in(),
            "message": f"S3 bucket is not accessible ({e}).",
        }


def aws_is_logged_in():
    """Return True if AWS credentials can call STS GetCallerIdentity."""
    try:
        boto3.client("sts").get_caller_identity()
        return True
    except Exception:
        return False


def run_aws_login(remote=False, output_placeholder=None):
    """Run aws login and return (ok, combined_output). Streams into output_placeholder when given."""
    cmd = ["aws", "login"]
    if remote:
        cmd.append("--remote")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if output_placeholder is not None:
                output_placeholder.code("".join(lines), language="bash")
        returncode = proc.wait(timeout=600)
        output = "".join(lines).strip() or f"(no output; exit code {returncode})"
        if output_placeholder is not None:
            output_placeholder.code(output, language="bash")
        return returncode == 0, output
    except FileNotFoundError:
        return False, "aws CLI not found. Install AWS CLI v2.32.0+ and ensure it is on PATH."
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        return False, "aws login timed out after 10 minutes."
    except Exception as e:
        return False, f"aws login failed: {e}"


_TILE_CACHE_SIZE_ARCMIN = 5.0  # size of the data slab fetched from S3 / disk once per source

LEGACY_SURVEY_FITS_URL = (
    "https://www.legacysurvey.org/viewer/fits-cutout"
    "?ra={ra:.6f}&dec={dec:.6f}&layer={layer}&size={size_px}&pixscale={pixscale}"
)
# 0.262 arcsec/pixel is the native Legacy Survey pixscale
LEGACY_PIXSCALE = 0.262


def fetch_legacy_cutout(ra, dec, size_arcmin, layer="ls-dr11", timeout=30, stop_event=None):
    """Fetch a multi-band FITS cutout from the Legacy Survey viewer.

    Returns (hdul, raw_bytes) or raises an exception.
    size_arcmin is converted to pixels at LEGACY_PIXSCALE arcsec/pixel.
    stop_event: optional threading.Event; if set before the request, raises InterruptedError.
    """
    from io import BytesIO
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Fetch stopped by user.")
    size_px = max(64, int(round(size_arcmin * 60.0 / LEGACY_PIXSCALE)))
    size_px = min(size_px, 3000)   # viewer hard cap
    url = LEGACY_SURVEY_FITS_URL.format(
        ra=ra, dec=dec, layer=layer, size_px=size_px, pixscale=LEGACY_PIXSCALE
    )
    req = urllib.request.Request(url, headers={"User-Agent": "EMUSE/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Fetch stopped by user.")
    if len(raw) < 2880:   # smaller than one FITS block → empty / error response
        raise ValueError(f"Legacy Survey returned no data for layer={layer} at RA={ra}, Dec={dec}")
    buf = BytesIO(raw)
    buf.seek(0)
    hdul = fits.open(buf)
    return hdul, raw


def fetch_optical_for_cutout(ra, dec, size_arcmin, stop_event=None):
    """Try ls-dr11 first; fall back to unwise-neo7 if optical not available.

    Returns dict with keys:
      optical_hdul, optical_bytes, optical_layer,
      ir_hdul, ir_bytes, ir_layer   (ir keys are None if not fetched)
    or raises RuntimeError if neither survey has data.
    stop_event: optional threading.Event to abort mid-fetch.
    """
    result = {
        "optical_hdul": None, "optical_bytes": None, "optical_layer": None,
        "ir_hdul": None,      "ir_bytes": None,      "ir_layer": None,
    }

    # --- Try optical (ls-dr11) ---
    try:
        hdul, raw = fetch_legacy_cutout(ra, dec, size_arcmin, layer="ls-dr11",
                                        stop_event=stop_event)
        # ls-dr11 is 3-band (g, r, z); check first extension has real data
        data = hdul[0].data if hdul[0].data is not None else (hdul[1].data if len(hdul) > 1 else None)
        if data is not None and np.any(data != 0):
            result["optical_hdul"] = hdul
            result["optical_bytes"] = raw
            result["optical_layer"] = "ls-dr11"
    except InterruptedError:
        raise
    except Exception:
        pass

    # Check stop between bands
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Fetch stopped by user.")

    # --- Try IR (unwise-neo7) ---
    try:
        hdul_ir, raw_ir = fetch_legacy_cutout(ra, dec, size_arcmin, layer="unwise-neo7",
                                              stop_event=stop_event)
        data_ir = hdul_ir[0].data if hdul_ir[0].data is not None else (hdul_ir[1].data if len(hdul_ir) > 1 else None)
        if data_ir is not None and np.any(data_ir != 0):
            result["ir_hdul"] = hdul_ir
            result["ir_bytes"] = raw_ir
            result["ir_layer"] = "unwise-neo7"
    except InterruptedError:
        raise
    except Exception:
        pass

    if result["optical_hdul"] is None and result["ir_hdul"] is None:
        raise RuntimeError(
            f"No Legacy Survey data (ls-dr11 or unwise-neo7) found at RA={ra:.5f}, Dec={dec:.5f}."
        )
    return result


def _optical_hdu_to_rgb(hdul, layer):
    """Convert a Legacy Survey multi-band FITS HDU into an RGB uint8 array (H×W×3).

    ls-dr11:    bands 0/1/2 → g/r/z → displayed as B/G/R (blue=g, green=r, red=z)
    unwise-neo7: bands 0/1  → W1/W2 → false-colour (blue=W1, red=W2, green=mean)
    """
    hdu = hdul[0]
    data = hdu.data  # shape (nband, H, W) for multi-band
    if data is None and len(hdul) > 1:
        data = hdul[1].data
    if data is None:
        return None

    data = np.array(data, dtype=float)
    if data.ndim == 2:
        data = data[np.newaxis]   # treat as single band

    def _asinh_stretch(band):
        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
        p_lo = float(np.nanpercentile(band, 0.5))
        p_hi = float(np.nanpercentile(band, 99.75))
        span = p_hi - p_lo
        if span <= 0:
            return np.zeros_like(band)
        soft = 5.0
        norm = np.clip((band - p_lo) / span, 0, None)
        stretched = np.arcsinh(soft * norm) / np.arcsinh(soft)
        return np.clip(stretched, 0.0, 1.0)

    if "unwise" in layer:
        w1 = _asinh_stretch(data[0]) if data.shape[0] > 0 else np.zeros(data.shape[-2:])
        w2 = _asinh_stretch(data[1]) if data.shape[0] > 1 else w1
        r_ch = w2
        g_ch = (w1 + w2) * 0.5
        b_ch = w1
    else:
        # g r z → B G R  (standard astronomical optical composite)
        g  = _asinh_stretch(data[0]) if data.shape[0] > 0 else np.zeros(data.shape[-2:])
        r  = _asinh_stretch(data[1]) if data.shape[0] > 1 else g
        z  = _asinh_stretch(data[2]) if data.shape[0] > 2 else r
        r_ch, g_ch, b_ch = z, r, g

    rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb





def _style_wcs_axes(ax):
    """Apply consistent WCSAxes cosmetics using FITS-native axis labels and formats."""
    ra_ax  = ax.coords[0]
    dec_ax = ax.coords[1]
    # Use FITS native label (e.g. "RA---SIN", "DEC--SIN") — don't override
    ra_ax.set_axislabel("Right Ascension (J2000)", fontsize=13, color="white")
    dec_ax.set_axislabel("Declination (J2000)", fontsize=13, color="white")
    # Keep FITS-default tick format (hms for RA, dms for Dec)
    ra_ax.set_ticks(color="white", size=6)
    dec_ax.set_ticks(color="white", size=6)
    ra_ax.set_ticklabel(color="white", fontsize=11, exclude_overlapping=True)
    dec_ax.set_ticklabel(color="white", fontsize=11, exclude_overlapping=True)
    ra_ax.grid(color="white", alpha=0.18, linestyle="--", linewidth=0.5)
    dec_ax.grid(color="white", alpha=0.18, linestyle="--", linewidth=0.5)


def render_radio_figure(radio_data, radio_wcs, ra, dec, size_arcmin, radio_pct_lo=95.0, radio_pct_hi=99.9, zoom_arcmin=None):
    """Render a standalone radio cutout with WCS axes, asinh stretch, and colorbar.

    The percentile range controls the colour stretch: lo sets the background
    noise floor, hi clips the bright emission peak.
    zoom_arcmin: if given, clips the displayed axes to this angular size centred
                 on (ra, dec) without changing the data or stretch.
    Returns a PNG bytes buffer (io.BytesIO).
    """
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from astropy.visualization.wcsaxes import WCSAxes

    radio = np.nan_to_num(np.array(radio_data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    nz = radio[radio != 0]
    r_lo = float(np.nanpercentile(nz, radio_pct_lo)) if nz.size else 0.0
    r_hi = float(np.nanpercentile(nz, radio_pct_hi)) if nz.size else 1.0
    if r_hi <= r_lo:
        r_hi = r_lo + 1e-9

    norm = mcolors.PowerNorm(gamma=0.5, vmin=r_lo, vmax=r_hi, clip=True)

    fig = plt.figure(figsize=(10, 9), dpi=130, facecolor="black")
    ax = WCSAxes(fig, [0.18, 0.15, 0.68, 0.70], wcs=radio_wcs)
    fig.add_axes(ax)

    im = ax.imshow(radio, origin="lower", cmap="inferno", norm=norm,
                   interpolation="lanczos", aspect="equal")
    ax.scatter([ra], [dec], transform=ax.get_transform("icrs"),
               s=80, marker="+", color="cyan", linewidths=1.5, zorder=10)

    # ---- Zoom: restrict displayed axes without touching data or stretch ----
    display_size = zoom_arcmin if (zoom_arcmin is not None and zoom_arcmin > 0) else size_arcmin
    display_size = min(display_size, size_arcmin)  # can't zoom out beyond the cutout
    try:
        centre_pix = radio_wcs.all_world2pix([[ra, dec]], 0)[0]
        half_pix = (display_size / 2.0) * 60.0 / abs(radio_wcs.wcs.cdelt[1] * 3600.0)
        ax.set_xlim(centre_pix[0] - half_pix, centre_pix[0] + half_pix)
        ax.set_ylim(centre_pix[1] - half_pix, centre_pix[1] + half_pix)
    except Exception:
        pass  # fall back to full extent if WCS query fails

    _style_wcs_axes(ax)
    ax.set_title(
        f"Radio (ASKAP/EMU)\nRA={ra:.5f}°   Dec={dec:.5f}°   zoom={display_size:.1f}′",
        color="white", fontsize=13, pad=10, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.88)
    cbar.set_label("Intensity (Jy/beam)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=10)
    cbar.outline.set_edgecolor("white")

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, pad_inches=0.2,
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def render_multiwavelength_figure(
    radio_data,
    radio_wcs,
    optical_hdul,
    optical_layer,
    ra,
    dec,
    size_arcmin,
    radio_pct_lo=95.0,
    radio_pct_hi=99.9,
    zoom_arcmin=None,
):
    """Produce a matplotlib figure overlaying radio contours on an optical/IR RGB background.

    Both images share the radio WCS frame (optical is reprojected to match).
    Radio is shown as white contours only so the optical/IR colours remain visible.
    zoom_arcmin: if given, clips the displayed axes to this angular size centred
                 on (ra, dec) without changing the data or stretch.
    Returns a PNG bytes buffer (io.BytesIO).
    """
    from io import BytesIO
    from reproject import reproject_interp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization.wcsaxes import WCSAxes

    radio = np.nan_to_num(np.array(radio_data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ny, nx = radio.shape

    # ---- Reproject optical bands to radio WCS ----
    opt_data = optical_hdul[0].data
    if opt_data is None and len(optical_hdul) > 1:
        opt_data = optical_hdul[1].data
        opt_header = optical_hdul[1].header
    else:
        opt_header = optical_hdul[0].header

    opt_data = np.array(opt_data, dtype=float)
    if opt_data.ndim == 2:
        opt_data = opt_data[np.newaxis]

    target_header = radio_wcs.to_header()
    target_header["NAXIS"] = 2
    target_header["NAXIS1"] = nx
    target_header["NAXIS2"] = ny

    def _asinh_stretch(band):
        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
        nz = band[band != 0]
        if not nz.size:
            return np.zeros_like(band)
        lo = float(np.nanpercentile(nz, 0.5))
        hi = float(np.nanpercentile(nz, 99.75))
        span = hi - lo
        if span <= 0:
            return np.zeros_like(band)
        norm = np.clip((band - lo) / span, 0, None)
        return np.clip(np.arcsinh(5.0 * norm) / np.arcsinh(5.0), 0.0, 1.0)

    reproj_bands = []
    for b in range(opt_data.shape[0]):
        band_hdu = fits.PrimaryHDU(data=opt_data[b], header=opt_header)
        rb, _ = reproject_interp(band_hdu, target_header)
        reproj_bands.append(_asinh_stretch(np.nan_to_num(rb, nan=0.0)))

    if "unwise" in optical_layer:
        w1 = reproj_bands[0] if len(reproj_bands) > 0 else np.zeros((ny, nx))
        w2 = reproj_bands[1] if len(reproj_bands) > 1 else w1
        r_ch, g_ch, b_ch = w2, (w1 + w2) * 0.5, w1
    else:
        g_ = reproj_bands[0] if len(reproj_bands) > 0 else np.zeros((ny, nx))
        r_ = reproj_bands[1] if len(reproj_bands) > 1 else g_
        z_ = reproj_bands[2] if len(reproj_bands) > 2 else r_
        r_ch, g_ch, b_ch = z_, r_, g_

    rgb = np.clip(np.stack([r_ch, g_ch, b_ch], axis=-1), 0.0, 1.0)

    # ---- Contour levels from radio percentile range ----
    nz_radio = radio[radio != 0]
    r_lo = float(np.nanpercentile(nz_radio, radio_pct_lo)) if nz_radio.size else 0.0
    r_hi = float(np.nanpercentile(nz_radio, radio_pct_hi)) if nz_radio.size else 1.0
    if r_hi <= r_lo:
        r_hi = r_lo + 1e-9
    r_lo_safe = max(r_lo, 1e-12)
    if r_hi > r_lo_safe:
        levels = np.logspace(np.log10(r_lo_safe), np.log10(r_hi), 6)
    else:
        levels = np.linspace(r_lo, r_hi, 6)

    # ---- Plot ----
    fig = plt.figure(figsize=(10, 9), dpi=130, facecolor="black")
    ax = WCSAxes(fig, [0.18, 0.15, 0.70, 0.70], wcs=radio_wcs)
    fig.add_axes(ax)

    ax.imshow(rgb, origin="lower", interpolation="lanczos", aspect="equal")
    ax.contour(radio, levels=levels, colors="white", linewidths=0.9,
               alpha=0.9, origin="lower")

    # Crosshair on target
    ax.scatter([ra], [dec], transform=ax.get_transform("icrs"),
               s=80, marker="+", color="cyan", linewidths=1.5, zorder=10)

    # ---- Axes ----
    _style_wcs_axes(ax)
    layer_label = "Optical (DECaLS DR11)" if "unwise" not in optical_layer else "IR (unWISE neo7)"

    # ---- Zoom: restrict displayed axes without touching data or stretch ----
    display_size = zoom_arcmin if (zoom_arcmin is not None and zoom_arcmin > 0) else size_arcmin
    display_size = min(display_size, size_arcmin)
    try:
        centre_pix = radio_wcs.all_world2pix([[ra, dec]], 0)[0]
        half_pix = (display_size / 2.0) * 60.0 / abs(radio_wcs.wcs.cdelt[1] * 3600.0)
        ax.set_xlim(centre_pix[0] - half_pix, centre_pix[0] + half_pix)
        ax.set_ylim(centre_pix[1] - half_pix, centre_pix[1] + half_pix)
    except Exception:
        pass

    ax.set_title(
        f"Radio + {layer_label}\nRA={ra:.5f}°   Dec={dec:.5f}°   zoom={display_size:.1f}′",
        color="white", fontsize=13, pad=10, fontweight="bold",
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, pad_inches=0.2,
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def reslice_cutout_from_cache(sbid, ra, dec, cutout_size_arcmin):
    """Re-cut from the cached tile data without any S3/disk I/O.

    Returns (cutout_files, cutout_previews, cutout_meta) written into the session cutout dir,
    or None if there is no valid cache for this source (cache miss — caller must stream from S3).
    """
    cache = st.session_state.get("cutout_tile_cache")
    if (
        cache is None
        or cache.get("sbid") != str(sbid)
        or abs(cache.get("ra", 0) - ra) > 1e-6
        or abs(cache.get("dec", 0) - dec) > 1e-6
    ):
        return None  # cache miss

    data = cache["data"]
    wcs = cache["wcs"]
    dradio = get_radio_cutout_hdu(data, wcs, ra, dec, cutout_size_arcmin=cutout_size_arcmin)
    preview_img = radio_cutout_to_png(dradio)
    if dradio is None or preview_img is None:
        return [], [], []

    downloads_dir = get_session_cutout_dir()
    fitsfile = os.path.join(downloads_dir, f"cutout_{sbid}_{ra:.5f}_{dec:.5f}.fits")
    dradio.writeto(fitsfile, overwrite=True)
    meta = {
        "sbid": sbid,
        "ra": ra,
        "dec": dec,
        "cutout_size_arcmin": cutout_size_arcmin,
        "fits_file": fitsfile,
    }
    return [fitsfile], [preview_img], [meta]


def run_emu_tile_cutout_pipeline(sbid, ra, dec, images_source, cutout_size_arcmin=CUTOUT_SIZE_ARCMIN):
    """Generate one radio cutout from EMU tiles and store it in session state.

    On first call for a source: streams the tile from S3 (or reads locally), caches the raw
    tile data array + WCS in session state, then slices at the requested size.
    On subsequent calls for the same source (slider moves): re-slices from the cache instantly,
    no S3 I/O.
    """
    # ---- Try cache first (size-change fast path) ----
    cached_result = reslice_cutout_from_cache(sbid, ra, dec, cutout_size_arcmin)
    if cached_result is not None:
        cutout_files, cutout_previews, cutout_meta = cached_result
        st.session_state.cutout_fig_path = None
        st.session_state.cutout_downloads_dir = get_session_cutout_dir()
        st.session_state.cutout_files = cutout_files
        st.session_state.cutout_previews = cutout_previews
        st.session_state.cutout_meta = cutout_meta
        st.session_state.cutout_generated_at = time.time()
        if cutout_previews:
            st.success(
                f"Cutout re-sliced from cached tile at {cutout_size_arcmin:.1f}′ "
                f"(no S3 download needed)."
            )
        else:
            st.warning("No cutout could be generated from the cached tile at this position.")
        return bool(cutout_previews)

    # ---- Cache miss: stream tile and populate cache ----
    downloads_dir = get_session_cutout_dir()
    for name in os.listdir(downloads_dir):
        if name.startswith("cutout_") and name.endswith(".fits"):
            try:
                os.remove(os.path.join(downloads_dir, name))
            except OSError:
                pass
    start_time = time.time()
    progress_bar = st.progress(0, text="Starting cutout generation from EMU tiles...")
    status_placeholder = st.empty()

    # Always stream/read a 30 arcmin slab so the cache covers any user-selected size.
    fetch_size = max(cutout_size_arcmin, _TILE_CACHE_SIZE_ARCMIN)
    cutout_files, cutout_previews, cutout_meta = generate_single_cutout_from_emu_tiles(
        sbid,
        ra,
        dec,
        images_source,
        downloads_dir,
        progress_bar,
        status_placeholder,
        cutout_size_arcmin=cutout_size_arcmin,
        fetch_size_arcmin=fetch_size,
    )
    st.session_state.cutout_fig_path = None
    st.session_state.cutout_downloads_dir = downloads_dir
    st.session_state.cutout_files = cutout_files
    st.session_state.cutout_previews = cutout_previews
    st.session_state.cutout_meta = cutout_meta
    st.session_state.cutout_generated_at = time.time()
    elapsed_time = time.time() - start_time
    progress_bar.progress(1.0, text="Cutout generation complete!")
    if cutout_previews:
        status_placeholder.success(
            f"Cutout generated from EMU tiles in {elapsed_time:.1f} seconds."
        )
    else:
        status_placeholder.warning(
            "No cutout could be generated from EMU tiles for this source."
        )
    return bool(cutout_previews)


def list_s3_fits_keys(s3_client, bucket, prefix):
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = os.path.basename(key).lower()
            if name.endswith(".fits") and "_wise" not in name:
                keys.append(key)
    return keys


def get_s3_fits_key_index(s3_client, bucket, prefix, cache_dir=None):
    """Return S3 FITS keys, using a local disk cache (if cache_dir given) or session cache."""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        index_path = os.path.join(cache_dir, f".s3_fits_index_{bucket}.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and cached:
                    return cached
            except Exception:
                pass

    if "emu_s3_fits_keys" in st.session_state and st.session_state.emu_s3_fits_keys:
        return st.session_state.emu_s3_fits_keys

    keys = list_s3_fits_keys(s3_client, bucket, prefix)
    st.session_state.emu_s3_fits_keys = keys

    if cache_dir is not None:
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(keys, f)
        except Exception:
            pass
    return keys


def find_tile_for_sbid(sbid, tile_paths):
    token = str(sbid)
    matches = [path for path in tile_paths if token in os.path.basename(path)]
    return matches[0] if matches else None


def stream_s3_tile_to_memory(s3_client, bucket, key, status_placeholder=None):
    """Stream an S3 FITS tile directly into a BytesIO buffer (no disk write).

    Returns a BytesIO object positioned at the start, ready for fits.open().
    """
    from io import BytesIO
    if status_placeholder is not None:
        status_placeholder.info(f"Streaming EMU tile from S3: {os.path.basename(key)} ...")
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    buf = BytesIO(resp["Body"].read())
    buf.seek(0)
    if status_placeholder is not None:
        status_placeholder.info(f"Tile streamed into memory: {os.path.basename(key)}")
    return buf


def open_emu_tile(tile_source):
    """Open an EMU FITS tile from either a local file path or a file-like object (BytesIO).

    Returns (hdul, hdu, data, wcs).  The caller is responsible for closing hdul.
    """
    hdul = fits.open(tile_source, memmap=False, relax=True)
    hdu = hdul[0]
    if hdu.data is None and len(hdul) > 1:
        hdu = hdul[1]
    data = np.squeeze(hdu.data)
    if data.ndim > 2:
        data = data[0]
    wcs = WCS(hdu.header)
    if wcs.naxis > 2:
        wcs = wcs.celestial
    return hdul, hdu, data, wcs


def get_radio_cutout_hdu(data, wcs, ra, dec, cutout_size_arcmin=CUTOUT_SIZE_ARCMIN, pixel_size_arcsec=PIXEL_SIZE_ARCSEC):
    angular = int(cutout_size_arcmin / (pixel_size_arcsec / 60.0))
    img_size = (angular, angular)
    pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    try:
        cutout = Cutout2D(data, position=pos, size=img_size, fill_value=np.nan, wcs=wcs)
    except Exception:
        return None
    cutout.data[np.isnan(cutout.data)] = 0
    cutout_wcs = cutout.wcs
    cutout_wcs.wcs.crpix = (angular / 2.0, angular / 2.0)
    header = cutout_wcs.to_header()
    header["CRVAL1"] = ra
    header["CRVAL2"] = dec
    return fits.PrimaryHDU(data=cutout.data, header=header)


def radio_array_to_png(r):
    """RGB preview stretched around the central source, not the brightest neighbour."""
    r = np.array(r, dtype=float)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    if r.size == 0 or r.ndim != 2:
        return None
    ny, nx = r.shape
    cy, cx = ny // 2, nx // 2
    med = float(np.nanmedian(r))
    mad = float(np.nanmedian(np.abs(r - med)))
    rms = 1.4826 * mad if mad > 0 else float(np.nanstd(r))
    lo = med
    min_span = max(5.0 * rms, 1e-8)

    yy, xx = np.ogrid[:ny, :nx]
    rad = max(15, int(min(ny, nx) * 0.08))
    inner = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
    center_pix = r[inner]
    if center_pix.size == 0:
        center_pix = r.ravel()

    def _center_highs(pixels):
        return [float(np.nanpercentile(pixels, p)) for p in (99.9, 99.5, 99.0)]

    highs = _center_highs(center_pix)
    if (highs[1] - lo) < 3.0 * max(rms, 1e-12):
        rad2 = max(rad * 2, int(min(ny, nx) * 0.2))
        inner2 = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad2 ** 2
        grown = r[inner2]
        if grown.size:
            highs = _center_highs(grown)

    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    soft = 8.0
    for clipid, hi_raw in enumerate(highs):
        hi = max(hi_raw, lo + min_span)
        clipped = np.clip(r, lo, hi)
        y = clipped - lo
        span = hi - lo
        stretched = np.arcsinh(soft * y / span) / np.arcsinh(soft)
        img[:, :, 2 - clipid] = np.uint8(np.clip(stretched, 0.0, 1.0) * 255.0)
    return Image.fromarray(img, "RGB")


def radio_cutout_to_png(dradio):
    if dradio is None or dradio.data is None:
        return None
    return radio_array_to_png(dradio.data)


def generate_single_cutout_from_emu_tiles(
    sbid,
    ra,
    dec,
    images_source,
    downloads_dir,
    progress_bar,
    status_placeholder,
    cutout_size_arcmin=CUTOUT_SIZE_ARCMIN,
    fetch_size_arcmin=None,
):
    """Build one radio cutout from a local or S3 EMU tile.

    S3 mode: streams the tile directly into memory — no tile file is written to disk.
    Local mode: reads the tile file directly from the configured images directory.
    The small per-source FITS cutout is always written to downloads_dir for user download.

    fetch_size_arcmin: size of the slab stored in the session tile cache so that future
    slider adjustments can re-slice without re-streaming from S3.  Defaults to
    max(cutout_size_arcmin, _TILE_CACHE_SIZE_ARCMIN).
    """
    from io import BytesIO

    if fetch_size_arcmin is None:
        fetch_size_arcmin = max(cutout_size_arcmin, _TILE_CACHE_SIZE_ARCMIN)
    fetch_size_arcmin = max(fetch_size_arcmin, cutout_size_arcmin)

    os.makedirs(downloads_dir, exist_ok=True)
    open_tiles = {}
    s3_client = None

    progress_bar.progress(0.05, text="Locating EMU tile...")
    if images_source["mode"] == "local":
        tile_index = [
            path
            for path in glob.glob(os.path.join(images_source["images_dir"], "*.fits"))
            if "_wise" not in os.path.basename(path).lower()
        ]
    else:
        s3_client = boto3.client("s3")
        status_placeholder.info("Resolving EMU tile keys in S3 (uses session index cache when available)...")
        tile_index = get_s3_fits_key_index(
            s3_client,
            images_source["bucket"],
            images_source["prefix"],
        )

    tile_ref = find_tile_for_sbid(sbid, tile_index)
    if tile_ref is None:
        status_placeholder.warning(f"No EMU tile found for SBID {sbid}.")
        return [], [], []

    try:
        progress_bar.progress(0.30, text="Loading EMU tile...")

        if images_source["mode"] == "s3":
            progress_bar.progress(0.35, text="Streaming tile from S3 into memory...")
            tile_source = stream_s3_tile_to_memory(
                s3_client,
                images_source["bucket"],
                tile_ref,
                status_placeholder=status_placeholder,
            )
        else:
            tile_source = tile_ref  # local path string

        progress_bar.progress(0.55, text="Cutting out selected source...")
        status_placeholder.info(
            f"Cutting out SBID={sbid}, RA={ra:.5f}, Dec={dec:.5f}, size={cutout_size_arcmin:.1f}'"
        )

        open_tiles[id(tile_source)] = open_emu_tile(tile_source)
        _hdul, _hdu, data, wcs = open_tiles[id(tile_source)]

        # ---- Populate tile cache with a 30 arcmin slab ----
        # Store only this smaller sub-array (not the full GB tile) so subsequent
        # size changes can re-slice from memory instantly.
        cache_hdu = get_radio_cutout_hdu(data, wcs, ra, dec, cutout_size_arcmin=fetch_size_arcmin)
        if cache_hdu is not None:
            st.session_state.cutout_tile_cache = {
                "sbid": str(sbid),
                "ra": ra,
                "dec": dec,
                "data": cache_hdu.data.copy(),
                "wcs": WCS(cache_hdu.header),
            }

        # ---- Slice at the actually requested size ----
        dradio = get_radio_cutout_hdu(
            data, wcs, ra, dec, cutout_size_arcmin=cutout_size_arcmin
        )
        preview_img = radio_cutout_to_png(dradio)
        if dradio is None or preview_img is None:
            status_placeholder.warning("Cutout failed at this position (outside tile or empty).")
            return [], [], []

        fitsfile = os.path.join(
            downloads_dir, f"cutout_{sbid}_{ra:.5f}_{dec:.5f}.fits"
        )
        dradio.writeto(fitsfile, overwrite=True)
        meta = {
            "sbid": sbid,
            "ra": ra,
            "dec": dec,
            "cutout_size_arcmin": cutout_size_arcmin,
            "fits_file": fitsfile,
        }
        progress_bar.progress(0.95, text="Writing FITS cutout...")
        return [fitsfile], [preview_img], [meta]
    finally:
        for hdul, _hdu, _data, _wcs in open_tiles.values():
            try:
                hdul.close()
            except Exception:
                pass

                pass


def cutout_source_labels(results_df):
    labels = []
    for i, row in results_df.iterrows():
        labels.append(
            f"{len(labels) + 1}. SBID {row['SBID']} | RA {row['RA']} | Dec {row['Dec']}"
        )
    return labels


def parse_cutout_source_choice(results_df, choice_label):
    idx = int(str(choice_label).split(".", 1)[0]) - 1
    row = results_df.iloc[idx]
    return str(row["SBID"]), float(row["RA"]), float(row["Dec"])


def render_cutout_source_controls(results_df):
    labels = cutout_source_labels(results_df)
    if not labels:
        st.warning("No sources available to cut out.")
        return None

    # ---- Initialise / guard the source choice ----
    if "cutout_source_choice" not in st.session_state:
        st.session_state.cutout_source_choice = labels[0]
    elif st.session_state.cutout_source_choice not in labels:
        st.session_state.cutout_source_choice = labels[0]

    # Track previous choice in a separate key so we can detect a real change
    # after the selectbox has updated st.session_state.cutout_source_choice.
    if "cutout_source_choice_prev" not in st.session_state:
        st.session_state.cutout_source_choice_prev = st.session_state.cutout_source_choice

    st.selectbox(
        "Source (SBID, RA, Dec)",
        options=labels,
        key="cutout_source_choice",
        help="Choose one source from the current search table.",
    )
    current_choice = st.session_state.cutout_source_choice

    # Detect a genuine dropdown change and repopulate the editable boxes.
    if current_choice != st.session_state.cutout_source_choice_prev:
        st.session_state.cutout_source_choice_prev = current_choice
        st.session_state.cutout_tile_cache = None
        clear_optical_state()
        _sbid, _ra, _dec = parse_cutout_source_choice(results_df, current_choice)
        st.session_state.cutout_edit_sbid   = str(_sbid)
        st.session_state.cutout_edit_ra     = float(_ra)
        st.session_state.cutout_edit_dec    = float(_dec)
        st.session_state.cutout_size_arcmin = float(CUTOUT_SIZE_ARCMIN)

    # Seed boxes the very first time
    if "cutout_edit_sbid" not in st.session_state:
        _sbid, _ra, _dec = parse_cutout_source_choice(results_df, current_choice)
        st.session_state.cutout_edit_sbid   = str(_sbid)
        st.session_state.cutout_edit_ra     = float(_ra)
        st.session_state.cutout_edit_dec    = float(_dec)
        st.session_state.cutout_size_arcmin = float(CUTOUT_SIZE_ARCMIN)

    st.caption("Edit any field below to override the selected source before creating the cutout.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.text_input(
            "SBID",
            key="cutout_edit_sbid",
            help="Scheduling Block ID of the tile to use.",
        )
    with c2:
        st.number_input(
            "RA (deg)", min_value=0.0, max_value=360.0,
            step=0.00001, format="%.5f",
            key="cutout_edit_ra",
            help="Right Ascension of the cutout centre in degrees.",
        )
    with c3:
        st.number_input(
            "Dec (deg)", min_value=-90.0, max_value=90.0,
            step=0.00001, format="%.5f",
            key="cutout_edit_dec",
            help="Declination of the cutout centre in degrees.",
        )
    with c4:
        st.number_input(
            "Size (arcmin)", min_value=1.0, max_value=30.0,
            step=0.5, format="%.1f",
            key="cutout_size_arcmin",
            help="Angular size of the square radio cutout (max 30′).",
        )
    return True


def generate_casda_single_cutout(sbid, ra, dec, username, password, cutout_size_arcmin):
    downloads_dir = get_session_cutout_dir()
    for name in os.listdir(downloads_dir):
        if name.startswith("cutout_") and name.endswith(".fits"):
            try:
                os.remove(os.path.join(downloads_dir, name))
            except OSError:
                pass

    casda = Casda()
    _orig_getpass = getpass.getpass

    def _autofill_getpass(prompt="Password: ", stream=None):
        return password

    getpass.getpass = _autofill_getpass
    try:
        casda.login(username=username)
    finally:
        getpass.getpass = _orig_getpass

    radius = max(cutout_size_arcmin / 2.0, 0.5) * u.arcmin
    coord = SkyCoord(ra, dec, unit="deg")
    result = Casda.query_region(coord, radius=2 * u.arcmin)
    pattern = "image.i.EMU*taylor.0.restored.conv.fits"
    mask = [fnmatch.fnmatch(fn, pattern) for fn in result["filename"]]
    data = result[mask]
    if len(data) == 0:
        return [], [], []
    url_list = casda.cutout(data[:1], coordinates=coord, radius=radius)
    filelist = casda.download_files(url_list, savedir=downloads_dir)
    cutout_files, cutout_previews, cutout_meta = [], [], []
    for fitsfile in filelist:
        if not (fitsfile.endswith(".fits") and os.path.exists(fitsfile)):
            continue
        hdul = fits.open(fitsfile)
        hdu_index = 0
        if hdul[hdu_index].data is None and len(hdul) > 1:
            hdu_index = 1
        data_img = np.nan_to_num(hdul[hdu_index].data)
        if data_img.ndim == 4:
            data_plot = data_img[0, 0, :, :]
        elif data_img.ndim == 3:
            data_plot = data_img[0, :, :]
        else:
            data_plot = data_img
        preview_img = radio_array_to_png(data_plot)
        hdul.close()
        if preview_img is None:
            continue
        cutout_files.append(fitsfile)
        cutout_previews.append(preview_img)
        cutout_meta.append(
            {
                "sbid": sbid,
                "ra": ra,
                "dec": dec,
                "cutout_size_arcmin": cutout_size_arcmin,
                "fits_file": fitsfile,
            }
        )
        break
    return cutout_files, cutout_previews, cutout_meta


bootstrap_download_cleanup()

if input_option == "Text":
    search_for = st.text_input("Enter object to search for:", "A bent tailed radio galaxy")
    if st.button("Search", key="text_search"):
        with st.spinner("Searching..."):
            try:
                expanded_queries, gemini_used, gemini_status = build_text_query(
                    search_for,
                    tokenizer,
                    model,
                    use_gemini_llm=use_gemini_llm,
                    force_gemini=False,
                    gemini_api_key=gemini_api_key,
                )
            except Exception as e:
                st.error(f"Text query building failed: {e}")
                st.stop()

            # Ensure new searches don't reuse stale checkbox/text state from prior query.
            reset_interpreted_query_widget_state()
            clear_session_cutout_state_and_files()
            st.session_state.interpreted_queries = expanded_queries
            st.session_state.gemini_used = gemini_used
            st.session_state.gemini_status = gemini_status
            # Keep custom query box empty by default for each new interpreted set.
            st.session_state.editable_final_queries = ""
            for i, q in enumerate(expanded_queries):
                st.session_state[f"query_checkbox_{i}"] = True
                st.session_state[f"query_text_{i}"] = q

            sb_ra_dec, filtered_probs = run_text_similarity_search(expanded_queries)
            st.session_state.sb_ra_dec = sb_ra_dec
            st.session_state.filtered_probs = filtered_probs
            st.session_state.input_option = input_option
            st.session_state.show_search_results = True

    interpreted_queries = st.session_state.get("interpreted_queries", [])
    if interpreted_queries:
        with st.expander("🔍 Interpreted query", expanded=True):
            st.write(f"Gemini used: {'Yes' if st.session_state.get('gemini_used', False) else 'No'}")
            st.caption(st.session_state.get("gemini_status", ""))
            st.write("Tick and edit queries in one place:")

            selected_queries = []
            for i, query in enumerate(interpreted_queries):
                checkbox_key = f"query_checkbox_{i}"
                text_key = f"query_text_{i}"
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = True
                if text_key not in st.session_state:
                    st.session_state[text_key] = query
                row_left, row_right = st.columns([1, 5])
                with row_left:
                    keep_this = st.checkbox(
                        "Use",
                        key=checkbox_key,
                        label_visibility="collapsed",
                    )
                with row_right:
                    edited_query = st.text_input(
                        f"Query {i+1}",
                        key=text_key,
                        label_visibility="collapsed",
                    )
                if keep_this and edited_query.strip():
                    selected_queries.append(edited_query.strip())

            st.text_area(
                "Optional: add more custom queries (one per line)",
                key="editable_final_queries",
                help="Extra custom query phrases to append.",
            )

            search_edited = st.button(
                "Search again using selected/edited queries",
                key="search_again_with_edited_queries",
                use_container_width=True,
            )

        if search_edited:
            with st.spinner("Searching with selected/edited queries..."):
                edited_text = st.session_state.get("editable_final_queries", "")
                final_queries = []
                for q in selected_queries:
                    if q.lower() not in {x.lower() for x in final_queries}:
                        final_queries.append(q)
                for line in edited_text.splitlines():
                    q = line.strip()
                    if q and q.lower() not in {x.lower() for x in final_queries}:
                        final_queries.append(q)
                if not final_queries:
                    st.warning("Please tick at least one interpreted query or enter custom queries.")
                else:
                    st.caption(f"Using {len(final_queries)} query phrase(s) for text embedding.")
                    sb_ra_dec, filtered_probs = run_text_similarity_search(final_queries)
                    clear_session_cutout_state_and_files()
                    st.session_state.sb_ra_dec = sb_ra_dec
                    st.session_state.filtered_probs = filtered_probs
                    st.session_state.input_option = input_option
                    st.session_state.show_search_results = True
elif input_option == "Image":
    _iu1, _iumid, _iu3 = st.columns([1, 4, 1])
    with _iumid:
        uploaded_file = st.file_uploader("Upload an image to search for similar objects...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        _ip1, _ipmid, _ip3 = st.columns([1, 4, 1])
        with _ipmid:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        image_upload = preprocess(Image.open(uploaded_file)).unsqueeze(0)
        _ib1, _ibmid, _ib3 = st.columns([1, 4, 1])
        with _ibmid:
            _search_clicked = st.button("Search", key="image_search", use_container_width=True)
        if _search_clicked:
            with st.spinner("Searching..."):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_feature = model.encode_image(image_upload)
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)
                    image_probs = (100.0 * all_image_features @ image_feature.T)
                image_probs_np = image_probs.numpy() / image_probs.numpy().max()
                idx_above_prob = np.where(image_probs_np > above_prob_of)[0]
                idx_above_prob_sorted = idx_above_prob[np.argsort(image_probs_np[idx_above_prob].flatten())[::-1]]
                sb_ra_dec = [idx_dict.get(val, "Key not found") for val in idx_above_prob_sorted]
                filtered_probs = image_probs_np[idx_above_prob_sorted].flatten()
                clear_session_cutout_state_and_files()
                st.session_state.sb_ra_dec = sb_ra_dec
                st.session_state.filtered_probs = filtered_probs
                st.session_state.input_option = input_option
                st.session_state.show_search_results = True

# Use persisted results if search has just run or if we're coming back after pressing "generate cutouts"
sb_ra_dec = st.session_state.get("sb_ra_dec", None)
filtered_probs = st.session_state.get("filtered_probs", None)
show_search_results = st.session_state.get("show_search_results", False)
if sb_ra_dec is not None and filtered_probs is not None and show_search_results:
    # Always reapply galactic filter and top_n since user could have changed sidebar
    if remove_galactic and len(sb_ra_dec) > 0:
        ra_dec_list = [(entry.split('_')[1], entry.split('_')[2]) for entry in sb_ra_dec]
        ra_dec_arr = np.array(ra_dec_list, dtype=float)
        coords = SkyCoord(ra=ra_dec_arr[:, 0] * u.deg, dec=ra_dec_arr[:, 1] * u.deg, frame='icrs')
        galactic_coords = coords.galactic
        galactic_latitudes = np.abs(galactic_coords.b.deg)
        filtered_indices = np.where(galactic_latitudes > 10)[0]
        filtered_sb_ra_dec = np.array(sb_ra_dec)[filtered_indices]
        filtered_probs = filtered_probs[filtered_indices]
    else:
        filtered_sb_ra_dec = sb_ra_dec

    _new_n_filtered = len(filtered_sb_ra_dec)
    if st.session_state.get("n_filtered_sources") != _new_n_filtered:
        st.session_state["n_filtered_sources"] = _new_n_filtered
        st.rerun()

    if _new_n_filtered < top_n:
        top_n = _new_n_filtered
    st.subheader(f"Top {top_n} similar sources:")

    df = pd.DataFrame(columns=['SBID', 'RA', 'Dec', 'Integrated Flux (mJy)', 'CatWISE Potential Host', 'Probability'])

    for i, (sb, prob) in enumerate(zip(filtered_sb_ra_dec[:top_n], filtered_probs[:top_n]), 1):
        sb_parts = sb.split('_')
        sb_id = sb_parts[0]
        ra = float(sb_parts[1])
        dec = float(sb_parts[2])
        flux = float(sb_parts[3])
        catwise = sb_parts[4]
        new_row = pd.DataFrame({'SBID': [sb_id], 'RA': [f'{ra:.5f}'], 'Dec': [f'{dec:.5f}'], 'Integrated Flux (mJy)': [f'{flux:.2f}'], 'CatWISE Potential Host': [f'{catwise}'], 'Probability': [f'{prob:.2f}']})
        df = pd.concat([df, new_row], ignore_index=True)

    df_cleaned = df.drop_duplicates(subset=["RA", "Dec"])
    st.session_state.results_df = df_cleaned  # cache table for cutouts and session
    st.dataframe(df_cleaned, use_container_width=True, hide_index=False)

    # --- Three-button action row ---
    _col_dl, _col_cut, _col_gem = st.columns(3)
    with _col_dl:
        st.download_button(
            "⬇ Download Table",
            data=df_cleaned.to_csv(index=False),
            file_name="emuse_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_table_btn",
        )
    with _col_cut:
        if st.button("🖼 Generate Cutout", use_container_width=True, key="generate_cutouts_btn"):
            st.session_state.cutout_flow_active = True
            st.session_state.show_credential_fields = False
            st.session_state.aws_login_completed = False
            st.session_state.casda_ready = False
            st.session_state.pop("aws_login_output", None)
            st.session_state.pop("aws_login_ok", None)
            st.session_state.pop("emu_s3_fits_keys", None)
            st.session_state.cutout_previews = []
            st.session_state.cutout_meta = []
            st.session_state.cutout_files = []
            st.session_state.emu_images_source = probe_emu_images_source()
            st.session_state.cutout_selector_ready = bool(
                st.session_state.emu_images_source.get("available")
            )
    with _col_gem:
        if st.button("🤖 Gemini Assistant", use_container_width=True, key="open_table_assistant_btn"):
            st.session_state.table_assistant_open = True

    render_gemini_table_assistant(df_cleaned, gemini_api_key)

    # --- Nice horizontal divider before cutouts section ---
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

else:
    # On page reloads/reruns, if we had results before, show them again so they don't vanish when using cutout button!
    if st.session_state.get("results_df", None) is not None:
        df_cleaned = st.session_state["results_df"]
        st.subheader(f"Top {min(top_n, len(df_cleaned))} similar sources (restored):")
        st.dataframe(df_cleaned.head(top_n), use_container_width=True, hide_index=False)

        # --- Three-button action row ---
        _col_dl2, _col_cut2, _col_gem2 = st.columns(3)
        with _col_dl2:
            st.download_button(
                "⬇ Download Table",
                data=df_cleaned.head(top_n).to_csv(index=False),
                file_name="emuse_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_table_btn2",
            )
        with _col_cut2:
            if st.button("🖼 Generate Cutout", use_container_width=True, key="generate_cutouts_btn"):
                st.session_state.cutout_flow_active = True
                st.session_state.show_credential_fields = False
                st.session_state.aws_login_completed = False
                st.session_state.casda_ready = False
                st.session_state.pop("aws_login_output", None)
                st.session_state.pop("aws_login_ok", None)
                st.session_state.pop("emu_s3_fits_keys", None)
                st.session_state.cutout_previews = []
                st.session_state.cutout_meta = []
                st.session_state.cutout_files = []
                st.session_state.emu_images_source = probe_emu_images_source()
                st.session_state.cutout_selector_ready = bool(
                    st.session_state.emu_images_source.get("available")
                )
        with _col_gem2:
            if st.button("🤖 Gemini Assistant", use_container_width=True, key="open_table_assistant_btn2"):
                st.session_state.table_assistant_open = True

        render_gemini_table_assistant(df_cleaned.head(top_n), gemini_api_key)

        # --- Nice horizontal divider before cutouts section ---
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# Only show cutout button if there are results
results_for_cutout = st.session_state.get("results_df", None)
if results_for_cutout is not None and not results_for_cutout.empty:
    if "casda_username" not in st.session_state:
        st.session_state.casda_username = ""
    if "casda_password" not in st.session_state:
        st.session_state.casda_password = ""
    if "cutouts_login_error" not in st.session_state:
        st.session_state.cutouts_login_error = False
    if "show_credential_fields" not in st.session_state:
        st.session_state.show_credential_fields = False
    if "cutout_flow_active" not in st.session_state:
        st.session_state.cutout_flow_active = False
    if "aws_login_completed" not in st.session_state:
        st.session_state.aws_login_completed = False
    if "cutout_selector_ready" not in st.session_state:
        st.session_state.cutout_selector_ready = False
    if "casda_ready" not in st.session_state:
        st.session_state.casda_ready = False
    if "aws_login_remote" not in st.session_state:
        st.session_state.aws_login_remote = False

    if st.session_state.get("cutout_flow_active"):
        st.session_state.cutout_selector_ready = bool(
            st.session_state.get("emu_images_source", {}).get("available")
        )

    images_source = st.session_state.get("emu_images_source")

    if st.session_state.cutout_flow_active and images_source is not None:
        if images_source.get("available"):
            st.session_state.cutout_selector_ready = True
        else:
            # S3 not accessible — go straight to CASDA credentials
            st.session_state.show_credential_fields = True
            st.session_state.aws_login_completed = True  # skip AWS login gate

    images_unavailable = bool(
        st.session_state.cutout_flow_active
        and images_source is not None
        and not images_source.get("available")
    )

    if images_unavailable and (
        st.session_state.show_credential_fields
        or st.session_state.cutouts_login_error
        or st.session_state.casda_ready
    ):
        st.session_state.show_credential_fields = True
        st.info(
            "Please enter your CASDA OPAL credentials to access data. If you don't have an account, register at https://data.csiro.au/domain/casda"
        )
        with st.form("casda_credentials_form", clear_on_submit=False):
            username = st.text_input("CASDA Username", key="casda_username")
            password = st.text_input("CASDA Password", type="password", key="casda_password")
            submitted = st.form_submit_button("Login to CASDA", use_container_width=True)

        if submitted:
            if username and password:
                casda = Casda()
                try:
                    _orig_getpass = getpass.getpass

                    def _autofill_getpass(prompt="Password: ", stream=None):
                        return password

                    getpass.getpass = _autofill_getpass
                    try:
                        casda.login(username=username)
                    finally:
                        getpass.getpass = _orig_getpass
                    st.session_state.cutouts_login_error = False
                    st.session_state.casda_ready = True
                    st.success("CASDA authentication successful. Choose a source below.")
                except Exception as e:
                    st.session_state.cutouts_login_error = True
                    st.session_state.casda_ready = False
                    st.warning(f"CASDA login failed: {e}")
            else:
                st.warning("Please enter both username and password before continuing.")

    can_select_cutout = st.session_state.get("cutout_selector_ready") or st.session_state.get("casda_ready")
    if can_select_cutout and render_cutout_source_controls(results_for_cutout):
        create_cutout = st.button(
            "Create this cutout",
            use_container_width=True,
            key="create_single_cutout_btn",
        )
        if create_cutout:
            sbid = str(st.session_state.get("cutout_edit_sbid", ""))
            ra   = float(st.session_state.get("cutout_edit_ra", 0.0))
            dec  = float(st.session_state.get("cutout_edit_dec", 0.0))
            size_arcmin = float(st.session_state.get("cutout_size_arcmin", CUTOUT_SIZE_ARCMIN))
            # Clear previous display so stale image and layer options are gone immediately
            st.session_state.cutout_previews = []
            st.session_state.cutout_meta = []
            st.session_state.cutout_files = []
            st.session_state.cutout_tile_cache = None
            # Reset slider states so they reseed to correct defaults for new cutout
            st.session_state.pop("display_zoom_arcmin", None)
            st.session_state.pop("multiwave_radio_pct_range", None)
            st.session_state.pop("multiwave_radio_pct_lo", None)
            st.session_state.pop("multiwave_radio_pct_hi", None)
            st.session_state.pop("multiwave_layer_choice_prev", None)
            clear_optical_state()
            with st.spinner("Generating cutout..."):
                if st.session_state.get("cutout_selector_ready") and images_source and images_source.get("available"):
                    run_emu_tile_cutout_pipeline(
                        sbid, ra, dec, images_source, cutout_size_arcmin=size_arcmin
                    )
                elif st.session_state.get("casda_ready"):
                    try:
                        start_time = time.time()
                        cutout_files, cutout_previews, cutout_meta = generate_casda_single_cutout(
                            sbid,
                            ra,
                            dec,
                            st.session_state.casda_username,
                            st.session_state.casda_password,
                            size_arcmin,
                        )
                        st.session_state.cutout_fig_path = None
                        st.session_state.cutout_downloads_dir = get_session_cutout_dir()
                        st.session_state.cutout_files = cutout_files
                        st.session_state.cutout_previews = cutout_previews
                        st.session_state.cutout_meta = cutout_meta
                        st.session_state.cutout_generated_at = time.time()
                        elapsed_time = time.time() - start_time
                        if cutout_previews:
                            st.success(f"CASDA cutout generated in {elapsed_time:.1f} seconds.")
                        else:
                            st.warning("CASDA returned no cutout for this position.")
                    except Exception as e:
                        st.warning(f"CASDA cutout failed: {e}")

    cutout_previews = st.session_state.get("cutout_previews", [])
    cutout_meta = st.session_state.get("cutout_meta", [])
    if cutout_previews:
        meta = cutout_meta[0] if cutout_meta else {}
        ra_val   = meta.get("ra", 0.0)
        dec_val  = meta.get("dec", 0.0)
        size_val = meta.get("cutout_size_arcmin", CUTOUT_SIZE_ARCMIN)
        fits_path = meta.get("fits_file")

        # ---- Read radio data / WCS from the session FITS file ----
        radio_data_for_plot = None
        radio_wcs_for_plot  = None
        if fits_path and os.path.exists(fits_path):
            try:
                with fits.open(fits_path) as _hdul:
                    _hdu = _hdul[0]
                    radio_data_for_plot = np.squeeze(_hdu.data).astype(float)
                    radio_wcs_for_plot  = WCS(_hdu.header)
                    if radio_wcs_for_plot.naxis > 2:
                        radio_wcs_for_plot = radio_wcs_for_plot.celestial
            except Exception:
                pass

        # ---- Optical/IR: load from session files if already fetched for this source ----
        _cached_ra  = st.session_state.get("multiwave_optical_ra")
        _cached_dec = st.session_state.get("multiwave_optical_dec")
        optical_cache_valid = (
            _cached_ra is not None
            and _cached_dec is not None
            and abs(_cached_ra - ra_val) < 1e-6
            and abs(_cached_dec - dec_val) < 1e-6
        )

        optical_hdu   = st.session_state.get("optical_hdu")
        ir_hdu        = st.session_state.get("ir_hdu")
        optical_layer = st.session_state.get("optical_layer")
        has_optical   = optical_hdu is not None
        has_ir        = ir_hdu is not None

        # ---- Heading ----
        st.markdown("<div style='text-align:center;'><h2>Cutout &amp; Multiwavelength View</h2></div>",
                    unsafe_allow_html=True)

        # ---- All controls centred in the same [1,6,1] column as the image ----
        _c1, _cmid, _c3 = st.columns([1, 6, 1])
        with _cmid:

            # Fetch checkbox + size input on one row
            _chk_col, _sz_col = st.columns([3, 2])
            with _chk_col:
                want_optical = st.checkbox(
                    "Also fetch optical & IR data (Legacy Survey)",
                    value=st.session_state.get("fetch_optical_ir_requested", False),
                    key="fetch_optical_ir_checkbox",
                    help=(
                        "When checked and you press 'Fetch optical/IR', the app will download "
                        "optical (ls-dr11) and infrared (unwise-neo7) cutouts from the "
                        "Legacy Survey and save them to your session folder."
                    ),
                )
                st.session_state.fetch_optical_ir_requested = want_optical

            with _sz_col:
                opt_ir_size = st.number_input(
                    "Optical/IR size (arcmin)",
                    min_value=3.0,
                    max_value=15.0,
                    step=0.5,
                    format="%.1f",
                    key="optical_ir_size_arcmin",
                    help="Cutout size for optical and IR images (max 15′). Change and re-fetch to update.",
                    disabled=not want_optical,
                )

            # Fetch button — full width of the centre column
            fetch_clicked = False
            if want_optical:
                _size_changed = (
                    optical_cache_valid
                    and abs(st.session_state.get("optical_ir_size_arcmin", OPTICAL_CUTOUT_SIZE_ARCMIN)
                            - st.session_state.get("multiwave_optical_size", OPTICAL_CUTOUT_SIZE_ARCMIN)) > 0.01
                )
                _btn_label = (
                    "Re-fetch optical/IR (new size)" if (optical_cache_valid and _size_changed)
                    else ("Re-fetch optical/IR" if optical_cache_valid else "Fetch optical/IR from Legacy Survey")
                )
                fetch_clicked = st.button(
                    _btn_label,
                    key="fetch_optical_ir_btn",
                    use_container_width=True,
                )

        # ---- Optical/IR fetch logic — threaded so a Stop button can abort it ----
        if want_optical and fetch_clicked:
            _opt_size = float(st.session_state.get("optical_ir_size_arcmin", OPTICAL_CUTOUT_SIZE_ARCMIN))
            _opt_size = min(_opt_size, OPTICAL_CUTOUT_SIZE_ARCMIN)  # hard cap at 15 arcmin
            import threading

            _fetch_result = {}
            _fetch_exc    = {}
            _stop_event   = threading.Event()

            def _do_fetch():
                try:
                    # Pass the stop event; fetch_legacy_cutout checks it between bands
                    result = fetch_optical_for_cutout(ra_val, dec_val, _opt_size,
                                                      stop_event=_stop_event)
                    _fetch_result["data"] = result
                except Exception as _ex:
                    _fetch_exc["error"] = _ex

            _thread = threading.Thread(target=_do_fetch, daemon=True)
            _thread.start()

            # Show centred status + Stop button while thread runs
            _sf1, _sf2, _sf3 = st.columns([1, 6, 1])
            with _sf2:
                _status_ph = st.empty()
                _stop_ph   = st.empty()
                _status_ph.markdown(
                    "<p style='text-align:center; color:#aaa;'>⏳ Fetching optical/IR data from Legacy Survey…</p>",
                    unsafe_allow_html=True,
                )
                _stop_pressed = _stop_ph.button("⏹ Stop fetch", key="stop_fetch_btn",
                                                 use_container_width=True)
                if _stop_pressed:
                    _stop_event.set()

            _thread.join()  # wait (already stopped if user pressed Stop)
            _status_ph.empty()
            _stop_ph.empty()

            if _stop_event.is_set():
                _sf1, _sf2, _sf3 = st.columns([1, 6, 1])
                with _sf2:
                    st.warning("Fetch was stopped by user.")
            elif "error" in _fetch_exc:
                _sf1, _sf2, _sf3 = st.columns([1, 6, 1])
                with _sf2:
                    st.warning(f"Could not fetch Legacy Survey data: {_fetch_exc['error']}  —  showing radio only.")
                st.session_state.optical_hdu = None
                st.session_state.ir_hdu = None
                st.session_state.multiwave_optical_ra  = ra_val
                st.session_state.multiwave_optical_dec = dec_val
            else:
                opt_result   = _fetch_result["data"]
                _session_dir = get_session_cutout_dir()

                if opt_result["optical_hdul"] is not None:
                    _opt_fname = (
                        f"optical_{opt_result['optical_layer']}_{ra_val:.5f}_{dec_val:.5f}.fits"
                    )
                    _opt_path = os.path.join(_session_dir, _opt_fname)
                    with open(_opt_path, "wb") as _f:
                        _f.write(opt_result["optical_bytes"])
                    st.session_state.optical_hdu          = opt_result["optical_hdul"]
                    st.session_state.optical_fits_bytes   = opt_result["optical_bytes"]
                    st.session_state.optical_fits_filename = _opt_fname
                    st.session_state.optical_fits_path    = _opt_path
                    st.session_state.optical_layer        = opt_result["optical_layer"]
                else:
                    st.session_state.optical_hdu          = None
                    st.session_state.optical_fits_bytes   = None
                    st.session_state.optical_fits_filename = None
                    st.session_state.optical_fits_path    = None
                    st.session_state.optical_layer        = None

                if opt_result["ir_hdul"] is not None:
                    _ir_fname = (
                        f"ir_{opt_result['ir_layer']}_{ra_val:.5f}_{dec_val:.5f}.fits"
                    )
                    _ir_path = os.path.join(_session_dir, _ir_fname)
                    with open(_ir_path, "wb") as _f:
                        _f.write(opt_result["ir_bytes"])
                    st.session_state.ir_hdu          = opt_result["ir_hdul"]
                    st.session_state.ir_fits_bytes   = opt_result["ir_bytes"]
                    st.session_state.ir_fits_filename = _ir_fname
                    st.session_state.ir_fits_path    = _ir_path
                else:
                    st.session_state.ir_hdu          = None
                    st.session_state.ir_fits_bytes   = None
                    st.session_state.ir_fits_filename = None
                    st.session_state.ir_fits_path    = None

                st.session_state.multiwave_optical_ra   = ra_val
                st.session_state.multiwave_optical_dec  = dec_val
                st.session_state.multiwave_optical_size = _opt_size

            # Refresh locals after fetch
            optical_hdu   = st.session_state.get("optical_hdu")
            ir_hdu        = st.session_state.get("ir_hdu")
            optical_layer = st.session_state.get("optical_layer")
            has_optical   = optical_hdu is not None
            has_ir        = ir_hdu is not None

        # ---- Layer selector — centred via CSS + [1,6,1] column ----
        # Order: Radio only → Radio + Infrared → Radio + Optical
        layer_options = ["Radio only"]
        if has_ir:
            layer_options.append("Infrared (unwise-neo7)")
        if has_optical:
            layer_options.append("Optical (ls-dr11)")

        _l1, _lmid, _l3 = st.columns([1, 6, 1])
        with _lmid:
            if len(layer_options) > 1:
                st.markdown(
                    "<p style='text-align:center; margin-bottom:4px; font-weight:600;'>"
                    "Display layer</p>",
                    unsafe_allow_html=True,
                )
                # Build equal-width button columns — always centered regardless of CSS
                _btn_cols = st.columns(len(layer_options))
                for _i, _opt in enumerate(layer_options):
                    with _btn_cols[_i]:
                        _is_active = st.session_state.get("multiwave_layer_choice", layer_options[0]) == _opt
                        _btn_style = "primary" if _is_active else "secondary"
                        if st.button(_opt, key=f"layer_btn_{_i}", use_container_width=True, type=_btn_style):
                            st.session_state.multiwave_layer_choice = _opt
                            st.rerun()
                chosen_layer = st.session_state.get("multiwave_layer_choice", layer_options[0])
                # Keep choice valid if options changed (e.g. optical removed)
                if chosen_layer not in layer_options:
                    chosen_layer = layer_options[0]
                    st.session_state.multiwave_layer_choice = chosen_layer
            else:
                chosen_layer = layer_options[0]
                st.session_state.multiwave_layer_choice = chosen_layer

        # ---- Percentile slider — only for radio-only view ----
        if chosen_layer == "Radio only" and radio_data_for_plot is not None:
            # Seed once; slider widget owns the value after that.
            if "multiwave_radio_pct_lo" not in st.session_state:
                st.session_state.multiwave_radio_pct_lo = 95.0
            if "multiwave_radio_pct_hi" not in st.session_state:
                st.session_state.multiwave_radio_pct_hi = 99.9
            if "multiwave_radio_pct_range" not in st.session_state:
                st.session_state.multiwave_radio_pct_range = (
                    float(st.session_state.multiwave_radio_pct_lo),
                    float(st.session_state.multiwave_radio_pct_hi),
                )
            # Match image column layout [1,6,1]; slider lives in the centre 6-unit
            # column and is itself split [1,4,1] so it occupies the middle ~4/6 ≈ ⅔
            # of the image width and is centred.
            _ps1, _ps2, _ps3 = st.columns([1, 6, 1])
            with _ps2:
                _psa, _psb, _psc = st.columns([1, 4, 1])
                with _psb:
                    st.slider(
                        "Radio stretch — percentile range",
                        min_value=90.0,
                        max_value=99.99,
                        step=0.05,
                        format="%.2f%%",
                        key="multiwave_radio_pct_range",
                        help="Lower value sets the noise floor; right value clips bright peaks.",
                    )
            pct_lo = float(st.session_state.multiwave_radio_pct_range[0])
            pct_hi = float(st.session_state.multiwave_radio_pct_range[1])
            st.session_state.multiwave_radio_pct_lo = pct_lo
            st.session_state.multiwave_radio_pct_hi = pct_hi
        else:
            pct_lo = float(st.session_state.get("multiwave_radio_pct_lo", 95.0))
            pct_hi = float(st.session_state.get("multiwave_radio_pct_hi", 99.9))

        # ---- Zoom slider — shared across all display layers ----
        _zoom_min = 3.0
        _zoom_max = min(30.0, size_val)
        _zoom_max = max(_zoom_max, _zoom_min)
        if "display_zoom_arcmin" not in st.session_state:
            st.session_state.display_zoom_arcmin = min(size_val, _zoom_max)
        _prev_layer = st.session_state.get("multiwave_layer_choice_prev", "Radio only")
        if _prev_layer == "Radio only" and chosen_layer != "Radio only":
            _opt_fetch_size = float(st.session_state.get("optical_ir_size_arcmin", OPTICAL_CUTOUT_SIZE_ARCMIN))
            st.session_state.display_zoom_arcmin = min(max(_opt_fetch_size, _zoom_min), _zoom_max)
        st.session_state.multiwave_layer_choice_prev = chosen_layer
        _stored = float(st.session_state.display_zoom_arcmin)
        if _stored < _zoom_min or _stored > _zoom_max:
            st.session_state.display_zoom_arcmin = min(max(_stored, _zoom_min), _zoom_max)
        _zs1, _zs2, _zs3 = st.columns([1, 6, 1])
        with _zs2:
            _zsa, _zsb, _zsc = st.columns([1, 4, 1])
            with _zsb:
                zoom_arcmin = st.slider(
                    "Zoom (arcmin)",
                    min_value=_zoom_min,
                    max_value=_zoom_max,
                    step=0.5,
                    format="%.1f′",
                    key="display_zoom_arcmin",
                    help="Zoom into the centre of the image. Moves the axis limits only — no data or stretch recalculation.",
                )

        # ---- Render figure ----
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            if chosen_layer == "Radio only" or radio_data_for_plot is None:
                # Radio-only: matplotlib figure with WCS axes + colorbar
                if radio_data_for_plot is not None and radio_wcs_for_plot is not None:
                    try:
                        fig_buf = render_radio_figure(
                            radio_data=radio_data_for_plot,
                            radio_wcs=radio_wcs_for_plot,
                            ra=ra_val, dec=dec_val,
                            size_arcmin=size_val,
                            radio_pct_lo=pct_lo,
                            radio_pct_hi=pct_hi,
                            zoom_arcmin=zoom_arcmin,
                        )
                        st.image(fig_buf, use_container_width=True, clamp=False)
                    except Exception as _fe:
                        st.warning(f"Radio figure failed ({_fe}); showing plain preview.")
                        st.image(cutout_previews[0], use_container_width=True, clamp=True)
                else:
                    st.image(cutout_previews[0], use_container_width=True, clamp=True)
            else:
                active_hdu = (
                    optical_hdu if "Optical" in chosen_layer and has_optical else ir_hdu
                )
                active_layer = (
                    optical_layer if "Optical" in chosen_layer and has_optical else "unwise-neo7"
                )
                if active_hdu is None:
                    st.warning("Overlay data not available; showing radio only.")
                    st.image(cutout_previews[0], use_container_width=True, clamp=True)
                else:
                    try:
                        fig_buf = render_multiwavelength_figure(
                            radio_data=radio_data_for_plot,
                            radio_wcs=radio_wcs_for_plot,
                            optical_hdul=active_hdu,
                            optical_layer=active_layer,
                            ra=ra_val, dec=dec_val,
                            size_arcmin=size_val,
                            radio_pct_lo=pct_lo,
                            radio_pct_hi=pct_hi,
                            zoom_arcmin=zoom_arcmin,
                        )
                        st.image(fig_buf, use_container_width=True, clamp=False)
                    except Exception as _fe:
                        st.warning(f"Multiwavelength figure failed ({_fe}); showing radio only.")
                        st.image(cutout_previews[0], use_container_width=True, clamp=True)

        # ---- Download row ----
        _opt_path = st.session_state.get("optical_fits_path")
        _ir_path  = st.session_state.get("ir_fits_path")
        _has_optical = bool(_opt_path and os.path.exists(_opt_path))
        _has_ir      = bool(_ir_path  and os.path.exists(_ir_path))
        _has_radio   = bool(fits_path and os.path.exists(fits_path))

        if _has_optical or _has_ir:
            # Full 3-column layout
            dl_cols = st.columns(3)
            if _has_radio:
                with open(fits_path, "rb") as _fh:
                    _radio_bytes = _fh.read()
                with dl_cols[0]:
                    st.download_button(
                        "⬇ Radio FITS",
                        data=_radio_bytes,
                        file_name=os.path.basename(fits_path),
                        mime="application/fits",
                        use_container_width=True,
                        key="dl_radio_fits_btn",
                    )
            if _has_optical:
                with open(_opt_path, "rb") as _fh:
                    _opt_bytes = _fh.read()
                with dl_cols[1]:
                    st.download_button(
                        "⬇ Optical FITS (ls-dr11)",
                        data=_opt_bytes,
                        file_name=st.session_state.get("optical_fits_filename", "optical.fits"),
                        mime="application/fits",
                        use_container_width=True,
                        key="dl_optical_fits_btn",
                    )
            if _has_ir:
                with open(_ir_path, "rb") as _fh:
                    _ir_bytes = _fh.read()
                with dl_cols[2]:
                    st.download_button(
                        "⬇ IR FITS (unwise-neo7)",
                        data=_ir_bytes,
                        file_name=st.session_state.get("ir_fits_filename", "ir.fits"),
                        mime="application/fits",
                        use_container_width=True,
                        key="dl_ir_fits_btn",
                    )
        elif _has_radio:
            # Radio only — center the single button
            with open(fits_path, "rb") as _fh:
                _radio_bytes = _fh.read()
            _dl_l, _dl_m, _dl_r = st.columns([1, 2, 1])
            with _dl_m:
                st.download_button(
                    "⬇ Download Radio FITS",
                    data=_radio_bytes,
                    file_name=os.path.basename(fits_path),
                    mime="application/fits",
                    use_container_width=True,
                    key="dl_radio_fits_btn",
                )

    elif st.session_state.get("cutout_fig_path", None):
        st.image(st.session_state.cutout_fig_path, caption="Cutout", use_container_width=True)
