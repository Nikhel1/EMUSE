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
import tempfile
import time

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
import astropy.units as u

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

    if st.button("Open Gemini assistant for this table", key="open_table_assistant_btn", use_container_width=True):
        st.session_state.table_assistant_open = True

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
    
    model_url =  f'https://drive.google.com/uc?id=1e1O-5774mkoGYZYC1gsXiGqDeu7KtOGs'
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
top_n = st.sidebar.slider("Number of top results to display", 1, 5000, 200)
use_gemini_llm = False
gemini_api_key = ""
if "gemini_api_key_saved" not in st.session_state:
    st.session_state.gemini_api_key_saved = ""
st.sidebar.markdown("---")
st.sidebar.subheader("Gemini Settings")
if input_option == "Text":
    use_gemini_llm = st.sidebar.checkbox("Use Gemini", value=True)
else:
    st.sidebar.caption("Image search does not require Gemini.")
    st.sidebar.caption("Gemini is used for table Q&A assistant only.")
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

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
with st.sidebar.expander(" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ℹ️ &nbsp;&nbsp; How to Use EMUSE"):
    st.markdown("""
    ### Search Methods

    #### Text Search
    - Select 'Text' from the sidebar options
    - Enter a description of the astronomical object you're looking for (e.g., "A bent tailed radio galaxy")
    - Click 'Search' to find matching objects from the EMU Survey

    #### Image Search  
    - Select 'Image' from the sidebar options
    - Upload a reference image (.jpg, .jpeg, or .png format). The image can just be the screenshot of 
    your favorite radio source in EMU or any other survey
    - Click 'Search' to find visually similar objects

    ### Search Parameters

    #### Remove Galactic Sources
    - When checked, filters out objects within 10 degrees of the galactic plane
    - Helps focus on extragalactic sources
    - Recommended for most searches

    #### Minimum Probability
    - Sets the confidence threshold for matches (0.0 to 1.0)
    - Higher values (e.g., 0.9) give more precise but fewer results
    - Lower values include more results but may be less accurate

    #### Number of Top Results
    - Controls how many matching objects to display
    - Range: 1 to 5000 results
    - Default: 200 results
    - Adjust based on your needs and search specificity
    """)

# Persist search results/cutout state across reruns
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "cutout_fig_path" not in st.session_state:
    st.session_state.cutout_fig_path = None
if "show_search_results" not in st.session_state:
    st.session_state.show_search_results = False

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
                row_left, row_right = st.columns([1, 5])
                with row_left:
                    keep_this = st.checkbox(
                        "Use",
                        value=st.session_state.get(f"query_checkbox_{i}", True),
                        key=f"query_checkbox_{i}",
                        label_visibility="collapsed",
                    )
                with row_right:
                    edited_query = st.text_input(
                        f"Query {i+1}",
                        value=st.session_state.get(f"query_text_{i}", query),
                        key=f"query_text_{i}",
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
                    st.session_state.sb_ra_dec = sb_ra_dec
                    st.session_state.filtered_probs = filtered_probs
                    st.session_state.input_option = input_option
                    st.session_state.show_search_results = True
elif input_option == "Image":
    uploaded_file = st.file_uploader("Upload an image to to search for similar objects...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        image_upload = preprocess(Image.open(uploaded_file)).unsqueeze(0)
        if st.button("Search", key="image_search"):
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

    st.success(f"Found {len(filtered_sb_ra_dec)} sources {'outside galactic regions ' if remove_galactic else ''}above probability of {above_prob_of}.")
    if len(filtered_sb_ra_dec)<top_n:
        top_n = len(filtered_sb_ra_dec)
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

    # REMOVED Download CSV button here

    render_gemini_table_assistant(df_cleaned, gemini_api_key)

    # --- Nice horizontal divider before cutouts section ---
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

else:
    # On page reloads/reruns, if we had results before, show them again so they don't vanish when using cutout button!
    if st.session_state.get("results_df", None) is not None:
        df_cleaned = st.session_state["results_df"]
        st.subheader(f"Top {min(top_n, len(df_cleaned))} similar sources (restored):")
        st.dataframe(df_cleaned.head(top_n), use_container_width=True, hide_index=False)
        # REMOVED Download CSV button here
        render_gemini_table_assistant(df_cleaned.head(top_n), gemini_api_key)

        # --- Nice horizontal divider before cutouts section ---
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

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


# Only show cutout button if there are results
results_for_cutout = st.session_state.get("results_df", None)
if results_for_cutout is not None and not results_for_cutout.empty:
    # --- BEGIN CASDA CREDENTIALS STATEFUL PATCH ---
    # Persist CASDA username/password fields across reruns
    if "casda_username" not in st.session_state:
        st.session_state.casda_username = ""
    if "casda_password" not in st.session_state:
        st.session_state.casda_password = ""
    if "cutouts_login_error" not in st.session_state:
        st.session_state.cutouts_login_error = False

    generate_cutouts = st.button("Generate 3x3 Cutouts for Top 9 Sources", use_container_width=True, key="generate_cutouts_btn")
    cutouts_already = st.session_state.get("cutout_fig_path", None) is not None

    # If either asked for cutout or already have cutout, ask for credentials and/or show cutout
    show_credential_fields = generate_cutouts or cutouts_already or st.session_state.cutouts_login_error

    # To persist: show_credential_fields determines if credentials dialog is open
    if "show_credential_fields" not in st.session_state:
        st.session_state.show_credential_fields = False

    if generate_cutouts:
        st.session_state.show_credential_fields = True

    if show_credential_fields or st.session_state.show_credential_fields:
        st.session_state.show_credential_fields = True
        st.info(
            "Please enter your CASDA OPAL credentials to access data. If you don't have an account, register at https://data.csiro.au/domain/casda"
        )
        # Put the username and password fields inside the form!
        with st.form("casda_credentials_form", clear_on_submit=False):
            username = st.text_input("CASDA Username", key="casda_username")
            password = st.text_input("CASDA Password", type="password", key="casda_password")
            submitted = st.form_submit_button("Login & Generate Cutouts", use_container_width=True)

        # Handle form submission for credentials and cutout generation
        if submitted:
            if username and password:
                st.session_state.cutout_fig_path = None
                downloads_dir = os.path.join(os.getcwd(), "Downloads")
                os.makedirs(downloads_dir, exist_ok=True)

                casdatap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
                casda = Casda()
                try:
                    # Some casda clients will trigger a prompt for password via getpass – we can monkeypatch getpass to provide our password automatically
                    _orig_getpass = getpass.getpass

                    def _autofill_getpass(prompt='Password: ', stream=None):
                        return password

                    getpass.getpass = _autofill_getpass
                    try:
                        casda.login(username=username)
                    finally:
                        getpass.getpass = _orig_getpass

                    st.session_state.cutouts_login_error = False
                except Exception as e:
                    st.session_state.cutouts_login_error = True
                    st.warning(f"CASDA login failed: {e}")
                    st.stop()

                # Auth done, generating cutouts with progress!
                st.success("✅ Authentication successful. Now generating cutouts...")

                # Timing the cutout generation
                start_time = time.time()

                ra_vals = results_for_cutout["RA"].astype(float).values[:9]
                dec_vals = results_for_cutout["Dec"].astype(float).values[:9]
                cutout_files = []
                cutout_previews = []
                cutout_meta = []

                # Add a progress bar in Streamlit
                progress_bar = st.progress(0, text="Starting cutout generation...")

                total = min(9, len(ra_vals))
                # To consistently update status text
                status_placeholder = st.empty()

                for i, (ra, dec) in enumerate(zip(ra_vals, dec_vals)):
                    progress_bar.progress(i / total, text=f"Fetching cutout {i+1} of {total}")

                    status_placeholder.info(f"Fetching cutout {i+1} of {total} (RA={ra:.5f}, Dec={dec:.5f})...")
                    coord = SkyCoord(ra, dec, unit="deg")
                    try:
                        result = Casda.query_region(coord, radius=2 * u.arcmin)
                        pattern = 'image.i.EMU*taylor.0.restored.conv.fits'
                        mask = [fnmatch.fnmatch(fn, pattern) for fn in result['filename']]
                        data = result[mask]
                        if len(data) == 0:
                            continue
                        url_list = casda.cutout(data[:1], coordinates=coord, radius=4 * u.arcmin)
                        filelist = casda.download_files(url_list, savedir=downloads_dir)
                        for fitsfile in filelist:
                            if fitsfile.endswith(".fits") and os.path.exists(fitsfile):
                                hdul = fits.open(fitsfile)
                                hdu_index = 0
                                if hdul[hdu_index].data is None and len(hdul) > 1:
                                    hdu_index = 1
                                data_img = hdul[hdu_index].data
                                data_img = np.nan_to_num(data_img)
                                if data_img.ndim == 4:
                                    data_plot = data_img[0, 0, :, :]
                                elif data_img.ndim == 3:
                                    data_plot = data_img[0, :, :]
                                else:
                                    data_plot = data_img
                                preview_img = prepare_cutout_preview(data_plot)
                                cutout_previews.append(preview_img)
                                cutout_meta.append(
                                    {
                                        "ra": ra,
                                        "dec": dec,
                                        "fits_file": fitsfile,
                                    }
                                )
                                hdul.close()
                                cutout_files.append(fitsfile)
                                break
                    except Exception as e:
                        continue
                    if len(cutout_previews) >= 9:
                        break
                st.session_state.cutout_fig_path = None
                st.session_state.cutout_downloads_dir = downloads_dir
                st.session_state.cutout_files = cutout_files
                st.session_state.cutout_previews = cutout_previews
                st.session_state.cutout_meta = cutout_meta

                elapsed_time = time.time() - start_time
                progress_bar.progress(1.0, text="Cutout generation complete!")
                status_placeholder.success(f"✅ Cutout generation complete in {elapsed_time:.1f} seconds.")

            else:
                st.warning("Please enter both username and password before continuing.")

        # The following block caused NameError due to credentials_ready not being defined.
        # It is removed/replaced: we simply rely on the above "if submitted:" block to handle credential/cutout logic.
        # If more complex workflow is needed (e.g. another way to trigger cutout generation),
        # you can manage state with st.session_state, e.g. st.session_state.cutouts_submit_triggered.

    # --- END CASDA CREDENTIALS STATEFUL PATCH ---

    # Show cutouts as app-friendly cards if available
    cutout_previews = st.session_state.get("cutout_previews", [])
    cutout_meta = st.session_state.get("cutout_meta", [])
    if cutout_previews:
        st.subheader("Top Cutouts")
        st.caption("Enhanced contrast preview of top sources")
        for row_start in range(0, len(cutout_previews), 3):
            cols = st.columns(3)
            for j in range(3):
                idx = row_start + j
                if idx >= len(cutout_previews):
                    continue
                with cols[j]:
                    meta = cutout_meta[idx] if idx < len(cutout_meta) else {}
                    st.image(
                        cutout_previews[idx],
                        caption=f"#{idx+1}  RA={meta.get('ra', 0):.5f}, Dec={meta.get('dec', 0):.5f}",
                        use_container_width=True,
                        clamp=True,
                    )
                    if meta.get("fits_file"):
                        st.caption(f"`{os.path.basename(meta['fits_file'])}`")
    elif st.session_state.get("cutout_fig_path", None):
        st.image(st.session_state.cutout_fig_path, caption="3x3 Cutouts", use_container_width=True)
    if cutout_previews or st.session_state.get("cutout_fig_path", None):
        if st.button("Clean up downloaded cutouts", key="cleanup_cutouts_btn"):
            try:
                for file in st.session_state.get("cutout_files", []):
                    if os.path.exists(file):
                        os.remove(file)
            except Exception:
                pass
            try:
                downloads_dir = st.session_state.get("cutout_downloads_dir", None)
                if downloads_dir and os.path.isdir(downloads_dir):
                    shutil.rmtree(downloads_dir)
            except Exception:
                pass
            st.session_state.cutout_fig_path = None
            st.session_state.cutout_previews = []
            st.session_state.cutout_meta = []
            st.success("Cleaned up cutout files.")
