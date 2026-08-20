import streamlit as st
import torch
from PIL import Image
import open_clip
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import gdown
import os
import tempfile
import time
import glob
import getpass
import fnmatch
import shutil
import matplotlib.pyplot as plt
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus
from astropy.io import fits

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
    </style>
    """, unsafe_allow_html=True)

# Display EMU logo
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("emu.png", use_container_width=True)

#col1, col2, col3 = st.columns([1,2,1])
#with col2:
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
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir='./clip_pretrained/')
    
    model_url =  f'https://drive.google.com/uc?id=1e1O-5774mkoGYZYC1gsXiGqDeu7KtOGs'
    #model_url =  f'https://drive.google.com/uc?id=1k0MNw1hyBDejxOovKwhQCPRmJil13ut5'
    model_file = 'epoch_99.pt'
    gdown.download(model_url, model_file, quiet=False)
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    #feature_url =  f'https://drive.google.com/uc?id=1ihgHSS043G60ozg6v32rYUJJFx1uqs_H'
    feature_url =  f'https://drive.google.com/uc?id=1rob5Yzza5m0MRIdCA8xB6QAJje016KJd'
    feature_file = 'all_sbid_image_features.pt'
    gdown.download(feature_url, feature_file, quiet=False)
    all_image_features = torch.load(feature_file)

    #idx_url =  f'https://drive.google.com/uc?id=1o-JWXmfUN1F6VMO6Lq-5U69qLDpyEMQ-'
    #idx_file = 'allidx_sbid_ra_dec.pkl'
    #idx_url =  f'https://drive.google.com/uc?id=14fwWW3KkkRfhAyaBVQeEKszx2iGLTCJc'
    idx_url =  f'https://drive.google.com/uc?id=12Vf7rUsBpRCkJd6FZBV0DMp8GZ-KLCs3'
    idx_file = 'allidx_sbid_ra_dec_flux_catwise.pkl'
    gdown.download(idx_url, idx_file, quiet=False)
    idx_dict = pd.read_pickle(idx_url)
    return model, preprocess, tokenizer, all_image_features, idx_dict

model, preprocess, tokenizer, all_image_features, idx_dict = load_model_and_data()

# Input options
st.sidebar.header("Search Options")
input_option = st.sidebar.radio("Choose input type:", ("Image","Text"))

# Common parameters
remove_galactic = st.sidebar.checkbox("Remove galactic sources", value=True)
above_prob_of = st.sidebar.slider("Minimum probability", 0.0, 1.0, 0.9, 0.01)
top_n = st.sidebar.slider("Number of top results to display", 1, 5000, 200)

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

if input_option == "Text":
    search_for = st.text_input("Enter object to search for:", "A bent tailed radio galaxy")
    if st.button("Search", key="text_search"):
        with st.spinner("Searching..."):
            text = ["star forming radio galaxy", "bent-tail radio galaxy", "a peculiar radio galaxy", 
                    "an FR-I", "an FR-II", "a compact circular radio galaxy", "a cat", search_for]
            text_token = tokenizer(text)
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = model.encode_text(text_token)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * all_image_features @ text_features.T).softmax(dim=-1)
            text_probs_np = text_probs.numpy()
            idx_above_prob = np.where(text_probs_np[:,-1] > above_prob_of)[0]
            idx_above_prob_sorted = idx_above_prob[np.argsort(text_probs_np[idx_above_prob, -1].flatten())[::-1]]
            sb_ra_dec = [idx_dict.get(val, "Key not found") for val in idx_above_prob_sorted]
            filtered_probs = text_probs_np[idx_above_prob_sorted, -1].flatten()
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

    # Download button for the currently shown df
    csv = df_cleaned.to_csv(index=False)
    st.download_button(
        label="Download table as CSV",
        data=csv,
        file_name="similar_sources.csv",
        mime="text/csv",
        use_container_width=True,
        on_click=None
    )
else:
    # On page reloads/reruns, if we had results before, show them again so they don't vanish when using cutout button!
    if st.session_state.get("results_df", None) is not None:
        df_cleaned = st.session_state["results_df"]
        st.subheader(f"Top {min(top_n, len(df_cleaned))} similar sources (restored):")
        st.dataframe(df_cleaned.head(top_n), use_container_width=True, hide_index=False)
        csv = df_cleaned.head(top_n).to_csv(index=False)
        st.download_button(
            label="Download table as CSV",
            data=csv,
            file_name="similar_sources.csv",
            mime="text/csv",
            use_container_width=True,
            on_click=None
        )

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
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                n_fetched = 0

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
                        st.write(url_list)
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
                                ax = axes.flat[n_fetched]
                                im = ax.imshow(
                                    data_plot, origin="lower", cmap="gray"
                                )
                                ax.set_title(f"RA={ra:.5f} Dec={dec:.5f}")
                                ax.set_xlabel("Pixel X")
                                ax.set_ylabel("Pixel Y")
                                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                                n_fetched += 1
                                hdul.close()
                                cutout_files.append(fitsfile)
                                break
                    except Exception as e:
                        continue
                    if n_fetched >= 9:
                        break

                for ax in axes.flat[n_fetched:]:
                    ax.axis("off")

                plt.tight_layout()
                cutout_fig_path = os.path.join(downloads_dir, "cutouts.png")
                fig.savefig(cutout_fig_path)
                plt.close(fig)
                st.session_state.cutout_fig_path = cutout_fig_path
                st.session_state.cutout_downloads_dir = downloads_dir
                st.session_state.cutout_files = cutout_files

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

    # Show cutout plot if available
    if st.session_state.get("cutout_fig_path", None):
        st.image(st.session_state.cutout_fig_path, caption="3x3 Cutouts", use_column_width=True)
        # Clean up
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
            st.success("Cleaned up cutout files.")
