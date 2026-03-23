import streamlit as st
import torch
from PIL import Image
import open_clip
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import gdown

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

if 'sb_ra_dec' in locals():
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
    st.dataframe(df_cleaned, use_container_width=True, hide_index=False)
    
    # Add download button for the dataframe
    csv = df_cleaned.to_csv(index=False)
    st.download_button(
        label="Download table as CSV",
        data=csv,
        file_name="similar_sources.csv",
        mime="text/csv",
        use_container_width=True,
        on_click=None
    )

    st.markdown("""
    ### View Images in Aladin Portal
    You can explore these sources in detail through the [EMU Survey Aladin Portal](https://emu-survey.org/progress/aladin.html). 
    Just copy the RA and Dec coordinates from the table above and enter them into the portal to visualize the source. 
    We are also working on adding a feature to display combined radio and infrared images for these RA and Dec positions directly within the app.
    """)
    
    st.image("AladinDisplay.png", caption="Example of source visualization in Aladin Portal", use_container_width=True)

#st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style='text-align: center;'>
        <p style='color: #34495E; font-size: 0.9em; margin-top: 20px;'>
            &copy; Nikhel Gupta | CSIRO
        </p>
    </div>
    """, unsafe_allow_html=True)
