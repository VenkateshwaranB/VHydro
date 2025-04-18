import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import base64
from PIL import Image
from io import BytesIO
import sys
import importlib.util
import logging

# Import Firebase Authentication module
from firebase_auth import authenticate, user_account_page, logout

# IMPORTANT: Set page configuration at the very beginning before any other Streamlit call
st.set_page_config(
    page_title="VHydro - Hydrocarbon Quality Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check authentication status
is_authenticated = authenticate()

# If user is not authenticated, stop execution here
if not is_authenticated:
    st.stop()

# User is authenticated, continue with the app
st.success(f"Welcome, {st.session_state.get('email', 'User')}!")

# Now we can safely try to import VHydro - handle missing dependencies gracefully
try:
    # Add current directory to path to import VHydro
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from VHydro_final import VHydro
    VHYDRO_AVAILABLE = True
    logger.info("Successfully imported VHydro module")
except ImportError as e:
    VHYDRO_AVAILABLE = False
    logger.error(f"Error importing VHydro module: {e}")
    st.error(f"Failed to import VHydro module: {e}")

# Function to create temp directory
def create_temp_dir():
    return tempfile.mkdtemp()

# Function to get image as base64 for embedding
def get_image_as_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

# Function to display images with a caption
def display_image_with_caption(image_path, caption="", width=None):
    try:
        image = Image.open(image_path)
        if width:
            w, h = image.size
            ratio = width / w
            image = image.resize((width, int(h * ratio)))
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        st.image(buffered, caption=caption, use_column_width=width is None)
    except Exception as e:
        logger.error(f"Error displaying image {image_path}: {e}")
        st.error(f"Could not display image: {e}")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #0c326f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem !important;
        color: #0c326f;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem !important;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #0c326f;
        color: white;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1e4d9e;
        color: white;
    }
    .logo-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #888;
        font-size: 0.8rem;
    }
    div.stTabs button {
        font-weight: bold;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c326f 0%, #1e4d9e 100%);
    }
    
    /* Make ALL sidebar text white */
    [data-testid="stSidebar"] {
        color: white !important;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Style sidebar title explicitly */
    [data-testid="stSidebar"] .sidebar-content h1 {
        color: white !important;
        font-weight: 600;
    }
    
    /* Style radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
    
    /* Style sliders in sidebar */
    [data-testid="stSidebar"] .stSlider label {
        color: white !important;
    }
    
    /* Style number inputs in sidebar */
    [data-testid="stSidebar"] .stNumberInput label {
        color: white !important;
    }
    
    /* Style buttons in sidebar */
    [data-testid="stSidebar"] button {
        background-color: rgba(255, 255, 255, 0.1);
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Force white text for expander content */
    [data-testid="stSidebar"] .st-expander .st-expander-content {
        color: white !important;
    }
    
    /* About section styling */
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #0c326f;
    }
    
    /* User account section styling */
    .user-account {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .user-email {
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .user-actions {
        display: flex;
        justify-content: space-between;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.5rem !important;
        color: #0c326f;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.8rem !important;
        color: #0c326f;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    .section-header {
        font-size: 1.5rem !important;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid #0c326f;
        padding-left: 10px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding-top: 1rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def header_with_logo(logo_path):
    # Custom CSS for the header area
    st.markdown("""
    <style>
    .banner-container {
        width: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
    }
    .banner-container img {
        width: 100%;
        height: auto;
        object-fit: cover;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        if os.path.exists(logo_path):
            # Display the logo as a full-width banner using HTML
            st.markdown(f"""
            <div class="banner-container">
                <img src="data:image/png;base64,{get_image_base64(logo_path)}" alt="VHydro Banner">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Logo image not found. Expected at: {logo_path}")
    except Exception as e:
        logger.error(f"Error loading logo: {e}")
        st.warning(f"Error loading logo: {e}")
    
    st.markdown("<h1 class='main-header'>VHydro - Hydrocarbon Quality Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Advanced Graph Convolutional Network for Petrophysical Analysis</p>", unsafe_allow_html=True)

# Helper function to convert an image to base64 for inline HTML display
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Sidebar navigation
def create_sidebar():
    try:
        # Try to load logo for sidebar
        logo_path = "src/VHydro_Logo.png"
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, width=100)
        else:
            st.sidebar.info("Logo not found at: " + logo_path)
    except Exception as e:
        logger.error(f"Error loading sidebar logo: {e}")

    # Display user account info in sidebar
    st.sidebar.markdown('<div class="user-account">', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="user-email">👤 {st.session_state.get("email", "User")}</div>', unsafe_allow_html=True)
    
    # Add logout and account buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Logout", key="sidebar_logout"):
            logout()
            st.experimental_rerun()
    with col2:
        if st.button("Account", key="sidebar_account"):
            st.session_state['current_page'] = "Account"
            st.experimental_rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page_options = ["Home", "Dataset Preparation", "Model Workflow", "Analysis Tool", 
                    "Results Visualization", "Account"]
    
    # Use selectbox instead of radio for better mobile experience
    page = st.sidebar.selectbox(
        "Select a Section",
        page_options,
        index=page_options.index(st.session_state.get('current_page', 'Home'))
    )
    
    # Update current page in session state
    st.session_state['current_page'] = page
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Configuration")
    
    # Cluster configuration
    min_clusters = st.sidebar.slider("Min Clusters", 2, 15, 5)
    max_clusters = st.sidebar.slider("Max Clusters", min_clusters, 15, 10)
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        train_ratio = st.slider("Training Data Ratio", 0.5, 0.9, 0.8, 0.05)
        val_ratio = st.slider("Validation Data Ratio", 0.05, 0.3, 0.1, 0.05)
        test_ratio = st.slider("Test Data Ratio", 0.05, 0.3, 0.1, 0.05)
        hidden_channels = st.number_input("Hidden Channels", 4, 64, 16, 4)
        num_runs = st.number_input("Number of Runs", 1, 10, 4, 1)
    
    # Adjust test_ratio to make sure ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        test_ratio = max(0.05, 1.0 - train_ratio - val_ratio)
        st.sidebar.warning(f"Adjusted test ratio to {test_ratio:.2f} to ensure total equals 1.0")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **VHydro** predicts hydrocarbon quality zones using petrophysical properties 
    and Graph Convolutional Networks.
    """)
    
    if not VHYDRO_AVAILABLE:
        st.sidebar.warning("⚠️ VHydro module is not available. Some features will be disabled.")
    
    return {
        "page": page,
        "min_clusters": min_clusters,
        "max_clusters": max_clusters,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "hidden_channels": hidden_channels,
        "num_runs": num_runs
    }

# Account page
def account_page():
    """Render the user account page"""
    st.markdown('<h1 class="main-header">Account Settings</h1>', unsafe_allow_html=True)
    
    # Show user account page from the firebase_auth module
    user_account_page()
def main():
    # Check if user is authenticated
    is_authenticated = authenticate()
    
    if not is_authenticated:
        # If not authenticated, stop execution (auth page was already shown by authenticate())
        st.stop()
    
    # User is authenticated, continue with app
    config = create_sidebar()
    
    # Display selected page based on sidebar selection
    if config["page"] == "Home":
        home_page()
    elif config["page"] == "Dataset Preparation":
        dataset_preparation_page()  # This function must be defined or imported
    elif config["page"] == "Model Workflow":
        model_workflow_page()
# # Main function
# def main():
#     # Create sidebar and get configuration
#     config = create_sidebar()
    
#     # Display selected page
#     if config["page"] == "Home":
#         home_page()
#     elif config["page"] == "Dataset Preparation":
#         dataset_preparation_page()
#     elif config["page"] == "Model Workflow":
#         model_workflow_page()
#     elif config["page"] == "Analysis Tool":
#         analysis_tool_page(config)
    elif config["page"] == "Results Visualization":
        results_visualization_page()
    elif config["page"] == "Account":
        account_page()
    
    # Footer
    st.markdown("<div class='footer'>VHydro - Advanced Hydrocarbon Quality Prediction © 2025</div>", unsafe_allow_html=True)

# Rest of your functions (home_page, dataset_preparation_page, etc.) remain the same
# ...

if __name__ == "__main__":
    main()
