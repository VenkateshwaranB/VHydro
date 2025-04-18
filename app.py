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
    /* Modern Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2855 0%, #164584 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 3px 0px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar content styling */
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1rem;
    }
    
    /* Logo container */
    .sidebar-logo {
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .sidebar-logo img {
        max-width: 80%;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        background: white;
        padding: 8px;
        transition: transform 0.3s ease;
    }
    
    .sidebar-logo img:hover {
        transform: scale(1.05);
    }
    
    /* Section headers */
    .sidebar-header {
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* User account panel */
    .user-account {
        background-color: rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 15px;
        margin: 0 0 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .user-account:hover {
        background-color: rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .user-email {
        font-weight: 600;
        font-size: 1rem;
        color: white !important;
        margin-bottom: 12px;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .user-email svg {
        margin-right: 8px;
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* User buttons */
    .user-buttons {
        display: flex;
        gap: 8px;
    }
    
    .sidebar-button {
        flex: 1;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        padding: 8px 15px;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        text-align: center;
        cursor: pointer;
        box-shadow: 0 2px 3px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar-button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .sidebar-button-primary {
        background: rgba(59, 130, 246, 0.4);
    }
    
    .sidebar-button-secondary {
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Navigation dropdown styling */
    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 1.5rem;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 8px;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div > div {
        color: white !important;
    }
    
    /* Slider styling */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #4285f4;
    }
    
    /* Separator line */
    .sidebar-separator {
        height: 1px;
        background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0) 100%);
        margin: 1.5rem 0;
    }
    
    /* Info box styling */
    .sidebar-info {
        background: rgba(66, 133, 244, 0.1);
        border-left: 3px solid #4285f4;
        padding: 12px;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Settings expander styling */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: white !important;
        padding: 10px 15px;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 0 0 8px 8px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: none;
        margin-top: -1rem;
    }
    
    /* Make sure all text in sidebar is white */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Warning styling */
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(255, 183, 77, 0.2) !important;
        color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(255, 183, 77, 0.3) !important;
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
    # Add user icon SVG
    user_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
      <path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6m2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0m4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4m-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10s-3.516.68-4.168 1.332c-.678.678-.83 1.418-.832 1.664z"/>
    </svg>
    """
    
    # Logo section with enhanced styling
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    try:
        logo_path = "src/VHydro_Logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=160)
        else:
            # Fallback if logo not found
            st.markdown("""
            <div style="width:100%; text-align:center; background:white; border-radius:10px; padding:20px; margin-bottom:15px;">
                <h2 style="color:#0a2855; margin:0;">VHydro</h2>
                <p style="color:#0a2855; margin:5px 0 0 0; font-size:0.8rem;">Hydrocarbon Quality Prediction</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error loading sidebar logo: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # User account section
    st.markdown('<div class="user-account">', unsafe_allow_html=True)
    st.markdown(f'<div class="user-email">{user_icon} {st.session_state.get("email", "User")}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-buttons">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Logout", key="sidebar_logout", use_container_width=True):
            logout()
            st.rerun()
    with col2:
        if st.button("Account", key="sidebar_account", use_container_width=True):
            st.session_state['current_page'] = "Account"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Navigation section
    st.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)
    
    page_options = ["Home", "Dataset Preparation", "Model Workflow", "Analysis Tool", 
                    "Results Visualization", "Account"]
    
    # Use selectbox with enhanced styling for navigation
    page = st.selectbox(
        "Select a Section",
        page_options,
        index=page_options.index(st.session_state.get('current_page', 'Home'))
    )
    
    # Update current page in session state
    st.session_state['current_page'] = page
    
    # Model configuration section
    st.markdown('<div class="sidebar-header">Model Configuration</div>', unsafe_allow_html=True)
    
    # Cluster configuration sliders
    min_clusters = st.slider("Min Clusters", 2, 15, 5)
    max_clusters = st.slider("Max Clusters", min_clusters, 15, 10)
    
    # Advanced settings in an expander
    with st.expander("Advanced Settings"):
        train_ratio = st.slider("Training Data Ratio", 0.5, 0.9, 0.8, 0.05)
        val_ratio = st.slider("Validation Data Ratio", 0.05, 0.3, 0.1, 0.05)
        test_ratio = st.slider("Test Data Ratio", 0.05, 0.3, 0.1, 0.05)
        
        # Add a visual separator
        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
        
        hidden_channels = st.number_input("Hidden Channels", 4, 64, 16, 4)
        num_runs = st.number_input("Number of Runs", 1, 10, 4, 1)
    
    # Adjust test_ratio to make sure ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        test_ratio = max(0.05, 1.0 - train_ratio - val_ratio)
        st.warning(f"Adjusted test ratio to {test_ratio:.2f} to ensure total equals 1.0")
    
    # Add a styled info box
    st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("""
    **VHydro** predicts hydrocarbon quality zones using petrophysical properties 
    and Graph Convolutional Networks.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Warning for missing VHydro module
    if not VHYDRO_AVAILABLE:
        st.warning("⚠️ VHydro module is not available. Some features will be disabled.")
    
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
# Home page
def home_page():
    logo_path = "src/Building a Greener World.png"  # Update path as needed
    header_with_logo(logo_path)
    
    st.markdown("<h2 class='sub-header'>About VHydro</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>What is VHydro?</h3>
        <p>
        <b>VHydro</b> is an advanced tool for hydrocarbon quality prediction using well log data.
        It combines traditional petrophysical analysis with modern machine learning techniques
        to provide accurate predictions of reservoir quality.
        </p>
        <p>
        The tool uses Graph Convolutional Networks (GCN) to model the complex relationships
        between different petrophysical properties and depth values, enabling more accurate
        classification of hydrocarbon potential zones.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to display workflow diagram
    workflow_image_path = "src/Workflow.png"
    
    if os.path.exists(workflow_image_path):
        st.markdown("<h2 class='sub-header'>Workflow Overview</h2>", unsafe_allow_html=True)
        display_image_with_caption(workflow_image_path, "VHydro Workflow")
    else:
        st.warning(f"Workflow image not found. Expected at: {workflow_image_path}")
    
    st.markdown("<h2 class='sub-header'>Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Petrophysical Property Calculation**
          - Shale Volume
          - Porosity
          - Water/Oil Saturation
          - Permeability
        
        - **Facies Classification**
          - K-means Clustering
          - Silhouette Score Optimization
          - Depth-based Facies Mapping
        """)
        
    with col2:
        st.markdown("""
        - **Graph-based Machine Learning**
          - Graph Convolutional Networks
          - Node and Edge Feature Extraction
          - Hydrocarbon Quality Classification
        
        - **Visualization and Reporting**
          - Facies Visualization
          - Prediction Accuracy Metrics
          - Classification Reports
        """)
    
    st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
    st.markdown("""
    1. Navigate to the **Dataset Preparation** section to understand the data requirements
    2. Use the **Analysis Tool** to upload and process your well log data
    3. Run the model with your preferred configuration
    4. Explore the results in the **Results Visualization** section
    
    Use the sidebar to navigate between different sections of the application.
    """)

# Account page
def account_page():
    """Render the user account page"""
    st.markdown('<h1 class="main-header">Account Settings</h1>', unsafe_allow_html=True)
    
    # Show user account page from the firebase_auth module
    user_account_page()

# Dataset preparation page
def dataset_preparation_page():
    """Render the dataset preparation page"""
    st.markdown('<h1 class="main-header">Dataset Preparation</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">Upload your LAS file containing well log data. The file will be processed to calculate petrophysical properties necessary for hydrocarbon potential prediction.</p>', unsafe_allow_html=True)
    
    # Rest of your dataset preparation page code...
    st.markdown("This page would contain information about dataset preparation.")

# Model workflow page
def model_workflow_page():
    """Render the model workflow page"""
    st.markdown('<h1 class="main-header">Model Workflow</h1>', unsafe_allow_html=True)
    
    # Rest of your model workflow page code...
    st.markdown("This page would contain information about the model workflow.")

# Analysis tool page
def analysis_tool_page(config):
    """Render the analysis tool page"""
    st.markdown('<h1 class="main-header">Analysis Tool</h1>', unsafe_allow_html=True)
    
    # Rest of your analysis tool page code...
    st.markdown("This page would contain the analysis tool.")

# Results visualization page
def results_visualization_page():
    """Render the results visualization page"""
    st.markdown('<h1 class="main-header">Results Visualization</h1>', unsafe_allow_html=True)
    
    # Rest of your results visualization page code...
    st.markdown("This page would contain results visualization.")

# Main function
def main():
    # Create sidebar and get configuration
    config = create_sidebar()
    
    # Display selected page
    if config["page"] == "Home":
        home_page()
    elif config["page"] == "Dataset Preparation":
        dataset_preparation_page()
    elif config["page"] == "Model Workflow":
        model_workflow_page()
    elif config["page"] == "Analysis Tool":
        analysis_tool_page(config)
    elif config["page"] == "Results Visualization":
        results_visualization_page()
    elif config["page"] == "Account":
        account_page()
    
    # Footer
    st.markdown("<div class='footer'>VHydro - Advanced Hydrocarbon Quality Prediction © 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
