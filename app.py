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
    /* Main app background and colors */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecfb 100%);
        border-radius: 10px;
    }
    
    /* Colorful section headers */
    .main-header {
        font-size: 2.5rem !important;
        background: linear-gradient(90deg, #0e4194, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.8rem !important;
        color: #0e4194;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    .section-header {
        font-size: 1.5rem !important;
        background: linear-gradient(90deg, #2c3e50, #4a6491);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid #0e4194;
        padding-left: 10px;
    }
    
    /* Card styling for content sections */
    .content-card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        border-top: 4px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .content-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    .card-blue {
        border-top-color: #0e4194;
    }
    
    .card-teal {
        border-top-color: #00bcd4;
    }
    
    .card-purple {
        border-top-color: #673ab7;
    }
    
    .card-orange {
        border-top-color: #ff9800;
    }
    
    /* Modern sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2855 0%, #164584 100%);
        box-shadow: 2px 0px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar sections */
    .sidebar-section {
        margin-bottom: 2rem;
    }
    
    .sidebar-header {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Logo container */
    .sidebar-logo {
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-logo img {
        width: 80%;
        border-radius: 10px;
        padding: 10px;
        background: white;
    }
    
    /* User section */
    .user-info {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .user-avatar {
        background: #3a7bd5;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
    }
    
    .user-name {
        color: white;
        font-weight: 500;
    }
    
    /* Toggle button styling */
    .toggle-button {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 15px;
        margin-bottom: 8px;
        width: 100%;
        text-align: left;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: background 0.3s ease;
    }
    
    .toggle-button:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .toggle-button-active {
        background: rgba(58, 123, 213, 0.6);
    }
    
    .toggle-button-icon {
        width: 20px;
        text-align: center;
    }
    
    /* Configuration sliders */
    [data-testid="stSidebar"] .stSlider > div {
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #3a7bd5;
    }
    
    /* Footer area */
    .sidebar-footer {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 1rem;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.8rem;
        text-align: center;
        background: rgba(0, 0, 0, 0.1);
    }
    
    /* Make sure all text in sidebar is white */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Colorful buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0e4194, #3a7bd5);
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #0c3880, #3373c8);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Info boxes */
    .info-box {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #0e4194;
    }
    
    /* Metrics and KPIs */
    .metric-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0e4194;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
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
    
    /* Login form styling */
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .login-form label {
        font-weight: 500;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .login-form input {
        width: 100%;
        padding: 0.75rem 1rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .login-button {
        width: 100%;
        padding: 0.75rem;
        background: linear-gradient(90deg, #0e4194, #3a7bd5);
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
        cursor: pointer;
    }
    
    .login-button:hover {
        background: linear-gradient(90deg, #0c3880, #3373c8);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        border: 1px solid #e9ecef;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #0e4194;
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
    """Create the sidebar with toggle buttons instead of dropdown"""
    # Logo section
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    try:
        logo_path = "src/VHydro_Logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path)
        else:
            # Fallback if logo not found
            st.markdown("""
            <div style="text-align:center;">
                <h2 style="color:white; margin:0;">VHydro</h2>
                <p style="color:white; margin:5px 0 0 0; opacity:0.8; font-size:0.8rem;">Hydrocarbon Prediction</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error loading sidebar logo: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # User account section - shown only if logged in
    if st.session_state.get('logged_in', False):
        st.markdown('<div class="user-info">', unsafe_allow_html=True)
        email = st.session_state.get('email', 'User')
        initial = email[0].upper() if email and len(email) > 0 else 'U'
        
        st.markdown(f"""
        <div class="user-avatar">{initial}</div>
        <div class="user-name">{email}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Logout button
        if st.button("Logout", key="sidebar_logout", use_container_width=True):
            logout()
            st.rerun()

    # Navigation section with toggle buttons
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)
    
    # Page options
    pages = {
        "Home": {"icon": "🏠", "active": False},
        "Dataset Preparation": {"icon": "📊", "active": False},
        "Model Workflow": {"icon": "🔄", "active": False},
        "Analysis Tool": {"icon": "🔍", "active": False},
        "Results Visualization": {"icon": "📈", "active": False},
    }
    
    # Set active page based on session state
    current_page = st.session_state.get('current_page', 'Home')
    if current_page in pages:
        pages[current_page]["active"] = True
    
    # Create toggle buttons for each page
    for page_name, page_info in pages.items():
        button_class = "toggle-button toggle-button-active" if page_info["active"] else "toggle-button"
        if st.markdown(f"""
        <button class="{button_class}" onclick="
            this.closest('section').querySelector('[data-testid=stFormSubmitButton] button').click();">
            <span>{page_name}</span>
            <span class="toggle-button-icon">{page_info["icon"]}</span>
        </button>
        """, unsafe_allow_html=True):
            st.session_state['current_page'] = page_name
            st.rerun()
    
    # Create a hidden form submit button that will be triggered by the custom buttons
    with st.form(key="navigation_form"):
        submit_button = st.form_submit_button("Navigate", type="primary")
        if submit_button:
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model configuration section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">Model Configuration</div>', unsafe_allow_html=True)
    
    # Cluster configuration with colorful sliders
    min_clusters = st.slider("Min Clusters", 2, 15, 5)
    max_clusters = st.slider("Max Clusters", min_clusters, 15, 10)
    
    # Advanced settings in an expander
    with st.expander("Advanced Settings"):
        train_ratio = st.slider("Training Ratio", 0.5, 0.9, 0.8, 0.05)
        val_ratio = st.slider("Validation Ratio", 0.05, 0.3, 0.1, 0.05)
        test_ratio = st.slider("Test Ratio", 0.05, 0.3, 0.1, 0.05)
        
        # Calculate the total and adjust if needed
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            test_ratio = max(0.05, 1.0 - train_ratio - val_ratio)
            st.warning(f"Adjusted test ratio to {test_ratio:.2f}")
            
        hidden_channels = st.number_input("Hidden Channels", 4, 64, 16, 4)
        num_runs = st.number_input("Number of Runs", 1, 10, 4, 1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add info box at the bottom
    st.markdown("""
    <div style="background:rgba(255,255,255,0.1); border-radius:5px; padding:10px; margin-top:20px;">
        <p style="font-size:0.8rem; margin:0;">
            <strong>VHydro</strong> uses Graph Convolutional Networks to predict hydrocarbon quality zones.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Warning for missing VHydro module
    if not VHYDRO_AVAILABLE:
        st.warning("⚠️ VHydro module unavailable")
    
    return {
        "page": current_page,
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
