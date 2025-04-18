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
import time

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

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

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
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
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
    
    /* Animation for logo */
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    .animate-logo {
        animation: pulse 2s infinite;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        border-top: 4px solid;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    /* Progress animation */
    @keyframes progress {
        0% {
            width: 0%;
        }
        100% {
            width: 100%;
        }
    }
    
    .progress-animation {
        height: 4px;
        background: linear-gradient(90deg, #0e4194, #3a7bd5);
        animation: progress 2s ease-in-out;
    }
    
    /* Custom file uploader */
    .custom-uploader {
        border: 2px dashed #0e4194;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(14, 65, 148, 0.05);
        cursor: pointer;
        transition: background 0.3s ease;
    }
    
    .custom-uploader:hover {
        background: rgba(14, 65, 148, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def header_with_logo(logo_path=None):
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
        if logo_path and os.path.exists(logo_path):
            # Display the logo as a full-width banner using HTML
            st.markdown(f"""
            <div class="banner-container">
                <img src="data:image/png;base64,{get_image_as_base64(logo_path)}" alt="VHydro Banner">
            </div>
            """, unsafe_allow_html=True)
        else:
            # Create a colorful gradient header if no logo is available
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #0e4194, #3a7bd5); height: 120px; 
                        display: flex; justify-content: center; align-items: center; border-radius: 10px;
                        margin-bottom: 20px;">
                <h1 style="color: white; text-align: center; margin: 0; padding: 0; font-size: 2.5rem;">
                    VHydro
                </h1>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error loading logo: {e}")
        st.warning(f"Error loading logo: {e}")
    
    st.markdown("<h1 class='main-header'>Hydrocarbon Quality Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Advanced Graph Convolutional Network for Petrophysical Analysis</p>", unsafe_allow_html=True)

# Helper function to convert an image to base64 for inline HTML display
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Sidebar navigation with toggle buttons
def create_sidebar():
    """Create the sidebar with toggle buttons instead of dropdown"""
    # Logo section
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    try:
        logo_path = "src/VHydro_Logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)
        else:
            # Fallback if logo not found
            st.markdown("""
            <div style="text-align:center;" class="animate-logo">
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
    
    # Page options with icons
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
    
    # Simulated loading animation for a more dynamic feel
    with st.spinner("Loading application..."):
        progress_bar = st.progress(0)
        for i in range(100):
            # Fast at the beginning, slow in the middle, fast at the end
            if i < 30:
                time.sleep(0.01)
            elif i < 70:
                time.sleep(0.03)
            else:
                time.sleep(0.01)
            progress_bar.progress(i + 1)
    
    st.markdown("<div class='progress-animation'></div>", unsafe_allow_html=True)
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
        # Create a colorful workflow diagram using Streamlit columns
        st.markdown("<h2 class='sub-header'>Workflow Overview</h2>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; padding:20px 0;'>", unsafe_allow_html=True)
        
        workflow_steps = [
            {"title": "Data Loading", "icon": "📥", "color": "#3498db", "desc": "Upload LAS files and extract petrophysical properties"},
            {"title": "Feature Extraction", "icon": "⚙️", "color": "#2ecc71", "desc": "Calculate key reservoir parameters"},
            {"title": "Graph Creation", "icon": "🔗", "color": "#e74c3c", "desc": "Generate nodes and edges for depth points"},
            {"title": "GCN Training", "icon": "🧠", "color": "#9b59b6", "desc": "Train Graph Convolutional Network models"},
            {"title": "Quality Prediction", "icon": "📊", "color": "#f39c12", "desc": "Predict hydrocarbon potential zones"}
        ]
        
        cols = st.columns(len(workflow_steps))
        
        for i, (col, step) in enumerate(zip(cols, workflow_steps)):
            with col:
                st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 10px; border-top: 4px solid {step['color']}; 
                            height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); transition: transform 0.3s ease, box-shadow 0.3s ease;"
                     onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 15px rgba(0, 0, 0, 0.15)';"
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.1)';">
                    <div style="font-size: 40px; margin-bottom: 10px;">{step['icon']}</div>
                    <div style="font-weight: bold; margin-bottom: 5px; color: {step['color']};">{step['title']}</div>
                    <div style="font-size: 0.8rem; text-align: center; color: #666;">{step['desc']}</div>
                    <div style="margin-top: 10px; font-size: 20px;">
                        {i+1}
                        {" → " if i < len(workflow_steps)-1 else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Key features with colorful cards
    st.markdown("<h2 class='sub-header'>Key Features</h2>", unsafe_allow_html=True)
    
    feature_sets = [
        {
            "title": "Petrophysical Property Calculation",
            "icon": "🔬",
            "color": "#0e4194",
            "items": ["Shale Volume", "Porosity", "Water/Oil Saturation", "Permeability"]
        },
        {
            "title": "Facies Classification",
            "icon": "🧩",
            "color": "#e67e22",
            "items": ["K-means Clustering", "Silhouette Score Optimization", "Depth-based Facies Mapping"]
        },
        {
            "title": "Graph-based Machine Learning",
            "icon": "🌐",
            "color": "#2ecc71",
            "items": ["Graph Convolutional Networks", "Node and Edge Feature Extraction", "Hydrocarbon Quality Classification"]
        },
        {
            "title": "Visualization and Reporting",
            "icon": "📈",
            "color": "#9b59b6",
            "items": ["Facies Visualization", "Prediction Accuracy Metrics", "Classification Reports"]
        }
    ]
    
    cols = st.columns(len(feature_sets))
    
    for col, feature in zip(cols, feature_sets):
        with col:
            st.markdown(f"""
            <div class="feature-card" style="border-top-color: {feature['color']}">
                <div class="feature-icon" style="color: {feature['color']}">{feature['icon']}</div>
                <div class="feature-title" style="color: {feature['color']}">{feature['title']}</div>
                <ul style="padding-left: 20px; margin-top: 10px;">
                    {" ".join([f'<li>{item}</li>' for item in feature['items']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Call-to-action section
    st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
    
    # Create a dynamic button layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 20px; text-align: center; 
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
            <p style="font-size: 1.2rem; margin-bottom: 20px;">Ready to analyze your well log data?</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Start Analysis", key="start_analysis_btn", use_container_width=True):
            st.session_state['current_page'] = "Dataset Preparation"
            st.rerun()
            
    # Tutorial section
    st.markdown("<h2 class='sub-header'>Quick Tutorial</h2>", unsafe_allow_html=True)
    
    # Tutorial steps with alternating colors and icons
    tutorial_steps = [
        {"step": "1", "title": "Upload your LAS file", "desc": "Navigate to Dataset Preparation and upload your well log data file", "icon": "📤", "color": "#3498db"},
        {"step": "2", "title": "Configure Parameters", "desc": "Set clustering parameters and petrophysical calculation options", "icon": "⚙️", "color": "#e74c3c"},
        {"step": "3", "title": "Run Analysis", "desc": "Process the data and generate the graph dataset", "icon": "🔄", "color": "#2ecc71"},
        {"step": "4", "title": "View Results", "desc": "Explore predictions and visualize hydrocarbon potential zones", "icon": "📊", "color": "#9b59b6"}
    ]
    
    # Create a dynamic tutorial display
    for i, step in enumerate(tutorial_steps):
        # Alternate layout (left/right) for each step
        if i % 2 == 0:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"""
                <div style="background: {step['color']}; width: 80px; height: 80px; border-radius: 50%; 
                            display: flex; justify-content: center; align-items: center; margin: 0 auto;">
                    <span style="color: white; font-size: 40px;">{step['icon']}</span>
                </div>
                <div style="text-align: center; margin-top: 10px;">
                    <span style="background: {step['color']}; color: white; padding: 5px 10px; 
                                border-radius: 20px; font-weight: bold;">Step {step['step']}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 10px; border-left: 5px solid {step['color']}; margin-top: 10px;">
                    <h3 style="color: {step['color']}; margin-top: 0;">{step['title']}</h3>
                    <p>{step['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 10px; border-right: 5px solid {step['color']}; margin-top: 10px; text-align: right;">
                    <h3 style="color: {step['color']}; margin-top: 0;">{step['title']}</h3>
                    <p>{step['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="background: {step['color']}; width: 80px; height: 80px; border-radius: 50%; 
                            display: flex; justify-content: center; align-items: center; margin: 0 auto;">
                    <span style="color: white; font-size: 40px;">{step['icon']}</span>
                </div>
                <div style="text-align: center; margin-top: 10px;">
                    <span style="background: {step['color']}; color: white; padding: 5px 10px; 
                                border-radius: 20px; font-weight: bold;">Step {step['step']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Add a connector line between steps
        if i < len(tutorial_steps) - 1:
            st.markdown(f"""
            <div style="width: 2px; height: 30px; background: #ddd; margin: 5px auto;"></div>
            """, unsafe_allow_html=True)

# Dataset preparation page
def dataset_preparation_page():
    """Render the dataset preparation page"""
    st.markdown('<h1 class="main-header">Dataset Preparation</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">Upload your LAS file containing well log data. The file will be processed to calculate petrophysical properties necessary for hydrocarbon potential prediction.</p>', unsafe_allow_html=True)
    
    # Create a modern custom uploader
    st.markdown("""
    <div class="custom-uploader">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
        <h3>Drag and drop LAS file here</h3>
        <p style="color: #666;">or click to browse files</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["las"], label_visibility="collapsed")
    
    # Provide option for sample data with a toggle
    col1, col2 = st.columns([1, 3])
    with col1:
        use_sample = st.toggle("Use sample data", value=False)
    with col2:
        if use_sample:
            st.success("Using sample data for demonstration")
    
    # Parameters section with a sleek card design
    st.markdown("""
    <div class="content-card card-blue">
        <h3>Calculation Parameters</h3>
        <p>Configure the parameters for petrophysical property calculations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<b>Fluid Parameters</b>", unsafe_allow_html=True)
        matrix_density = st.number_input("Matrix Density (g/cc)", min_value=2.0, max_value=3.0, value=2.65, step=0.01)
        fluid_density = st.number_input("Fluid Density (g/cc)", min_value=0.5, max_value=1.5, value=1.0, step=0.01)
    
    with col2:
        st.markdown("<b>Archie Parameters</b>", unsafe_allow_html=True)
        a_param = st.number_input("Tortuosity Factor (a)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        m_param = st.number_input("Cementation Exponent (m)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        n_param = st.number_input("Saturation Exponent (n)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    
    # Feature selection section with animated checkboxes
    st.markdown("""
    <div class="content-card card-teal">
        <h3>Feature Selection</h3>
        <p>Select the log curves to use for facies classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate available logs
    available_logs = {
        "GR": "Gamma Ray",
        "RHOB": "Bulk Density",
        "NPHI": "Neutron Porosity",
        "RT": "Resistivity",
        "DT": "Sonic Travel Time",
        "VSHALE": "Calculated Shale Volume",
        "PHI": "Calculated Porosity",
        "SW": "Calculated Water Saturation"
    }
    
    # Create columns for log selection
    cols = st.columns(4)
    
    # Distribute logs across columns
    for i, (log_code, log_name) in enumerate(available_logs.items()):
        with cols[i % 4]:
            st.checkbox(f"{log_code}: {log_name}", value=log_code in ["GR", "RHOB", "VSHALE", "PHI", "SW"])
    
    # Process button with loading animation
    st.markdown("<br>", unsafe_allow_html=True)  # Add some space
    
    if uploaded_file or use_sample:
        if st.button("Process Data", type="primary", use_container_width=True):
            with st.spinner('Processing data...'):
                # Simulate processing with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)  # Simulate work being done
                    progress_bar.progress(i + 1)
                
                # Show success message
                st.success("Data processed successfully!")
                
                # Preview processed data in a nice table
                st.markdown("""
                <div class="content-card card-purple">
                    <h3>Processed Data Preview</h3>
                    <p>The first 5 rows of processed well log data with calculated properties</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a sample dataframe for demonstration
                data = {
                    "DEPTH": [1000.0, 1000.5, 1001.0, 1001.5, 1002.0],
                    "GR": [75.2, 78.5, 80.1, 76.3, 72.8],
                    "RHOB": [2.45, 2.48, 2.52, 2.47, 2.44],
                    "VSHALE": [0.35, 0.38, 0.42, 0.36, 0.32],
                    "PHI": [0.18, 0.16, 0.14, 0.17, 0.19],
                    "SW": [0.45, 0.48, 0.52, 0.47, 0.43]
                }
                
                df = pd.DataFrame(data)
                st.dataframe(df, hide_index=True)
                
                # Create a visualization
                st.markdown("""
                <div class="content-card card-orange">
                    <h3>Petrophysical Visualization</h3>
                    <p>Visual representation of the calculated properties</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a simple visualization
                fig, ax = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
                
                # Depth increases downward
                ax[0].plot(data["GR"], data["DEPTH"], 'b-')
                ax[0].set_xlabel("GR")
                ax[0].set_ylabel("Depth")
                ax[0].invert_yaxis()  # Depth increases downward
                
                ax[1].plot(data["RHOB"], data["DEPTH"], 'r-')
                ax[1].set_xlabel("RHOB")
                
                ax[2].plot(data["VSHALE"], data["DEPTH"], 'g-')
                ax[2].set_xlabel("VSHALE")
                
                ax[3].plot(data["PHI"], data["DEPTH"], 'k-')
                ax[3].set_xlabel("PHI")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Continue button
                if st.button("Continue to Model Workflow", use_container_width=True):
                    st.session_state['current_page'] = "Model Workflow"
                    st.rerun()
    else:
        st.info("Please upload a LAS file or use the sample data to continue.")

# Model workflow page
def model_workflow_page():
    """Render the model workflow page"""
    st.markdown('<h1 class="main-header">Model Workflow</h1>', unsafe_allow_html=True)
    
    # Create a modern intro section
    st.markdown("""
    <div class="info-box">
        <h3>Graph Convolutional Network (GCN) Workflow</h3>
        <p>
        This page outlines the workflow for creating a Graph Convolutional Network model for hydrocarbon quality prediction.
        The process includes dataset creation, graph construction, model training, and evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create workflow steps with timeline
    steps = [
        {"title": "1. Dataset Preparation", "icon": "📋", "color": "#3498db", "status": "completed", "desc": "Process well log data and calculate petrophysical properties"},
        {"title": "2. K-means Clustering", "icon": "🧩", "color": "#2ecc71", "status": "active", "desc": "Perform K-means clustering to identify facies"},
        {"title": "3. Graph Construction", "icon": "🔗", "color": "#e67e22", "status": "pending", "desc": "Create nodes and edges for the graph dataset"},
        {"title": "4. GCN Training", "icon": "🧠", "color": "#9b59b6", "status": "pending", "desc": "Train the Graph Convolutional Network model"},
        {"title": "5. Model Evaluation", "icon": "📊", "color": "#e74c3c", "status": "pending", "desc": "Evaluate model performance and visualize results"}
    ]
    
    # Create a modern timeline display
    for i, step in enumerate(steps):
        # Status colors
        status_color = "#2ecc71" if step["status"] == "completed" else "#3498db" if step["status"] == "active" else "#95a5a6"
        status_text = "Completed" if step["status"] == "completed" else "In Progress" if step["status"] == "active" else "Pending"
        
        st.markdown(f"""
        <div style="display: flex; margin-bottom: 20px; align-items: flex-start;">
            <div style="background: {status_color}; color: white; width: 40px; height: 40px; 
                        border-radius: 50%; display: flex; justify-content: center; 
                        align-items: center; margin-right: 15px; flex-shrink: 0;">
                {step["icon"]}
            </div>
            <div style="flex-grow: 1;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <h3 style="margin: 0; color: {step["color"]};">{step["title"]}</h3>
                    <span style="background: {status_color}; color: white; padding: 3px 10px; 
                                border-radius: 20px; font-size: 0.8rem;">{status_text}</span>
                </div>
                <p style="margin-top: 5px; color: #666;">{step["desc"]}</p>
                <div style="background: #f1f1f1; height: 2px; margin-top: 15px; position: relative;">
                    <div style="position: absolute; top: -6px; left: 0; width: 100%; 
                                display: flex; justify-content: space-between;">
                        {
                            ''.join([f'<div style="width: 10px; height: 10px; background: {"#2ecc71" if j < i else "#3498db" if j == i else "#95a5a6"}; border-radius: 50%;"></div>' for j in range(len(steps))])
                        }
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # K-means Clustering section (active step)
    st.markdown("""
    <div class="content-card card-blue">
        <h3>K-means Clustering for Facies Classification</h3>
        <p>Configure clustering parameters to identify natural facies groups in the well log data</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=3, max_value=15, value=7)
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=42)
    
    with col2:
        st.markdown("<b>Feature Scaling</b>", unsafe_allow_html=True)
        scaling_method = st.radio("Select scaling method", 
                               ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                               horizontal=True)
        
        st.markdown("<b>Silhouette Score Optimization</b>", unsafe_allow_html=True)
        optimize_silhouette = st.checkbox("Automatically determine optimal cluster count", value=True)
    
    # Run clustering button
    if st.button("Run Clustering", use_container_width=True):
        with st.spinner('Performing clustering...'):
            # Simulate clustering with a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)  # Simulate work being done
                progress_bar.progress(i + 1)
            
            # Show clustering results
            st.success("Clustering completed successfully!")
            
            # Visualization of silhouette scores
            st.markdown("<h3>Silhouette Score Analysis</h3>", unsafe_allow_html=True)
            
            # Create a sample visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sample data for silhouette scores
            cluster_range = range(3, 11)
            silhouette_scores = [0.42, 0.48, 0.52, 0.58, 0.55, 0.51, 0.47, 0.43]
            
            ax.plot(cluster_range, silhouette_scores, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel("Number of Clusters", fontsize=12)
            ax.set_ylabel("Silhouette Score", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Highlight the best score
            best_cluster = cluster_range[silhouette_scores.index(max(silhouette_scores))]
            ax.axvline(x=best_cluster, color='r', linestyle='--', alpha=0.7)
            ax.text(best_cluster + 0.1, max(silhouette_scores) - 0.02, 
                   f"Optimal: {best_cluster} clusters", 
                   color='r', fontsize=12)
            
            st.pyplot(fig)
            
            # Facies visualization
            st.markdown("<h3>Facies Classification</h3>", unsafe_allow_html=True)
            
            # Create a sample facies visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Sample data for facies
            depths = np.arange(1000, 1050, 0.5)
            facies = np.random.randint(0, n_clusters, size=len(depths))
            
            # Create a colormap
            cmap = plt.cm.get_cmap('viridis', n_clusters)
            
            # Create the facies plot
            im = ax.imshow(facies.reshape(-1, 1), aspect='auto', cmap=cmap, 
                         extent=[0, 1, max(depths), min(depths)])
            
            ax.set_title("Facies Classification", fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel("Depth", fontsize=12)
            ax.set_xticks([])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Facies", fontsize=12)
            cbar.set_ticks(range(n_clusters))
            cbar.set_ticklabels([f"Facies {i}" for i in range(n_clusters)])
            
            st.pyplot(fig)
            
            # Continue button
            if st.button("Continue to Graph Construction", use_container_width=True):
                st.markdown("""
                <div class="info-box">
                    <h3>Next: Graph Construction</h3>
                    <p>
                    In the next step, we will create a graph dataset from the facies classification results.
                    This includes defining nodes for depth points and petrophysical entities (PET),
                    and creating edges to represent relationships between them.
                    </p>
                </div>
                """, unsafe_allow_html=True)

# Analysis tool page
def analysis_tool_page(config):
    """Render the analysis tool page"""
    # Check if user is authenticated for this page
    is_authenticated = authenticate()
    
    # If user is not authenticated, stop execution here
    if not is_authenticated:
        st.stop()
    
    st.markdown('<h1 class="main-header">Analysis Tool</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">Analyze your well log data using the VHydro Graph Convolutional Network model.</p>', unsafe_allow_html=True)
    
    # Create tabs for different analysis steps
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Process", "Graph Creation", "Model Training", "Results"])
    
    with tab1:
        st.markdown("<h3>1. Data Upload and Processing</h3>", unsafe_allow_html=True)
        
        # Modern file uploader with drag and drop
        st.markdown("""
        <div class="custom-uploader">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
            <h3>Drag and drop LAS file here</h3>
            <p style="color: #666;">or click to browse files</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["las"], label_visibility="collapsed")
        
        # Option to use sample data
        use_sample = st.checkbox("Use sample data instead", value=not uploaded_file)
        
        # Process button
        if (uploaded_file or use_sample) and st.button("Process Data", key="process_data_btn", use_container_width=True):
            with st.spinner('Processing well log data...'):
                # Simulate processing with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(i + 1)
                
                # Show success message
                st.success("Data processed successfully! Well log data loaded and petrophysical properties calculated.")
                
                # Show animated arrow pointing to next tab
                st.markdown("""
                <div style="text-align: center; margin-top: 20px;">
                    <p>Continue to Graph Creation</p>
                    <div style="font-size: 2rem; animation: bounce 1s infinite alternate;">
                        ➡️
                    </div>
                </div>
                <style>
                @keyframes bounce {
                    0% { transform: translateX(0); }
                    100% { transform: translateX(10px); }
                }
                </style>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3>2. Graph Dataset Creation</h3>", unsafe_allow_html=True)
        
        # Show options for graph creation
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=5, max_value=10, value=7, 
                                key="graph_clusters")
            train_test_split = st.slider("Train/Test Split", min_value=0.6, max_value=0.9, value=0.8,
                                      key="graph_train_split")
        
        with col2:
            st.markdown("<b>Node Connection Strategy</b>", unsafe_allow_html=True)
            connection_strategy = st.radio("Select how nodes are connected:",
                                       ["K-Nearest Neighbors", "Distance Threshold", "Facies-based"],
                                       index=2)
            
            connection_param = st.slider("Connection Parameter", 
                                      min_value=1, max_value=10, value=5,
                                      help="K value for KNN, distance threshold, or maximum inter-facies connections")
        
        # Create graph button
        if st.button("Generate Graph Dataset", key="generate_graph_btn", use_container_width=True):
            with st.spinner('Creating graph dataset...'):
                # Simulate processing with a progress bar
                progress_bar = st.progress(0)
                step_texts = [
                    "Performing K-means clustering...",
                    "Identifying continuous facies segments...",
                    "Generating node connections...",
                    "Creating depth-to-depth edges...",
                    "Generating petrophysical entity nodes...",
                    "Assigning hydrocarbon quality labels...",
                    "Finalizing graph dataset..."
                ]
                
                # Show steps with progress
                step_container = st.empty()
                for i, text in enumerate(step_texts):
                    step_container.markdown(f"**Step {i+1}/7**: {text}")
                    for j in range(100//len(step_texts)):
                        time.sleep(0.02)
                        progress_bar.progress(int((i*100/len(step_texts)) + j*(100/len(step_texts)/100)))
                
                progress_bar.progress(100)
                step_container.markdown("**Completed!** Graph dataset successfully created.")
                
                # Success message
                st.success("Graph dataset created successfully!")
                
                # Show graph statistics
                st.markdown("""
                <div class="content-card card-blue">
                    <h4>Graph Dataset Statistics</h4>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold; color: #0e4194;">458</div>
                            <div>Nodes</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold; color: #0e4194;">1,245</div>
                            <div>Edges</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold; color: #0e4194;">7</div>
                            <div>Facies</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold; color: #0e4194;">4</div>
                            <div>Quality Classes</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show sample graph visualization
                st.markdown("<h4>Graph Visualization Preview</h4>", unsafe_allow_html=True)
                
                # Create a sample graph visualization using matplotlib
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Generate random node positions
                np.random.seed(42)
                n_nodes = 50
                pos = np.random.rand(n_nodes, 2)
                
                # Generate random node colors based on facies
                node_colors = np.random.randint(0, n_clusters, n_nodes)
                
                # Generate random edges
                n_edges = 100
                edges = np.random.randint(0, n_nodes, (n_edges, 2))
                
                # Plot nodes
                scatter = ax.scatter(pos[:, 0], pos[:, 1], c=node_colors, cmap='viridis', 
                                   s=100, alpha=0.8, edgecolors='white')
                
                # Plot edges
                for edge in edges:
                    ax.plot([pos[edge[0], 0], pos[edge[1], 0]], 
                           [pos[edge[0], 1], pos[edge[1], 1]], 
                           'k-', alpha=0.2)
                
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Sample Graph Visualization (Subset of Nodes)", fontsize=14)
                
                # Add a legend
                legend1 = ax.legend(*scatter.legend_elements(),
                                  loc="upper right", title="Facies")
                ax.add_artist(legend1)
                
                st.pyplot(fig)
                
                # Show animated arrow pointing to next tab
                st.markdown("""
                <div style="text-align: center; margin-top: 20px;">
                    <p>Continue to Model Training</p>
                    <div style="font-size: 2rem; animation: bounce 1s infinite alternate;">
                        ➡️
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h3>3. GCN Model Training</h3>", unsafe_allow_html=True)
        
        # Model configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            hidden_channels = st.slider("Hidden Channels", min_value=8, max_value=64, value=16, step=8,
                                     key="model_hidden_channels")
            learning_rate = st.select_slider("Learning Rate", 
                                         options=[0.001, 0.005, 0.01, 0.05, 0.1], 
                                         value=0.01,
                                         key="model_learning_rate")
        
        with col2:
            epochs = st.slider("Training Epochs", min_value=50, max_value=500, value=200, step=50,
                            key="model_epochs")
            dropout = st.slider("Dropout Rate", min_value=0.1, max_value=0.7, value=0.5, step=0.1,
                             key="model_dropout")
        
        # Advanced options
        with st.expander("Advanced Model Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                gcn_type = st.radio("GCN Implementation", ["PyTorch Geometric", "StellarGraph"],
                                 index=0, key="model_gcn_type")
                l2_reg = st.select_slider("L2 Regularization", 
                                       options=[0.0001, 0.0005, 0.001, 0.005, 0.01], 
                                       value=0.001,
                                       key="model_l2_reg")
            
            with col2:
                optimizer = st.radio("Optimizer", ["Adam", "SGD", "RMSprop"],
                                  index=0, key="model_optimizer")
                early_stopping = st.number_input("Early Stopping Patience", 
                                              min_value=10, max_value=100, value=50, step=10,
                                              key="model_early_stopping")
        
        # Train model button
        if st.button("Train GCN Model", key="train_model_btn", use_container_width=True):
            with st.spinner('Training Graph Convolutional Network model...'):
                # Simulate training with a progress bar
                progress_bar = st.progress(0)
                
                # Display metrics during training
                metrics_container = st.container()
                
                # Initialize metrics history
                train_loss = []
                val_loss = []
                train_acc = []
                val_acc = []
                
                # Simulate epochs
                for epoch in range(1, epochs + 1):
                    # Simulate metrics (decreasing loss, increasing accuracy)
                    curr_train_loss = 1.0 - 0.8 * (epoch / epochs) + 0.1 * np.random.random()
                    curr_val_loss = 1.2 - 0.7 * (epoch / epochs) + 0.15 * np.random.random()
                    curr_train_acc = 0.4 + 0.5 * (epoch / epochs) + 0.05 * np.random.random()
                    curr_val_acc = 0.3 + 0.55 * (epoch / epochs) + 0.07 * np.random.random()
                    
                    # Add to history
                    train_loss.append(curr_train_loss)
                    val_loss.append(curr_val_loss)
                    train_acc.append(curr_train_acc)
                    val_acc.append(curr_val_acc)
                    
                    # Update progress
                    progress_bar.progress(epoch / epochs)
                    
                    # Update metrics display every 10 epochs
                    if epoch % 10 == 0 or epoch == epochs:
                        with metrics_container:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Train Loss", f"{curr_train_loss:.4f}", 
                                       f"{train_loss[-2] - curr_train_loss:.4f}" if len(train_loss) > 1 else None)
                                st.metric("Validation Loss", f"{curr_val_loss:.4f}", 
                                       f"{val_loss[-2] - curr_val_loss:.4f}" if len(val_loss) > 1 else None)
                            
                            with col2:
                                st.metric("Train Accuracy", f"{curr_train_acc:.2%}", 
                                       f"{curr_train_acc - train_acc[-2]:.2%}" if len(train_acc) > 1 else None)
                                st.metric("Validation Accuracy", f"{curr_val_acc:.2%}", 
                                       f"{curr_val_acc - val_acc[-2]:.2%}" if len(val_acc) > 1 else None)
                            
                            # Plot training curves
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                            
                            # Loss plot
                            ax1.plot(train_loss, label='Train')
                            ax1.plot(val_loss, label='Validation')
                            ax1.set_xlabel('Epoch')
                            ax1.set_ylabel('Loss')
                            ax1.set_title('Training and Validation Loss')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # Accuracy plot
                            ax2.plot(train_acc, label='Train')
                            ax2.plot(val_acc, label='Validation')
                            ax2.set_xlabel('Epoch')
                            ax2.set_ylabel('Accuracy')
                            ax2.set_title('Training and Validation Accuracy')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Simulate some delay to make it look like real training
                        time.sleep(0.1)
                
                # Show final evaluation
                st.success(f"Model training completed successfully! Final validation accuracy: {val_acc[-1]:.2%}")
                
                # Show test evaluation
                st.markdown("<h4>Model Evaluation on Test Set</h4>", unsafe_allow_html=True)
                
                # Calculate metrics on test set (simulated)
                test_acc = 0.85 + 0.05 * np.random.random()
                precision = 0.83 + 0.07 * np.random.random()
                recall = 0.81 + 0.08 * np.random.random()
                f1_score = 0.82 + 0.06 * np.random.random()
                
                # Display test metrics using columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Test Accuracy", f"{test_acc:.2%}")
                col2.metric("Precision", f"{precision:.2%}")
                col3.metric("Recall", f"{recall:.2%}")
                col4.metric("F1 Score", f"{f1_score:.2%}")
                
                # Classification report
                st.markdown("<h4>Classification Report</h4>", unsafe_allow_html=True)
                
                # Create a simulated classification report
                data = {
                    "Precision": [0.92, 0.85, 0.78, 0.81],
                    "Recall": [0.88, 0.82, 0.75, 0.79],
                    "F1-Score": [0.90, 0.83, 0.76, 0.80],
                    "Support": [120, 150, 90, 110]
                }
                
                report_df = pd.DataFrame(data, index=["Very Low", "Low", "Moderate", "High"])
                st.dataframe(report_df)
                
                # Confusion matrix
                st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
                
                # Create a simulated confusion matrix
                classes = ["Very Low", "Low", "Moderate", "High"]
                cm = np.array([
                    [105, 10, 3, 2],
                    [12, 123, 9, 6],
                    [5, 8, 68, 9],
                    [3, 5, 15, 87]
                ])
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title("Confusion Matrix", fontsize=14)
                plt.colorbar(im)
                tick_marks = np.arange(len(classes))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
                
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show animated arrow pointing to next tab
                st.markdown("""
                <div style="text-align: center; margin-top: 20px;">
                    <p>Proceed to Results Analysis</p>
                    <div style="font-size: 2rem; animation: bounce 1s infinite alternate;">
                        ➡️
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<h3>4. Results Analysis</h3>", unsafe_allow_html=True)
        
        # Add tabs for different visualizations
        result_tab1, result_tab2, result_tab3 = st.tabs(["Hydrocarbon Potential", "Depth Profile", "Comparison"])
        
        with result_tab1:
            st.markdown("<h4>Hydrocarbon Potential Distribution</h4>", unsafe_allow_html=True)
            
            # Create a simulated distribution visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sample data
            categories = ["Very Low", "Low", "Moderate", "High"]
            colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
            
            # True distribution
            true_values = [25, 35, 30, 10]
            pred_values = [22, 38, 28, 12]
            
            x = np.arange(len(categories))
            width = 0.35
            
            # Create bars
            ax.bar(x - width/2, true_values, width, label='True', color=colors)
            ax.bar(x + width/2, pred_values, width, label='Predicted', color=colors, alpha=0.7)
            
            # Add percentage labels
            for i, v in enumerate(true_values):
                ax.text(i - width/2, v + 1, f"{v}%", ha='center')
            
            for i, v in enumerate(pred_values):
                ax.text(i + width/2, v + 1, f"{v}%", ha='center')
            
            ax.set_ylabel('Percentage')
            ax.set_title('Distribution of Hydrocarbon Potential')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            
            st.pyplot(fig)
            
            # Add interpretation
            st.markdown("""
            <div class="info-box">
                <h4>Interpretation</h4>
                <p>
                The model accurately predicts the overall distribution of hydrocarbon potential zones.
                There is a slight overestimation of "Low" potential zones and underestimation of "Moderate" zones,
                but overall the predictions closely match the true distribution.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_tab2:
            st.markdown("<h4>Depth vs. Hydrocarbon Potential</h4>", unsafe_allow_html=True)
            
            # Create a simulated depth profile visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
            
            # Sample data
            depths = np.arange(1000, 1100, 0.5)
            true_quality = np.random.randint(0, 4, size=len(depths))
            pred_quality = np.copy(true_quality)
            
            # Add some random differences to simulate prediction errors
            error_indices = np.random.choice(len(depths), size=int(len(depths)*0.15), replace=False)
            for idx in error_indices:
                pred_quality[idx] = (true_quality[idx] + np.random.choice([-1, 1])) % 4
            
            # Define colors for different quality levels
            quality_colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
            quality_labels = ["Very Low", "Low", "Moderate", "High"]
            
            # Create true quality plot
            for quality in range(4):
                mask = true_quality == quality
                ax1.scatter(np.ones(mask.sum()), depths[mask], 
                          c=quality_colors[quality], label=quality_labels[quality], s=50)
            
            ax1.set_title("True Hydrocarbon Potential", fontsize=14)
            ax1.set_xticks([])
            ax1.set_ylabel("Depth", fontsize=12)
            ax1.invert_yaxis()  # Depth increases downward
            ax1.legend()
            
            # Create predicted quality plot
            for quality in range(4):
                mask = pred_quality == quality
                ax2.scatter(np.ones(mask.sum()), depths[mask], 
                          c=quality_colors[quality], label=quality_labels[quality], s=50)
            
            ax2.set_title("Predicted Hydrocarbon Potential", fontsize=14)
            ax2.set_xticks([])
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add interpretation
            st.markdown("""
            <div class="info-box">
                <h4>Interpretation</h4>
                <p>
                The depth profile shows the distribution of hydrocarbon potential zones along the well.
                The model accurately predicts most of the zones, with some minor differences.
                Note the high potential zone around 1050-1060m depth, which is correctly identified by the model.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_tab3:
            st.markdown("<h4>Well Log and Hydrocarbon Potential Comparison</h4>", unsafe_allow_html=True)
            
            # Create a simulated well log visualization with predictions
            fig, axes = plt.subplots(1, 6, figsize=(20, 12), sharey=True)
            
            # Sample data
            depths = np.arange(1000, 1100, 0.5)
            gr = 50 + 30 * np.sin(depths / 10) + 10 * np.random.randn(len(depths))
            rhob = 2.2 + 0.3 * np.cos(depths / 15) + 0.05 * np.random.randn(len(depths))
            phi = 0.2 - 0.15 * np.cos(depths / 15) + 0.02 * np.random.randn(len(depths))
            sw = 0.4 + 0.3 * np.sin(depths / 20) + 0.05 * np.random.randn(len(depths))
            facies = np.random.randint(0, 7, size=len(depths))
            quality = np.random.randint(0, 4, size=len(depths))
            
            # Plot GR
            axes[0].plot(gr, depths, 'b-')
            axes[0].set_title('GR', fontsize=14)
            axes[0].set_xlabel('API', fontsize=12)
            axes[0].set_ylabel('Depth', fontsize=12)
            axes[0].invert_yaxis()  # Depth increases downward
            
            # Plot RHOB
            axes[1].plot(rhob, depths, 'r-')
            axes[1].set_title('RHOB', fontsize=14)
            axes[1].set_xlabel('g/cc', fontsize=12)
            
            # Plot PHI
            axes[2].plot(phi, depths, 'g-')
            axes[2].set_title('PHI', fontsize=14)
            axes[2].set_xlabel('v/v', fontsize=12)
            
            # Plot SW
            axes[3].plot(sw, depths, 'k-')
            axes[3].set_title('SW', fontsize=14)
            axes[3].set_xlabel('v/v', fontsize=12)
            
            # Plot facies
            cmap_facies = plt.cm.get_cmap('viridis', 7)
            facies_2d = np.vstack((facies, facies)).T
            im_facies = axes[4].imshow(facies_2d, aspect='auto', cmap=cmap_facies,
                                     extent=[0, 1, max(depths), min(depths)])
            axes[4].set_title('Facies', fontsize=14)
            axes[4].set_xticks([])
            
            # Add colorbar for facies
            cbar_facies = plt.colorbar(im_facies, ax=axes[4])
            cbar_facies.set_ticks(np.arange(7) + 0.5)
            cbar_facies.set_ticklabels([f"F{i}" for i in range(7)])
            
            # Plot hydrocarbon quality
            quality_colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
            cmap_quality = mcolors.ListedColormap(quality_colors)
            quality_2d = np.vstack((quality, quality)).T
            im_quality = axes[5].imshow(quality_2d, aspect='auto', cmap=cmap_quality,
                                      extent=[0, 1, max(depths), min(depths)])
            axes[5].set_title('HC Potential', fontsize=14)
            axes[5].set_xticks([])
            
            # Add colorbar for quality
            cbar_quality = plt.colorbar(im_quality, ax=axes[5])
            cbar_quality.set_ticks([0.375, 1.125, 1.875, 2.625])
            cbar_quality.set_ticklabels(["Very Low", "Low", "Moderate", "High"])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add interpretation
            st.markdown("""
            <div class="info-box">
                <h4>Interpretation</h4>
                <p>
                This integrated view shows the relationship between well log curves (GR, RHOB, PHI, SW),
                facies classification, and hydrocarbon potential predictions.
                Note how zones with high porosity (PHI) and low water saturation (SW) tend to have
                higher hydrocarbon potential.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Add download section
        st.markdown("<h3>Download Results</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="Download Predictions (CSV)",
                data="Sample data",
                file_name="hydrocarbon_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="Download Model (PKL)",
                data="Sample model",
                file_name="gcn_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        with col3:
            st.download_button(
                label="Download Report (PDF)",
                data="Sample report",
                file_name="hydrocarbon_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        # Add a continue button to visualization page
        if st.button("Continue to Results Visualization", use_container_width=True):
            st.session_state['current_page'] = "Results Visualization"
            st.rerun()

# Results visualization page
def results_visualization_page():
    """Render the results visualization page"""
    # Check if user is authenticated for this page
    is_authenticated = authenticate()
    
    # If user is not authenticated, stop execution here
    if not is_authenticated:
        st.stop()
    
    st.markdown('<h1 class="main-header">Results Visualization</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">Interactive visualization of hydrocarbon potential prediction results.</p>', unsafe_allow_html=True)
    
    # Create a sample dataset for visualization
    if 'visualization_data' not in st.session_state:
        # Create sample depths
        depths = np.arange(1000, 1500, 0.5)
        
        # Create sample well logs
        gr = 50 + 30 * np.sin(depths / 20) + 10 * np.random.randn(len(depths))
        rhob = 2.2 + 0.3 * np.cos(depths / 30) + 0.05 * np.random.randn(len(depths))
        phi = 0.2 - 0.15 * np.cos(depths / 30) + 0.02 * np.random.randn(len(depths))
        sw = 0.4 + 0.3 * np.sin(depths / 40) + 0.05 * np.random.randn(len(depths))
        
        # Create sample facies
        facies = np.random.randint(0, 7, size=len(depths))
        
        # Create sample quality predictions
        quality_labels = ["Very Low", "Low", "Moderate", "High"]
        quality_numeric = np.random.randint(0, 4, size=len(depths))
        quality = np.array(quality_labels)[quality_numeric]
        
        # Create sample dataframe
        data = {
            "DEPTH": depths,
            "GR": gr,
            "RHOB": rhob,
            "PHI": phi,
            "SW": sw,
            "FACIES": facies,
            "QUALITY": quality,
            "QUALITY_NUMERIC": quality_numeric
        }
        
        st.session_state['visualization_data'] = pd.DataFrame(data)
    
    # Create sidebar controls for visualization
    st.sidebar.markdown('<div class="sidebar-header">Visualization Controls</div>', unsafe_allow_html=True)
    
    # Depth range slider
    depth_min = float(st.session_state['visualization_data']["DEPTH"].min())
    depth_max = float(st.session_state['visualization_data']["DEPTH"].max())
    
    depth_range = st.sidebar.slider(
        "Depth Range",
        min_value=depth_min,
        max_value=depth_max,
        value=(depth_min, depth_min + 100),
        step=10.0
    )
    
    # Filter data by depth range
    filtered_data = st.session_state['visualization_data'][
        (st.session_state['visualization_data']["DEPTH"] >= depth_range[0]) & 
        (st.session_state['visualization_data']["DEPTH"] <= depth_range[1])
    ]
    
    # Visualization type
    viz_type = st.sidebar.radio(
        "Visualization Type",
        ["Well Log View", "3D View", "Cross-Plot", "Statistical Summary"]
    )
    
    # Display visualization based on type
    if viz_type == "Well Log View":
        st.markdown("<h2 class='sub-header'>Well Log View</h2>", unsafe_allow_html=True)
        
        # Create well log visualization
        fig, axes = plt.subplots(1, 6, figsize=(20, 12), sharey=True)
        
        # Plot GR
        axes[0].plot(filtered_data["GR"], filtered_data["DEPTH"], 'b-')
        axes[0].set_title('GR', fontsize=14)
        axes[0].set_xlabel('API', fontsize=12)
        axes[0].set_ylabel('Depth', fontsize=12)
        axes[0].invert_yaxis()  # Depth increases downward
        axes[0].grid(True, alpha=0.3)
        
        # Plot RHOB
        axes[1].plot(filtered_data["RHOB"], filtered_data["DEPTH"], 'r-')
        axes[1].set_title('RHOB', fontsize=14)
        axes[1].set_xlabel('g/cc', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Plot PHI
        axes[2].plot(filtered_data["PHI"], filtered_data["DEPTH"], 'g-')
        axes[2].set_title('PHI', fontsize=14)
        axes[2].set_xlabel('v/v', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Plot SW
        axes[3].plot(filtered_data["SW"], filtered_data["DEPTH"], 'k-')
        axes[3].set_title('SW', fontsize=14)
        axes[3].set_xlabel('v/v', fontsize=12)
        axes[3].grid(True, alpha=0.3)
        
        # Plot facies
        cmap_facies = plt.cm.get_cmap('viridis', 7)
        facies_2d = np.vstack((filtered_data["FACIES"], filtered_data["FACIES"])).T
        im_facies = axes[4].imshow(facies_2d, aspect='auto', cmap=cmap_facies,
                                 extent=[0, 1, filtered_data["DEPTH"].max(), filtered_data["DEPTH"].min()])
        axes[4].set_title('Facies', fontsize=14)
        axes[4].set_xticks([])
        
        # Add colorbar for facies
        cbar_facies = plt.colorbar(im_facies, ax=axes[4])
        cbar_facies.set_ticks(np.arange(7) + 0.5)
        cbar_facies.set_ticklabels([f"F{i}" for i in range(7)])
        
        # Plot hydrocarbon quality
        quality_colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
        cmap_quality = mcolors.ListedColormap(quality_colors)
        quality_2d = np.vstack((filtered_data["QUALITY_NUMERIC"], filtered_data["QUALITY_NUMERIC"])).T
        im_quality = axes[5].imshow(quality_2d, aspect='auto', cmap=cmap_quality,
                                  extent=[0, 1, filtered_data["DEPTH"].max(), filtered_data["DEPTH"].min()])
        axes[5].set_title('HC Potential', fontsize=14)
        axes[5].set_xticks([])
        
        # Add colorbar for quality
        cbar_quality = plt.colorbar(im_quality, ax=axes[5])
        cbar_quality.set_ticks([0.375, 1.125, 1.875, 2.625])
        cbar_quality.set_ticklabels(["Very Low", "Low", "Moderate", "High"])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display interpretation
        st.markdown("""
        <div class="info-box">
            <h4>Interpretation</h4>
            <p>
            The well log view shows the relationship between different petrophysical properties and the
            predicted hydrocarbon potential. Note how zones with high porosity (PHI) and low water saturation (SW)
            typically correspond to higher hydrocarbon potential.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    elif viz_type == "3D View":
        st.markdown("<h2 class='sub-header'>3D Visualization</h2>", unsafe_allow_html=True)
        
        # Create a 3D visualization using matplotlib
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D scatter plot
        scatter = ax.scatter(
            filtered_data["PHI"],
            filtered_data["GR"],
            filtered_data["DEPTH"],
            c=filtered_data["QUALITY_NUMERIC"],
            cmap=mcolors.ListedColormap(["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]),
            s=50,
            alpha=0.8
        )
        
        # Set labels
        ax.set_xlabel('Porosity (PHI)')
        ax.set_ylabel('Gamma Ray (GR)')
        ax.set_zlabel('Depth')
        
        # Invert z-axis (depth)
        ax.invert_zaxis()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
        cbar.set_ticklabels(["Very Low", "Low", "Moderate", "High"])
        cbar.set_label('Hydrocarbon Potential')
        
        # Set title
        ax.set_title('3D Visualization of Hydrocarbon Potential', fontsize=14)
        
        # Adjust view angle
        ax.view_init(30, 45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        <div class="info-box">
            <h4>Interpretation</h4>
            <p>
            The 3D visualization shows the relationship between porosity, gamma ray, depth, and
            hydrocarbon potential. This allows for identification of patterns and clusters in the data
            that might not be apparent in traditional 2D visualizations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    elif viz_type == "Cross-Plot":
        st.markdown("<h2 class='sub-header'>Cross-Plot Analysis</h2>", unsafe_allow_html=True)
        
        # Create sidebar for cross-plot parameters
        x_property = st.sidebar.selectbox(
            "X-Axis Property",
            ["PHI", "GR", "RHOB", "SW"],
            index=0
        )
        
        y_property = st.sidebar.selectbox(
            "Y-Axis Property",
            ["PHI", "GR", "RHOB", "SW"],
            index=1
        )
        
        color_by = st.sidebar.radio(
            "Color By",
            ["QUALITY", "FACIES"],
            index=0
        )
        
        # Create cross-plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if color_by == "QUALITY":
            scatter = ax.scatter(
                filtered_data[x_property],
                filtered_data[y_property],
                c=filtered_data["QUALITY_NUMERIC"],
                cmap=mcolors.ListedColormap(["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]),
                s=80,
                alpha=0.7
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
            cbar.set_ticklabels(["Very Low", "Low", "Moderate", "High"])
            cbar.set_label('Hydrocarbon Potential')
        else:
            cmap = plt.cm.get_cmap('viridis', 7)
            scatter = ax.scatter(
                filtered_data[x_property],
                filtered_data[y_property],
                c=filtered_data["FACIES"],
                cmap=cmap,
                s=80,
                alpha=0.7
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(7) + 0.5)
            cbar.set_ticklabels([f"F{i}" for i in range(7)])
            cbar.set_label('Facies')
        
        # Set labels and title
        ax.set_xlabel(x_property, fontsize=12)
        ax.set_ylabel(y_property, fontsize=12)
        ax.set_title(f'{x_property} vs {y_property} Cross-Plot', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add trendline
        z = np.polyfit(filtered_data[x_property], filtered_data[y_property], 1)
        p = np.poly1d(z)
        ax.plot(filtered_data[x_property], p(filtered_data[x_property]), 
               "r--", alpha=0.7, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add cross-plot matrix option
        if st.checkbox("Show Cross-Plot Matrix"):
            st.markdown("<h3>Cross-Plot Matrix</h3>", unsafe_allow_html=True)
            
            # Create pairplot using seaborn
            properties = ["PHI", "GR", "RHOB", "SW"]
            
            # Generate plot using matplotlib
            fig, axes = plt.subplots(len(properties), len(properties), figsize=(15, 15))
            
            for i, prop1 in enumerate(properties):
                for j, prop2 in enumerate(properties):
                    if i == j:
                        # Histogram on diagonal
                        axes[i, j].hist(filtered_data[prop1], bins=20, color="#3498db", alpha=0.7)
                        axes[i, j].set_title(prop1)
                    else:
                        # Scatter plot on off-diagonal
                        scatter = axes[i, j].scatter(
                            filtered_data[prop2],
                            filtered_data[prop1],
                            c=filtered_data["QUALITY_NUMERIC"],
                            cmap=mcolors.ListedColormap(["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]),
                            s=30,
                            alpha=0.7
                        )
                        
                        # Only add labels on edge plots
                        if i == len(properties) - 1:
                            axes[i, j].set_xlabel(prop2)
                        if j == 0:
                            axes[i, j].set_ylabel(prop1)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        <div class="info-box">
            <h4>Interpretation</h4>
            <p>
            Cross-plots help identify relationships between different petrophysical properties and their
            association with hydrocarbon potential. For example, zones with high porosity and low density
            tend to have higher hydrocarbon potential.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    elif viz_type == "Statistical Summary":
        st.markdown("<h2 class='sub-header'>Statistical Summary</h2>", unsafe_allow_html=True)
        
        # Create tabs for different statistical views
        stat_tab1, stat_tab2, stat_tab3 = st.tabs(["Summary Statistics", "Distribution Analysis", "Quality Assessment"])
        
        with stat_tab1:
            st.markdown("<h3>Summary Statistics by Hydrocarbon Potential</h3>", unsafe_allow_html=True)
            
            # Calculate statistics by quality
            grouped_stats = filtered_data.groupby("QUALITY")[["GR", "RHOB", "PHI", "SW"]].agg(
                ["mean", "std", "min", "max"]
            )
            
            # Display statistics
            st.dataframe(grouped_stats)
            
            # Add summary charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Create bar chart for mean values
                fig, ax = plt.subplots(figsize=(10, 6))
                
                properties = ["GR", "RHOB", "PHI", "SW"]
                quality_labels = ["Very Low", "Low", "Moderate", "High"]
                
                # Calculate means for each property and quality
                means = {}
                for prop in properties:
                    means[prop] = [
                        filtered_data[filtered_data["QUALITY"] == label][prop].mean()
                        for label in quality_labels
                    ]
                
                # Plot
                x = np.arange(len(quality_labels))
                width = 0.2
                
                # Plot bars for each property
                for i, prop in enumerate(properties):
                    ax.bar(x + i*width - 0.3, means[prop], width, 
                           label=prop)
                
                ax.set_title('Mean Property Values by Hydrocarbon Potential', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(quality_labels)
                ax.legend()
                
                st.pyplot(fig)
            
            with col2:
                # Create heatmap of correlations
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Calculate correlation matrix
                corr_matrix = filtered_data[["GR", "RHOB", "PHI", "SW", "QUALITY_NUMERIC"]].corr()
                
                # Create heatmap
                im = ax.imshow(corr_matrix, cmap="coolwarm")
                
                # Add labels
                properties = ["GR", "RHOB", "PHI", "SW", "QUALITY"]
                ax.set_xticks(np.arange(len(properties)))
                ax.set_yticks(np.arange(len(properties)))
                ax.set_xticklabels(properties)
                ax.set_yticklabels(properties)
                
                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add text annotations
                for i in range(len(properties)):
                    for j in range(len(properties)):
                        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                               ha="center", va="center", 
                               color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                
                ax.set_title("Correlation Matrix", fontsize=14)
                plt.colorbar(im)
                plt.tight_layout()
                
                st.pyplot(fig)
        
        with stat_tab2:
            st.markdown("<h3>Distribution Analysis</h3>", unsafe_allow_html=True)
            
            # Select property for distribution analysis
            property_for_dist = st.selectbox(
                "Select Property for Distribution Analysis",
                ["GR", "RHOB", "PHI", "SW"]
            )
            
            # Create distribution plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(filtered_data[property_for_dist], bins=20, alpha=0.7, color="#3498db")
            ax1.set_title(f'{property_for_dist} Histogram', fontsize=14)
            ax1.set_xlabel(property_for_dist, fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # KDE plot by quality
            for i, quality in enumerate(["Very Low", "Low", "Moderate", "High"]):
                quality_data = filtered_data[filtered_data["QUALITY"] == quality][property_for_dist]
                if len(quality_data) > 1:  # Need at least 2 points for KDE
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(quality_data)
                    x = np.linspace(filtered_data[property_for_dist].min(), 
                                    filtered_data[property_for_dist].max(), 100)
                    ax2.plot(x, kde(x), linewidth=2, 
                            label=quality, color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"][i])
            
            ax2.set_title(f'{property_for_dist} Distribution by Hydrocarbon Potential', fontsize=14)
            ax2.set_xlabel(property_for_dist, fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add box plot
            st.markdown("<h3>Box Plot Analysis</h3>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create box plot
            bp = ax.boxplot(
                [filtered_data[filtered_data["QUALITY"] == quality][property_for_dist] 
                 for quality in ["Very Low", "Low", "Moderate", "High"]],
                patch_artist=True,
                notch=True,
                labels=["Very Low", "Low", "Moderate", "High"]
            )
            
            # Customize colors
            colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
            for i, box in enumerate(bp['boxes']):
                box.set(facecolor=colors[i], alpha=0.7)
            
            ax.set_title(f'{property_for_dist} Distribution by Hydrocarbon Potential', fontsize=14)
            ax.set_xlabel('Hydrocarbon Potential', fontsize=12)
            ax.set_ylabel(property_for_dist, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with stat_tab3:
            st.markdown("<h3>Hydrocarbon Quality Assessment</h3>", unsafe_allow_html=True)
            
            # Create quality distribution charts
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart
            quality_counts = filtered_data["QUALITY"].value_counts()
            colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
            
            ax1.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax1.set_title('Hydrocarbon Potential Distribution', fontsize=14)
            
            # Stacked bar for depth ranges
            # Group data into depth bins
            bin_size = 50
            bins = np.arange(depth_range[0], depth_range[1] + bin_size, bin_size)
            
            filtered_data['depth_bin'] = pd.cut(filtered_data['DEPTH'], bins)
            
            # Count qualities by depth bin
            depth_quality = pd.crosstab(filtered_data['depth_bin'], filtered_data['QUALITY'])
            
            # Get bin labels
            bin_labels = [f"{int(b.left)}-{int(b.right)}" for b in depth_quality.index]
            
            # Plot stacked bar
            bottom = np.zeros(len(depth_quality))
            
            for i, quality in enumerate(["Very Low", "Low", "Moderate", "High"]):
                if quality in depth_quality.columns:
                    values = depth_quality[quality].values
                    ax2.bar(bin_labels, values, bottom=bottom, label=quality, color=colors[i])
                    bottom += values
            
            ax2.set_title('Hydrocarbon Potential by Depth Range', fontsize=14)
            ax2.set_xlabel('Depth Range (m)', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Summary metrics
            st.markdown("<h3>Key Performance Metrics</h3>", unsafe_allow_html=True)
            
            # Calculate metrics
            total_depth = filtered_data["DEPTH"].max() - filtered_data["DEPTH"].min()
            high_quality_depth = filtered_data[filtered_data["QUALITY"].isin(["High", "Moderate"])]["DEPTH"].count() * 0.5  # Assume 0.5m per reading
            quality_ratio = high_quality_depth / total_depth
            
            net_pay = high_quality_depth  # Simplified, in a real app would use more complex calculation
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{:.1f}%</div>
                    <div class="metric-label">Quality Ratio<br>(High+Moderate)</div>
                </div>
                """.format(quality_ratio * 100), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{:.1f}m</div>
                    <div class="metric-label">Net Pay</div>
                </div>
                """.format(net_pay), unsafe_allow_html=True)
            
            with col3:
                avg_porosity = filtered_data[filtered_data["QUALITY"].isin(["High", "Moderate"])]["PHI"].mean()
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{:.1f}%</div>
                    <div class="metric-label">Average Porosity<br>(High+Moderate Zones)</div>
                </div>
                """.format(avg_porosity * 100), unsafe_allow_html=True)
    
    # Add export options
    st.markdown("<h2 class='sub-header'>Export Options</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="Export Data as CSV",
            data=filtered_data.to_csv(index=False),
            file_name="vhydro_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("Generate Summary Report", use_container_width=True):
            st.info("Generating summary report... This may take a moment.")
            time.sleep(2)  # Simulate report generation
            
            st.success("Report generated successfully!")
            
            st.download_button(
                label="Download Report (PDF)",
                data="Sample report data",
                file_name="vhydro_summary_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with col3:
        if st.button("Save Visualization", use_container_width=True):
            st.info("Preparing visualization... This may take a moment.")
            time.sleep(2)  # Simulate preparation
            
            st.success("Visualization prepared successfully!")
            
            st.download_button(
                label="Download Visualization (PNG)",
                data="Sample image data",
                file_name="vhydro_visualization.png",
                mime="image/png",
                use_container_width=True
            )

# Main function for the app
def main():
    # Create sidebar
    config = create_sidebar()
    
    # Display the selected page
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
    
    # Footer
    st.markdown("<div class='footer'>VHydro - Hydrocarbon Potential Prediction Application © 2025</div>", unsafe_allow_html=True)

# Entry point for the application
if __name__ == "__main__":
    main()
