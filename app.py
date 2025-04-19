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
        background: linear-gradient(180deg, #0e1c36 0%, #1b3366 100%);
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
    
    /* Navigation buttons styling */
    .nav-button {
        background-color: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white !important;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
        display: flex;
        align-items: center;
    }
    
    .nav-button:hover, .nav-button.active {
        background-color: rgba(66, 133, 244, 0.3);
        border-color: rgba(66, 133, 244, 0.6);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .nav-button.active {
        background-color: rgba(66, 133, 244, 0.5);
        border-left: 4px solid #4285f4;
    }
    
    .nav-button-icon {
        margin-right: 10px;
        width: 20px;
        text-align: center;
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
    
    /* Main content area styling */
    .main-header {
        color: #0e4194;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #0e4194;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(14, 65, 148, 0.2);
    }
    
    .section-header {
        color: #0e4194;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.2rem 0 0.8rem 0;
    }
    
    .description {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    
    .info-box {
        background-color: #f8f9fa;
        border-left: 5px solid #0e4194;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.9rem;
        margin-top: 30px;
        border-top: 1px solid #eee;
    }
    
    /* Card styling for features */
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .feature-title {
        color: #0e4194;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 10px;
        border-bottom: 2px solid rgba(14, 65, 148, 0.2);
        padding-bottom: 5px;
    }
    
    /* Login form styling */
    .auth-form {
        max-width: 400px;
        margin: 0 auto;
        background: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .auth-header {
        text-align: center;
        margin-bottom: 20px;
        color: #0e4194;
    }
    
    /* Banner styling */
    .banner-container {
        width: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .banner-container img {
        width: 100%;
        height: auto;
        object-fit: cover;
    }
    
    /* Document upload area */
    .upload-area {
        border: 2px dashed #0e4194;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: rgba(14, 65, 148, 0.05);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background-color: rgba(14, 65, 148, 0.1);
        border-color: #0e4194;
    }
    
    /* Results container */
    .results-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    
    /* Scientific visualization styling */
    .visualization-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .visualization-title {
        color: #0e4194;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 10px;
        text-align: center;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #0e4194 !important;
    }
</style>
""", unsafe_allow_html=True)

# Try to import VHydro - handle missing dependencies gracefully
try:
    # Add current directory to path to import VHydro
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from VHydro_final import VHydro
    VHYDRO_AVAILABLE = True
    logger.info("Successfully imported VHydro module")
except ImportError as e:
    VHYDRO_AVAILABLE = False
    logger.error(f"Error importing VHydro module: {e}")

# Import Firebase Authentication module
try:
    from firebase_auth import authenticate, user_account_page, logout, login, signup
    AUTH_AVAILABLE = True
    logger.info("Successfully imported Firebase authentication module")
except ImportError as e:
    AUTH_AVAILABLE = False
    logger.error(f"Error importing Firebase authentication module: {e}")
    
    # Define fallback authentication functions
    def authenticate():
        return True
        
    def user_account_page():
        st.info("Firebase authentication module not available. Using guest access.")
        
    def logout():
        if "user_id" in st.session_state:
            del st.session_state["user_id"]
        if "email" in st.session_state:
            del st.session_state["email"]
        if "logged_in" in st.session_state:
            del st.session_state["logged_in"]

def header_with_logo(logo_path):
    # Custom CSS for the header area
    try:
        if os.path.exists(logo_path):
            # Display the logo as a full-width banner using HTML
            st.markdown(f"""
            <div class="banner-container">
                <img src="data:image/png;base64,{get_image_as_base64(logo_path)}" alt="VHydro Banner">
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
def get_image_as_base64(image_path):
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

    # User account section - only show if logged in
    if st.session_state.get("logged_in", False):
        st.markdown('<div class="user-account">', unsafe_allow_html=True)
        st.markdown(f'<div class="user-email">{user_icon} {st.session_state.get("email", "Guest")}</div>', unsafe_allow_html=True)
        
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
    
    # Define navigation buttons with icons
    nav_buttons = [
        {"name": "Home", "icon": "🏠"},
        {"name": "Dataset Preparation", "icon": "📊"},
        {"name": "Model Workflow", "icon": "⚙️"},
        {"name": "Analysis Tool", "icon": "🔬"},
        {"name": "Results Visualization", "icon": "📈"},
        {"name": "Account", "icon": "👤"}
    ]
    
    # Display the navigation buttons
    current_page = st.session_state.get('current_page', 'Home')
    for button in nav_buttons:
        button_class = "nav-button active" if current_page == button["name"] else "nav-button"
        if st.markdown(f"""
        <button class="{button_class}" onclick="this.form.requestSubmit()">
            <span class="nav-button-icon">{button["icon"]}</span> {button["name"]}
        </button>
        """, unsafe_allow_html=True):
            st.session_state['current_page'] = button["name"]
            st.rerun()
    
    # Create a form for click handling
    with st.form(key="nav_form"):
        for i, button in enumerate(nav_buttons):
            if st.form_submit_button(label=button["name"], key=f"nav_{i}"):
                st.session_state['current_page'] = button["name"]
                st.rerun()
    
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
        "page": current_page,
        "min_clusters": min_clusters,
        "max_clusters": max_clusters,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "hidden_channels": hidden_channels,
        "num_runs": num_runs
    }

# Authentication page
def auth_page():
    st.markdown("<h1 class='main-header'>VHydro Login</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please login to access the analysis and visualization tools</p>", unsafe_allow_html=True)
    
    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        st.markdown('<h2 class="auth-header">Login</h2>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if AUTH_AVAILABLE:
                    success, message = login(email, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    # Fallback login for demo purposes
                    if email and password:
                        st.session_state["user_id"] = email
                        st.session_state["email"] = email
                        st.session_state["logged_in"] = True
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Please enter both email and password")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sign up option
        st.markdown("<p style='text-align: center; margin-top: 20px;'>Don't have an account?</p>", unsafe_allow_html=True)
        if st.button("Create Account", use_container_width=True):
            st.session_state["auth_mode"] = "signup"
            st.rerun()
            
        if st.session_state.get("auth_mode") == "signup":
            st.markdown('<div class="auth-form" style="margin-top: 30px;">', unsafe_allow_html=True)
            st.markdown('<h2 class="auth-header">Create Account</h2>', unsafe_allow_html=True)
            
            with st.form("signup_form"):
                new_email = st.text_input("Email", key="signup_email")
                new_password = st.text_input("Password", type="password", key="signup_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
                
                # Password requirements notice
                st.markdown("""
                <div style="font-size: 0.8em; color: #6c757d; margin-bottom: 15px;">
                Password must be at least 8 characters with 1 uppercase letter, 1 lowercase letter, and 1 digit.
                </div>
                """, unsafe_allow_html=True)
                
                signup_button = st.form_submit_button("Sign Up", use_container_width=True)
                
                if signup_button:
                    if AUTH_AVAILABLE:
                        success, message = signup(new_email, new_password, confirm_password)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        # Fallback signup for demo purposes
                        if new_password != confirm_password:
                            st.error("Passwords don't match")
                        elif len(new_password) < 8:
                            st.error("Password must be at least 8 characters")
                        else:
                            st.session_state["user_id"] = new_email
                            st.session_state["email"] = new_email
                            st.session_state["logged_in"] = True
                            st.success("Account created successfully!")
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("Back to Login", use_container_width=True):
                st.session_state.pop("auth_mode", None)
                st.rerun()

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
    
    # Use a 2x2 grid for feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Petrophysical Property Calculation</div>
            <ul>
                <li>Shale Volume</li>
                <li>Porosity</li>
                <li>Water/Oil Saturation</li>
                <li>Permeability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card" style="margin-top: 20px;">
            <div class="feature-title">Facies Classification</div>
            <ul>
                <li>K-means Clustering</li>
                <li>Silhouette Score Optimization</li>
                <li>Depth-based Facies Mapping</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Graph-based Machine Learning</div>
            <ul>
                <li>Graph Convolutional Networks</li>
                <li>Node and Edge Feature Extraction</li>
                <li>Hydrocarbon Quality Classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card" style="margin-top: 20px;">
            <div class="feature-title">Visualization and Reporting</div>
            <ul>
                <li>Facies Visualization</li>
                <li>Prediction Accuracy Metrics</li>
                <li>Classification Reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
    
    # Create a better getting started section with cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">1. Prepare Data</div>
            <p>Upload your well log data in LAS format and validate required curves.</p>
            <p>Navigate to the <b>Dataset Preparation</b> section to understand data requirements.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">2. Run Analysis</div>
            <p>Calculate petrophysical properties and run the GCN model.</p>
            <p>Use the <b>Analysis Tool</b> to process your data and generate predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">3. Visualize Results</div>
            <p>Explore facies classifications and hydrocarbon quality predictions.</p>
            <p>Visit the <b>Results Visualization</b> section to interpret findings.</p>
        </div>
        """, unsafe_allow_html=True)

# Account page
def account_page():
    """Render the user account page"""
    st.markdown('<h1 class="main-header">Account Settings</h1>', unsafe_allow_html=True)
    
    # Check if user is logged in
    if not st.session_state.get("logged_in", False):
        st.warning("You need to log in to access your account settings.")
        if st.button("Go to Login"):
            st.session_state['current_page'] = "Login"
            st.rerun()
        return
    
    # Show user account page from the firebase_auth module
    user_account_page()

# Dataset preparation page
def dataset_preparation_page():
    """Render the dataset preparation page"""
    st.markdown('<h1 class="main-header">Dataset Preparation</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">Prepare your well log data for hydrocarbon quality prediction. The system requires specific log curves to calculate petrophysical properties necessary for accurate predictions.</p>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data Requirements", "File Format", "Sample Data"])
    
    with tab1:
        st.markdown("<h3 class='section-header'>Required Log Curves</h3>", unsafe_allow_html=True)
        
        # Create a table with required log curves
        requirements_data = {
            "Curve": ["GR/CGR", "RHOB", "NPHI", "LLD/ILD", "DEPT"],
            "Description": [
                "Gamma Ray or Computed Gamma Ray",
                "Bulk Density",
                "Neutron Porosity",
                "Deep Resistivity",
                "Depth"
            ],
            "Purpose": [
                "Shale volume calculation",
                "Density porosity calculation",
                "Effective porosity calculation",
                "Water/oil saturation calculation",
                "Spatial reference for facies classification"
            ]
        }
        
        requirements_df = pd.DataFrame(requirements_data)
        st.table(requirements_df)
        
        st.markdown("<h3 class='section-header'>Data Quality Requirements</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <ul>
                <li><b>Data Completeness:</b> Minimal gaps in required log curves</li>
                <li><b>Depth Consistency:</b> Consistent depth sampling</li>
                <li><b>Cleaned Data:</b> Pre-processed to remove outliers and erroneous values</li>
                <li><b>Depth Range:</b> Continuous depth range for the zone of interest</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        st.markdown("<h3 class='section-header'>LAS File Format</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <p>VHydro accepts well log data in <b>LAS</b> (Log ASCII Standard) format, which is the industry standard for storing well log data.</p>
        
        <div class="info-box">
            <h4>LAS File Components</h4>
            <ul>
                <li><b>Version Information:</b> LAS file version</li>
                <li><b>Well Information:</b> Well name, location, date, etc.</li>
                <li><b>Curve Information:</b> Names and units of log curves</li>
                <li><b>Parameter Information:</b> Additional parameters</li>
                <li><b>Data Section:</b> Actual log data values</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display sample LAS file structure
        st.markdown("<h4 class='section-header'>Example LAS File Structure</h4>", unsafe_allow_html=True)
        
        sample_las = """~VERSION INFORMATION
VERS.   2.0 :   CWLS LOG ASCII STANDARD - VERSION 2.0
WRAP.   NO  :   ONE LINE PER DEPTH STEP

~WELL INFORMATION
STRT.M   1670.0 :
STOP.M   1660.0 :
STEP.M   -0.1  :
NULL.    -999.25:
WELL.    EXAMPLE WELL:
FLD .    EXAMPLE FIELD:

~CURVE INFORMATION
DEPT.M   :   Depth
GR  .GAPI:   Gamma Ray
RHOB.K/M3:   Bulk Density
NPHI.V/V :   Neutron Porosity
LLD .OHMM:   Deep Resistivity

~PARAMETER INFORMATION
...

~A  DEPT     GR      RHOB    NPHI    LLD
1670.000   75.075   2.561   0.246   12.863
1669.900   74.925   2.563   0.245   13.042
...
"""
        
        st.code(sample_las, language='text')
        
    with tab3:
        st.markdown("<h3 class='section-header'>Sample Data</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <p>You can use the following sample data to test the VHydro workflow:</p>
        """, unsafe_allow_html=True)
        
        # Create download buttons for sample data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">Sample Well #1</div>
                <p>Sandstone reservoir with moderate hydrocarbon potential</p>
                <p>Contains all required log curves</p>
            </div>
            """, unsafe_allow_html=True)
            
            # In a real application, you would provide a download button here
            st.download_button(
                label="Download Sample #1", 
                data="Sample data would be here", 
                file_name="sample_well_1.las",
                mime="text/plain",
                key="sample1"
            )
            
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">Sample Well #2</div>
                <p>Carbonate reservoir with high hydrocarbon potential</p>
                <p>Contains all required log curves plus additional data</p>
            </div>
            """, unsafe_allow_html=True)
            
            # In a real application, you would provide a download button here
            st.download_button(
                label="Download Sample #2", 
                data="Sample data would be here", 
                file_name="sample_well_2.las",
                mime="text/plain",
                key="sample2"
            )

# Model workflow page
def model_workflow_page():
    """Render the model workflow page"""
    st.markdown('<h1 class="main-header">Model Workflow</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">Understanding how VHydro processes data and generates predictions is essential for interpreting results. This page explains the workflow from data input to final prediction.</p>', unsafe_allow_html=True)
    
    # Display model workflow diagram
    workflow_image_path = "src/model_workflow.png"
    dataset_workflow_image_path = "src/dataset_preparation_workflow.png"
    
    # Main Workflow Overview
    st.markdown("<h2 class='sub-header'>Workflow Overview</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>VHydro follows a multi-stage workflow to predict hydrocarbon quality from well log data:</p>
        <ol>
            <li><b>Data Loading and Validation:</b> Import LAS file and validate required curves</li>
            <li><b>Petrophysical Property Calculation:</b> Calculate key reservoir properties</li>
            <li><b>Facies Classification:</b> Group similar rock types using K-means clustering</li>
            <li><b>Graph Construction:</b> Create node connections based on depth relationships</li>
            <li><b>GCN Model Training:</b> Train the Graph Convolutional Network model</li>
            <li><b>Hydrocarbon Quality Prediction:</b> Generate predictions for each depth point</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Display the main workflow diagram if available
    if os.path.exists(workflow_image_path):
        display_image_with_caption(workflow_image_path, "VHydro Model Workflow")
    else:
        st.info("Workflow diagram not available")
    
    # Dataset Preparation Workflow
    st.markdown("<h2 class='sub-header'>Dataset Preparation</h2>", unsafe_allow_html=True)
    
    # Display the dataset preparation workflow diagram if available
    if os.path.exists(dataset_workflow_image_path):
        display_image_with_caption(dataset_workflow_image_path, "Dataset Preparation Workflow")
    else:
        st.info("Dataset preparation workflow diagram not available")
    
    st.markdown("""
    <div class="info-box">
        <p>The dataset preparation stage involves:</p>
        <ul>
            <li><b>Log Curve Selection:</b> Identifying and extracting relevant log curves</li>
            <li><b>Data Cleaning:</b> Handling missing values and outliers</li>
            <li><b>Feature Scaling:</b> Normalizing features for improved model performance</li>
            <li><b>Data Splitting:</b> Creating training, validation, and testing sets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Petrophysical Property Calculation
    st.markdown("<h2 class='sub-header'>Petrophysical Property Calculation</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>VHydro calculates the following petrophysical properties:</p>
        
        <h4>Shale Volume (Vsh)</h4>
        <p>Calculated from Gamma Ray logs using the formula:</p>
        <pre>Vsh = (GR - GRmin) / (GRmax - GRmin)</pre>
        <p>Tertiary rocks correction: Vsh = 0.083 * (2^(3.7 * Vsh) - 1)</p>
        
        <h4>Density Porosity (φd)</h4>
        <p>Calculated from Bulk Density logs using the formula:</p>
        <pre>φd = (ρmatrix - ρbulk) / (ρmatrix - ρfluid)</pre>
        <p>Where ρmatrix is matrix density and ρfluid is fluid density</p>
        
        <h4>Effective Porosity (φeff)</h4>
        <p>Corrected for shale content:</p>
        <pre>φeff = φd - (Vsh * 0.3)</pre>
        
        <h4>Water Saturation (Sw)</h4>
        <p>Calculated using Archie's equation:</p>
        <pre>Sw = ((a * (φeff^-m)) / (Rt * Rw))^(1/n)</pre>
        <p>Where a, m, n are constants, Rt is true resistivity, and Rw is formation water resistivity</p>
        
        <h4>Oil Saturation (So)</h4>
        <p>Calculated as:</p>
        <pre>So = 1 - Sw</pre>
        
        <h4>Permeability (K)</h4>
        <p>Estimated using porosity-based correlation:</p>
        <pre>K = 0.00004 * exp(57.117 * φeff)</pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Graph Construction and GCN Model
    st.markdown("<h2 class='sub-header'>Graph Convolutional Network Model</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Graph Construction</h4>
        <p>VHydro constructs a graph where:</p>
        <ul>
            <li><b>Nodes:</b> Depth points with associated petrophysical properties</li>
            <li><b>Edges:</b> Connections between related depth points based on facies classification</li>
            <li><b>Node Features:</b> Encoded petrophysical properties (PE_1, PE_2, etc.)</li>
            <li><b>Edge Features:</b> Relationships between facies at different depths</li>
        </ul>
        
        <h4>GCN Architecture</h4>
        <p>The Graph Convolutional Network consists of:</p>
        <ul>
            <li><b>Input Layer:</b> Node features from petrophysical properties</li>
            <li><b>Hidden Layers:</b> Graph convolutional layers that aggregate information from neighboring nodes</li>
            <li><b>Output Layer:</b> Classification layer for hydrocarbon quality prediction</li>
            <li><b>Regularization:</b> Dropout and batch normalization to prevent overfitting</li>
        </ul>
        
        <h4>Training Process</h4>
        <p>The model is trained using:</p>
        <ul>
            <li><b>Loss Function:</b> Categorical cross-entropy</li>
            <li><b>Optimizer:</b> Adam with learning rate scheduler</li>
            <li><b>Early Stopping:</b> Monitoring validation accuracy with patience</li>
            <li><b>Multiple Runs:</b> Training several models to select the best performer</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Analysis tool page
def analysis_tool_page(config):
    """Render the analysis tool page"""
    # Check if user is logged in - require login for analysis
    if not st.session_state.get("logged_in", False):
        st.warning("You need to log in to access the analysis tool.")
        if st.button("Go to Login"):
            st.session_state['current_page'] = "Login"
            st.rerun()
        return
    
    st.markdown('<h1 class="main-header">Analysis Tool</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">Upload your well log data in LAS format and run the VHydro analysis to predict hydrocarbon quality zones.</p>', unsafe_allow_html=True)
    
    # Check if VHydro is available
    if not VHYDRO_AVAILABLE:
        st.error("The VHydro module is not available. Analysis functionality is disabled.")
        st.info("Please check the logs for more information on why the module failed to load.")
        return
    
    # Create tabs for the analysis workflow
    tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Property Calculation", "Facies Classification", "GCN Model"])
    
    with tab1:
        st.markdown("<h3 class='section-header'>Upload LAS File</h3>", unsafe_allow_html=True)
        
        # File upload section
        st.markdown("""
        <div class="upload-area">
            <h4>Drag and drop your LAS file here</h4>
            <p>Or click to browse files</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a LAS file", type=["las"], key="las_uploader")
        
        if uploaded_file is not None:
            # Create a temp directory to save the file
            temp_dir = create_temp_dir()
            las_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file
            with open(las_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            
            # Store the path in session state
            st.session_state["las_path"] = las_path
            st.session_state["output_dir"] = os.path.join(temp_dir, "output")
            
            # Show file info
            st.markdown("<h4>File Information</h4>", unsafe_allow_html=True)
            
            # In a real application, you would parse the LAS file here
            st.json({
                "File Name": uploaded_file.name,
                "File Size": f"{uploaded_file.size / 1024:.2f} KB",
                "Upload Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Proceed button
            if st.button("Proceed to Property Calculation", key="proceed_to_prop_calc"):
                st.session_state["analysis_stage"] = "property_calculation"
    
    with tab2:
        st.markdown("<h3 class='section-header'>Petrophysical Property Calculation</h3>", unsafe_allow_html=True)
        
        # Check if user has uploaded a file
        if not st.session_state.get("las_path"):
            st.warning("Please upload a LAS file first.")
            return
        
        st.markdown("""
        <div class="info-box">
            <p>This step calculates key petrophysical properties from your well log data:</p>
            <ul>
                <li>Shale Volume (Vsh)</li>
                <li>Porosity (φ)</li>
                <li>Water Saturation (Sw)</li>
                <li>Oil Saturation (So)</li>
                <li>Permeability (K)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Option to configure calculation parameters
        with st.expander("Calculation Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                matrix_density = st.number_input("Matrix Density (g/cm³)", min_value=2.0, max_value=3.0, value=2.65, step=0.01)
                fluid_density = st.number_input("Fluid Density (g/cm³)", min_value=0.5, max_value=1.5, value=1.0, step=0.01)
            
            with col2:
                a_const = st.number_input("Archie Constant (a)", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
                m_const = st.number_input("Cementation Exponent (m)", min_value=1.5, max_value=2.5, value=2.0, step=0.1)
                n_const = st.number_input("Saturation Exponent (n)", min_value=1.5, max_value=2.5, value=2.0, step=0.1)
        
        # Run calculation button
        if st.button("Calculate Properties", key="calc_properties"):
            # In a real application, you would run the VHydro calculation here
            st.info("Calculating petrophysical properties...")
            
            # Show a progress bar for demonstration
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate calculation progress
                progress_bar.progress(i + 1)
                time.sleep(0.02)
            
            st.success("Petrophysical properties calculated successfully!")
            
            # Display sample results (in a real app, this would be actual results)
            sample_results = pd.DataFrame({
                "DEPTH": np.arange(1000, 1020, 1),
                "VSHALE": np.random.uniform(0.1, 0.5, 20),
                "PHI": np.random.uniform(0.05, 0.25, 20),
                "SW": np.random.uniform(0.3, 0.7, 20),
                "SO": np.random.uniform(0.3, 0.7, 20),
                "PERM": np.random.uniform(0.01, 100, 20)
            })
            
            st.dataframe(sample_results)
            
            # Store in session state and allow proceeding
            st.session_state["property_data"] = sample_results
            st.session_state["analysis_stage"] = "facies_classification"
            
            # Download button for results
            st.download_button(
                label="Download Properties",
                data=sample_results.to_csv(index=False),
                file_name="petrophysical_properties.csv",
                mime="text/csv",
                key="download_properties"
            )
    
    with tab3:
        st.markdown("<h3 class='section-header'>Facies Classification</h3>", unsafe_allow_html=True)
        
        # Check if properties have been calculated
        if not st.session_state.get("property_data") is not None:
            st.warning("Please calculate petrophysical properties first.")
            return
        
        st.markdown("""
        <div class="info-box">
            <p>This step identifies natural rock types (facies) using K-means clustering:</p>
            <ul>
                <li>Groups similar depth points based on petrophysical properties</li>
                <li>Optimizes the number of clusters using silhouette scores</li>
                <li>Generates depth-based facies maps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        
        with col1:
            min_clusters = st.number_input("Minimum Clusters", min_value=2, max_value=15, value=config["min_clusters"], step=1)
            feature_cols = st.multiselect("Features for Clustering", 
                                         options=["VSHALE", "PHI", "SW", "SO", "PERM", "GR", "DENSITY"],
                                         default=["VSHALE", "PHI", "SW", "GR", "DENSITY"])
        
        with col2:
            max_clusters = st.number_input("Maximum Clusters", min_value=min_clusters, max_value=15, value=config["max_clusters"], step=1)
        
        # Run clustering button
        if st.button("Run Facies Classification", key="run_clustering"):
            # In a real application, you would run the clustering here
            st.info("Running facies classification...")
            
            # Show a progress bar for demonstration
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate clustering progress
                progress_bar.progress(i + 1)
                time.sleep(0.02)
            
            st.success("Facies classification completed successfully!")
            
            # Display silhouette scores
            st.markdown("<h4>Silhouette Scores</h4>", unsafe_allow_html=True)
            
            # Sample silhouette scores (in a real app, these would be actual results)
            silhouette_df = pd.DataFrame({
                "Clusters": list(range(min_clusters, max_clusters + 1)),
                "Silhouette Score": np.random.uniform(0.4, 0.7, max_clusters - min_clusters + 1)
            })
            
            # Plot silhouette scores
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(silhouette_df["Clusters"], silhouette_df["Silhouette Score"], marker='o')
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Cluster Optimization")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Store in session state and allow proceeding
            st.session_state["silhouette_scores"] = silhouette_df
            st.session_state["best_clusters"] = silhouette_df["Silhouette Score"].idxmax() + min_clusters
            st.session_state["analysis_stage"] = "gcn_model"
            
            st.info(f"Optimal number of clusters: {st.session_state['best_clusters']}")
    
    with tab4:
        st.markdown("<h3 class='section-header'>Graph Convolutional Network Model</h3>", unsafe_allow_html=True)
        
        # Check if facies classification has been done
        if "best_clusters" not in st.session_state:
            st.warning("Please complete facies classification first.")
            return
        
        st.markdown("""
        <div class="info-box">
            <p>This step builds and trains a Graph Convolutional Network model:</p>
            <ul>
                <li>Constructs a graph from depth points and their relationships</li>
                <li>Trains a GCN model to predict hydrocarbon quality</li>
                <li>Evaluates model performance and generates final predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Model parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.number_input("Number of Clusters", min_value=min_clusters, max_value=max_clusters, 
                                       value=st.session_state.get("best_clusters", config["min_clusters"]), step=1)
            hidden_channels = st.number_input("Hidden Channels", min_value=4, max_value=64, value=config["hidden_channels"], step=4)
        
        with col2:
            num_runs = st.number_input("Number of Runs", min_value=1, max_value=10, value=config["num_runs"], step=1)
            epochs = st.number_input("Maximum Epochs", min_value=50, max_value=500, value=200, step=50)
        
        # Run model button
        if st.button("Train GCN Model", key="train_model"):
            # In a real application, you would train the GCN model here
            st.info("Training Graph Convolutional Network model...")
            
            # Show a progress bar for demonstration
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate training progress
                progress_bar.progress(i + 1)
                time.sleep(0.05)
            
            st.success("GCN model trained successfully!")
            
            # Display model performance
            st.markdown("<h4>Model Performance</h4>", unsafe_allow_html=True)
            
            # Sample history (in a real app, this would be actual results)
            history = {
                "loss": np.random.uniform(0.2, 0.8, 50),
                "acc": np.linspace(0.6, 0.95, 50) + np.random.normal(0, 0.02, 50),
                "val_acc": np.linspace(0.55, 0.9, 50) + np.random.normal(0, 0.03, 50),
                "test_acc": 0.88
            }
            
            # Plot training history
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss plot
            ax[0].plot(history["loss"], label="Training Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[0].set_title("Training Loss")
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            ax[1].plot(history["acc"], label="Training Accuracy")
            ax[1].plot(history["val_acc"], label="Validation Accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            ax[1].set_title("Model Accuracy")
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display test accuracy
            st.info(f"Test Accuracy: {history['test_acc']:.4f}")
            
            # Store in session state
            st.session_state["model_history"] = history
            st.session_state["analysis_stage"] = "prediction"
            
            # Display classification report
            st.markdown("<h4>Classification Report</h4>", unsafe_allow_html=True)
            
            # Sample classification report (in a real app, this would be actual results)
            report = {
                "class": ["Very_Low", "Low", "Moderate", "High", "Very_High"],
                "precision": np.random.uniform(0.75, 0.95, 5),
                "recall": np.random.uniform(0.75, 0.95, 5),
                "f1-score": np.random.uniform(0.75, 0.95, 5),
                "support": np.random.randint(20, 100, 5)
            }
            
            report_df = pd.DataFrame(report)
            st.table(report_df)
            
            # Download buttons for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Classification Report",
                    data=report_df.to_csv(index=False),
                    file_name="classification_report.csv",
                    mime="text/csv",
                    key="download_report"
                )
            
            with col2:
                st.download_button(
                    label="Download Predictions",
                    data="Sample predictions data",
                    file_name="hydrocarbon_predictions.csv",
                    mime="text/csv",
                    key="download_predictions"
                )
            
            # Show prediction visualization
            st.markdown("<h4>Hydrocarbon Quality Prediction</h4>", unsafe_allow_html=True)
            
            # Create a simple visualization of the results
            # In a real app, this would be based on actual data
            depth = np.arange(1000, 1200)
            predictions = np.random.randint(0, 5, size=200)  # 0=Very_Low, 4=Very_High
            
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = plt.cm.get_cmap('viridis', 5)
            
            # Create a depth vs prediction plot
            sc = ax.scatter(np.ones_like(depth), depth, c=predictions, cmap=cmap, 
                           s=100, marker='s')
            
            # Customize the plot
            ax.set_yticks(np.arange(1000, 1201, 25))
            ax.set_yticklabels(np.arange(1000, 1201, 25))
            ax.set_ylabel("Depth")
            ax.set_xlim(0.9, 1.1)
            ax.set_xticks([])
            ax.invert_yaxis()  # Invert y-axis to show depth increasing downward
            
            # Add a colorbar
            cbar = plt.colorbar(sc)
            cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
            cbar.set_label('Hydrocarbon Quality')
            
            plt.tight_layout()
            st.pyplot(fig)

# Results visualization page
def results_visualization_page():
    """Render the results visualization page"""
    # Check if user is logged in - require login for visualization
    if not st.session_state.get("logged_in", False):
        st.warning("You need to log in to access the results visualization.")
        if st.button("Go to Login"):
            st.session_state['current_page'] = "Login"
            st.rerun()
        return
    
    st.markdown('<h1 class="main-header">Results Visualization</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">Visualize and interpret hydrocarbon quality predictions and facies classifications from VHydro analysis.</p>', unsafe_allow_html=True)
    
    # Check if analysis has been completed
    if "model_history" not in st.session_state:
        st.warning("No analysis results found. Please run the analysis tool first.")
        
        # Add demo option
        if st.button("Load Demo Results"):
            # Simulate analysis results for demonstration
            st.session_state["analysis_stage"] = "prediction"
            st.session_state["model_history"] = {
                "loss": np.random.uniform(0.2, 0.8, 50),
                "acc": np.linspace(0.6, 0.95, 50) + np.random.normal(0, 0.02, 50),
                "val_acc": np.linspace(0.55, 0.9, 50) + np.random.normal(0, 0.03, 50),
                "test_acc": 0.88
            }
            st.rerun()
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Facies Visualization", "Quality Prediction", "Model Performance"])
    
    with tab1:
        st.markdown("<h3 class='section-header'>Facies Classification</h3>", unsafe_allow_html=True)
        
        # Create visualization
        st.markdown("""
        <div class="visualization-container">
            <div class="visualization-title">Facies Classification by Depth</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a dropdown to select cluster configuration
        cluster_options = [5, 6, 7, 8, 9, 10]
        selected_cluster = st.selectbox("Select Cluster Configuration", options=cluster_options, index=0)
        
        # Generate a random classification for demonstration
        depth = np.arange(1000, 1200)
        facies = np.random.randint(0, selected_cluster, size=200)
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.get_cmap('viridis', selected_cluster)
        
        # Create a depth vs facies plot
        sc = ax.scatter(np.ones_like(depth), depth, c=facies, cmap=cmap, 
                       s=100, marker='s')
        
        # Customize the plot
        ax.set_yticks(np.arange(1000, 1201, 25))
        ax.set_yticklabels(np.arange(1000, 1201, 25))
        ax.set_ylabel("Depth")
        ax.set_xlim(0.9, 1.1)
        ax.set_xticks([])
        ax.invert_yaxis()  # Invert y-axis to show depth increasing downward
        
        # Add a colorbar
        cbar = plt.colorbar(sc)
        cbar.set_ticks(np.arange(selected_cluster) + 0.5)
        cbar.set_ticklabels([f"Facies {i+1}" for i in range(selected_cluster)])
        cbar.set_label('Facies Classification')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab2:
        st.markdown("<h3 class='section-header'>Hydrocarbon Quality Prediction</h3>", unsafe_allow_html=True)
        
        # Create visualization
        st.markdown("""
        <div class="visualization-container">
            <div class="visualization-title">Hydrocarbon Quality Prediction by Depth</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create options for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            view_type = st.radio("View Type", ["Single Column", "Multi-Column", "Heat Map"])
        
        with col2:
            color_scheme = st.selectbox("Color Scheme", ["viridis", "plasma", "inferno", "magma", "cividis"])
        
        # Generate sample data for visualization
        depth = np.arange(1000, 1200)
        quality_labels = ["Very_Low", "Low", "Moderate", "High", "Very_High"]
        quality = np.random.randint(0, 5, size=200)
        
        if view_type == "Single Column":
            # Create the visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = plt.cm.get_cmap(color_scheme, 5)
            
            # Create a depth vs quality plot
            sc = ax.scatter(np.ones_like(depth), depth, c=quality, cmap=cmap, 
                           s=100, marker='s')
            
            # Customize the plot
            ax.set_yticks(np.arange(1000, 1201, 25))
            ax.set_yticklabels(np.arange(1000, 1201, 25))
            ax.set_ylabel("Depth")
            ax.set_xlim(0.9, 1.1)
            ax.set_xticks([])
            ax.invert_yaxis()  # Invert y-axis to show depth increasing downward
            
            # Add a colorbar
            cbar = plt.colorbar(sc)
            cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(quality_labels)
            cbar.set_label('Hydrocarbon Quality')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        elif view_type == "Multi-Column":
            # Create a multi-column view comparing different cluster configurations
            fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
            
            for i, ax in enumerate(axes):
                # Generate random data for each column (in a real app, this would be actual results)
                cluster_quality = np.random.randint(0, 5, size=200)
                
                cmap = plt.cm.get_cmap(color_scheme, 5)
                sc = ax.scatter(np.ones_like(depth), depth, c=cluster_quality, cmap=cmap, 
                               s=100, marker='s')
                
                # Customize the plot
                if i == 0:
                    ax.set_yticks(np.arange(1000, 1201, 25))
                    ax.set_yticklabels(np.arange(1000, 1201, 25))
                    ax.set_ylabel("Depth")
                ax.set_xlim(0.9, 1.1)
                ax.set_xticks([])
                ax.set_title(f"Clusters: {i+5}")
                ax.invert_yaxis()
            
            # Add a colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(sc, cax=cbar_ax)
            cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(quality_labels)
            cbar.set_label('Hydrocarbon Quality')
            
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            st.pyplot(fig)
            
        else:  # Heat Map
            # Create a heat map view
            quality_data = np.random.randint(0, 5, size=(200, 3))  # 3 different models
            
            fig, ax = plt.subplots(figsize=(12, 8))
            cmap = plt.cm.get_cmap(color_scheme, 5)
            
            im = ax.imshow(quality_data, aspect='auto', cmap=cmap, 
                          extent=[0, 3, 1200, 1000])
            
            # Customize the plot
            ax.set_yticks(np.arange(1000, 1201, 25))
            ax.set_yticklabels(np.arange(1000, 1201, 25))
            ax.set_ylabel("Depth")
            ax.set_xticks([0.5, 1.5, 2.5])
            ax.set_xticklabels(["Model A", "Model B", "Model C"])
            
            # Add a colorbar
            cbar = plt.colorbar(im)
            cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(quality_labels)
            cbar.set_label('Hydrocarbon Quality')
            
            plt.tight_layout()
            st.pyplot(fig)
        
    with tab3:
        st.markdown("<h3 class='section-header'>Model Performance</h3>", unsafe_allow_html=True)
        
        # Create visualization
        st.markdown("""
        <div class="visualization-container">
            <div class="visualization-title">GCN Model Training Performance</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get model history from session state
        history = st.session_state["model_history"]
        
        # Create the visualization
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax[0].plot(history["loss"], 'b-', label='training loss', linewidth=2)
        ax[0].set_title("Model Loss", fontsize=14)
        ax[0].set_xlabel("Epoch", fontsize=12)
        ax[0].set_ylabel("Loss", fontsize=12)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        ax[1].plot(history["acc"], 'b-', label='training', linewidth=2)
        ax[1].plot(history["val_acc"], 'r-', label='validation', linewidth=2)
        ax[1].axhline(y=history["test_acc"], color='g', linestyle='-', label='test', linewidth=2)
        ax[1].set_title("Model Accuracy", fontsize=14)
        ax[1].set_xlabel("Epoch", fontsize=12)
        ax[1].set_ylabel("Accuracy", fontsize=12)
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Confusion matrix
        st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
        
        # Generate a random confusion matrix for demonstration
        labels = ["Very_Low", "Low", "Moderate", "High", "Very_High"]
        cm = np.random.randint(5, 30, size=(5, 5))
        np.fill_diagonal(cm, np.random.randint(30, 50, size=5))  # Higher values on diagonal
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Count')
        
        # Set tick marks and labels
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)
        
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
        
        # Model comparison across different cluster configurations
        st.markdown("<h4>Model Comparison</h4>", unsafe_allow_html=True)
        
        # Generate random data for comparison
        cluster_configs = [5, 6, 7, 8, 9, 10]
        accuracies = np.random.uniform(0.75, 0.95, len(cluster_configs))
        f1_scores = np.random.uniform(0.70, 0.93, len(cluster_configs))
        
        # Create a bar chart for comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(cluster_configs))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Accuracy')
        ax.bar(x + width/2, f1_scores, width, label='F1 Score')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"Clusters: {c}" for c in cluster_configs])
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance by Cluster Configuration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# Login page handler
def login_page():
    """Handle the login page"""
    if st.session_state.get("logged_in", False):
        # User is already logged in, redirect to home
        st.success("You are already logged in")
        st.session_state['current_page'] = "Home"
        st.rerun()
    else:
        # Show authentication page
        auth_page()

# Main function
def main():
    # Initialize session state variables if not already set
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Home"
    
    # Create sidebar and get configuration
    config = create_sidebar()
    
    # Check if we need to display a specific page based on session state
    current_page = st.session_state.get('current_page', 'Home')
    
    # Display selected page
    if current_page == "Home":
        home_page()
    elif current_page == "Dataset Preparation":
        dataset_preparation_page()
    elif current_page == "Model Workflow":
        model_workflow_page()
    elif current_page == "Analysis Tool":
        analysis_tool_page(config)
    elif current_page == "Results Visualization":
        results_visualization_page()
    elif current_page == "Account":
        account_page()
    elif current_page == "Login":
        login_page()
    
    # Footer
    st.markdown("<div class='footer'>VHydro - Advanced Hydrocarbon Quality Prediction © 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
