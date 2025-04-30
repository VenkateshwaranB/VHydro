import streamlit as st
import os
import numpy as np
import pandas as pd
import time
from PIL import Image
import base64
import logging

# Configure matplotlib for faster loading
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/matplotlib_cache'
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration at the very beginning
st.set_page_config(
    page_title="StrataGraph",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache expensive operations
@st.cache_data
def load_css():
    css = """
    /* Main styling */
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }
    h1, h2, h3, h4, h5, h6 { color: #0e4194; }
    
    /* Logo centering fix */
    [data-testid="stSidebar"] .element-container:first-child {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] .element-container:first-child img {
        max-width: 80% !important;
        margin: 0 auto;
    }
    
    /* Fallback logo styling */
    .logo-fallback {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 1.5rem auto;
        text-align: center;
    }
    
    .logo-icon {
        background-color: rgba(255, 255, 255, 0.1);
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    
    .logo-title {
        color: white;
        font-size: 1.5rem;
        margin: 0.5rem 0 0.25rem 0;
    }
    
    .logo-subtitle {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #0e4194 0%, #153a6f 100%); 
    }
    
    /* Sidebar navigation styling */
    .sidebar-nav {
        margin-top: 1rem;
    }
    
    .nav-item {
        padding: 0.5rem 1rem;
        margin-bottom: 0.25rem;
        border-radius: 4px;
        cursor: pointer;
        color: rgba(255, 255, 255, 0.8);
        transition: all 0.2s ease;
    }
    
    .nav-item:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    .nav-item.active {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border-left: 3px solid white;
    }
    
    .nav-sub-item {
        padding: 0.4rem 1rem 0.4rem 2rem;
        margin-bottom: 0.15rem;
        border-radius: 4px;
        cursor: pointer;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }
    
    .nav-sub-item:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    .nav-sub-item.active {
        background-color: rgba(255, 255, 255, 0.15);
        color: white;
        border-left: 2px solid white;
    }
    
    /* Coming soon tag */
    .coming-soon-tag {
        background-color: rgba(255, 152, 0, 0.2);
        color: rgb(255, 152, 0);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 8px;
        vertical-align: middle;
    }
    
    /* Navigation section styling */
    .nav-section {
        color: #8c9196;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 18px 0 8px 10px;
        margin: 0;
    }
    
    /* Style for soon tag */
    .soon-tag {
        background-color: #f8cd5a;
        color: #664d03;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 10px;
        margin-left: 8px;
    }
    
    /* Footer text */
    .footer-text {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.8rem;
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    /* Version section */
    .version-section {
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    
    .version-section h4 {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .version-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        color: white;
    }
    
    .version-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 10px;
    }
    
    .active-version {
        background-color: #4CAF50;
    }
    
    .coming-version {
        background-color: #FFA500;
    }
    
    /* Home page styling */
    .hero-section {
        background-color: #0e4194;
        color: white;
        padding: 60px 20px;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
    }
    
    .hero-title {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .hero-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
    }
    
    .primary-button {
        background-color: white;
        color: #0e4194;
        padding: 0.6rem 1.5rem;
        border-radius: 5px;
        text-decoration: none;
        font-weight: 600;
    }
    
    .secondary-button {
        background-color: transparent;
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 5px;
        text-decoration: none;
        border: 1px solid white;
        font-weight: 600;
    }
    
    .section-title {
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
    }
    
    .card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.8rem;
    }
    
    .feature-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
    }
    
    /* Workflow step styling */
    .workflow-step {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background-color: #e6f3ff;
        border-radius: 5px;
        border-left: 4px solid #0d6efd;
    }
    
    .step-number {
        margin-right: 1rem;
        font-weight: bold;
        color: #0d6efd;
    }
    
    .step-content .step-title {
        font-weight: bold;
        color: #0d6efd;
    }
    
    .step-content .step-description {
        color: #495057;
    }
    
    /* Statistics cards */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stat-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stat-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0d6efd;
    }
    
    /* CO2 storage section */
    .co2-section {
        background-color: #1a2332;
        color: white;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2.5rem;
    }
    
    /* Call to action section */
    .cta-section {
        background-color: #0e4194;
        color: white;
        padding: 3rem 2rem;
        margin: 2rem -1rem -1rem -1rem;
        text-align: center;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Helper function to logout
def logout():
    for key in ["email", "logged_in", "auth_mode"]:
        if key in st.session_state:
            del st.session_state[key]

# Preload and cache images
@st.cache_data
def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {e}")
        return None

# Create a sidebar navigation system with reliable subsection display
def create_sidebar():
    # Logo and title
    try:
        st.sidebar.image("src/StrataGraph_White_Logo.png", width=130)
    except:
        st.sidebar.markdown(
            """
            <div class="logo-fallback">
                <div class="logo-icon">
                    <span style="color: white; font-size: 24px;">SG</span>
                </div>
                <h2 class="logo-title">StrataGraph</h2>
                <p class="logo-subtitle">VHydro 1.0</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Initialize current page if not exists
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
    # Track expanded state of VHydro in session state
    if "vhydro_expanded" not in st.session_state:
        st.session_state["vhydro_expanded"] = False
    
    # Define all pages
    main_pages = ["Home", "VHydro", "CO2 Storage Applications", "Help and Contact", "About Us"]
    vhydro_pages = ["VHydro Overview", "Data Preparation", "Petrophysical Properties", 
                   "Facies Classification", "Hydrocarbon Potential Using GCN"]
    
    # Check if current page is in VHydro section
    is_vhydro_page = st.session_state["current_page"] in vhydro_pages
    if is_vhydro_page and not st.session_state["vhydro_expanded"]:
        st.session_state["vhydro_expanded"] = True
    
    # MAIN NAVIGATION SECTION
    st.sidebar.markdown('<div class="nav-section">MAIN NAVIGATION</div>', unsafe_allow_html=True)
    
    # Home navigation item
    home_active = st.session_state["current_page"] == "Home"
    if st.sidebar.button("🏠 Home", key="nav_home", use_container_width=True, 
                       type="secondary" if not home_active else "primary"):
        st.session_state["current_page"] = "Home"
        st.rerun()
    
    # VHydro navigation item
    vhydro_active = st.session_state["current_page"] in vhydro_pages or st.session_state["current_page"] == "VHydro"
    if st.sidebar.button("📈 VHydro", key="nav_vhydro", use_container_width=True,
                       type="secondary" if not vhydro_active else "primary"):
        st.session_state["current_page"] = "VHydro Overview"
        st.rerun()
    
    # CO2 Storage navigation item
    co2_active = st.session_state["current_page"] == "CO2 Storage Applications"
    co2_button = st.sidebar.button("⭕ CO2 Storage", key="nav_co2", use_container_width=True,
                                type="secondary" if not co2_active else "primary")
    # Create a container to show the "Soon" badge next to CO2 Storage
    st.sidebar.markdown("""
    <div style="margin-top: -4px; margin-bottom: 10px; text-align: right; padding-right: 10px;">
        <span class="soon-tag">Soon</span>
    </div>
    """, unsafe_allow_html=True)
    
    if co2_button:
        st.session_state["current_page"] = "CO2 Storage Applications"
        st.rerun()
    
    # MODULES SECTION
    st.sidebar.markdown('<div class="nav-section">MODULES</div>', unsafe_allow_html=True)
    
    # Data Preparation navigation item
    data_active = st.session_state["current_page"] == "Data Preparation"
    if st.sidebar.button("📝 Data Preparation", key="nav_data_prep", use_container_width=True,
                       type="secondary" if not data_active else "primary"):
        st.session_state["current_page"] = "Data Preparation"
        st.rerun()
    
    # Petrophysical Properties navigation item
    petro_active = st.session_state["current_page"] == "Petrophysical Properties"
    if st.sidebar.button("⚙️ Petrophysical Properties", key="nav_petro", use_container_width=True,
                       type="secondary" if not petro_active else "primary"):
        st.session_state["current_page"] = "Petrophysical Properties"
        st.rerun()
    
    # Facies Classification navigation item
    facies_active = st.session_state["current_page"] == "Facies Classification"
    if st.sidebar.button("📊 Facies Classification", key="nav_facies", use_container_width=True,
                       type="secondary" if not facies_active else "primary"):
        st.session_state["current_page"] = "Facies Classification"
        st.rerun()
    
    # GCN Analysis navigation item
    gcn_active = st.session_state["current_page"] == "Hydrocarbon Potential Using GCN"
    if st.sidebar.button("🌐 GCN Analysis", key="nav_gcn", use_container_width=True,
                       type="secondary" if not gcn_active else "primary"):
        st.session_state["current_page"] = "Hydrocarbon Potential Using GCN"
        st.rerun()
    
    # SUPPORT SECTION
    st.sidebar.markdown('<div class="nav-section">SUPPORT</div>', unsafe_allow_html=True)
    
    # Help and Contact navigation item
    help_active = st.session_state["current_page"] == "Help and Contact"
    if st.sidebar.button("❓ Help and Contact", key="nav_help", use_container_width=True,
                       type="secondary" if not help_active else "primary"):
        st.session_state["current_page"] = "Help and Contact"
        st.rerun()
    
    # About Us navigation item
    about_active = st.session_state["current_page"] == "About Us"
    if st.sidebar.button("ℹ️ About Us", key="nav_about", use_container_width=True,
                       type="secondary" if not about_active else "primary"):
        st.session_state["current_page"] = "About Us"
        st.rerun()
    
    # Versions section
    st.sidebar.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 5px; margin-top: 20px;">
            <div style="font-weight: bold; color: white; margin-bottom: 10px;">Versions</div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #4CAF50; margin-right: 10px;"></div>
                <span style="color: white;">VHydro 1.0 (Current)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #FFA500; margin-right: 10px;"></div>
                <span style="color: white;">CO2 Storage 2.0 (Coming Soon)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Footer
    st.sidebar.markdown(
        """
        <div style="color: rgba(255, 255, 255, 0.7); font-size: 12px; text-align: center; margin-top: 30px;">
            © 2025 StrataGraph. All rights reserved.<br>
            Version 1.0.0
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Analysis parameters
    min_clusters = 5
    max_clusters = 10
    
    if st.session_state["current_page"] == "Facies Classification":
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        st.sidebar.markdown('<div style="color: white; font-weight: bold;">Analysis Parameters</div>', unsafe_allow_html=True)
        min_clusters = st.sidebar.slider("Min Clusters", 2, 15, 5)
        max_clusters = st.sidebar.slider("Max Clusters", min_clusters, 15, 10)
    
    return {
        "page": st.session_state["current_page"],
        "min_clusters": min_clusters,
        "max_clusters": max_clusters
    }

def home_page():
    # Hero section with blue background
    st.markdown("""
    <div style="background-color: #0e4194; color: white; padding: 60px 20px 80px 20px; margin: -1rem -1rem 2rem -1rem; text-align: center;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">StrataGraph</h1>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">Subsurface strata properties represented as graph datasets for advanced deep learning applications</p>
        <div style="display: flex; justify-content: center; gap: 1rem;">
            <a href="/?page=Data Preparation" style="background-color: white; color: #0e4194; padding: 0.6rem 1.5rem; border-radius: 5px; text-decoration: none; font-weight: 600;">Get Started</a>
            <a href="/?page=VHydro Overview" style="background-color: transparent; color: white; padding: 0.6rem 1.5rem; border-radius: 5px; text-decoration: none; border: 1px solid white; font-weight: 600;">Learn More</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # About StrataGraph section
    st.markdown("""
    <h2 style="font-size: 2rem; margin-bottom: 1.5rem;">About StrataGraph</h2>
    <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
        StrataGraph is a pioneering tool for subsurface analysis that represents complex geological relationships as graph datasets. By transforming
        conventional well log data into graph-based structures, StrataGraph enables more accurate predictions for both hydrocarbon exploration and
        carbon storage applications.
    </p>
    <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 2.5rem;">
        Our first release (VHydro 1.0) focuses on hydrocarbon quality prediction using Graph Convolutional Networks (GCNs). The upcoming StrataGraph
        2.0 will expand these capabilities to CO2 storage potential analysis.
    </p>
    """, unsafe_allow_html=True)
    
    # Key Features section
    st.markdown("""
    <h2 style="font-size: 1.8rem; margin-bottom: 1.5rem;">Key Features</h2>
    <ul style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 2.5rem; list-style-type: none; padding-left: 0;">
        <li style="margin-bottom: 0.8rem;">• Graph-based representation of well log data</li>
        <li style="margin-bottom: 0.8rem;">• Physics-Informed Graph Neural Networks (PI-GNNs)</li>
        <li style="margin-bottom: 0.8rem;">• Advanced facies classification using K-means clustering</li>
        <li style="margin-bottom: 0.8rem;">• Interactive visualization of geological relationships</li>
        <li style="margin-bottom: 0.8rem;">• Comprehensive analysis workflow from raw data to predictions</li>
    </ul>
    """, unsafe_allow_html=True)
    
    # VHydro Methodology section
    st.markdown("""
    <h2 style="font-size: 1.8rem; margin-bottom: 1.5rem;">VHydro Methodology</h2>
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 2.5rem;">
        <h3 style="font-size: 1.3rem; margin-bottom: 1rem;">Workflow</h3>
        
        <div style="display: flex; align-items: center; margin-bottom: 1rem; padding: 1rem; background-color: #e6f3ff; border-radius: 5px; border-left: 4px solid #0d6efd;">
            <div style="margin-right: 1rem; font-weight: bold; color: #0d6efd;">1.</div>
            <div>
                <div style="font-weight: bold; color: #0d6efd;">Data Preparation</div>
                <div style="color: #495057;">Upload and process well log data (LAS files)</div>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 1rem; padding: 1rem; background-color: #e6f3ff; border-radius: 5px; border-left: 4px solid #0d6efd;">
            <div style="margin-right: 1rem; font-weight: bold; color: #0d6efd;">2.</div>
            <div>
                <div style="font-weight: bold; color: #0d6efd;">Petrophysical Analysis</div>
                <div style="color: #495057;">Calculate key properties like porosity, permeability</div>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 1rem; padding: 1rem; background-color: #e6f3ff; border-radius: 5px; border-left: 4px solid #0d6efd;">
            <div style="margin-right: 1rem; font-weight: bold; color: #0d6efd;">3.</div>
            <div>
                <div style="font-weight: bold; color: #0d6efd;">Facies Classification</div>
                <div style="color: #495057;">Group similar depth points using K-means clustering</div>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 1rem; padding: 1rem; background-color: #e6f3ff; border-radius: 5px; border-left: 4px solid #0d6efd;">
            <div style="margin-right: 1rem; font-weight: bold; color: #0d6efd;">4.</div>
            <div>
                <div style="font-weight: bold; color: #0d6efd;">Graph Construction</div>
                <div style="color: #495057;">Create graph representation with nodes and edges</div>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 1rem; padding: 1rem; background-color: #e6f3ff; border-radius: 5px; border-left: 4px solid #0d6efd;">
            <div style="margin-right: 1rem; font-weight: bold; color: #0d6efd;">5.</div>
            <div>
                <div style="font-weight: bold; color: #0d6efd;">GCN Training</div>
                <div style="color: #495057;">Train graph neural network to predict hydrocarbon quality</div>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem; padding: 1rem; background-color: #e6f3ff; border-radius: 5px; border-left: 4px solid #0d6efd;">
            <div style="margin-right: 1rem; font-weight: bold; color: #0d6efd;">6.</div>
            <div>
                <div style="font-weight: bold; color: #0d6efd;">Results Visualization</div>
                <div style="color: #495057;">Interpret and visualize prediction results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to load the workflow image
    workflow_path = "src/Workflow.png"
    try:
        st.image(workflow_path, use_container_width=True, caption="VHydro Analysis Workflow")
    except:
        st.info("Workflow visualization diagram will be displayed here.")
    
    # Technical Implementation section
    st.markdown("""
    <h2 style="font-size: 1.8rem; margin-bottom: 1.5rem;">Technical Implementation</h2>
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 2.5rem;">
        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
            VHydro implements Graph Convolutional Networks (GCNs) using PyTorch Geometric and StellarGraph frameworks. 
            The model architecture includes:
        </p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1rem;">
            <div style="background-color: white; padding: 1rem; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="color: #6c757d; font-size: 0.9rem;">Hidden Channels</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #0d6efd;">16</div>
            </div>
            
            <div style="background-color: white; padding: 1rem; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="color: #6c757d; font-size: 0.9rem;">Layers</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #0d6efd;">2</div>
            </div>
            
            <div style="background-color: white; padding: 1rem; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="color: #6c757d; font-size: 0.9rem;">Dropout Rate</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #0d6efd;">0.5</div>
            </div>
            
            <div style="background-color: white; padding: 1rem; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="color: #6c757d; font-size: 0.9rem;">Learning Rate</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #0d6efd;">0.01</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Applications section
    st.markdown("""
    <h2 style="font-size: 1.8rem; margin-bottom: 1.5rem;">Applications</h2>
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 2.5rem;">
        <ul style="font-size: 1.1rem; line-height: 1.6; list-style-type: none; padding-left: 0;">
            <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                <span style="background-color: #28a745; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
                Accurate prediction of hydrocarbon quality zones
            </li>
            <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                <span style="background-color: #28a745; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
                Spatial relationship modeling between geological features
            </li>
            <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                <span style="background-color: #28a745; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
                Enhanced facies classification with preserved spatial context
            </li>
            <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                <span style="background-color: #28a745; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
                Quantification of geological uncertainty
            </li>
            <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                <span style="background-color: #28a745; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
                Integration with traditional petrophysical analysis
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # StrataGraph 2.0 section
    st.markdown("""
    <div style="background-color: #1a2332; color: white; padding: 2rem; border-radius: 8px; margin-bottom: 2.5rem;">
        <h2 style="font-size: 1.8rem; margin-bottom: 1rem;">StrataGraph 2.0 - CO2 Storage Potential Analysis</h2>
        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
            Building upon VHydro's graph-based geological modeling, StrataGraph 2.0 will focus on carbon capture and storage applications with these advanced features:
        </p>
        
        <ul style="font-size: 1.1rem; line-height: 1.6; list-style-type: none; padding-left: 0; margin-bottom: 1.5rem;">
            <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                <span style="background-color: #007bff; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
                CO2 storage capacity prediction
            </li>
            <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                <span style="background-color: #007bff; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
                Caprock integrity analysis using geomechanical properties
            </li>
            <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                <span style="background-color: #007bff; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
                Risk assessment for long-term storage
            </li>
            <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                <span style="background-color: #007bff; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
                Physics-Informed GNNs for geomechanical simulations
            </li>
        </ul>
        
        <p style="font-size: 1.2rem; margin-bottom: 1.5rem;">Coming in 2025</p>
        
        <a href="#" style="background-color: #007bff; color: white; padding: 0.6rem 1.5rem; border-radius: 5px; text-decoration: none; font-weight: 600; display: inline-block; margin-bottom: 1.5rem;">Join Waitlist</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Call-to-action section
    st.markdown("""
    <div style="background-color: #0e4194; color: white; padding: 3rem 2rem; margin: 2rem -1rem -1rem -1rem; text-align: center;">
        <h2 style="font-size: 2rem; margin-bottom: 1rem;">Ready to Analyze Your Subsurface Data?</h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">Start with VHydro 1.0 today and transform your well log data into powerful graph-based insights.</p>
        <a href="/?page=Data Preparation" style="background-color: white; color: #0e4194; padding: 0.8rem 2rem; border-radius: 5px; text-decoration: none; font-weight: 600; display: inline-block;">Get Started with VHydro</a>
    </div>
    """, unsafe_allow_html=True)

def vhydro_overview_page():
    st.markdown("""
    <div style="background-color: #0e4194; color: white; padding: 40px 20px; margin: -1rem -1rem 2rem -1rem; text-align: center;">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">VHydro</h1>
        <p style="font-size: 1.2rem;">Hydrocarbon Quality Prediction Using Graph Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h2 style="font-size: 1.8rem; margin-bottom: 1rem;">About VHydro</h2>
    <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
        VHydro is StrataGraph's flagship module for predicting hydrocarbon quality zones using Graph Convolutional Networks (GCNs).
        This advanced approach models the complex relationships between petrophysical properties and depth to provide more accurate 
        predictions than traditional methods.
    </p>
    """, unsafe_allow_html=True)
    
    # Tab section
    tab1, tab2, tab3 = st.tabs(["Overview", "Workflow", "Technical Details"])
    
    with tab1:
        st.markdown("""
        <h3 style="font-size: 1.5rem; margin-bottom: 1rem;">VHydro Features</h3>
        <ul style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
            <li>Graph-based representation of well log data</li>
            <li>Advanced facies classification using K-means clustering</li>
            <li>GCN-based quality prediction model</li>
            <li>Interactive visualization of results</li>
            <li>Complete workflow from data preparation to prediction</li>
        </ul>
        
        <h3 style="font-size: 1.5rem; margin-bottom: 1rem;">Benefits</h3>
        <ul style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
            <li>Higher accuracy compared to traditional methods</li>
            <li>Preservation of spatial relationships between data points</li>
            <li>Better handling of geological heterogeneity</li>
            <li>Integrated approach combining petrophysics and machine learning</li>
        </ul>
        """, unsafe_allow_html=True)
        
    with tab2:
        # Display the workflow diagram
        # Try to load the workflow image
        workflow_path = "src/Workflow.png"
        try:
            st.image(workflow_path, use_container_width=True)
        except:
            st.info("Workflow diagram will be displayed here.")
            
        st.markdown("""
        <h3 style="font-size: 1.5rem; margin-bottom: 1rem;">VHydro Workflow</h3>
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <ol style="font-size: 1.1rem; line-height: 1.8;">
                <li><strong>Data Preparation</strong>: Upload and process well log data (LAS files)</li>
                <li><strong>Petrophysical Analysis</strong>: Calculate key properties like porosity, permeability</li>
                <li><strong>Facies Classification</strong>: Group similar depth points using K-means clustering</li>
                <li><strong>Graph Construction</strong>: Create a graph representation of the well data</li>
                <li><strong>GCN Training</strong>: Train the model to predict hydrocarbon quality</li>
                <li><strong>Visualization</strong>: Interpret and visualize results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
    with tab3:
        st.markdown("""
        <h3 style="font-size: 1.5rem; margin-bottom: 1rem;">Technical Implementation</h3>
        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
            VHydro uses PyTorch Geometric and StellarGraph frameworks to implement Graph Convolutional Networks tailored for geoscience applications.
            The model architecture is designed specifically for handling geological data with spatial relationships.
        </p>
        """, unsafe_allow_html=True)
        
        # Model parameters using simple metrics displayed in a grid
        st.markdown("""
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;">
                <div style="color: #6c757d; font-size: 0.9rem;">Hidden Channels</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #0d6efd;">16</div>
            </div>
            
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;">
                <div style="color: #6c757d; font-size: 0.9rem;">Layers</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #0d6efd;">2</div>
            </div>
            
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;">
                <div style="color: #6c757d; font-size: 0.9rem;">Dropout Rate</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #0d6efd;">0.5</div>
            </div>
            
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;">
                <div style="color: #6c757d; font-size: 0.9rem;">Learning Rate</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #0d6efd;">0.01</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="font-size: 1.3rem; margin-bottom: 1rem;">Model Architecture</h4>
        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">
            The GCN model consists of multiple graph convolutional layers that process node features and edge connections.
            Each node represents a depth point with associated petrophysical properties, while edges represent geological relationships.
        </p>
        
        <h4 style="font-size: 1.3rem; margin-bottom: 1rem;">Training Process</h4>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            The model is trained using supervised learning with early stopping to prevent overfitting.
            Training data is split into training, validation, and test sets to ensure robust performance evaluation.
        </p>
        """, unsafe_allow_html=True)
    
    # Button to start the analysis workflow
    st.markdown("<h3 style='font-size: 1.5rem; margin: 2rem 0 1rem 0;'>Start VHydro Analysis</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Begin Data Preparation", key="begin_analysis_btn", use_container_width=True):
            st.session_state["current_page"] = "Data Preparation"
            st.rerun()

def data_preparation_page():
    st.markdown("""
    <div style="background-color: #0e4194; color: white; padding: 40px 20px; margin: -1rem -1rem 2rem -1rem; text-align: center;">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">Data Preparation</h1>
        <p style="font-size: 1.2rem;">Prepare your well log data for VHydro analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.5rem; margin-bottom: 1rem;">VHydro Data Preparation</h2>
        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
            VHydro requires specific log curves to calculate petrophysical properties needed for accurate predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use tabs to organize content but load lazily
    tab1, tab2 = st.tabs(["Required Curves", "Upload Data"])
    
    with tab1:
        # Simple table instead of complex HTML
        curves = [
            ["GR/CGR", "Gamma Ray", "Shale volume calculation"],
            ["RHOB", "Bulk Density", "Density porosity calculation"],
            ["NPHI", "Neutron Porosity", "Effective porosity calculation"],
            ["LLD/ILD", "Deep Resistivity", "Water/oil saturation calculation"],
            ["DEPT", "Depth", "Spatial reference for facies"]
        ]
        
        df = pd.DataFrame(curves, columns=["Curve", "Description", "Purpose"])
        st.table(df)
    
    with tab2:
        # File upload component
        uploaded_file = st.file_uploader("Choose a LAS file", type=["las"])
        
        if uploaded_file is not None:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            
            # Store in session state
            st.session_state["uploaded_file"] = uploaded_file.name
            
            # Show file info
            file_info = {
                "File Name": uploaded_file.name,
                "File Size": f"{uploaded_file.size / 1024:.2f} KB",
                "Upload Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Use dataframe for display instead of HTML
            st.dataframe(pd.DataFrame(list(file_info.items()), 
                        columns=["Property", "Value"]))
            
            # Proceed button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Process Data", use_container_width=True):
                    with st.spinner("Processing data..."):
                        # Simulate processing with minimal updates
                        progress_bar = st.progress(0)
                        for i in range(0, 101, 25):
                            progress_bar.progress(i)
                            time.sleep(0.1)
                    
                    st.success("Data processing complete!")
                    st.session_state["analysis_stage"] = "property_calculation"
                    st.session_state["current_page"] = "Petrophysical Properties"
                    st.rerun()

def petrophysical_properties_page():
    st.markdown("""
    <div style="background-color: #0e4194; color: white; padding: 40px 20px; margin: -1rem -1rem 2rem -1rem; text-align: center;">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">Petrophysical Properties</h1>
        <p style="font-size: 1.2rem;">Calculate key reservoir properties from well log data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user has uploaded a file
    if "uploaded_file" not in st.session_state:
        st.warning("Please upload a LAS file first.")
        
        # Button to go back to upload tab
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Go to Data Preparation", use_container_width=True):
                st.session_state["current_page"] = "Data Preparation"
                st.rerun()
        return
    
    st.markdown("""
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">
            This step calculates key petrophysical properties from your well log data:
        </p>
        <ul style="font-size: 1.1rem; line-height: 1.6;">
            <li>Shale Volume (Vsh)</li>
            <li>Porosity (φ)</li>
            <li>Water Saturation (Sw)</li>
            <li>Oil Saturation (So)</li>
            <li>Permeability (K)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Option to configure calculation parameters
    with st.expander("Show Calculation Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            matrix_density = st.number_input("Matrix Density (g/cm³)", min_value=2.0, max_value=3.0, value=2.65, step=0.01)
            fluid_density = st.number_input("Fluid Density (g/cm³)", min_value=0.5, max_value=1.5, value=1.0, step=0.01)
        
        with col2:
            a_const = st.number_input("Archie Constant (a)", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
            m_const = st.number_input("Cementation Exponent (m)", min_value=1.5, max_value=2.5, value=2.0, step=0.1)
    
    # Run calculation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Calculate Properties", use_container_width=True):
            with st.spinner("Calculating petrophysical properties..."):
                # Simulate calculation progress with fewer updates
                progress_bar = st.progress(0)
                for i in range(0, 101, 25):
                    progress_bar.progress(i)
                    time.sleep(0.1)
            
            st.success("Petrophysical properties calculated successfully!")
            
            # Display sample results
            sample_results = pd.DataFrame({
                "DEPTH": np.arange(1000, 1010, 1),
                "VSHALE": np.random.uniform(0.1, 0.5, 10),
                "PHI": np.random.uniform(0.05, 0.25, 10),
                "SW": np.random.uniform(0.3, 0.7, 10),
                "SO": np.random.uniform(0.3, 0.7, 10),
                "PERM": np.random.uniform(0.01, 100, 10)
            })
            
            st.dataframe(sample_results)
            
            # Add a download button for the properties
            csv = sample_results.to_csv(index=False)
            st.download_button(
                label="Download Properties CSV",
                data=csv,
                file_name="petrophysical_properties.csv",
                mime="text/csv"
            )
            
            # Store in session state
            st.session_state["property_data"] = True
            st.session_state["analysis_stage"] = "facies_classification"
            
            # Button to proceed to next step
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Proceed to Facies Classification", use_container_width=True):
                    st.session_state["current_page"] = "Facies Classification"
                    st.rerun()
