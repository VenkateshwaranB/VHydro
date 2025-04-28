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
    .colored-header { background: linear-gradient(90deg, #0e4194 0%, #3a6fc4 100%); color: white; 
                     padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
    .card { border-radius: 10px; padding: 20px; margin-bottom: 20px; background: white; 
           box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); }
    .feature-card { border-radius: 10px; padding: 20px; margin-bottom: 20px; background: white; 
                   box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); height: 100%; 
                   border-left: 5px solid #0e4194; }
    .feature-header { font-weight: bold; color: #0e4194; margin-bottom: 10px; font-size: 1.2rem; }
    
    
    /* CO2 Storage section styling */
    .co2-section {
        background: linear-gradient(to right, rgba(13, 31, 51, 0.9), rgba(29, 68, 111, 0.9)), url('https://placehold.co/600x400');
        background-size: cover;
        color: white;
        padding: 35px;
        border-radius: 15px;
        margin: 30px 0;
        position: relative;
        overflow: hidden;
    }
    
    .co2-section h2 {
        color: white;
        margin-bottom: 15px;
        font-size: 2rem;
        position: relative;
    }
    
    .co2-section h2::after {
        content: '';
        display: block;
        width: 60px;
        height: 4px;
        background: #4CAF50;
        margin-top: 10px;
    }
    
    .co2-section p {
        font-size: 1.1rem;
        line-height: 1.5;
        margin-bottom: 20px;
        max-width: 80%;
    }
    
    .co2-features {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 25px;
    }
    
    .co2-feature {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        width: calc(50% - 10px);
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
    }
    
    .co2-feature h3 {
        color: #4CAF50;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.2rem;
    }
    
    .co2-feature p {
        margin: 0;
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .release-date {
        margin-top: 30px;
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.85);
    }
    
    /* Team and profile cards for About page */
    .team-section {
        margin-top: 30px;
        margin-bottom: 30px;
    }
    
    .profile-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        height: 100%;
        border-left: 5px solid #0e4194;
    }
    
    .profile-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        border-bottom: 1px solid #e9ecef;
        padding-bottom: 15px;
    }
    
    .profile-avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background-color: #0e4194;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 36px;
        margin-right: 20px;
    }
    
    .profile-title h3 {
        margin: 0;
        color: #0e4194;
    }
    
    .profile-title p {
        margin: 5px 0 0 0;
        color: #6c757d;
    }
    
    .profile-bio {
        margin-bottom: 15px;
    }
    
    .profile-links {
        display: flex;
        gap: 10px;
    }
    
    .profile-link {
        padding: 5px 10px;
        border-radius: 5px;
        background-color: #f1f3f5;
        text-decoration: none;
        color: #495057;
        font-size: 0.9rem;
    }
    
    .profile-link:hover {
        background-color: #e9ecef;
    }
    
    .supervisor-section, .collaborator-section {
        margin-top: 30px;
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
    
    /* Waitlist form styling */
    .waitlist-form {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .co2-feature {
            width: 100%;
        }
        
        .co2-section p {
            max-width: 100%;
        }
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

def create_sidebar():
    # Logo and title
    try:
        st.sidebar.image("src/StrataGraph_White_Logo.png", width=130)
    except:
        # Fallback for logo
        st.sidebar.markdown(
            """
            <div style="text-align: center; margin: 20px 0;">
                <div style="background-color: #f0f2f6; width: 60px; height: 60px; display: inline-flex; align-items: center; justify-content: center; border-radius: 8px; margin: 0 auto;">
                    <span style="color: #8c9196; font-size: 24px;">SG</span>
                </div>
                <h2 style="color: white; font-size: 20px; margin-top: 10px; margin-bottom: 0;">StrataGraph</h2>
                <p style="color: #8c9196; font-size: 14px; margin-top: 4px;">VHydro 1.0</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Initialize current page if not exists
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
    # Add custom CSS for the sidebar styling
    st.markdown("""
    <style>
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
    
    /* Hide default Streamlit button styling */
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # MAIN NAVIGATION SECTION
    st.sidebar.markdown('<div class="nav-section">MAIN NAVIGATION</div>', unsafe_allow_html=True)
    
    # Define which pages are in each section
    main_pages = ["Home", "VHydro", "CO2 Storage Applications"]
    vhydro_pages = ["VHydro Overview", "VHydro"]
    modules_pages = ["Data Preparation", "Petrophysical Properties", "Facies Classification", "Hydrocarbon Potential Using GCN"]
    support_pages = ["Help and Contact", "About Us"]
    
    # Define the navigation items with icons
    main_nav = [
        {"page": "Home", "icon": "🏠", "label": "Home"},
        {"page": "VHydro", "icon": "📈", "label": "VHydro"},
        {"page": "CO2 Storage Applications", "icon": "⭕", "label": "CO2 Storage", "tag": '<span class="soon-tag">Soon</span>'}
    ]
    
    # Create clickable navigation items for main section
    for item in main_nav:
        page = item["page"]
        is_current = st.session_state["current_page"] == page or (page == "VHydro" and st.session_state["current_page"] in vhydro_pages)
        
        # Create a container for each nav item to style it properly
        container = st.sidebar.container()
        
        # Use columns to create the icon + text layout
        col1, col2 = container.columns([1, 9])
        
        with col1:
            st.markdown(f'<div style="text-align: center; font-size: 18px;">{item["icon"]}</div>', unsafe_allow_html=True)
        
        with col2:
            label_html = f'{item["label"]}{item.get("tag", "")}'
            
            # Handle click with a button disguised as a nav item
            if st.button(
                item["label"], 
                key=f"nav_{page.replace(' ', '_').lower()}", 
                use_container_width=True
            ):
                st.session_state["current_page"] = page
                st.rerun()
                
            # Apply styling based on active state
            bg_color = "rgba(144, 202, 249, 0.1)" if is_current else "transparent"
            border_left = "3px solid #90CAF9" if is_current else "none"
            
            # Inject the styling for the container
            st.markdown(f"""
            <style>
            div[data-testid="column"]:has(button#{f"nav_{page.replace(' ', '_').lower()}"}:first-of-type) {{
                background-color: {bg_color};
                border-left: {border_left};
                border-radius: 4px;
                padding: 8px 0;
                margin: 2px 0;
            }}
            button#{f"nav_{page.replace(' ', '_').lower()}"} {{
                color: #f0f2f6 !important;
                text-align: left !important;
                font-weight: 500 !important;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            # Inject the badge if needed
            if "tag" in item:
                st.markdown(f"""
                <style>
                button#{f"nav_{page.replace(' ', '_').lower()}"} + div {{
                    display: inline !important;
                    position: absolute !important;
                    right: 10px !important;
                    top: 10px !important;
                }}
                button#{f"nav_{page.replace(' ', '_').lower()}"} + div:after {{
                    content: '{item["tag"]}';
                    background-color: #f8cd5a;
                    color: #664d03;
                    font-size: 10px;
                    font-weight: 600;
                    padding: 2px 6px;
                    border-radius: 10px;
                    margin-left: 8px;
                }}
                </style>
                """, unsafe_allow_html=True)
    
    # MODULES SECTION
    st.sidebar.markdown('<div class="nav-section">MODULES</div>', unsafe_allow_html=True)
    
    # Module navigation items
    modules_nav = [
        {"page": "Data Preparation", "icon": "📈", "label": "Data Preparation"},
        {"page": "Petrophysical Properties", "icon": "⚙️", "label": "Petrophysical Properties"},
        {"page": "Facies Classification", "icon": "📊", "label": "Facies Classification"},
        {"page": "Hydrocarbon Potential Using GCN", "icon": "🌐", "label": "GCN Analysis"}
    ]
    
    # Create clickable navigation items for modules section
    for item in modules_nav:
        page = item["page"]
        is_current = st.session_state["current_page"] == page
        
        # Create a container for each nav item
        container = st.sidebar.container()
        
        # Use columns to create the icon + text layout
        col1, col2 = container.columns([1, 9])
        
        with col1:
            st.markdown(f'<div style="text-align: center; font-size: 18px;">{item["icon"]}</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button(
                item["label"], 
                key=f"nav_{page.replace(' ', '_').lower()}", 
                use_container_width=True
            ):
                st.session_state["current_page"] = page
                st.rerun()
            
            # Apply styling based on active state
            bg_color = "rgba(144, 202, 249, 0.1)" if is_current else "transparent"
            border_left = "3px solid #90CAF9" if is_current else "none"
            
            # Inject the styling for the container
            st.markdown(f"""
            <style>
            div[data-testid="column"]:has(button#{f"nav_{page.replace(' ', '_').lower()}"}:first-of-type) {{
                background-color: {bg_color};
                border-left: {border_left};
                border-radius: 4px;
                padding: 8px 0;
                margin: 2px 0;
            }}
            button#{f"nav_{page.replace(' ', '_').lower()}"} {{
                color: #f0f2f6 !important;
                text-align: left !important;
                font-weight: 500 !important;
            }}
            </style>
            """, unsafe_allow_html=True)
    
    # SUPPORT SECTION
    st.sidebar.markdown('<div class="nav-section">SUPPORT</div>', unsafe_allow_html=True)
    
    # Support navigation items
    support_nav = [
        {"page": "Help and Contact", "icon": "❓", "label": "Help and Contact"},
        {"page": "About Us", "icon": "ℹ️", "label": "About Us"}
    ]
    
    # Create clickable navigation items for support section
    for item in support_nav:
        page = item["page"]
        is_current = st.session_state["current_page"] == page
        
        # Create a container for each nav item
        container = st.sidebar.container()
        
        # Use columns to create the icon + text layout
        col1, col2 = container.columns([1, 9])
        
        with col1:
            st.markdown(f'<div style="text-align: center; font-size: 18px;">{item["icon"]}</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button(
                item["label"], 
                key=f"nav_{page.replace(' ', '_').lower()}", 
                use_container_width=True
            ):
                st.session_state["current_page"] = page
                st.rerun()
            
            # Apply styling based on active state
            bg_color = "rgba(144, 202, 249, 0.1)" if is_current else "transparent"
            border_left = "3px solid #90CAF9" if is_current else "none"
            
            # Inject the styling for the container
            st.markdown(f"""
            <style>
            div[data-testid="column"]:has(button#{f"nav_{page.replace(' ', '_').lower()}"}:first-of-type) {{
                background-color: {bg_color};
                border-left: {border_left};
                border-radius: 4px;
                padding: 8px 0;
                margin: 2px 0;
            }}
            button#{f"nav_{page.replace(' ', '_').lower()}"} {{
                color: #f0f2f6 !important;
                text-align: left !important;
                font-weight: 500 !important;
            }}
            </style>
            """, unsafe_allow_html=True)
    
    # Add footer to sidebar
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="color: rgba(255, 255, 255, 0.7); font-size: 12px; text-align: center; margin-top: 30px;">
            © 2025 StrataGraph. All rights reserved.<br>
            Version 1.0.0
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Analysis parameters for Facies Classification page
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
    # Try to load the banner image
    banner_path = "src/StrataGraph_Banner.png"
    try:
        st.image(banner_path, use_container_width=True)
    except:
        st.markdown("""
        <div class="colored-header">
            <h1>StrataGraph</h1>
            <p>Subsurface strata properties represented as a graph dataset for deep learning applications</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>About StrataGraph</h2>
        <p>StrataGraph is a cutting-edge platform for geoscience modeling and analysis, combining advanced machine learning techniques with traditional geological and petrophysical analysis.</p>
        <p>Our innovative approach utilizes graph-based data structures to represent complex subsurface relationships, enabling more accurate predictions and insights for both hydrocarbon exploration and carbon storage applications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # First section: VHydro
    st.markdown("""
    <div class="card">
        <h2>StrataGraph 1.0 - VHydro</h2>
        <p>Our first release focuses on hydrocarbon quality prediction using Graph Convolutional Networks (GCNs) that model complex relationships between different petrophysical properties and depth values.</p>
        <p>VHydro 1.0 enables accurate prediction of hydrocarbon zones using a graph-based approach that captures the spatial relationships between well log measurements.</p>
        <p>This approach was introduced in our paper: <a href="https://link.springer.com/article/10.1007/s11053-024-10311-x" target="_blank">Hydrocarbon Potential Prediction Using Novel Graph Dataset</a>, which combines petrophysical and facies features to classify potential zones using GCN.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # VHydro Workflow section
    st.markdown("<h2>VHydro Workflow</h2>", unsafe_allow_html=True)
    
    # Try to load the workflow image
    workflow_path = "src/Workflow.png"
    try:
        st.image(workflow_path, use_container_width=True)
    except:
        st.warning("Workflow image not available.")
    
    # Single button to explore VHydro
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Explore VHydro Analysis Tool", key="explore_vhydro_btn", 
                    use_container_width=True):
            st.session_state["current_page"] = "VHydro Overview"
            st.rerun()

    # Second section: CO2 Storage (Coming Soon) - Enhanced version based on image
    st.markdown("""
    <div class="co2-section">
        <span class="coming-soon-tag">COMING SOON</span>
        <h2>StrataGraph 2.0 - CO2 Storage Potential Analysis</h2>
        <p>Building upon VHydro's graph-based geological modeling, StrataGraph 2.0 will focus on carbon capture and storage applications with these advanced features:</p>
        
        <div class="co2-features">
            <div class="co2-feature">
                <h3>CO2 Storage Capacity Prediction</h3>
                <p>Advanced modeling of reservoir storage capacity using graph neural networks trained on petrophysical properties.</p>
            </div>
            
            <div class="co2-feature">
                <h3>Caprock Integrity Analysis</h3>
                <p>Assessment of caprock integrity using geomechanical properties to ensure long-term CO2 containment.</p>
            </div>
            
            <div class="co2-feature">
                <h3>Risk Assessment</h3>
                <p>Comprehensive risk assessment for long-term storage using graph-based connectivity analysis.</p>
            </div>
            
            <div class="co2-feature">
                <h3>Physics-Informed GNNs</h3>
                <p>Integration of physical laws with data-driven approaches for more accurate geomechanical simulations.</p>
            </div>
        </div>
        
        <div class="release-date">Anticipated Release: Q3 2025</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Join Waitlist Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Join StrataGraph 2.0 Waitlist", key="co2_waitlist_btn", 
                    use_container_width=True):
            st.session_state["show_waitlist_form"] = True
    
    # Waitlist form if button clicked
    if st.session_state.get("show_waitlist_form", False):
        st.markdown("""
        <style>
        .waitlist-form {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        </style>
        <div class="waitlist-form">
            <h3>Join the StrataGraph 2.0 Waitlist</h3>
            <p>Be the first to know when our CO2 Storage Potential Analysis tools become available.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Full Name")
            st.text_input("Email Address")
        with col2:
            st.text_input("Organization")
            st.selectbox("Primary Interest", ["Carbon Storage", "Hydrocarbon Exploration", "Research", "Education", "Other"])
        
        if st.button("Submit", use_container_width=True):
            st.success("Thank you for joining the waitlist! We'll notify you when StrataGraph 2.0 becomes available.")
            st.session_state["show_waitlist_form"] = False

def vhydro_overview_page():
    st.markdown("""
    <div class="colored-header">
        <h1>VHydro</h1>
        <p>Hydrocarbon Quality Prediction Using Graph Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>About VHydro</h2>
        <p>VHydro is StrataGraph's flagship module for predicting hydrocarbon quality zones using Graph Convolutional Networks (GCNs).</p>
        <p>This advanced approach models the complex relationships between petrophysical properties and depth to provide more accurate predictions than traditional methods.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add tabs for the different sections
    tab1, tab2, tab3 = st.tabs(["Overview", "Workflow", "Technical Details"])
    
    with tab1:
        st.markdown("""
        <h3>VHydro Features</h3>
        <ul>
            <li>Graph-based representation of well log data</li>
            <li>Advanced facies classification using K-means clustering</li>
            <li>GCN-based quality prediction model</li>
            <li>Interactive visualization of results</li>
        </ul>
        """, unsafe_allow_html=True)
        
    with tab2:
        # Try to load the workflow image
        workflow_path = "src/Workflow.png"
        try:
            st.image(workflow_path, use_container_width=True)
        except:
            st.warning("Workflow image not available.")
            
        st.markdown("""
        <h3>VHydro Workflow</h3>
        <ol>
            <li><strong>Data Preparation</strong>: Upload and process well log data</li>
            <li><strong>Petrophysical Analysis</strong>: Calculate key properties like porosity, permeability</li>
            <li><strong>Facies Classification</strong>: Group similar depth points using K-means clustering</li>
            <li><strong>Graph Construction</strong>: Create a graph representation of the well data</li>
            <li><strong>GCN Training</strong>: Train the model to predict hydrocarbon quality</li>
            <li><strong>Visualization</strong>: Interpret and visualize results</li>
        </ol>
        """, unsafe_allow_html=True)
        
    with tab3:
        st.markdown("""
        <h3>Technical Implementation</h3>
        <p>VHydro uses PyTorch Geometric and StellarGraph frameworks to implement Graph Convolutional Networks tailored for geoscience applications.</p>
        """, unsafe_allow_html=True)
        
        # Model parameters using simple metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Hidden Channels", "16")
        with col2: st.metric("Layers", "2")
        with col3: st.metric("Dropout Rate", "0.5")
        with col4: st.metric("Learning Rate", "0.01")
    
    # Button to start the analysis workflow
    st.markdown("<h3>Start VHydro Analysis</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Begin Data Preparation", key="begin_analysis_btn", use_container_width=True):
            st.session_state["current_page"] = "Data Preparation"
            st.rerun()

def data_preparation_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Data Preparation</h1>
        <p>Prepare your well log data for VHydro analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>VHydro Data Preparation</h2>
        <p>VHydro requires specific log curves to calculate petrophysical properties needed for accurate predictions:</p>
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
    <div class="colored-header">
        <h1>Petrophysical Properties</h1>
        <p>Calculate key reservoir properties from well log data</p>
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
    <div class="card">
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

def facies_classification_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Facies Classification</h1>
        <p>Identify geological facies using machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if properties have been calculated
    if "property_data" not in st.session_state:
        st.warning("Please calculate petrophysical properties first.")
        
        # Button to go back to property calculation tab
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Go to Petrophysical Properties", use_container_width=True):
                st.session_state["current_page"] = "Petrophysical Properties"
                st.rerun()
        return
    
    st.markdown("""
    <div class="card">
        <p>This step identifies natural rock types (facies) using K-means clustering:</p>
        <ul>
            <li>Groups similar depth points based on petrophysical properties</li>
            <li>Optimizes the number of clusters using silhouette scores</li>
            <li>Generates depth-based facies maps</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Clustering parameters
    with st.expander("Clustering Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            min_clusters = st.number_input("Minimum Clusters", min_value=2, max_value=15, value=5, step=1)
            feature_cols = st.multiselect("Features for Clustering", 
                                         options=["VSHALE", "PHI", "SW", "SO", "PERM", "GR", "DENSITY"],
                                         default=["VSHALE", "PHI", "SW", "GR", "DENSITY"])
        
        with col2:
            max_clusters = st.number_input("Maximum Clusters", min_value=min_clusters, max_value=15, value=10, step=1)
            algorithm = st.selectbox("Clustering Algorithm", ["K-means", "Agglomerative", "DBSCAN"])
    
    # Run clustering button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Run Facies Classification", use_container_width=True):
            with st.spinner("Running facies classification..."):
                # Simpler progress indicator
                progress_bar = st.progress(0)
                for i in range(0, 101, 20):
                    progress_bar.progress(i)
                    time.sleep(0.1)
            
            st.success("Facies classification completed successfully!")
            
            # Generate simulated silhouette scores
            silhouette_scores = {
                i: np.random.uniform(0.4, 0.7) for i in range(min_clusters, max_clusters + 1)
            }
            
            # Find optimal clusters
            optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
            st.info(f"Optimal number of clusters: {optimal_clusters}")
            
            # Create a simple facies dataset for download
            facies_df = pd.DataFrame({
                "DEPTH": np.arange(1000, 1100),
                "FACIES": np.random.randint(0, optimal_clusters, size=100)
            })
            
            # Visualization in an expander to keep the UI clean
            with st.expander("Facies Visualization", expanded=True):
                # Create a simple visualization
                plt.figure(figsize=(10, 7))
                plt.imshow(facies_df['FACIES'].values.reshape(-1, 1), aspect='auto', cmap='viridis',
                          extent=[0, 1, facies_df['DEPTH'].max(), facies_df['DEPTH'].min()])
                plt.title(f"Facies Classification (Clusters: {optimal_clusters})")
                plt.ylabel("Depth")
                plt.xticks([])
                plt.colorbar(label="Facies")
                st.pyplot(plt)
            
            # Show sample data
            st.subheader("Sample Data")
            st.dataframe(facies_df.head())
            
            # Download button
            csv = facies_df.to_csv(index=False)
            st.download_button(
                label="Download Facies CSV",
                data=csv,
                file_name="facies_classification.csv",
                mime="text/csv"
            )
            
            # Store in session state
            st.session_state["facies_data"] = True
            st.session_state["best_clusters"] = optimal_clusters
            st.session_state["analysis_stage"] = "gcn_model"
            
            # Button to proceed to next step
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Proceed to Hydrocarbon Potential Prediction", use_container_width=True):
                    st.session_state["current_page"] = "Hydrocarbon Potential Using GCN"
                    st.rerun()

def hydrocarbon_prediction_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Hydrocarbon Potential Prediction</h1>
        <p>Graph Convolutional Network (GCN) Model for Quality Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if facies classification has been done
    if "facies_data" not in st.session_state:
        st.warning("Please complete facies classification first.")
        
        # Button to go back to facies classification tab
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Go to Facies Classification", use_container_width=True):
                st.session_state["current_page"] = "Facies Classification"
                st.rerun()
        return
    
    st.markdown("""
    <div class="card">
        <p>This step builds and trains a Graph Convolutional Network model:</p>
        <ul>
            <li>Constructs a graph from depth points and their relationships</li>
            <li>Trains a GCN model to predict hydrocarbon quality</li>
            <li>Evaluates model performance and generates final predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model parameters in expandable section
    with st.expander("Model Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.number_input("Number of Clusters", 
                                       min_value=2, 
                                       max_value=15, 
                                       value=int(st.session_state.get("best_clusters", 7)), 
                                       step=1)
            hidden_channels = st.number_input("Hidden Channels", min_value=4, max_value=64, value=16, step=4)
            
        with col2:
            num_runs = st.number_input("Number of Runs", min_value=1, max_value=10, value=4, step=1)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                value=0.01
            )
    
    # Advanced parameters in toggle
    with st.expander("Advanced Model Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.5, step=0.1)
            epochs = st.number_input("Maximum Epochs", min_value=50, max_value=500, value=200, step=50)
        
        with col2:
            patience = st.number_input("Early Stopping Patience", min_value=5, max_value=100, value=20, step=5)
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
    
    # Run model button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Train GCN Model", key="train_gcn_btn", use_container_width=True):
            with st.spinner("Training GCN model..."):
                # Simpler progress indicator with fewer updates
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                steps = ["Preparing graph structure...", "Creating features...", 
                        "Training model...", "Finalizing predictions..."]
                
                for i, step in enumerate(steps):
                    progress_bar.progress(i * 25)
                    status_text.info(step)
                    time.sleep(0.5)
                
                progress_bar.progress(100)
            
            # Clear status area and show success message
            status_text.empty()
            st.success("GCN model trained successfully!")
            
            # Use tabs for organizing results
            tabs = st.tabs(["Model Performance", "Quality Predictions", "Classification Report"])
            
            with tabs[0]:
                # Model performance metrics
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Test Accuracy", "0.88")
                with col2: st.metric("F1 Score", "0.86")
                with col3: st.metric("AUC", "0.92")
                
                # Learning curves visualization
                st.subheader("Learning Curves")
                
                # Create sample data for learning curves
                epochs = np.arange(1, 101)
                train_loss = 1.0 - 0.8 * np.exp(-epochs/30) + 0.05 * np.random.randn(100)
                val_loss = 1.2 - 0.7 * np.exp(-epochs/25) + 0.1 * np.random.randn(100)
                train_acc = 0.3 + 0.6 * (1 - np.exp(-epochs/30)) + 0.03 * np.random.randn(100)
                val_acc = 0.2 + 0.6 * (1 - np.exp(-epochs/35)) + 0.05 * np.random.randn(100)
                
                # Create and display the learning curves
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
                ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training and Validation Loss')
                ax1.legend()
                ax1.grid(alpha=0.3)
                
                ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
                ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Training and Validation Accuracy')
                ax2.legend()
                ax2.grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tabs[1]:
                # Create sample prediction data
                predictions = pd.DataFrame({
                    "DEPTH": np.arange(1000, 1010),
                    "PREDICTED_QUALITY": np.random.choice(
                        ["Very Low", "Low", "Moderate", "High", "Very High"], 
                        size=10
                    )
                })
                
                # Display predictions table
                st.subheader("Hydrocarbon Quality Predictions")
                st.dataframe(predictions)
                
                # Download button
                csv = predictions.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="quality_predictions.csv",
                    mime="text/csv"
                )
                
                # Quality visualization
                st.subheader("Quality Distribution Visualization")
                
                # Sample data for visualization
                quality_levels = ["Very Low", "Low", "Moderate", "High", "Very High"]
                quality_counts = [np.random.randint(5, 30) for _ in range(5)]
                colors = ['#FF5757', '#FFBD59', '#4FB47A', '#5271FF', '#B89393']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(quality_levels, quality_counts, color=colors)
                ax.set_xlabel('Hydrocarbon Quality')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Predicted Hydrocarbon Quality')
                
                # Add count labels above bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{int(height)}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Depth-based visualization
                st.subheader("Depth-based Quality Visualization")
                
                # Create sample data for depth visualization
                depths = np.arange(1000, 1100)
                quality_codes = np.random.randint(0, 5, size=100)  # 0 to 4 for the quality levels
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(5, 10))
                cmap = plt.cm.get_cmap('viridis', 5)
                
                # Create a 2D array for imshow
                quality_array = np.vstack((quality_codes, quality_codes)).T
                
                im = ax.imshow(quality_array, aspect='auto', cmap=cmap, 
                              extent=[0, 1, depths.max(), depths.min()])
                
                ax.set_ylabel('Depth')
                ax.set_xticks([])
                ax.set_title('Hydrocarbon Quality by Depth')
                
                # Create a custom colorbar with quality labels
                cbar = plt.colorbar(im, ax=ax, ticks=[0.4, 1.2, 2, 2.8, 3.6])
                cbar.set_ticklabels(quality_levels)
                
                st.pyplot(fig)
            
            with tabs[2]:
                # Sample classification report
                report_data = {
                    "Class": ["Very Low", "Low", "Moderate", "High", "Very High", "Average"],
                    "Precision": [0.85, 0.78, 0.92, 0.86, 0.91, 0.86],
                    "Recall": [0.82, 0.75, 0.90, 0.89, 0.84, 0.84],
                    "F1-Score": [0.83, 0.76, 0.91, 0.87, 0.87, 0.85],
                    "Support": [25, 31, 42, 28, 19, 145]
                }
                
                # Create and display dataframe
                report_df = pd.DataFrame(report_data)
                st.table(report_df)
                
                # Confusion matrix visualization
                st.subheader("Confusion Matrix")
                
                # Create sample confusion matrix data
                classes = ["Very Low", "Low", "Moderate", "High", "Very High"]
                cm = np.array([
                    [21, 3, 1, 0, 0],
                    [2, 24, 4, 1, 0],
                    [1, 3, 37, 1, 0],
                    [0, 1, 2, 24, 1],
                    [0, 0, 0, 2, 17]
                ])
                
                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm, cmap='Blues')
                
                # Add labels, title and ticks
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                ax.set_xticks(np.arange(len(classes)))
                ax.set_yticks(np.arange(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha="right")
                ax.set_yticklabels(classes)
                
                # Add text annotations to show the values
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        text = ax.text(j, i, cm[i, j],
                                      ha="center", va="center", color="white" if cm[i, j] > 10 else "black")
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Store in session state
            st.session_state["model_history"] = True
            st.session_state["analysis_complete"] = True
            
            # Download full results button
            st.subheader("Download Complete Analysis")
            st.markdown("Download a comprehensive report of the analysis results:")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="Download Analysis Report",
                    data="Sample report content that would be more detailed in a real application.",
                    file_name="vhydro_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

def co2_storage_page():
    st.markdown("""
    <div class="colored-header">
        <h1>CO2 Storage Applications</h1>
        <p>Carbon Capture, Utilization, and Storage (CCUS) Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced CO2 Storage section from the home page
    st.markdown("""
    <div class="co2-section">
        <span class="coming-soon-tag">COMING SOON</span>
        <h2>CO2 Storage Potential Analysis</h2>
        <p>Building upon VHydro's graph-based geological modeling, StrataGraph 2.0 will focus on carbon capture and storage applications with these advanced features:</p>
        
        <div class="co2-features">
            <div class="co2-feature">
                <h3>CO2 Storage Capacity Prediction</h3>
                <p>Advanced modeling of reservoir storage capacity using graph neural networks trained on petrophysical properties.</p>
            </div>
            
            <div class="co2-feature">
                <h3>Caprock Integrity Analysis</h3>
                <p>Assessment of caprock integrity using geomechanical properties to ensure long-term CO2 containment.</p>
            </div>
            
            <div class="co2-feature">
                <h3>Risk Assessment</h3>
                <p>Comprehensive risk assessment for long-term storage using graph-based connectivity analysis.</p>
            </div>
            
            <div class="co2-feature">
                <h3>Physics-Informed GNNs</h3>
                <p>Integration of physical laws with data-driven approaches for more accurate geomechanical simulations.</p>
            </div>
        </div>
        
        <div class="release-date">Anticipated Release: Q3 2025</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature preview
    st.markdown("""
    <div class="card">
        <h2>Technical Development</h2>
        <p>StrataGraph 2.0 will build upon the graph-based reservoir characterization from VHydro 1.0 with several scientific advancements:</p>
        <ul>
            <li>Physics-Informed Graph Neural Networks (PI-GNNs) for improved prediction accuracy</li>
            <li>Integration with existing carbon storage databases for more robust model training</li>
            <li>Enhanced visualization tools for long-term storage simulation monitoring</li>
            <li>Comprehensive risk assessment methodologies based on graph connectivity analysis</li>
            <li>Specialized caprock integrity models using geomechanical properties</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Simplified sign up form
    st.markdown("""
    <div class="waitlist-form">
        <h2>Stay Updated</h2>
        <p>Sign up to receive updates when CO2 Storage Applications becomes available.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Name")
        st.text_input("Email")
    with col2:
        st.text_input("Organization")
        st.selectbox("Area of Interest", ["Carbon Storage", "Hydrocarbon Production", "Research", "Education", "Other"])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Notify Me", use_container_width=True):
            st.success("Thank you for your interest! We'll notify you when StrataGraph 2.0 becomes available.")

def help_contact_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Help and Contact</h1>
        <p>Get support and connect with our team</p>
    </div>
    """, unsafe_allow_html=True)
    
    # FAQ section
    st.markdown("""
    <div class="card">
        <h2>Frequently Asked Questions</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("What file formats are supported?"):
        st.markdown("""
        Currently, StrataGraph supports the following file formats:
        - LAS (Log ASCII Standard) for well log data
        - CSV for tabular data
        - Excel spreadsheets for tabular data
        """)
    
    with st.expander("How accurate is the hydrocarbon prediction model?"):
        st.markdown("""
        The GCN-based hydrocarbon prediction model typically achieves 85-92% accuracy on test datasets,
        depending on data quality and completeness. The model is trained using both supervised and
        unsupervised learning approaches to ensure robust predictions.
        """)
    
    with st.expander("What are the system requirements?"):
        st.markdown("""
        StrataGraph is a web-based application that runs in your browser. The recommended specifications are:
        - Modern web browser (Chrome, Firefox, Edge)
        - Minimum 8GB RAM for optimal performance with large datasets
        - Internet connection for cloud-based processing
        """)
    
    with st.expander("Can I use my own custom clustering algorithm?"):
        st.markdown("""
        Yes, StrataGraph is designed to be flexible. While the default implementation uses K-means clustering, 
        advanced users can implement custom clustering algorithms by modifying the code or using the API endpoints.
        Contact our support team for guidance on implementing custom algorithms.
        """)
    
    with st.expander("Is my data secure?"):
        st.markdown("""
        Yes, data security is a top priority. StrataGraph uses industry-standard encryption for data transfer and storage.
        All data processing occurs within secure environments, and you maintain complete ownership of your data.
        We do not share your data with third parties.
        """)
    
    # Contact form
    st.markdown("""
    <div class="card">
        <h2>Contact Us</h2>
        <p>Have questions or need assistance? Reach out to our team.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Name")
        st.text_input("Email")
    with col2:
        st.text_input("Subject")
        st.selectbox("Category", ["Technical Support", "Feature Request", "Billing", "Other"])
    
    st.text_area("Message", height=150)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Send Message", use_container_width=True):
            st.success("Your message has been sent. We'll get back to you soon!")

def about_us_page():
    st.markdown("""
    <div class="colored-header">
        <h1>About Us</h1>
        <p>Learn about StrataGraph and our mission</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Our Mission</h2>
        <p>StrataGraph is committed to revolutionizing geoscience analysis through advanced machine learning techniques. 
        Our mission is to provide geoscientists and engineers with powerful, intuitive tools that transform complex subsurface 
        data into actionable insights through innovative graph-based approaches.</p>
        
        <h2>Our Vision</h2>
        <p>We envision a future where geological and petrophysical data analysis is enhanced by the power of graph-based 
        deep learning, enabling more accurate predictions for both hydrocarbon exploration and carbon storage applications. 
        By bridging traditional geoscience with cutting-edge AI, we aim to contribute to both energy security and climate solutions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Research and publications
    st.markdown("""
    <div class="card">
        <h2>Research and Publications</h2>
        <p>Our technology is built on peer-reviewed research:</p>
        <ul>
            <li><a href="https://link.springer.com/article/10.1007/s11053-024-10311-x" target="_blank">Hydrocarbon Potential Prediction Using Novel Graph Dataset</a> - 
            This paper introduces our approach to hydrocarbon potential prediction using Graph Convolutional Networks.</li>
            <li>Graph-based methodologies for subsurface characterization using machine learning techniques.</li>
            <li>Forthcoming research on CO2 storage potential assessment using graph-based approaches.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Lead Developer & Research Team
    st.markdown('<h2 class="team-section">Lead Developer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="profile-card">
            <div class="profile-header">
                <div class="profile-avatar">VB</div>
                <div class="profile-title">
                    <h3>Venkateshwaran Baskaran, Ph.D.</h3>
                    <p>Computational Geoscientist</p>
                </div>
            </div>
            <div class="profile-bio">
                <p>Ph.D. in Petroleum Geoscience from Universiti Teknologi PETRONAS. Specializes in graph-based machine learning 
                applications for subsurface analysis and geological modeling.</p>
            </div>
            <div class="profile-links">
                <a href="https://www.linkedin.com/in/venkateshwaran-baskaran/" target="_blank" class="profile-link">LinkedIn</a>
                <a href="https://scholar.google.com/citations?user=YOURID" target="_blank" class="profile-link">Google Scholar</a>
                <a href="https://github.com/venkateshwaranb" target="_blank" class="profile-link">GitHub</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Research Advisors & Supervisors
    st.markdown('<h2 class="supervisor-section">Research Advisors & Supervisors</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="profile-card">
            <div class="profile-header">
                <div class="profile-avatar">AH</div>
                <div class="profile-title">
                    <h3>Dr. AKM Eashanul Haque</h3>
                    <p>Assistant Professor</p>
                    <p>Universiti Teknologi PETRONAS</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="profile-card">
            <div class="profile-header">
                <div class="profile-avatar">HR</div>
                <div class="profile-title">
                    <h3>Dr. Hariharan Ramachandran</h3>
                    <p>Research Associate</p>
                    <p>Heriot Watt University</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="profile-card">
            <div class="profile-header">
                <div class="profile-avatar">NA</div>
                <div class="profile-title">
                    <h3>Dr. Numair Ahmed Siddiqui</h3>
                    <p>Associate Professor</p>
                    <p>Universiti Teknologi PETRONAS</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="profile-card">
            <div class="profile-header">
                <div class="profile-avatar">RK</div>
                <div class="profile-title">
                    <h3>Dr. Ramkumar Krishnan</h3>
                    <p>Professor</p>
                    <p>Periyar University</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Collaborators & Contributors
    st.markdown('<h2 class="collaborator-section">Collaborators & Contributors</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="profile-card">
            <div class="profile-header">
                <div class="profile-avatar">SG</div>
                <div class="profile-title">
                    <h3>Sugavanam</h3>
                    <p>ExLog, Kuwait</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="profile-card">
            <div class="profile-header">
                <div class="profile-avatar">MB</div>
                <div class="profile-title">
                    <h3>Manobalaji</h3>
                    <p>Curtin University</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="profile-card">
            <div class="profile-header">
                <div class="profile-avatar">JO</div>
                <div class="profile-title">
                    <h3>John Olutoki</h3>
                    <p>Universiti Teknologi PETRONAS</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Technologies section
    st.markdown("""
    <div class="card">
        <h2>Our Technology</h2>
        <p>StrataGraph leverages the power of Graph Neural Networks (GNNs) to model complex relationships in subsurface data. Our technology stack includes:</p>
        <ul>
            <li>PyTorch Geometric and StellarGraph for graph-based deep learning</li>
            <li>Graph Convolutional Networks (GCNs) for spatial relationship modeling</li>
            <li>Advanced data visualization techniques for intuitive interpretation</li>
            <li>Cloud-based processing for scalable analysis</li>
            <li>Physics-Informed Graph Neural Networks (PI-GNNs) for geomechanical simulations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact information
    st.markdown("""
    <div class="card">
        <h2>Contact Information</h2>
        <p><strong>Email:</strong> info@stratagraph.ai</p>
        <p><strong>Research Lab:</strong> Universiti Teknologi PETRONAS, 32610 Seri Iskandar, Perak, Malaysia</p>
    </div>
    """, unsafe_allow_html=True)

# The main function to run the app
def main():
    # Load CSS
    load_css()
    
    # Create sidebar
    sidebar_options = create_sidebar()
    
    # Render the appropriate page based on navigation
    page = sidebar_options["page"]
    
    if page == "Home":
        home_page()
    elif page == "VHydro Overview":
        vhydro_overview_page()
    elif page == "Data Preparation":
        data_preparation_page()
    elif page == "Petrophysical Properties":
        petrophysical_properties_page()
    elif page == "Facies Classification":
        facies_classification_page()
    elif page == "Hydrocarbon Potential Using GCN":
        hydrocarbon_prediction_page()
    elif page == "VHydro":
        vhydro_overview_page()
    elif page == "CO2 Storage Applications":
        co2_storage_page()
    elif page == "Help and Contact":
        help_contact_page()
    elif page == "About Us":
        about_us_page()
    else:
        home_page()  # Default to home page

if __name__ == "__main__":
    main()
