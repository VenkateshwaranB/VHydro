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
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #0e4194 0%, #153a6f 100%); 
    }
    
    /* Sidebar header/logo area */
    .sidebar-logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem 0;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sidebar-logo-container img {
        max-width: 80%;
        margin-bottom: 1rem;
    }
    
    .sidebar-logo-container h1 {
        color: white !important;
        font-size: 1.8rem;
        margin: 0;
        padding: 0;
    }
    
    .version-tag {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 8px 0;
    }
    
    /* Navigation menu styling */
    .nav-container {
        color: white;
        margin-top: 2rem;
    }
    
    .nav-section {
        margin-bottom: 1rem;
    }
    
    .section-title {
        color: white !important;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        padding-left: 1rem;
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
    
    /* Coming soon section */
    .coming-soon-section {
        background: linear-gradient(rgba(30, 41, 59, 0.8), rgba(30, 41, 59, 0.8)), url('https://placehold.co/600x400');
        background-size: cover;
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin: 30px 0;
        filter: blur(0px); /* Container not blurred */
    }
    
    .coming-soon-section h2 {
        color: white;
        margin-bottom: 15px;
    }
    
    .coming-soon-section .content {
        filter: blur(3px); /* Content inside is blurred */
        pointer-events: none;
    }
    
    .footer-text {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.8rem;
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    /* Streamlit element styling overrides */
    div[data-baseweb="select"] > div {
        background-color: white;
        color: #0e4194;
    }
    
    div[class*="stRadio"] label {
        color: white;
    }
    
    .stButton button {
        background-color: #0e4194;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #3a6fc4;
        color: white;
    }
    
    /* Custom radio buttons */
    .custom-radio {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .custom-radio:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    .custom-radio.selected {
        background-color: rgba(255, 255, 255, 0.3);
        border-left: 3px solid white;
    }
    
    /* Version section */
    .version-section {
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    
    .version-section h4 {
        color: white !important;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
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
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Simple login function using hardcoded credentials
@st.cache_data
def login(email, password):
    # Demo users
    users = {"user@example.com": "password", "admin@stratagraph.com": "admin"}
    if email in users and users[email] == password:
        st.session_state["email"] = email
        st.session_state["logged_in"] = True
        return True
    return False

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

# Create a simplified sidebar navigation system
def create_sidebar():
    # Logo and title section
    st.sidebar.markdown(
        """
        <div class="sidebar-logo-container">
            <img src="https://i.imgur.com/TNl1gM7.png" alt="StrataGraph Logo">
            <h1>StrataGraph</h1>
            <div class="version-tag">VHydro 1.0</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Initialize current page if not exists
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
    # Simple navigation using radio buttons instead of custom HTML/buttons
    st.sidebar.markdown('<div class="nav-container">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    
    # Main navigation as a simple radio
    main_pages = ["Home", "VHydro", "CO2 Storage Applications", "Help and Contact", "About Us"]
    selected_main = st.sidebar.radio("", main_pages, index=main_pages.index(st.session_state["current_page"]) 
                                     if st.session_state["current_page"] in main_pages else 0,
                                     label_visibility="collapsed")
    
    # If VHydro is selected, show sub-pages
    vhydro_selected = False
    if selected_main == "VHydro":
        vhydro_selected = True
        st.sidebar.markdown('<div style="margin-left: 1.5rem;">', unsafe_allow_html=True)
        vhydro_pages = ["VHydro Overview", "Data Preparation", "Petrophysical Properties", 
                      "Facies Classification", "Hydrocarbon Potential Using GCN"]
        
        # Find the index of the current page in vhydro_pages if it exists
        current_index = 0
        if st.session_state["current_page"] in vhydro_pages:
            current_index = vhydro_pages.index(st.session_state["current_page"])
        
        selected_vhydro = st.sidebar.radio(
            "", vhydro_pages, index=current_index, label_visibility="collapsed"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Update session state with selected VHydro page
        if selected_vhydro != st.session_state["current_page"]:
            st.session_state["current_page"] = selected_vhydro
            st.rerun()
    
    # Update session state with selected main page
    if not vhydro_selected and selected_main != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_main
        st.rerun()
    
    # Version selection section
    st.sidebar.markdown(
        """
        <div class="version-section">
            <h4>Versions</h4>
            <div class="version-item">
                <div class="version-indicator active-version"></div>
                VHydro 1.0 (Current)
            </div>
            <div class="version-item">
                <div class="version-indicator coming-version"></div>
                CO2 Storage 2.0 (Coming Soon)
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Only show model configuration in analysis pages
    if st.session_state["current_page"] == "Facies Classification":
        st.sidebar.markdown('<div class="section-title">Analysis Parameters</div>', unsafe_allow_html=True)
        min_clusters = st.sidebar.slider("Min Clusters", 2, 15, 5)
        max_clusters = st.sidebar.slider("Max Clusters", min_clusters, 15, 10)
    else:
        min_clusters = 5
        max_clusters = 10
    
    # Footer
    st.sidebar.markdown(
        """
        <div class="footer-text">
            © 2025 StrataGraph. All rights reserved.<br>
            Version 1.0.0
        </div>
        """, 
        unsafe_allow_html=True
    )
    
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
            <p>Advanced Graph Convolutional Networks for Geoscience Modeling</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>About StrataGraph</h2>
        <p>StrataGraph is a cutting-edge platform for geoscience modeling and analysis, combining advanced machine learning techniques with traditional geological and petrophysical analysis.</p>
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
    
    # Second section: CO2 Storage (Coming Soon) - Now directly below VHydro
    st.markdown("""
    <div class="coming-soon-section">
        <h2>StrataGraph 2.0 - CO2 Storage Potential Analysis</h2>
        <div class="content">
            <p>Advanced carbon capture utilization and storage (CCUS) modules powered by Graph Neural Networks.</p>
            <ul>
                <li>CO2 storage capacity prediction</li>
                <li>Caprock integrity analysis using geomechanical properties</li>
                <li>Built upon VHydro reservoir identification techniques</li>
                <li>Long-term storage monitoring</li>
            </ul>
        </div>
        <h3>Coming Soon</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # VHydro Workflow section
    st.markdown("<h2>VHydro Workflow</h2>", unsafe_allow_html=True)
    
    # Try to load the workflow image
    workflow_path = "src/Workflow.png"
    try:
        st.image(workflow_path, use_container_width=True)
    except:
        st.warning("Workflow image not found.")
    
    # Button to explore VHydro - simplified single button
    if st.button("Explore VHydro Analysis Tool", key="explore_vhydro_btn"):
        st.session_state["current_page"] = "VHydro Overview"
        st.rerun()

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
            st.warning("Workflow image not found.")
            
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
        
    # Advanced Configuration Options - moved from home page
    if st.checkbox("Show Advanced Configuration Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Architecture")
            st.selectbox("GCN Version", ["RegularizedGCN", "Standard GCN", "GCN with Skip Connections"])
            st.number_input("Hidden Channels", min_value=8, max_value=128, value=16, step=8)
            st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.5, step=0.1)
            st.checkbox("Use Batch Normalization", value=True)
        
        with col2:
            st.markdown("### Training Configuration")
            st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05], value=0.01)
            st.number_input("Early Stopping Patience", min_value=10, max_value=100, value=50, step=5)
            st.number_input("Maximum Epochs", min_value=50, max_value=500, value=200, step=50)
            st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
    
    # Button to start the analysis workflow
    st.markdown("<h3>Start VHydro Analysis</h3>", unsafe_allow_html=True)
    if st.button("Begin Data Preparation", key="begin_analysis_btn"):
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
            if st.button("Process Data"):
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
        if st.button("Go to Data Preparation"):
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
    if st.checkbox("Show Calculation Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            matrix_density = st.number_input("Matrix Density (g/cm³)", min_value=2.0, max_value=3.0, value=2.65, step=0.01)
            fluid_density = st.number_input("Fluid Density (g/cm³)", min_value=0.5, max_value=1.5, value=1.0, step=0.01)
        
        with col2:
            a_const = st.number_input("Archie Constant (a)", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
            m_const = st.number_input("Cementation Exponent (m)", min_value=1.5, max_value=2.5, value=2.0, step=0.1)
    
    # Run calculation button
    if st.button("Calculate Properties"):
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
        if st.button("Proceed to Facies Classification"):
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
        if st.button("Go to Petrophysical Properties"):
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
    if st.button("Run Facies Classification"):
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
        
        # Button to show visualization (lazy loading of heavy content)
        if st.checkbox("Show Facies Visualization"):
            # Simple HTML/CSS based visualization instead of matplotlib for better performance
            st.markdown(f"""
            <div style="width:100%; height:400px; background: linear-gradient(180deg, #0e4194 0%, #3a6fc4 100%); 
                        border-radius:10px; color:white; text-align:center; padding:20px;">
                <h4>Facies Visualization (Cluster {optimal_clusters})</h4>
                <p>Interactive visualization would appear here in the full version.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a simple facies dataset for download
        facies_df = pd.DataFrame({
            "DEPTH": np.arange(1000, 1100),
            "FACIES": np.random.randint(0, optimal_clusters, size=100)
        })
        
        # Show sample data
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
        if st.button("Proceed to Hydrocarbon Potential Prediction"):
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
        if st.button("Go to Facies Classification"):
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
    
    # Model parameters
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
    if st.checkbox("Show Advanced Model Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.5, step=0.1)
            epochs = st.number_input("Maximum Epochs", min_value=50, max_value=500, value=200, step=50)
        
        with col2:
            patience = st.number_input("Early Stopping Patience", min_value=5, max_value=100, value=20, step=5)
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
    
    # Run model button
    if st.button("Train GCN Model", key="train_gcn_btn"):
        with st.spinner("Training GCN model..."):
            # Simpler progress indicator with fewer updates
            progress_bar = st.progress(0)
            status_area = st.empty()
            
            for i in range(0, 101, 25):
                progress_bar.progress(i)
                
                # Update status with fewer messages
                if i == 0:
                    status_area.info("Preparing graph structure...")
                elif i == 25:
                    status_area.info("Creating features...")
                elif i == 50:
                    status_area.info("Training model...")
                elif i == 75:
                    status_area.info("Finalizing predictions...")
                
                time.sleep(0.1)
        
        # Clear status area and show success message
        status_area.empty()
        st.success("GCN model trained successfully!")
        
        # Instead of creating tabs that load everything at once,
        # use expanders to lazy load content
        with st.expander("Model Performance"):
            # Create basic metrics with less visualization
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Test Accuracy", "0.88")
            with col2: st.metric("F1 Score", "0.86")
            with col3: st.metric("AUC", "0.92")
            
            # Option to view detailed plots
            if st.checkbox("Show Learning Curves"):
                # Use simple HTML/CSS representation instead of matplotlib
                st.markdown("""
                <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                    <div style="width: 48%; background-color: #f1f5f9; padding: 15px; border-radius: 10px;">
                        <h4 style="text-align:center;">Training Loss</h4>
                        <div style="height: 150px; background: linear-gradient(to bottom right, #0e4194, #3a6fc4); border-radius: 5px;"></div>
                    </div>
                    <div style="width: 48%; background-color: #f1f5f9; padding: 15px; border-radius: 10px;">
                        <h4 style="text-align:center;">Model Accuracy</h4>
                        <div style="height: 150px; background: linear-gradient(to top right, #10b981, #3a6fc4); border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("Quality Predictions"):
            # Create a simple prediction table
            predictions = pd.DataFrame({
                "DEPTH": np.arange(1000, 1010),
                "PREDICTED_QUALITY": np.random.choice(
                    ["Very Low", "Low", "Moderate", "High", "Very High"], 
                    size=10
                )
            })
            
            st.dataframe(predictions)
            
            # Download button
            csv = predictions.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="quality_predictions.csv",
                mime="text/csv"
            )
            
            # Visualization toggle to lazy load heavy content
            if st.checkbox("Show Quality Visualization"):
                # A very simplified visualization
                st.markdown("""
                <div style="width:100%; height:300px; background: linear-gradient(90deg, #ff0000, #ffaa00, #00ff00, #0000ff); 
                            border-radius:10px; text-align:center; padding:20px; color:white;">
                    <h4>Hydrocarbon Quality Visualization</h4>
                    <p>From Very Low (left) to Very High (right)</p>
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("Classification Report"):
            # Simple classification report
            report_data = {
                "Class": ["Very Low", "Low", "Moderate", "High", "Very High", "Average"],
                "Precision": [0.85, 0.78, 0.92, 0.86, 0.91, 0.86],
                "Recall": [0.82, 0.75, 0.90, 0.89, 0.84, 0.84],
                "F1-Score": [0.83, 0.76, 0.91, 0.87, 0.87, 0.85]
            }
            
            # Create and display dataframe
            report_df = pd.DataFrame(report_data)
            st.table(report_df)
        
        # Store in session state
        st.session_state["model_history"] = True
        st.session_state["analysis_complete"] = True

def co2_storage_page():
    st.markdown("""
    <div class="colored-header">
        <h1>CO2 Storage Applications</h1>
        <p>Carbon Capture, Utilization, and Storage (CCUS) Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="coming-soon-section">
        <h2>CO2 Storage Potential Analysis</h2>
        <div class="content">
            <p>Advanced carbon capture utilization and storage (CCUS) modules powered by Graph Neural Networks.</p>
            <ul>
                <li>CO2 storage capacity prediction</li>
                <li>Reservoir integrity analysis</li>
                <li>Long-term storage monitoring</li>
            </ul>
        </div>
        <h3>Coming Soon</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature preview
    st.markdown("""
    <div class="card">
        <h2>Upcoming Features</h2>
        <ul>
            <li>CO2 storage capacity estimation based on reservoir properties</li>
            <li>Caprock integrity assessment using Graph Neural Networks</li>
            <li>Integration with existing carbon storage databases</li>
            <li>Long-term storage simulation and monitoring</li>
            <li>Risk assessment and mitigation strategies</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sign up for updates
    st.markdown("""
    <div class="card">
        <h2>Stay Updated</h2>
        <p>Sign up to receive updates when CO2 Storage Applications becomes available.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("signup_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name")
            email = st.text_input("Email")
        with col2:
            organization = st.text_input("Organization")
            interest = st.selectbox("Area of Interest", ["Carbon Storage", "Hydrocarbon Production", "Research", "Other"])
        
        st.form_submit_button("Notify Me")

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
    
    # Contact form
    st.markdown("""
    <div class="card">
        <h2>Contact Us</h2>
        <p>Have questions or need assistance? Reach out to our team.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("contact_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name")
            email = st.text_input("Email")
        with col2:
            subject = st.text_input("Subject")
            category = st.selectbox("Category", ["Technical Support", "Feature Request", "Billing", "Other"])
        
        message = st.text_area("Message", height=150)
        
        submitted = st.form_submit_button("Send Message")
        if submitted:
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
        <p>StrataGraph is committed to revolutionizing geoscience analysis through advanced machine learning techniques. Our mission is to provide geoscientists and engineers with powerful, intuitive tools that transform complex subsurface data into actionable insights.</p>
        
        <h2>Our Team</h2>
        <p>We are a diverse team of geoscientists, data scientists, and software engineers passionate about applying cutting-edge technology to solve complex subsurface challenges.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Team section
    st.markdown("<h2>Leadership Team</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h3>Dr. Sarah Chen</h3>
            <p>Founder & CEO</p>
            <p>Ph.D. in Reservoir Engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h3>Dr. Michael Rodriguez</h3>
            <p>CTO</p>
            <p>Ph.D. in Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h3>Dr. Emily Patel</h3>
            <p>Chief Scientist</p>
            <p>Ph.D. in Geophysics</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technologies section
    st.markdown("""
    <div class="card">
        <h2>Our Technology</h2>
        <p>StrataGraph leverages the power of Graph Neural Networks (GNNs) to model complex relationships in subsurface data. Our technology stack includes:</p>
        <ul>
            <li>PyTorch and TensorFlow for deep learning</li>
            <li>Graph Convolutional Networks (GCNs) for spatial relationship modeling</li>
            <li>Advanced data visualization techniques for intuitive interpretation</li>
            <li>Cloud-based processing for scalable analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Partners and funding
    st.markdown("""
    <div class="card">
        <h2>Partners & Collaborators</h2>
        <p>We collaborate with leading academic institutions and industry partners to continuously improve our technology and deliver innovative solutions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact information
    st.markdown("""
    <div class="card">
        <h2>Contact Information</h2>
        <p><strong>Email:</strong> info@stratagraph.ai</p>
        <p><strong>Address:</strong> 123 Innovation Way, Houston, TX 77002</p>
        <p><strong>Phone:</strong> +1 (713) 555-0123</p>
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
    elif page == "VHydro":
        vhydro_page()
    elif page == "Data Preparation":
        data_preparation_page()
    elif page == "Petrophysical Properties":
        petrophysical_properties_page()
    elif page == "Facies Classification":
        facies_classification_page()
    elif page == "CO2 Storage Applications":
        co2_storage_page()
    elif page == "Hydrocarbon Potential Using GCN":
        hydrocarbon_prediction_page()
    elif page == "Help and Contact":
        help_contact_page()
    elif page == "About Us":
        about_us_page()
    else:
        home_page()  # Default to home page

if __name__ == "__main__":
    main()
