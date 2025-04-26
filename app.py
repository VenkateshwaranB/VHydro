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

# Create a sidebar navigation system
def create_sidebar():
    # Function to load and encode image
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as img_file:
                b64_data = base64.b64encode(img_file.read()).decode()
            return f"data:image/png;base64,{b64_data}"
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    # Path to your image
    image_base64 = get_base64_image("src/StrataGraph_White_Logo.png")
    
    # Inject image via HTML in the sidebar
    if image_base64:
        st.sidebar.markdown(
            f"""
            <div style="text-align: center;">
                <img src="{image_base64}" alt="StrataGraph Logo" style="width: 80%; max-width: 150px; margin-bottom: 10px;"/>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: white; font-size: 24px; margin-bottom: 5px;">StrataGraph</h1>
            <p style="color: rgba(255, 255, 255, 0.7); font-size: 14px;">VHydro 1.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize current page if not exists
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
    # Main navigation
    main_pages = ["Home", "VHydro", "CO2 Storage Applications", "Help and Contact", "About Us"]
    vhydro_pages = ["VHydro Overview", "Data Preparation", "Petrophysical Properties", 
                   "Facies Classification", "Hydrocarbon Potential Using GCN"]
    
    # Determine if we're in VHydro or one of its subpages
    in_vhydro = st.session_state["current_page"] == "VHydro" or st.session_state["current_page"] in vhydro_pages
    
    # Navigation header
    st.sidebar.markdown('<div style="color: white; margin-bottom: 10px;">Navigation</div>', unsafe_allow_html=True)
    
    # Display main navigation items
    for page in main_pages:
        # Add "Coming Soon" tag for CO2 Storage
        coming_soon = ""
        if page == "CO2 Storage Applications":
            coming_soon = '<span style="background-color: #FF9800; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 8px;">Coming Soon</span>'
        
        # Create button with the same styling as your image
        if st.sidebar.button(f"{page} {coming_soon}", key=f"nav_{page}", use_container_width=True, help=page):
            st.session_state["current_page"] = page
            st.rerun()
    
    # Show VHydro subpages when in VHydro section
    if in_vhydro:
        # Add some space
        st.sidebar.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Display VHydro subpages with indentation
        for subpage in vhydro_pages:
            if st.sidebar.button(f"  • {subpage}", key=f"subnav_{subpage}", use_container_width=True, help=subpage):
                st.session_state["current_page"] = subpage
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
        <div style="color: rgba(255, 255, 255, 0.7); font-size: 12px; text-align: center; position: absolute; bottom: 20px; left: 0; right: 0; padding: 0 20px;">
            © 2025 StrataGraph. All rights reserved.<br>
            Version 1.0.0
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Analysis parameters (only shown on certain pages)
    min_clusters = 5
    max_clusters = 10
    
    if st.session_state["current_page"] == "Facies Classification":
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        st.sidebar.markdown('<div style="color: white;">Analysis Parameters</div>', unsafe_allow_html=True)
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

    # Second section: CO2 Storage (Coming Soon) - directly below VHydro
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
    
    # Blurred coming soon section
    st.markdown("""
    <div class="coming-soon-section">
        <h2>CO2 Storage Potential Analysis</h2>
        <div class="content">
            <p>Advanced carbon capture utilization and storage (CCUS) modules powered by Graph Neural Networks.</p>
            <ul>
                <li>CO2 storage capacity prediction</li>
                <li>Caprock integrity analysis using geomechanical properties</li>
                <li>Built upon VHydro reservoir identification techniques</li>
                <li>Long-term storage monitoring</li>
            </ul>
        </div>
        <h3>Coming Soon in StrataGraph 2.0</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature preview
    st.markdown("""
    <div class="card">
        <h2>Upcoming Features</h2>
        <p>StrataGraph 2.0 will build upon the graph-based reservoir characterization from VHydro 1.0 to assess CO2 storage potential:</p>
        <ul>
            <li>CO2 storage capacity estimation based on reservoir properties</li>
            <li>Caprock integrity assessment using Graph Neural Networks</li>
            <li>Integration with existing carbon storage databases</li>
            <li>Long-term storage simulation and monitoring</li>
            <li>Risk assessment and mitigation strategies</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Simplified sign up form
    st.markdown("""
    <div class="card">
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
        st.selectbox("Area of Interest", ["Carbon Storage", "Hydrocarbon Production", "Research", "Other"])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("Notify Me", use_container_width=True)

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
        <p>StrataGraph is committed to revolutionizing geoscience analysis through advanced machine learning techniques. Our mission is to provide geoscientists and engineers with powerful, intuitive tools that transform complex subsurface data into actionable insights.</p>
        
        <h2>Our Vision</h2>
        <p>We envision a future where geological and petrophysical data analysis is enhanced by the power of graph-based deep learning, enabling more accurate predictions for both hydrocarbon exploration and carbon storage applications. By bridging traditional geoscience with cutting-edge AI, we aim to contribute to both energy security and climate solutions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Research and publications
    st.markdown("""
    <div class="card">
        <h2>Research and Publications</h2>
        <p>Our technology is built on peer-reviewed research:</p>
        <ul>
            <li><a href="https://link.springer.com/article/10.1007/s11053-024-10311-x" target="_blank">Hydrocarbon Potential Prediction Using Novel Graph Dataset</a> - This paper introduces our approach to hydrocarbon potential prediction using Graph Convolutional Networks.</li>
            <li>Forthcoming research on CO2 storage potential assessment using graph-based approaches.</li>
        </ul>
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
            <li>PyTorch Geometric and StellarGraph for graph-based deep learning</li>
            <li>Graph Convolutional Networks (GCNs) for spatial relationship modeling</li>
            <li>Advanced data visualization techniques for intuitive interpretation</li>
            <li>Cloud-based processing for scalable analysis</li>
        </ul>
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
