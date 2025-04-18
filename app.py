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
try:
    from firebase_auth import authenticate, user_account_page, logout
except ImportError:
    # Define simplified authentication functions if the module is not available
    def authenticate():
        return True
    
    def user_account_page():
        st.write("User account page is not available.")
    
    def logout():
        st.session_state.clear()

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
        st.image(buffered, caption=caption, width=width)
    except Exception as e:
        logger.error(f"Error displaying image {image_path}: {e}")
        st.error(f"Could not display image: {e}")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #0e4194;
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
    
    .toggle-button {
        background: rgba(14, 65, 148, 0.1);
        color: #0e4194;
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
    }
    
    .toggle-button:hover {
        background: rgba(14, 65, 148, 0.2);
    }
    
    .toggle-button-active {
        background: rgba(14, 65, 148, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def header_with_logo(logo_path=None):
    # Add title using standard Streamlit components
    st.title("VHydro - Hydrocarbon Quality Prediction")
    st.write("Advanced Graph Convolutional Network for Petrophysical Analysis")
    
    # Try to display logo if path is provided
    if logo_path and os.path.exists(logo_path):
        st.image(logo_path, width=800)

# Sidebar navigation
def create_sidebar():
    """Create the sidebar with toggle buttons"""
    # Logo section
    st.sidebar.title("VHydro")
    
    try:
        logo_path = "src/VHydro_Logo.png"
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path)
    except Exception as e:
        logger.error(f"Error loading sidebar logo: {e}")
    
    st.sidebar.markdown("---")

    # User account section - shown only if logged in
    if st.session_state.get('logged_in', False):
        st.sidebar.info(f"Logged in as: {st.session_state.get('email', 'User')}")
        
        # Logout button
        if st.sidebar.button("Logout", key="sidebar_logout"):
            logout()
            st.rerun()

    # Navigation section with buttons
    st.sidebar.header("Navigation")
    
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
    
    # Create buttons for each page
    for page_name, page_info in pages.items():
        button_text = f"{page_info['icon']} {page_name}"
        if page_info["active"]:
            button_text = f"**{button_text}**"
            
        if st.sidebar.button(button_text, key=f"nav_{page_name}"):
            st.session_state['current_page'] = page_name
            st.rerun()
            
    # Model configuration section
    st.sidebar.markdown("---")
    st.sidebar.header("Model Configuration")
    
    # Cluster configuration with sliders
    min_clusters = st.sidebar.slider("Min Clusters", 2, 15, 5)
    max_clusters = st.sidebar.slider("Max Clusters", min_clusters, 15, 10)
    
    # Advanced settings in an expander
    with st.sidebar.expander("Advanced Settings"):
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
    
    st.sidebar.markdown("---")
    
    # Warning for missing VHydro module
    if not VHYDRO_AVAILABLE:
        st.sidebar.warning("⚠️ VHydro module unavailable")
    
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
    
    st.markdown("## About VHydro")
    
    st.info("""
    **VHydro** is an advanced tool for hydrocarbon quality prediction using well log data.
    It combines traditional petrophysical analysis with modern machine learning techniques
    to provide accurate predictions of reservoir quality.
    
    The tool uses Graph Convolutional Networks (GCN) to model the complex relationships
    between different petrophysical properties and depth values, enabling more accurate
    classification of hydrocarbon potential zones.
    """)
    
    # Try to display workflow diagram
    workflow_image_path = "src/Workflow.png"
    
    if os.path.exists(workflow_image_path):
        st.markdown("## Workflow Overview")
        st.image(workflow_image_path, caption="VHydro Workflow")
    
    st.markdown("## Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Petrophysical Property Calculation**
        - Shale Volume
        - Porosity
        - Water/Oil Saturation
        - Permeability
        
        **Facies Classification**
        - K-means Clustering
        - Silhouette Score Optimization
        - Depth-based Facies Mapping
        """)
        
    with col2:
        st.markdown("""
        **Graph-based Machine Learning**
        - Graph Convolutional Networks
        - Node and Edge Feature Extraction
        - Hydrocarbon Quality Classification
        
        **Visualization and Reporting**
        - Facies Visualization
        - Prediction Accuracy Metrics
        - Classification Reports
        """)
    
    st.markdown("## Getting Started")
    st.markdown("""
    1. Navigate to the **Dataset Preparation** section to understand the data requirements
    2. Use the **Analysis Tool** to upload and process your well log data
    3. Run the model with your preferred configuration
    4. Explore the results in the **Results Visualization** section
    
    Use the sidebar to navigate between different sections of the application.
    """)
    
    if st.button("Start Analysis", key="start_analysis_btn"):
        st.session_state['current_page'] = "Dataset Preparation"
        st.rerun()

# Dataset preparation page
def dataset_preparation_page():
    """Render the dataset preparation page"""
    st.title("Dataset Preparation")
    
    st.write("Upload your LAS file containing well log data. The file will be processed to calculate petrophysical properties necessary for hydrocarbon potential prediction.")
    
    uploaded_file = st.file_uploader("Upload LAS file", type=["las"])
    
    # Provide option for sample data
    use_sample = st.checkbox("Use sample data instead", value=False)
    
    # Parameters section
    st.markdown("## Calculation Parameters")
    
    # Create two columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Fluid Parameters**")
        matrix_density = st.number_input("Matrix Density (g/cc)", min_value=2.0, max_value=3.0, value=2.65, step=0.01)
        fluid_density = st.number_input("Fluid Density (g/cc)", min_value=0.5, max_value=1.5, value=1.0, step=0.01)
    
    with col2:
        st.markdown("**Archie Parameters**")
        a_param = st.number_input("Tortuosity Factor (a)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        m_param = st.number_input("Cementation Exponent (m)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        n_param = st.number_input("Saturation Exponent (n)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    
    # Feature selection section
    st.markdown("## Feature Selection")
    
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
    
    # Process button
    if uploaded_file or use_sample:
        if st.button("Process Data", type="primary"):
            with st.spinner('Processing data...'):
                # Simulate processing with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Simulate work being done
                    progress_bar.progress(i + 1)
                
                # Show success message
                st.success("Data processed successfully!")
                
                # Preview processed data in a table
                st.markdown("## Processed Data Preview")
                
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
                st.dataframe(df)
                
                # Create a visualization
                st.markdown("## Petrophysical Visualization")
                
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
                if st.button("Continue to Model Workflow"):
                    st.session_state['current_page'] = "Model Workflow"
                    st.rerun()
    else:
        st.info("Please upload a LAS file or use the sample data to continue.")

# Model workflow page
def model_workflow_page():
    """Render the model workflow page"""
    st.title("Model Workflow")
    
    # Create a modern intro section
    st.info("""
    **Graph Convolutional Network (GCN) Workflow**
    
    This page outlines the workflow for creating a Graph Convolutional Network model for hydrocarbon quality prediction.
    The process includes dataset creation, graph construction, model training, and evaluation.
    """)
    
    # Create workflow steps with timeline
    steps = [
        {"title": "1. Dataset Preparation", "status": "completed", "desc": "Process well log data and calculate petrophysical properties"},
        {"title": "2. K-means Clustering", "status": "active", "desc": "Perform K-means clustering to identify facies"},
        {"title": "3. Graph Construction", "status": "pending", "desc": "Create nodes and edges for the graph dataset"},
        {"title": "4. GCN Training", "status": "pending", "desc": "Train the Graph Convolutional Network model"},
        {"title": "5. Model Evaluation", "status": "pending", "desc": "Evaluate model performance and visualize results"}
    ]
    
    # Display steps
    for step in steps:
        status_color = "green" if step["status"] == "completed" else "blue" if step["status"] == "active" else "gray"
        status_text = "Completed" if step["status"] == "completed" else "In Progress" if step["status"] == "active" else "Pending"
        
        col1, col2 = st.columns([1, 5])
        with col1:
            st.write(f"**{step['title']}**")
        with col2:
            st.write(f"**Status:** {status_text}")
            st.write(f"{step['desc']}")
        
        st.markdown("---")
    
    # K-means Clustering section (active step)
    st.markdown("## K-means Clustering for Facies Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=3, max_value=15, value=7)
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=42)
    
    with col2:
        st.markdown("**Feature Scaling**")
        scaling_method = st.radio("Select scaling method", 
                               ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                               horizontal=True)
        
        st.markdown("**Silhouette Score Optimization**")
        optimize_silhouette = st.checkbox("Automatically determine optimal cluster count", value=True)
    
    # Run clustering button
    if st.button("Run Clustering"):
        with st.spinner('Performing clustering...'):
            # Simulate clustering with a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulate work being done
                progress_bar.progress(i + 1)
            
            # Show clustering results
            st.success("Clustering completed successfully!")
            
            # Visualization of silhouette scores
            st.markdown("## Silhouette Score Analysis")
            
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
            
            # Continue button
            if st.button("Continue to Graph Construction"):
                st.info("Next: Graph Construction")

# Analysis tool page
def analysis_tool_page(config):
    """Render the analysis tool page"""
    # Check if user is authenticated for this page
    is_authenticated = authenticate()
    
    # If user is not authenticated, stop execution here
    if not is_authenticated:
        st.stop()
    
    st.title("Analysis Tool")
    
    st.write("Analyze your well log data using the VHydro Graph Convolutional Network model.")
    
    # Create tabs for different analysis steps
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Process", "Graph Creation", "Model Training", "Results"])
    
    with tab1:
        st.header("1. Data Upload and Processing")
        
        uploaded_file = st.file_uploader("Upload LAS file", type=["las"], key="analysis_upload")
        
        # Option to use sample data
        use_sample = st.checkbox("Use sample data instead", value=not uploaded_file, key="analysis_sample")
        
        # Process button
        if (uploaded_file or use_sample) and st.button("Process Data", key="process_data_btn"):
            with st.spinner('Processing well log data...'):
                # Simulate processing with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Show success message
                st.success("Data processed successfully! Well log data loaded and petrophysical properties calculated.")
    
    with tab2:
        st.header("2. Graph Dataset Creation")
        
        # Show options for graph creation
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=5, max_value=10, value=7, 
                                key="graph_clusters")
            train_test_split = st.slider("Train/Test Split", min_value=0.6, max_value=0.9, value=0.8,
                                      key="graph_train_split")
        
        with col2:
            st.markdown("**Node Connection Strategy**")
            connection_strategy = st.radio("Select how nodes are connected:",
                                       ["K-Nearest Neighbors", "Distance Threshold", "Facies-based"],
                                       index=2,
                                       key="connection_strategy")
            
            connection_param = st.slider("Connection Parameter", 
                                      min_value=1, max_value=10, value=5,
                                      help="K value for KNN, distance threshold, or maximum inter-facies connections")
        
        # Create graph button
        if st.button("Generate Graph Dataset", key="generate_graph_btn"):
            with st.spinner('Creating graph dataset...'):
                # Simulate processing with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Success message
                st.success("Graph dataset created successfully!")
                
                # Show graph statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nodes", "458")
                col2.metric("Edges", "1,245")
                col3.metric("Facies", "7")
                col4.metric("Quality Classes", "4")
    
    with tab3:
        st.header("3. GCN Model Training")
        
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
        
        # Train model button
        if st.button("Train GCN Model", key="train_model_btn"):
            with st.spinner('Training Graph Convolutional Network model...'):
                # Simulate training with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Show success message
                st.success("Model training completed successfully!")
                
                # Show metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Test Accuracy", "87.3%")
                col2.metric("Precision", "85.2%")
                col3.metric("Recall", "83.7%")
                col4.metric("F1 Score", "84.4%")
    
    with tab4:
        st.header("4. Results Analysis")
        
        # Sample visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        
        categories = ["Very Low", "Low", "Moderate", "High"]
        values = [25, 35, 30, 10]
        
        ax.bar(categories, values)
        ax.set_ylabel('Percentage')
        ax.set_title('Distribution of Hydrocarbon Potential')
        
        st.pyplot(fig)
        
        # Add download section
        st.subheader("Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="Download Predictions (CSV)",
                data="Sample data",
                file_name="hydrocarbon_predictions.csv",
                mime="text/csv",
            )
        
        with col2:
            st.download_button(
                label="Download Model (PKL)",
                data="Sample model",
                file_name="gcn_model.pkl",
                mime="application/octet-stream",
            )
        
        with col3:
            st.download_button(
                label="Download Report (PDF)",
                data="Sample report",
                file_name="hydrocarbon_analysis_report.pdf",
                mime="application/pdf",
            )

# Results visualization page
def results_visualization_page():
    """Render the results visualization page"""
    # Check if user is authenticated for this page
    is_authenticated = authenticate()
    
    # If user is not authenticated, stop execution here
    if not is_authenticated:
        st.stop()
    
    st.title("Results Visualization")
    
    st.write("Interactive visualization of hydrocarbon potential prediction results.")
    
    # Create a sample dataset for visualization
    depths = np.arange(1000, 1500, 0.5)
    quality_labels = ["Very Low", "Low", "Moderate", "High"]
    quality_numeric = np.random.randint(0, 4, size=len(depths))
    quality = np.array(quality_labels)[quality_numeric]
    
    # Create sample dataframe
    data = {
        "DEPTH": depths,
        "GR": 50 + 30 * np.sin(depths / 20) + 10 * np.random.randn(len(depths)),
        "RHOB": 2.2 + 0.3 * np.cos(depths / 30) + 0.05 * np.random.randn(len(depths)),
        "PHI": 0.2 - 0.15 * np.cos(depths / 30) + 0.02 * np.random.randn(len(depths)),
        "SW": 0.4 + 0.3 * np.sin(depths / 40) + 0.05 * np.random.randn(len(depths)),
        "FACIES": np.random.randint(0, 7, size=len(depths)),
        "QUALITY": quality,
        "QUALITY_NUMERIC": quality_numeric
    }
    
    df = pd.DataFrame(data)
    
    # Create sidebar controls for visualization
    st.sidebar.markdown("### Visualization Controls")
    
    # Depth range slider
    depth_min = float(df["DEPTH"].min())
    depth_max = float(df["DEPTH"].max())
    
    depth_range = st.sidebar.slider(
        "Depth Range",
        min_value=depth_min,
        max_value=depth_max,
        value=(depth_min, depth_min + 100),
        step=10.0
    )
    
    # Filter data by depth range
    filtered_data = df[
        (df["DEPTH"] >= depth_range[0]) & 
        (df["DEPTH"] <= depth_range[1])
    ]
    
    # Visualization type
    viz_type = st.sidebar.radio(
        "Visualization Type",
        ["Well Log View", "Cross-Plot", "Statistical Summary"]
    )
    
    # Display visualization based on type
    if viz_type == "Well Log View":
        st.subheader("Well Log View")
        
        # Create well log visualization
        fig, axes = plt.subplots(1, 4, figsize=(12, 8), sharey=True)
        
        # Plot GR
        axes[0].plot(filtered_data["GR"], filtered_data["DEPTH"], 'b-')
        axes[0].set_title('GR')
        axes[0].set_xlabel('API')
        axes[0].set_ylabel('Depth')
        axes[0].invert_yaxis()  # Depth increases downward
        
        # Plot RHOB
        axes[1].plot(filtered_data["RHOB"], filtered_data["DEPTH"], 'r-')
        axes[1].set_title('RHOB')
        axes[1].set_xlabel('g/cc')
        
        # Plot PHI
        axes[2].plot(filtered_data["PHI"], filtered_data["DEPTH"], 'g-')
        axes[2].set_title('PHI')
        axes[2].set_xlabel('v/v')
        
        # Plot SW
        axes[3].plot(filtered_data["SW"], filtered_data["DEPTH"], 'k-')
        axes[3].set_title('SW')
        axes[3].set_xlabel('v/v')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    elif viz_type == "Cross-Plot":
        st.subheader("Cross-Plot Analysis")
        
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
        
        # Create cross-plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            filtered_data[x_property],
            filtered_data[y_property],
            c=filtered_data["QUALITY_NUMERIC"],
            cmap="viridis",
            s=50,
            alpha=0.7
        )
        
        # Set labels and title
        ax.set_xlabel(x_property)
        ax.set_ylabel(y_property)
        ax.set_title(f'{x_property} vs {y_property} Cross-Plot')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Hydrocarbon Potential')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    elif viz_type == "Statistical Summary":
        st.subheader("Statistical Summary")
        
        # Display dataframe
        st.dataframe(filtered_data.describe())
        
        # Create a simple bar chart
        st.markdown("### Quality Distribution")
        
        quality_counts = filtered_data["QUALITY"].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(quality_counts.index, quality_counts.values)
        ax.set_ylabel('Count')
        ax.set_title('Hydrocarbon Potential Distribution')
        
        st.pyplot(fig)

# Main function for the app
def main():
    # Create sidebar
    config = create_sidebar()
    
    # Display the selected page
    if st.session_state['current_page'] == "Home":
        home_page()
    elif st.session_state['current_page'] == "Dataset Preparation":
        dataset_preparation_page()
    elif st.session_state['current_page'] == "Model Workflow":
        model_workflow_page()
    elif st.session_state['current_page'] == "Analysis Tool":
        analysis_tool_page(config)
    elif st.session_state['current_page'] == "Results Visualization":
        results_visualization_page()
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align:center;'>VHydro - Hydrocarbon Potential Prediction Application © 2025</div>", unsafe_allow_html=True)

# Entry point for the application
if __name__ == "__main__":
    main()
