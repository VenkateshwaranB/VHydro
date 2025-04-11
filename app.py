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

# Safely try to import VHydro - handle missing dependencies gracefully
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
</style>
""", unsafe_allow_html=True)

# Function to create layout with logo and title
def header_with_logo(logo_path):
    cols = st.columns([1, 2, 1])
    with cols[1]:
        try:
            if os.path.exists(logo_path):
                logo_image = Image.open(logo_path)
                st.image(logo_image, width='full')
            else:
                st.warning("Logo image not found. Expected at: " + logo_path)
        except Exception as e:
            logger.error(f"Error loading logo: {e}")
            st.warning(f"Error loading logo: {e}")
    
    st.markdown("<h1 class='main-header'>VHydro - Hydrocarbon Quality Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Advanced Graph Convolutional Network for Petrophysical Analysis</p>", unsafe_allow_html=True)

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

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Section",
        ["Home", "Dataset Preparation", "Model Workflow", "Analysis Tool", "Results Visualization"]
    )
    
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

# Home page
def home_page():
    logo_path = "src/Building a Greener World.png"
    header_with_logo(logo_path)
    
    st.markdown("<h2 class='sub-header'>About VHydro</h2>", unsafe_allow_html=True)
    
    with st.expander("What is VHydro?", expanded=True):
        st.markdown("""
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

# Dataset preparation page
def dataset_preparation_page():
    st.markdown("<h2 class='sub-header'>Dataset Preparation</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Before running the VHydro model, well log data must be properly prepared and organized.
    The model requires specific petrophysical properties that are calculated from standard
    well log measurements.
    """)
    
    # Try to display dataset preparation workflow
    dataset_workflow_path = "src/Graph Data Preparation.png"
    
    if os.path.exists(dataset_workflow_path):
        display_image_with_caption(dataset_workflow_path, "Dataset Preparation Workflow")
    else:
        st.warning(f"Dataset preparation workflow image not found. Expected at: {dataset_workflow_path}")
    
    st.markdown("<h3 class='section-header'>Required Log Data</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Basic Logs:**
        - Gamma Ray (GR)
        - Resistivity (LLD)
        - Density (RHOB)
        - Neutron Porosity (NPHI)
        """)
    
    with col2:
        st.markdown("""
        **Calculated Properties:**
        - Shale Volume (VSHALE)
        - Effective Porosity (PHI)
        - Water Saturation (WSAT)
        - Oil Saturation (OSAT)
        """)
    
    with col3:
        st.markdown("""
        **Quality Indicators:**
        - Permeability (PERM)
        - Hydrocarbon Quality
        - Facies Classification
        """)
    
    st.markdown("<h3 class='section-header'>Data Format</h3>", unsafe_allow_html=True)
    st.markdown("""
    VHydro accepts well log data in LAS (Log ASCII Standard) format, which is the industry standard
    for storing and exchanging well log data. The LAS file should contain depth information and
    relevant log curves.
    
    **Example LAS File Structure:**
    ```
    ~VERSION INFORMATION
    VERS.   2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
    WRAP.   NO  : ONE LINE PER DEPTH STEP
    
    ~WELL INFORMATION
    STRT.M   1670.0 : START DEPTH
    STOP.M   1690.0 : STOP DEPTH
    STEP.M      0.1 : STEP
    NULL.     -999.25 : NULL VALUE
    
    ~CURVE INFORMATION
    DEPT.M     : Depth
    GR.GAPI    : Gamma Ray
    LLD.OHMM   : Deep Resistivity
    RHOB.G/C3  : Bulk Density
    NPHI.V/V   : Neutron Porosity
    
    ~PARAMETER INFORMATION
    
    ~A  DEPT     GR      LLD     RHOB    NPHI
    1670.0  75.0    10.0    2.65    0.12
    1670.1  76.2    11.2    2.64    0.13
    ...
    ```
    """)
    
    st.markdown("<h3 class='section-header'>Petrophysical Calculations</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    VHydro performs the following petrophysical calculations automatically:
    
    1. **Shale Volume (VSHALE)** - Calculated using the gamma ray log:
       - $V_{sh} = 0.083 \times (2 ^ {2 \times 3.7 \times GR_{normalized}} - 1)$
       - Where $GR_{normalized} = \frac{GR - GR_{min}}{GR_{max} - GR_{min}}$
    
    2. **Density Porosity (PHI)** - Calculated from the density log:
       - $\phi_D = \frac{\rho_{matrix} - \rho_{log}}{\rho_{matrix} - \rho_{fluid}}$
       - Typical values: $\rho_{matrix} = 2.65 g/cm^3$, $\rho_{fluid} = 1.0 g/cm^3$
    
    3. **Water Saturation (WSAT)** - Calculated using Archie's equation:
       - $S_w = \left(\frac{a \times R_w}{\phi^m \times R_t}\right)^{1/n}$
       - Where $a = 1$, $m = 2$, $n = 2$ are typical Archie parameters
    
    4. **Oil Saturation (OSAT)**:
       - $S_o = 1 - S_w$
    
    5. **Permeability (PERM)**:
       - $K = 0.00004 \times e^{57.117 \times \phi_{effective}}$
    """)
    
    with st.expander("Data Preprocessing Steps"):
        st.markdown("""
        1. **Data Cleaning**
           - Handling missing values
           - Removing outliers
           - Normalizing curves
        
        2. **Feature Selection**
           - Identifying relevant logs for clustering
           - Creating derived features
        
        3. **Depth Alignment**
           - Ensuring consistent depth steps
           - Handling depth shifts
        
        4. **Data Scaling**
           - Standardizing features before clustering
           - Normalizing values between 0 and 1 for specific algorithms
        """)

# Model workflow page
def model_workflow_page():
    st.markdown("<h2 class='sub-header'>Model Workflow</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    VHydro combines traditional petrophysical analysis with advanced Graph Convolutional Networks (GCN)
    to classify hydrocarbon quality zones. The workflow involves several key steps from facies classification
    to graph-based machine learning.
    """)
    
    # Try to display model workflow
    model_workflow_path = "src/Model.png"
    
    if os.path.exists(model_workflow_path):
        display_image_with_caption(model_workflow_path, "VHydro Model Workflow")
    else:
        st.warning(f"Model workflow image not found. Expected at: {model_workflow_path}")
    
    st.markdown("<h3 class='section-header'>K-means Clustering for Facies Classification</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    The first step is to classify the well log data into different facies using K-means clustering:
    
    1. **Feature Selection**: Relevant logs (GR, RHOB, NPHI, etc.) are selected for clustering
    2. **Feature Scaling**: Features are standardized to have mean=0 and std=1
    3. **Optimal Cluster Determination**: 
       - Multiple cluster counts (5-10) are tested
       - Silhouette scores are calculated to find the optimal number of clusters
    4. **Facies Classification**: Each depth point is assigned to a facies class
    """)
    
    st.markdown("<h3 class='section-header'>Graph Construction</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Node Types:**
        - Depth nodes: Represent specific depth points
        - PET nodes: Represent Petrophysical Entities
        
        **Edge Types:**
        - Depth-to-depth: Connect depths in the same cluster
        - Depth-to-PET: Connect depths to petrophysical properties
        """)
    
    with col2:
        st.markdown("""
        **Petrophysical Entities (PET):**
        - Permeability (5 classes)
        - Porosity (3 classes)
        - Volume Shale (2 classes)
        - Water Saturation (2 classes)
        - Oil Saturation (2 classes)
        """)
    
    st.markdown("""
    The graph structure enables the model to capture complex relationships between
    depths and petrophysical properties, as well as relationships between adjacent depths
    within the same facies.
    """)
    
    st.markdown("<h3 class='section-header'>Graph Convolutional Network (GCN)</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    VHydro uses a Graph Convolutional Network to classify hydrocarbon quality:
    
    1. **Model Architecture**:
       - Two graph convolutional layers with 16 hidden units each
       - ReLU activation functions
       - Dropout (0.5) for regularization
       - Final dense layer with softmax activation
    
    2. **Training Process**:
       - 80% of data for training, 10% for validation, 10% for testing
       - Early stopping to prevent overfitting
       - Adam optimizer with learning rate scheduling
       - Cross-entropy loss function
    
    3. **Output Classes**:
       - Very Low: Poor hydrocarbon quality
       - Low: Below average hydrocarbon quality
       - Moderate: Average hydrocarbon quality
       - High: Good hydrocarbon quality
       - Very High: Excellent hydrocarbon quality
    """)
    
    with st.expander("Technical Details: Graph Convolutional Layers"):
        st.markdown(r"""
        The graph convolutional layer performs the following operation:
        
        $$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$
        
        Where:
        - $H^{(l)}$ is the feature matrix at layer $l$
        - $\tilde{A} = A + I$ is the adjacency matrix with self-connections
        - $\tilde{D}$ is the degree matrix of $\tilde{A}$
        - $W^{(l)}$ is the weight matrix for layer $l$
        - $\sigma$ is the activation function (ReLU)
        
        This operation allows the model to learn from both node features and graph structure.
        """)

# Analysis tool page
def analysis_tool_page(config):
    st.markdown("<h2 class='sub-header'>Analysis Tool</h2>", unsafe_allow_html=True)
    
    if not VHYDRO_AVAILABLE:
        st.warning("""
        ⚠️ The VHydro module is not available in this deployment. The analysis tool requires the full 
        VHydro package with all dependencies (PyTorch, TensorFlow, StellarGraph, etc.).
        
        You can still upload your LAS file to preview the data, but model training and prediction 
        features are disabled.
        """)
    
    st.markdown("""
    Upload your LAS file to analyze well log data, perform petrophysical calculations,
    and run the VHydro model to predict hydrocarbon quality zones.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload LAS File", type=["las"])
    
    if uploaded_file is not None:
        # Create temporary directory for outputs
        temp_dir = create_temp_dir()
        
        # Save uploaded file
        las_file_path = os.path.join(temp_dir, "well_data.las")
        with open(las_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        
        # Initialize tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Petrophysical Properties", "Clustering", "Model Training"])
        
        try:
            if VHYDRO_AVAILABLE:
                # Create VHydro instance
                vh = VHydro(las_file_path, temp_dir)
                
                # Load LAS file
                load_success = vh.load_las_file()
                
                if not load_success:
                    st.error("Failed to load LAS file. Please check the file format.")
                    return
                
                # Preview data
                with tab1:
                    st.markdown("<h3 class='section-header'>Well Log Data Preview</h3>", unsafe_allow_html=True)
                    
                    # Show first few rows of data
                    st.dataframe(vh.df.head(10))
                    
                    # Basic statistics
                    st.markdown("<h4>Basic Statistics</h4>", unsafe_allow_html=True)
                    st.dataframe(vh.df.describe())
                    
                    # Available columns
                    st.markdown("<h4>Available Log Curves</h4>", unsafe_allow_html=True)
                    st.write(", ".join(vh.df.columns))
                    
                    # Choose columns for analysis
                    st.markdown("<h4>Select Features for Analysis</h4>", unsafe_allow_html=True)
                    available_cols = list(vh.df.columns)
                    default_cols = [col for col in ['GR', 'RHOB', 'LLD', 'NPHI', 'VSHALE', 'PHI', 'SW'] 
                                    if col in available_cols]
                    
                    selected_features = st.multiselect(
                        "Select features for clustering",
                        options=available_cols,
                        default=default_cols
                    )
                
                # Calculate petrophysical properties
                with tab2:
                    st.markdown("<h3 class='section-header'>Petrophysical Properties</h3>", unsafe_allow_html=True)
                    
                    if st.button("Calculate Petrophysical Properties"):
                        with st.spinner("Calculating properties..."):
                            vh.calculate_petrophysical_properties()
                            
                            if hasattr(vh, 'well_data') and vh.well_data is not None:
                                st.success("Petrophysical properties calculated successfully!")
                                
                                # Show calculated properties
                                st.dataframe(vh.well_data.head(10))
                                
                                # Prepare features for clustering
                                if selected_features:
                                    feature_success = vh.prepare_features(selected_features)
                                    if feature_success:
                                        st.success(f"Features prepared: {', '.join(selected_features)}")
                                    else:
                                        st.error("Error preparing features.")
                                else:
                                    st.warning("Please select features for clustering in the Data Preview tab.")
                            else:
                                st.error("Error calculating petrophysical properties.")
                
                # Clustering
                with tab3:
                    st.markdown("<h3 class='section-header'>Facies Classification with K-means Clustering</h3>", unsafe_allow_html=True)
                    
                    min_clusters = config["min_clusters"]
                    max_clusters = config["max_clusters"]
                    
                    if st.button("Perform K-means Clustering"):
                        with st.spinner(f"Performing clustering with {min_clusters}-{max_clusters} clusters..."):
                            # Check if features are prepared
                            if not hasattr(vh, 'features') or vh.features is None:
                                st.error("Features not prepared. Please calculate petrophysical properties first.")
                            else:
                                # Perform clustering
                                cluster_results = vh.perform_kmeans_clustering(
                                    min_clusters=min_clusters,
                                    max_clusters=max_clusters,
                                    save_plots=True
                                )
                                
                                if cluster_results:
                                    st.success("Clustering completed successfully!")
                                    
                                    # Show silhouette scores
                                    silhouette_scores = cluster_results['silhouette_scores']
                                    
                                    # Plot silhouette scores
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    clusters = list(silhouette_scores.keys())
                                    scores = list(silhouette_scores.values())
                                    
                                    ax.plot(clusters, scores, 'o-', linewidth=2, markersize=10)
                                    ax.set_xlabel('Number of Clusters', fontsize=14)
                                    ax.set_ylabel('Silhouette Score', fontsize=14)
                                    ax.set_xticks(clusters)
                                    ax.grid(True, alpha=0.3)
                                    ax.set_title('Silhouette Scores for Different Cluster Counts', fontsize=16)
                                    
                                    st.pyplot(fig)
                                    
                                    # Best cluster count
                                    best_cluster = max(silhouette_scores.items(), key=lambda x: x[1])[0]
                                    st.info(f"Best cluster count based on silhouette score: {best_cluster}")
                                    
                                    # Create cluster dataset for best cluster
                                    st.markdown("<h4>Creating Facies Classification</h4>", unsafe_allow_html=True)
                                    
                                    with st.spinner(f"Creating facies classification with {best_cluster} clusters..."):
                                        # Create datasets for all clusters
                                        for n_clusters in range(min_clusters, max_clusters + 1):
                                            vh.create_cluster_dataset(n_clusters)
                                            vh.identify_clustering_ranges(n_clusters)
                                            vh.generate_adjacency_matrix(n_clusters)
                                        
                                        st.success(f"Created facies classifications for clusters {min_clusters}-{max_clusters}")
                                        
                                        # Try to display elbow method plot if available
                                        elbow_plot_path = os.path.join(temp_dir, 'elbow_method.png')
                                        if os.path.exists(elbow_plot_path):
                                            st.markdown("<h4>Elbow Method for Optimal Cluster Count</h4>", unsafe_allow_html=True)
                                            display_image_with_caption(elbow_plot_path, "Elbow Method")
                                else:
                                    st.error("Error performing clustering.")
                
                # Model training
                with tab4:
                    st.markdown("<h3 class='section-header'>GCN Model Training</h3>", unsafe_allow_html=True)
                    
                    # Get configuration
                    train_ratio = config["train_ratio"]
                    val_ratio = config["val_ratio"]
                    test_ratio = config["test_ratio"]
                    hidden_channels = config["hidden_channels"]
                    num_runs = config["num_runs"]
                    
                    # Check if required ML libraries are available
                    ml_libs_available = True
                    try:
                        import torch
                        from torch_geometric.nn import GCNConv
                    except ImportError:
                        ml_libs_available = False
                        st.warning("""
                        ⚠️ PyTorch and/or PyTorch Geometric are not available in this deployment.
                        Full model training is disabled.
                        """)
                    
                    if ml_libs_available:
                        # Select clusters to run models for
                        cluster_options = list(range(min_clusters, max_clusters + 1))
                        selected_clusters = st.multiselect(
                            "Select clusters to train models for",
                            options=cluster_options,
                            default=[cluster_options[0], cluster_options[-1]]
                        )
                        
                        if st.button("Train GCN Models"):
                            if not selected_clusters:
                                st.warning("Please select at least one cluster configuration.")
                            else:
                                with st.spinner(f"Training models for {len(selected_clusters)} cluster configurations..."):
                                    try:
                                        # Initialize progress bar
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()
                                        
                                        # Train models for selected clusters
                                        best_models = {}
                                        for i, n_clusters in enumerate(selected_clusters):
                                            status_text.text(f"Training model for {n_clusters} clusters...")
                                            
                                            # Update progress
                                            progress = (i) / len(selected_clusters)
                                            progress_bar.progress(progress)
                                            
                                            # Run multiple models for each cluster
                                            best_accuracy = 0
                                            best_run = None
                                            
                                            for run_id in range(1, num_runs + 1):
                                                status_text.text(f"Training model for {n_clusters} clusters (Run {run_id}/{num_runs})...")
                                                
                                                # Train model
                                                model_results = vh.build_pyg_gcn_model(
                                                    n_clusters=n_clusters,
                                                    train_ratio=train_ratio,
                                                    val_ratio=val_ratio,
                                                    test_ratio=test_ratio,
                                                    hidden_channels=hidden_channels,
                                                    run_id=run_id
                                                )
                                                
                                                # Check if this run has better accuracy
                                                if 'test_acc' in model_results and model_results['test_acc'] > best_accuracy:
                                                    best_accuracy = model_results['test_acc']
                                                    best_run = run_id
                                            
                                            best_models[n_clusters] = {
                                                'run_id': best_run,
                                                'accuracy': best_accuracy
                                            }
                                        
                                        # Update progress to completion
                                        progress_bar.progress(1.0)
                                        status_text.text("Model training completed!")
                                        
                                        # Display results
                                        st.success("Models trained successfully!")
                                        
                                        # Display best model for each cluster
                                        st.markdown("<h4>Best Models</h4>", unsafe_allow_html=True)
                                        
                                        best_results = []
                                        for n_clusters, info in best_models.items():
                                            best_results.append({
                                                "Clusters": n_clusters,
                                                "Best Run": info["run_id"],
                                                "Test Accuracy": f"{info['accuracy']:.4f}"
                                            })
                                        
                                        st.table(pd.DataFrame(best_results))
                                        
                                        # Generate visualizations
                                        st.markdown("<h4>Visualizations</h4>", unsafe_allow_html=True)
                                        
                                        with st.spinner("Generating visualizations..."):
                                            # Get best run IDs
                                            best_run_ids = {n: info['run_id'] for n, info in best_models.items()}
                                            
                                            # Visualize loss and accuracy
                                            vh.visualize_loss_accuracy(
                                                n_clusters_list=selected_clusters,
                                                run_ids=best_run_ids
                                            )
                                            
                                            # Visualize facies
                                            vh.visualize_facies(
                                                n_clusters_list=selected_clusters
                                            )
                                            
                                            # Visualize predicted results
                                            vh.visualize_predicted_results(
                                                n_clusters_list=selected_clusters,
                                                run_ids=best_run_ids
                                            )
                                            
                                            st.success("Visualizations generated successfully!")
                                            
                                            # Display visualizations
                                            loss_acc_path = os.path.join(temp_dir, 'loss_accuracy_comparison.png')
                                            facies_path = os.path.join(temp_dir, 'facies_comparison.png')
                                            pred_path = os.path.join(temp_dir, 'prediction_comparison.png')
                                            
                                            if os.path.exists(loss_acc_path):
                                                st.markdown("<h5>Loss and Accuracy Comparison</h5>", unsafe_allow_html=True)
                                                display_image_with_caption(loss_acc_path, "Loss and Accuracy Comparison")
                                            
                                            if os.path.exists(facies_path):
                                                st.markdown("<h5>Facies Comparison</h5>", unsafe_allow_html=True)
                                                display_image_with_caption(facies_path, "Facies Comparison")
                                            
                                            if os.path.exists(pred_path):
                                                st.markdown("<h5>Prediction Comparison</h5>", unsafe_allow_html=True)
                                                display_image_with_caption(pred_path, "Prediction Comparison")
                                    
                                    except Exception as e:
                                        st.error(f"Error training models: {str(e)}")
                                        logger.error(f"Error training models: {e}", exc_info=True)
                    else:
                        st.info("""
                        This section requires PyTorch and PyTorch Geometric libraries to run the GCN models.
                        
                        To train models locally:
                        1. Install the required ML packages: `pip install torch torch-geometric stellargraph tensorflow`
                        2. Run the VHydro tool on your local machine
                        """)
            else:
                # If VHydro module is not available, show preview only
                try:
                    # Try to load the LAS file manually using lasio if available
                    try:
                        import lasio
                        las = lasio.read(las_file_path)
                        df = las.df()
                        df = df.reset_index()
                        
                        with tab1:
                            st.markdown("<h3 class='section-header'>Well Log Data Preview</h3>", unsafe_allow_html=True)
                            st.dataframe(df.head(10))
                            st.markdown("<h4>Basic Statistics</h4>", unsafe_allow_html=True)
                            st.dataframe(df.describe())
                            st.info("The VHydro module is not available for further processing. This is just a data preview.")
                    except ImportError:
                        with tab1:
                            st.error("The lasio library is not available to read LAS files.")
                    
                    with tab2, tab3, tab4:
                        st.warning("VHydro module is not available. These features are disabled in this deployment.")
                except Exception as e:
                    st.error(f"Error loading LAS file: {str(e)}")
                    logger.error(f"Error loading LAS file: {e}", exc_info=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error processing file: {e}", exc_info=True)
    else:
        st.info("Please upload a LAS file to begin analysis.")
        
        # Show example analysis
        with st.expander("Example Analysis"):
            st.markdown("""
            VHydro performs the following analysis steps on your well log data:
            
            1. **Data Loading and Validation**
               - Load LAS file
               - Validate required curves
               - Handle missing or invalid data
            
            2. **Petrophysical Property Calculation**
               - Calculate Shale Volume, Porosity, Water/Oil Saturation, and Permeability
               - Apply industry-standard equations and parameters
            
            3. **Facies Classification**
               - Apply K-means clustering to identify natural groupings in the data
               - Determine optimal number of clusters using silhouette scores
               - Generate depth-based facies maps
            
            4. **Graph Construction**
               - Create depth nodes and PET (Petrophysical Entity) nodes
               - Establish connections between related nodes
               - Generate adjacency matrices for GCN input
            
            5. **GCN Model Training**
               - Train Graph Convolutional Network models
               - Optimize hyperparameters
               - Evaluate model performance
            
            6. **Hydrocarbon Quality Prediction**
               - Classify depth points into quality categories
               - Identify high-potential zones
               - Generate detailed visualization and reports
            """)

# Results visualization page
def results_visualization_page():
    st.markdown("<h2 class='sub-header'>Results Visualization</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This section allows you to visualize previously generated results from VHydro analysis.
    Upload result files to view facies classifications, model performance, and hydrocarbon
    quality predictions.
    """)
    
    # File uploader for results
    uploaded_results = st.file_uploader("Upload Results Directory (zip)", type=["zip"])
    
    if uploaded_results is not None:
        # Create temporary directory for outputs
        temp_dir = create_temp_dir()
        
        # Save uploaded file
        zip_path = os.path.join(temp_dir, "results.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_results.getbuffer())
        
        # Extract ZIP file
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            st.success("Results uploaded and extracted successfully!")
            
            # Check for results structure
            results_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d)) and d.isdigit()]
            
            if not results_dirs:
                st.error("No valid results directories found. Please upload a ZIP file with the correct structure.")
                return
            
            # Find available cluster configurations
            available_clusters = sorted([int(d) for d in results_dirs])
            
            st.info(f"Found results for {len(available_clusters)} cluster configurations: {', '.join(map(str, available_clusters))}")
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Facies Classification", "Model Performance", "Hydrocarbon Quality"])
            
            # Facies classification tab
            with tab1:
                st.markdown("<h3 class='section-header'>Facies Classification</h3>", unsafe_allow_html=True)
                
                # Select cluster to visualize
                selected_cluster = st.selectbox(
                    "Select cluster configuration",
                    options=available_clusters
                )
                
                # Look for facies file
                facies_file1 = os.path.join(temp_dir, f'facies_for_{selected_cluster}.xlsx')
                facies_file2 = os.path.join(temp_dir, str(selected_cluster), f'facies_for_{selected_cluster}.xlsx')
                
                if os.path.exists(facies_file1) or os.path.exists(facies_file2):
                    # Load facies data
                    facies_file = facies_file1 if os.path.exists(facies_file1) else facies_file2
                    facies_df = pd.read_excel(facies_file)
                    
                    # Display facies data
                    st.dataframe(facies_df.head(20))
                    
                    # Plot facies
                    fig, ax = plt.subplots(figsize=(8, 12))
                    
                    # Check for column names
                    facies_col = 'Facies_pred' if 'Facies_pred' in facies_df.columns else 'Facies'
                    
                    if facies_col in facies_df.columns and 'DEPTH' in facies_df.columns:
                        # Get unique facies values
                        unique_facies = facies_df[facies_col].unique()
                        
                        # Create a colormap
                        cmap = plt.cm.get_cmap('viridis', len(unique_facies))
                        
                        # Plot facies as a color log
                        facies_array = np.vstack((facies_df[facies_col].values, facies_df[facies_col].values)).T
                        im = ax.imshow(facies_array, aspect='auto', cmap=cmap, 
                                    extent=[0, 1, facies_df['DEPTH'].max(), facies_df['DEPTH'].min()])
                        
                        ax.set_title(f"Facies Classification (Clusters: {selected_cluster})", fontsize=14)
                        ax.set_xlabel("")
                        ax.set_ylabel("Depth", fontsize=12)
                        ax.set_xticks([])
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label('Facies')
                        cbar.set_ticks(range(len(unique_facies)))
                        cbar.set_ticklabels(range(len(unique_facies)))
                        
                        st.pyplot(fig)
                    else:
                        st.warning(f"Could not find required columns in facies file. Available columns: {', '.join(facies_df.columns)}")
                else:
                    st.warning(f"Could not find facies file for cluster {selected_cluster}")
            
            # Model performance tab
            with tab2:
                st.markdown("<h3 class='section-header'>Model Performance</h3>", unsafe_allow_html=True)
                
                # Select cluster to visualize
                selected_cluster = st.selectbox(
                    "Select cluster configuration",
                    options=available_clusters,
                    key="model_performance_cluster"
                )
                
                # Look for history file
                history_file = os.path.join(temp_dir, str(selected_cluster), 'results', f'History_{selected_cluster}.xlsx')
                
                if os.path.exists(history_file):
                    # Load history data
                    history_df = pd.read_excel(history_file)
                    
                    # Display history data
                    st.dataframe(history_df.head(20))
                    
                    # Plot loss and accuracy
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                    
                    # Plot loss
                    if 'loss' in history_df.columns:
                        ax1.plot(history_df['loss'], 'b-', linewidth=2, label='Training Loss')
                        ax1.set_title(f"Loss (Clusters: {selected_cluster})", fontsize=14)
                        ax1.set_xlabel("Epoch", fontsize=12)
                        ax1.set_ylabel("Loss", fontsize=12)
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                    
                    # Plot accuracy
                    acc_cols = [col for col in history_df.columns if 'acc' in col.lower()]
                    if acc_cols:
                        for col in acc_cols:
                            if 'train' in col.lower():
                                ax2.plot(history_df[col], 'b-', linewidth=2, label='Train Accuracy')
                            elif 'val' in col.lower():
                                ax2.plot(history_df[col], 'r-', linewidth=2, label='Validation Accuracy')
                            elif 'test' in col.lower():
                                ax2.plot(history_df[col], 'g-', linewidth=2, label='Test Accuracy')
                            else:
                                ax2.plot(history_df[col], '-', linewidth=2, label=col)
                                
                        ax2.set_title(f"Accuracy (Clusters: {selected_cluster})", fontsize=14)
                        ax2.set_xlabel("Epoch", fontsize=12)
                        ax2.set_ylabel("Accuracy", fontsize=12)
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Look for classification report
                    class_report_file = os.path.join(temp_dir, str(selected_cluster), 'results', f'ClassReport_{selected_cluster}.xlsx')
                    
                    if os.path.exists(class_report_file):
                        st.markdown("<h4>Classification Report</h4>", unsafe_allow_html=True)
                        
                        # Load classification report
                        cr_df = pd.read_excel(class_report_file)
                        
                        # Display classification report
                        st.dataframe(cr_df)
                    else:
                        st.warning(f"Could not find classification report for cluster {selected_cluster}")
                else:
                    st.warning(f"Could not find history file for cluster {selected_cluster}")
            
            # Hydrocarbon quality tab
            with tab3:
                st.markdown("<h3 class='section-header'>Hydrocarbon Quality Prediction</h3>", unsafe_allow_html=True)
                
                # Select cluster to visualize
                selected_cluster = st.selectbox(
                    "Select cluster configuration",
                    options=available_clusters,
                    key="hydrocarbon_quality_cluster"
                )
                
                # Look for results file
                results_file = os.path.join(temp_dir, str(selected_cluster), 'results', f'Results_{selected_cluster}.xlsx')
                
                if os.path.exists(results_file):
                    # Load results data
                    results_df = pd.read_excel(results_file)
                    
                    # Display results data
                    st.dataframe(results_df.head(20))
                    
                    # Get unique classes
                    if 'True' in results_df.columns and 'Predicted' in results_df.columns:
                        true_classes = results_df['True'].unique()
                        
                        # Confusion matrix
                        st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
                        
                        # Calculate confusion matrix
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(results_df['True'], results_df['Predicted'])
                        
                        # Plot confusion matrix
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax.set_title(f"Confusion Matrix (Clusters: {selected_cluster})", fontsize=14)
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label('Count')
                        
                        # Set ticks and labels
                        tick_marks = np.arange(len(true_classes))
                        ax.set_xticks(tick_marks)
                        ax.set_yticks(tick_marks)
                        ax.set_xticklabels(true_classes, rotation=45, ha='right')
                        ax.set_yticklabels(true_classes)
                        
                        # Add text annotations
                        thresh = cm.max() / 2.
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax.text(j, i, format(cm[i, j], 'd'),
                                        ha="center", va="center",
                                        color="white" if cm[i, j] > thresh else "black")
                        
                        ax.set_ylabel('True Label', fontsize=12)
                        ax.set_xlabel('Predicted Label', fontsize=12)
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Quality distribution
                        st.markdown("<h4>Hydrocarbon Quality Distribution</h4>", unsafe_allow_html=True)
                        
                        # Plot quality distribution
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                        
                        # True quality distribution
                        true_counts = results_df['True'].value_counts().sort_index()
                        ax1.bar(true_counts.index, true_counts.values, color='blue', alpha=0.7)
                        ax1.set_title('True Quality Distribution', fontsize=14)
                        ax1.set_xlabel('Quality Class', fontsize=12)
                        ax1.set_ylabel('Count', fontsize=12)
                        ax1.tick_params(axis='x', rotation=45)
                        
                        # Predicted quality distribution
                        pred_counts = results_df['Predicted'].value_counts().sort_index()
                        ax2.bar(pred_counts.index, pred_counts.values, color='green', alpha=0.7)
                        ax2.set_title('Predicted Quality Distribution', fontsize=14)
                        ax2.set_xlabel('Quality Class', fontsize=12)
                        ax2.set_ylabel('Count', fontsize=12)
                        ax2.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Quality vs Depth
                        st.markdown("<h4>Hydrocarbon Quality vs Depth</h4>", unsafe_allow_html=True)
                        
                        if 'Node' in results_df.columns:
                            # Convert Node to numeric (remove dots from depth values)
                            results_df['Depth'] = results_df['Node'].apply(lambda x: float(str(x).replace('.', '')))
                            
                            # Plot true and predicted quality vs depth
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 16), sharey=True)
                            
                            # Create categorical color maps
                            from matplotlib.colors import ListedColormap
                            
                            # Get quality classes and assign colors
                            quality_classes = sorted(list(set(results_df['True'].unique()) | set(results_df['Predicted'].unique())))
                            quality_map = {quality: i for i, quality in enumerate(quality_classes)}
                            
                            # Define colors for different quality levels
                            quality_colors = plt.cm.viridis(np.linspace(0, 1, len(quality_classes)))
                            quality_cmap = ListedColormap(quality_colors)
                            
                            # Map quality classes to numeric values for coloring
                            results_df['True_num'] = results_df['True'].map(quality_map)
                            results_df['Predicted_num'] = results_df['Predicted'].map(quality_map)
                            
                            # Sort by depth
                            results_df_sorted = results_df.sort_values('Depth')
                            
                            # Plot true quality
                            scatter1 = ax1.scatter(
                                [0.5] * len(results_df_sorted), 
                                results_df_sorted['Depth'],
                                c=results_df_sorted['True_num'], 
                                cmap=quality_cmap,
                                s=100,
                                vmin=0, 
                                vmax=len(quality_classes) - 1
                            )
                            
                            ax1.set_title('True Quality', fontsize=14)
                            ax1.set_xlim(0, 1)
                            ax1.set_xticks([])
                            ax1.set_ylabel('Depth', fontsize=12)
                            
                            # Plot predicted quality
                            scatter2 = ax2.scatter(
                                [0.5] * len(results_df_sorted), 
                                results_df_sorted['Depth'],
                                c=results_df_sorted['Predicted_num'], 
                                cmap=quality_cmap,
                                s=100,
                                vmin=0, 
                                vmax=len(quality_classes) - 1
                            )
                            
                            ax2.set_title('Predicted Quality', fontsize=14)
                            ax2.set_xlim(0, 1)
                            ax2.set_xticks([])
                            
                            # Add colorbar
                            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                            cbar = fig.colorbar(scatter2, cax=cbar_ax, ticks=range(len(quality_classes)))
                            cbar.set_ticklabels(quality_classes)
                            cbar.set_label('Hydrocarbon Quality', fontsize=12)
                            
                            plt.tight_layout(rect=[0, 0, 0.9, 1])
                            st.pyplot(fig)
                        else:
                            st.warning("Could not find depth information in results file")
                    else:
                        st.warning("Could not find required columns in results file")
                else:
                    st.warning(f"Could not find results file for cluster {selected_cluster}")
        except Exception as e:
            st.error(f"Error extracting or processing results: {str(e)}")
            logger.error(f"Error extracting or processing results: {e}", exc_info=True)
    else:
        st.info("Please upload a ZIP file containing VHydro results.")
        
        # Example visualizations
        with st.expander("Example Visualizations"):
            st.markdown("""
            VHydro generates multiple visualizations to help interpret the results:
            
            1. **Facies Classification**
               - Depth-based facies logs
               - Cluster distribution plots
               - Facies property cross-plots
            
            2. **Model Performance**
               - Loss and accuracy curves
               - Confusion matrices
               - Classification reports with precision, recall, and F1-score
            
            3. **Hydrocarbon Quality Prediction**
               - Quality distribution by depth
               - Comparison of true vs. predicted quality
               - Identification of high-potential zones
            """)

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
    
    # Footer
    st.markdown("<div class='footer'>VHydro - Advanced Hydrocarbon Quality Prediction © 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
