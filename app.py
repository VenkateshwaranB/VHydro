import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# IMPORTANT: Set page configuration at the beginning
st.set_page_config(
    page_title="VHydro - Hydrocarbon Quality Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

# Custom CSS - simplified version
st.markdown("""
<style>
    /* Main app background and colors */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Colorful section headers */
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
    }
    
    /* Toggle button styling */
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

# Simplified sidebar with navigation
def create_sidebar():
    """Create the sidebar with toggle buttons"""
    st.sidebar.title("VHydro")
    st.sidebar.markdown("---")
    
    # Navigation section
    st.sidebar.markdown("### Navigation")
    
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
        button_class = "toggle-button toggle-button-active" if page_info["active"] else "toggle-button"
        
        # Create button with HTML for styling
        if st.sidebar.markdown(f"""
        <div class="{button_class}">
            <span>{page_name}</span>
            <span>{page_info["icon"]}</span>
        </div>
        """, unsafe_allow_html=True):
            st.session_state['current_page'] = page_name
            st.rerun()
    
    # Model configuration section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Configuration")
    
    # Cluster configuration
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

# Home page - simplified
def home_page():
    st.markdown('<h1 class="main-header">VHydro - Hydrocarbon Quality Prediction</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Advanced Graph Convolutional Network for Petrophysical Analysis</p>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>About VHydro</h2>", unsafe_allow_html=True)
    
    st.info("""
    **VHydro** is an advanced tool for hydrocarbon quality prediction using well log data.
    It combines traditional petrophysical analysis with modern machine learning techniques
    to provide accurate predictions of reservoir quality.
    
    The tool uses Graph Convolutional Networks (GCN) to model the complex relationships
    between different petrophysical properties and depth values, enabling more accurate
    classification of hydrocarbon potential zones.
    """)
    
    # Key features
    st.markdown("<h2 class='sub-header'>Key Features</h2>", unsafe_allow_html=True)
    
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
    
    # Getting started section
    st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
    
    if st.button("Start Analysis", key="start_analysis_btn"):
        st.session_state['current_page'] = "Dataset Preparation"
        st.rerun()

# Dataset preparation page - simplified
def dataset_preparation_page():
    """Render the dataset preparation page"""
    st.markdown('<h1 class="main-header">Dataset Preparation</h1>', unsafe_allow_html=True)
    
    st.markdown('Upload your LAS file containing well log data. The file will be processed to calculate petrophysical properties necessary for hydrocarbon potential prediction.', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload LAS file", type=["las"])
    
    # Provide option for sample data
    use_sample = st.checkbox("Use sample data instead", value=False)
    
    # Parameters section
    st.markdown("<h2 class='sub-header'>Calculation Parameters</h2>", unsafe_allow_html=True)
    
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
    st.markdown("<h2 class='sub-header'>Feature Selection</h2>", unsafe_allow_html=True)
    
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
                
                # Continue button
                if st.button("Continue to Model Workflow"):
                    st.session_state['current_page'] = "Model Workflow"
                    st.rerun()
    else:
        st.info("Please upload a LAS file or use the sample data to continue.")

# Model workflow page - simplified
def model_workflow_page():
    """Render the model workflow page"""
    st.markdown('<h1 class="main-header">Model Workflow</h1>', unsafe_allow_html=True)
    
    # Create a modern intro section
    st.info("""
    **Graph Convolutional Network (GCN) Workflow**
    
    This page outlines the workflow for creating a Graph Convolutional Network model for hydrocarbon quality prediction.
    The process includes dataset creation, graph construction, model training, and evaluation.
    """)
    
    # Create workflow steps
    steps = [
        {"title": "1. Dataset Preparation", "icon": "📋", "status": "completed", "desc": "Process well log data and calculate petrophysical properties"},
        {"title": "2. K-means Clustering", "icon": "🧩", "status": "active", "desc": "Perform K-means clustering to identify facies"},
        {"title": "3. Graph Construction", "icon": "🔗", "status": "pending", "desc": "Create nodes and edges for the graph dataset"},
        {"title": "4. GCN Training", "icon": "🧠", "status": "pending", "desc": "Train the Graph Convolutional Network model"},
        {"title": "5. Model Evaluation", "icon": "📊", "status": "pending", "desc": "Evaluate model performance and visualize results"}
    ]
    
    # Display steps
    for step in steps:
        status_color = "green" if step["status"] == "completed" else "blue" if step["status"] == "active" else "gray"
        status_text = "Completed" if step["status"] == "completed" else "In Progress" if step["status"] == "active" else "Pending"
        
        st.markdown(f"""
        <div style="display: flex; margin-bottom: 20px; align-items: flex-start;">
            <div style="margin-right: 15px; font-size: 24px;">
                {step["icon"]}
            </div>
            <div style="flex-grow: 1;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <h3 style="margin: 0;">{step["title"]}</h3>
                    <span style="color: {status_color};">{status_text}</span>
                </div>
                <p style="margin-top: 5px; color: #666;">{step["desc"]}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # K-means Clustering section (active step)
    st.markdown("<h2 class='sub-header'>K-means Clustering for Facies Classification</h2>", unsafe_allow_html=True)
    
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
            
            # Continue button
            if st.button("Continue to Graph Construction"):
                st.info("Next: Graph Construction")

# Analysis tool page - simplified
def analysis_tool_page(config):
    """Render the analysis tool page"""
    st.markdown('<h1 class="main-header">Analysis Tool</h1>', unsafe_allow_html=True)
    
    st.markdown('Analyze your well log data using the VHydro Graph Convolutional Network model.', unsafe_allow_html=True)
    
    # Create tabs for different analysis steps
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Process", "Graph Creation", "Model Training", "Results"])
    
    with tab1:
        st.markdown("### 1. Data Upload and Processing")
        
        uploaded_file = st.file_uploader("Upload LAS file", type=["las"])
        
        # Option to use sample data
        use_sample = st.checkbox("Use sample data instead", value=not uploaded_file)
        
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
        st.markdown("### 2. Graph Dataset Creation")
        
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
                                       index=2)
            
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
    
    with tab3:
        st.markdown("### 3. GCN Model Training")
        
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
    
    with tab4:
        st.markdown("### 4. Results Analysis")
        
        # Add tabs for different visualizations
        result_tab1, result_tab2 = st.tabs(["Hydrocarbon Potential", "Depth Profile"])
        
        with result_tab1:
            st.markdown("#### Hydrocarbon Potential Distribution")
            
            # Create a simple chart
            fig, ax = plt.subplots(figsize=(8, 5))
            
            categories = ["Very Low", "Low", "Moderate", "High"]
            values = [25, 35, 30, 10]
            
            ax.bar(categories, values, color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"])
            ax.set_ylabel('Percentage')
            ax.set_title('Distribution of Hydrocarbon Potential')
            
            st.pyplot(fig)
        
        with result_tab2:
            st.markdown("#### Depth vs. Hydrocarbon Potential")
            
            st.info("Depth profile visualization would appear here.")

# Results visualization page - simplified
def results_visualization_page():
    """Render the results visualization page"""
    st.markdown('<h1 class="main-header">Results Visualization</h1>', unsafe_allow_html=True)
    
    st.markdown('Interactive visualization of hydrocarbon potential prediction results.', unsafe_allow_html=True)
    
    # Create a sample dataset for visualization
    depths = np.arange(1000, 1100, 0.5)
    quality_labels = ["Very Low", "Low", "Moderate", "High"]
    quality_numeric = np.random.randint(0, 4, size=len(depths))
    quality = np.array(quality_labels)[quality_numeric]
    
    # Create sample dataframe
    data = {
        "DEPTH": depths,
        "GR": 50 + 30 * np.sin(depths / 10) + 10 * np.random.randn(len(depths)),
        "RHOB": 2.2 + 0.3 * np.cos(depths / 15) + 0.05 * np.random.randn(len(depths)),
        "PHI": 0.2 - 0.15 * np.cos(depths / 15) + 0.02 * np.random.randn(len(depths)),
        "SW": 0.4 + 0.3 * np.sin(depths / 20) + 0.05 * np.random.randn(len(depths)),
        "QUALITY": quality,
        "QUALITY_NUMERIC": quality_numeric
    }
    
    df = pd.DataFrame(data)
    
    # Sidebar controls for visualization
    st.sidebar.markdown("### Visualization Controls")
    
    # Depth range slider
    depth_min = float(df["DEPTH"].min())
    depth_max = float(df["DEPTH"].max())
    
    depth_range = st.sidebar.slider(
        "Depth Range",
        min_value=depth_min,
        max_value=depth_max,
        value=(depth_min, depth_min + 50),
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
        st.markdown("### Well Log View")
        
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
        st.markdown("### Cross-Plot Analysis")
        
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
        st.markdown("### Statistical Summary")
        
        # Display dataframe
        st.dataframe(filtered_data.describe())
        
        # Create a simple bar chart
        st.markdown("#### Quality Distribution")
        
        quality_counts = filtered_data["QUALITY"].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(quality_counts.index, quality_counts.values, color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"])
        ax.set_ylabel('Count')
        ax.set_title('Hydrocarbon Potential Distribution')
        
        st.pyplot(fig)

# Main function
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
    st.markdown("<div style='text-align:center; margin-top:30px; color:#666; font-size:0.8rem;'>VHydro - Hydrocarbon Potential Prediction Application © 2025</div>", unsafe_allow_html=True)

# Entry point for the application
if __name__ == "__main__":
    main()
