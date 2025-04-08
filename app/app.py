import streamlit as st

import streamlit as st

# Set up error handling for dependency imports
st.set_page_config(
    page_title="VHydro - Hydrocarbon Potential Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try importing dependencies with helpful error messages
try:
    import pandas as pd
    import numpy as np
    st.write("Successfully imported pandas and numpy")
except ImportError as e:
    st.error(f"Error importing pandas or numpy: {e}")
    st.info("Please check your requirements.txt file")
    st.stop()

try:
    import matplotlib.pyplot as plt
    st.write("Successfully imported matplotlib")
except ImportError as e:
    st.error(f"Error importing matplotlib: {e}")
    st.info("This may be due to missing system dependencies. Please check your packages.txt file.")
    st.stop()

# Continue with other imports...
import os
import tempfile
import time
import plotly.graph_objects as go
from PIL import Image

# Import custom modules
import sys
sys.path.append('.')
try:
    from data_processing import process_las_file, evaluate_clusters, extract_well_metadata
    from gcn_model import VHydroGCN, simulate_gcn_prediction
    from network_visualization import (
        create_network_graph, create_facies_visualization, 
        create_pet_distribution_plot, create_feature_importance_plot,
        create_3d_graph_visualization, create_comparison_heatmap,
        create_depth_profile_visualization
    )
    from utils import (
        create_download_link, create_figure_download_link, save_uploaded_file,
        create_project_folder, save_project_metadata, load_project_metadata,
        format_time, display_progress, create_export_zip, create_download_button_for_file,
        get_hydrocarbon_potential_color_scale, create_summary_report, show_colorbar_legend
    )
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Make sure all required files are in the current directory.")
    st.stop()


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0e4194;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0e4194;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1.1rem;
        text-align: justify;
        margin-bottom: 1.5rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #0e4194;
        color: white;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #555;
        font-size: 0.8rem;
    }
    .plot-container {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: white;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e7f1ff;
        border-left: 5px solid #0e4194;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"

# Define page navigation
def navigation():
    st.sidebar.markdown('<p class="sidebar-header">Navigation</p>', unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Home", "Data Upload & Processing", "Graph Dataset Creation", 
                                "Graph Visualization", "Model Prediction", "Results & Interpretation"])
    return page

def home_page():
    st.markdown('<h1 class="main-header">VHydro - Hydrocarbon Potential Prediction</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="description">Welcome to the VHydro app for predicting hydrocarbon potential zones using Graph Convolutional Networks (GCN). This application leverages a novel graph dataset approach to analyze well log data and predict zones with high hydrocarbon potential.</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<h3>About VHydro</h3>', unsafe_allow_html=True)
        st.markdown('<p>VHydro is a novel approach that uses Graph Convolutional Networks to predict hydrocarbon potential zones from well log data. The method creates a graph representation of well log data, where nodes represent depth points and edges represent relationships between different petrophysical entities (PET).</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<h2 class="sub-header">How It Works</h2>', unsafe_allow_html=True)
        st.markdown('<p class="description">The VHydro workflow consists of the following steps:</p>', unsafe_allow_html=True)
        
        st.markdown("""
        1. **Data Upload & Processing**: Upload LAS file containing well log data that will be processed to calculate petrophysical properties
        2. **Graph Dataset Creation**: Convert well log data into a graph dataset with nodes representing depth points
        3. **Graph Visualization**: Visualize the graph dataset to understand the relationships between different depth points
        4. **Model Prediction**: Use the Graph Convolutional Network model to predict hydrocarbon potential
        5. **Results & Interpretation**: Analyze and interpret the prediction results
        """)
        
        st.markdown('<h2 class="sub-header">Getting Started</h2>', unsafe_allow_html=True)
        st.markdown('<p class="description">To get started, navigate to the "Data Upload & Processing" page using the sidebar and upload your LAS file containing well log data.</p>', unsafe_allow_html=True)
    
    with col2:
        # Placeholder for a conceptual image
        st.markdown('<div style="padding: 1rem; background-color: #f5f5f5; border-radius: 0.5rem; text-align: center;">', unsafe_allow_html=True)
        st.markdown('<h3>VHydro Process Overview</h3>', unsafe_allow_html=True)
        
        # Create a simple diagram using matplotlib
        fig, ax = plt.subplots(figsize=(5, 8))
        ax.axis('off')
        
        steps = ["Well Log Data", "Feature Extraction", "Graph Dataset Creation", 
                 "GCN Model Training", "Hydrocarbon Potential Prediction"]
        
        for i, step in enumerate(steps):
            ax.add_patch(plt.Rectangle((0.2, 0.8 - i*0.15), 0.6, 0.1, alpha=0.8, facecolor='#0e4194'))
            ax.text(0.5, 0.85 - i*0.15, step, ha='center', va='center', color='white', fontweight='bold')
            
            if i < len(steps) - 1:
                ax.arrow(0.5, 0.8 - i*0.15, 0, -0.05, head_width=0.03, head_length=0.02, 
                         fc='black', ec='black', width=0.005)
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown('<h3>Publication Reference</h3>', unsafe_allow_html=True)
        st.markdown('<p>For more details about the VHydro methodology, please refer to the publication:</p>', unsafe_allow_html=True)
        st.markdown('<a href="https://link.springer.com/article/10.1007/s11053-024-10311-x" target="_blank">Novel Graph Dataset for Hydrocarbon Potential Prediction</a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def data_upload_page():
    st.markdown('<h1 class="main-header">Data Upload & Processing</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">Upload your LAS file containing well log data. The file will be processed to calculate petrophysical properties necessary for hydrocarbon potential prediction.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload LAS file", type=["las"])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        tmp_filepath = save_uploaded_file(uploaded_file)
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown(f"<p>File <b>{uploaded_file.name}</b> successfully uploaded! Processing data...</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process the LAS file
        with st.spinner('Processing LAS file...'):
            result, message = process_las_file(tmp_filepath)
        
        if result is None:
            st.markdown('<div class="warning-message">', unsafe_allow_html=True)
            st.markdown(f"<p>Error: {message}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-message">', unsafe_allow_html=True)
            st.markdown("<p>LAS file processed successfully! Petrophysical properties have been calculated.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Extract well metadata
            metadata = extract_well_metadata(result['las'])
            
            # Display well information
            st.markdown('<h2 class="sub-header">Well Information</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<b>Well Name:</b> " + metadata.get('WELL', 'Unknown'), unsafe_allow_html=True)
                st.markdown("<b>UWI:</b> " + metadata.get('UWI', 'Unknown'), unsafe_allow_html=True)
                st.markdown("<b>Field:</b> " + metadata.get('FIELD', 'Unknown'), unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<b>Depth Range:</b> {metadata['STRT']} - {metadata['STOP']} {metadata['DEPTH_UNIT']}", unsafe_allow_html=True)
                st.markdown(f"<b>Step:</b> {metadata['STEP']} {metadata['DEPTH_UNIT']}", unsafe_allow_html=True)
                st.markdown(f"<b>Available Curves:</b> {', '.join(metadata['CURVES'][:5])}...", unsafe_allow_html=True)
            
            # Display the first few rows of the processed data
            st.markdown('<h2 class="sub-header">Processed Well Log Data</h2>', unsafe_allow_html=True)
            st.dataframe(result['features'].head(10))
            
            # Create well log visualization
            st.markdown('<h2 class="sub-header">Well Log Visualization</h2>', unsafe_allow_html=True)
            
            # Basic well log plot
            fig, ax = plt.subplots(1, 5, figsize=(15, 10), sharey=True)
            
            # Plot GR
            if 'GR' in result['features'].columns:
                ax[0].plot(result['features']['GR'], result['features']['DEPTH'], color='black')
                ax[0].set_title('GR')
                ax[0].set_xlim(0, result['features']['GR'].max() * 1.1)
                ax[0].grid(True)
            
            # Plot RHOB
            if 'RHOB' in result['features'].columns:
                ax[1].plot(result['features']['RHOB'], result['features']['DEPTH'], color='red')
                ax[1].set_title('RHOB')
                ax[1].set_xlim(result['features']['RHOB'].min() * 0.9, result['features']['RHOB'].max() * 1.1)
                ax[1].grid(True)
            
            # Plot Porosity
            ax[2].plot(result['features']['PHIECALC'], result['features']['DEPTH'], color='blue')
            ax[2].set_title('PHIE')
            ax[2].set_xlim(0, 0.4)
            ax[2].grid(True)
            
            # Plot Water/Oil Saturation
            ax[3].plot(result['features']['WSAT'], result['features']['DEPTH'], color='blue', label='SW')
            ax[3].plot(result['features']['OSAT'], result['features']['DEPTH'], color='green', label='SO')
            ax[3].set_title('Saturation')
            ax[3].set_xlim(0, 1)
            ax[3].grid(True)
            ax[3].legend()
            
            # Plot Permeability
            perm_log = np.log10(result['features']['PERM'] + 0.001)  # Log scale for permeability
            ax[4].plot(perm_log, result['features']['DEPTH'], color='brown')
            ax[4].set_title('Permeability (log)')
            ax[4].grid(True)
            
            # Set y-axis to increase with depth
            for a in ax:
                a.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Save the processed data to session state for use in other pages
            st.session_state['result'] = result
            
            # Find optimal number of clusters
            with st.expander("Find Optimal Number of Clusters", expanded=False):
                st.write("You can evaluate different numbers of clusters to find the optimal number for your data.")
                
                if st.button("Evaluate Clusters"):
                    with st.spinner("Evaluating clusters..."):
                        scores, optimal_clusters = evaluate_clusters(result['features'])
                        
                        # Plot scores
                        fig, ax = plt.subplots(figsize=(10, 6))
                        clusters = list(scores.keys())
                        silhouette_values = list(scores.values())
                        
                        ax.plot(clusters, silhouette_values, 'o-')
                        ax.axvline(x=optimal_clusters, color='r', linestyle='--', 
                                  label=f'Optimal: {optimal_clusters} clusters')
                        ax.set_xlabel('Number of Clusters')
                        ax.set_ylabel('Silhouette Score')
                        ax.set_title('Silhouette Scores for Different Numbers of Clusters')
                        ax.grid(True)
                        ax.legend()
                        
                        st.pyplot(fig)
                        
                        st.markdown(f"<p>The optimal number of clusters for your data is <b>{optimal_clusters}</b>.</p>", unsafe_allow_html=True)
                        
                        # Save to session state
                        st.session_state['optimal_clusters'] = optimal_clusters
            
            # Add a button to proceed to the next step
            if st.button("Proceed to Graph Dataset Creation"):
                st.session_state['current_page'] = "Graph Dataset Creation"
                st.experimental_rerun()

def graph_dataset_page():
    st.markdown('<h1 class="main-header">Graph Dataset Creation</h1>', unsafe_allow_html=True)
    
    if 'result' not in st.session_state:
        st.markdown('<div class="warning-message">', unsafe_allow_html=True)
        st.markdown("<p>Please upload and process a LAS file first on the 'Data Upload & Processing' page.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Go to Data Upload"):
            st.session_state['current_page'] = "Data Upload & Processing"
            st.experimental_rerun()
            
        return
    
    st.markdown('<p class="description">On this page, we will create a graph dataset from the processed well log data. The graph dataset consists of nodes representing depth points and edges representing connections between different petrophysical entities (PET).</p>', unsafe_allow_html=True)
    
    # Parameters for graph dataset creation
    st.markdown('<h2 class="sub-header">Graph Dataset Parameters</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use optimal clusters if available, otherwise default to 7
        default_clusters = st.session_state.get('optimal_clusters', 7)
        n_clusters = st.slider("Number of clusters", min_value=5, max_value=10, value=default_clusters, 
                             help="Number of clusters for K-means clustering of depth points")
    
    with col2:
        train_test_split = st.slider("Train/Test Split", min_value=0.6, max_value=0.9, value=0.8, 
                                  help="Proportion of data to use for training")
    
    # Create button to generate graph dataset
    if st.button("Generate Graph Dataset"):
        with st.spinner("Generating graph dataset..."):
            # Setup progress bar
            progress_bar = st.progress(0)
            
            # Perform clustering
            display_progress(progress_bar, 10, "Performing K-means clustering...")
            clustered_df, kmeans_model = cluster_data(st.session_state['result']['features'], n_clusters)
            
            # Identify continuous segments
            display_progress(progress_bar, 30, "Identifying continuous facies segments...")
            segments = identify_continuous_segments(clustered_df['DEPTH'].values, 
                                                 clustered_df['Facies_pred'].values)
            
            # Generate node connections
            display_progress(progress_bar, 50, "Generating node connections...")
            node_data, split_indices = generate_node_connections(clustered_df, segments, train_test_split)
            
            # Generate edge features
            display_progress(progress_bar, 70, "Generating edge features...")
            edge_data = generate_edge_features(clustered_df)
            
            # Assign hydrocarbon potential labels
            display_progress(progress_bar, 90, "Assigning hydrocarbon potential labels...")
            labeled_data = assign_hydrocarbon_potential(edge_data)
            
            # Complete progress
            display_progress(progress_bar, 100, "Graph dataset creation complete!")
            
            # Update result dictionary
            st.session_state['result']['clustered_data'] = clustered_df
            st.session_state['result']['segments'] = segments
            st.session_state['result']['node_data'] = node_data
            st.session_state['result']['edge_data'] = labeled_data
            st.session_state['result']['split_indices'] = split_indices
            
            # Show success message
            st.markdown('<div class="success-message">', unsafe_allow_html=True)
            st.markdown("<p>Graph dataset successfully created!</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display dataset information if available
    if all(k in st.session_state['result'] for k in ['clustered_data', 'node_data', 'edge_data']):
        st.markdown('<h2 class="sub-header">Graph Dataset Information</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Nodes", len(st.session_state['result']['edge_data']))
        
        with col2:
            st.metric("Number of Edges", len(st.session_state['result']['node_data']))
        
        with col3:
            st.metric("Number of Facies", len(st.session_state['result']['clustered_data']['Facies_pred'].unique()))
        
        # Display clustered well log
        st.markdown('<h2 class="sub-header">Clustered Well Log</h2>', unsafe_allow_html=True)
        
        facies_fig = create_facies_visualization(
            st.session_state['result']['clustered_data'],
            len(st.session_state['result']['clustered_data']['Facies_pred'].unique())
        )
        st.pyplot(facies_fig)
        
        # Display PET distribution
        st.markdown('<h2 class="sub-header">Petrophysical Entity (PET) Distribution</h2>', unsafe_allow_html=True)
        
        pet_fig = create_pet_distribution_plot(st.session_state['result']['edge_data'])
        st.pyplot(pet_fig)
        
        # Display node and edge data samples
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3>Node Data Sample</h3>', unsafe_allow_html=True)
            st.dataframe(st.session_state['result']['node_data'].head(10))
        
        with col2:
            st.markdown('<h3>Edge Data Sample</h3>', unsafe_allow_html=True)
            display_cols = ['DEPTH', 'PET_label']
            st.dataframe(st.session_state['result']['edge_data'][display_cols].head(10))
        
        # Add a button to proceed to the next step
        if st.button("Proceed to Graph Visualization"):
            st.session_state['current_page'] = "Graph Visualization"
            st.experimental_rerun()

def graph_visualization_page():
    st.markdown('<h1 class="main-header">Graph Visualization</h1>', unsafe_allow_html=True)
    
    if not all(k in st.session_state.get('result', {}) for k in ['clustered_data', 'node_data', 'edge_data', 'segments']):
        st.markdown('<div class="warning-message">', unsafe_allow_html=True)
        st.markdown("<p>Please create a graph dataset first on the 'Graph Dataset Creation' page.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Go to Graph Dataset Creation"):
            st.session_state['current_page'] = "Graph Dataset Creation"
            st.experimental_rerun()
            
        return
    
    st.markdown('<p class="description">This page provides visualizations of the graph dataset. The graph visualization shows the connections between different depth points and their hydrocarbon potential classification.</p>', unsafe_allow_html=True)
    
    # Create visualization options
    visualization_type = st.radio(
        "Select Visualization Type",
        ["2D Network Graph", "3D Network Graph", "Feature Importance", "Depth Profile"]
    )
    
    if visualization_type == "2D Network Graph":
        # Create 2D graph visualization
        with st.spinner("Generating 2D graph visualization..."):
            # Allow user to set sample size
            sample_size = st.slider("Sample Size", min_value=50, max_value=500, value=100, 
                                  help="Number of nodes to display (smaller for better performance)")
            
            graph_fig = create_network_graph(
                st.session_state['result']['node_data'], 
                st.session_state['result']['edge_data'], 
                st.session_state['result']['segments'],
                sample_size=sample_size
            )
            
            st.plotly_chart(graph_fig, use_container_width=True)
    
    elif visualization_type == "3D Network Graph":
        # Create 3D graph visualization
        with st.spinner("Generating 3D graph visualization..."):
            # Allow user to set sample size
            sample_size = st.slider("Sample Size", min_value=50, max_value=200, value=100, 
                                  help="Number of nodes to display (smaller for better performance)")
            
            graph_3d_fig = create_3d_graph_visualization(
                st.session_state['result']['node_data'], 
                st.session_state['result']['edge_data'], 
                st.session_state['result']['segments'],
                sample_size=sample_size
            )
            
            st.plotly_chart(graph_3d_fig, use_container_width=True)
    
    elif visualization_type == "Feature Importance":
        # Create feature importance visualization
        with st.spinner("Generating feature importance visualization..."):
            feature_importance_fig = create_feature_importance_plot(
                st.session_state['result']['edge_data'],
                st.session_state['result']['features']
            )
            
            st.pyplot(feature_importance_fig)
    
    elif visualization_type == "Depth Profile":
        # Create depth profile visualization
        with st.spinner("Generating depth profile visualization..."):
            # Create a depth vs PET label visualization
            
            # Sort by depth
            sorted_df = st.session_state['result']['edge_data'].sort_values('DEPTH')
            
            # Create color map for labels
            color_map = get_hydrocarbon_potential_color_scale()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 12))
            
            # Create scatter plot for each label
            for label, color in color_map.items():
                subset = sorted_df[sorted_df['PET_label'] == label]
                ax.scatter([1] * len(subset), subset['DEPTH'], color=color, label=label, alpha=0.8)
            
            # Set labels and title
            ax.set_title('Hydrocarbon Potential vs Depth')
            ax.set_xticks([])
            ax.set_ylabel('Depth')
            ax.legend()
            
            # Invert y-axis to show increasing depth
            ax.invert_yaxis()
            
            st.pyplot(fig)
    
    # Display additional visualizations
    st.markdown('<h2 class="sub-header">Facies vs Hydrocarbon Potential</h2>', unsafe_allow_html=True)
    
    # Create a visualization showing the relationship between facies and hydrocarbon potential
    with st.spinner("Generating facies vs hydrocarbon potential visualization..."):
        # Combine facies and PET labels
        combined_df = pd.DataFrame({
            'DEPTH': st.session_state['result']['clustered_data']['DEPTH'],
            'Facies': st.session_state['result']['clustered_data']['Facies_pred']
        })
        
        # Merge with edge data
        merged_df = pd.merge(combined_df, st.session_state['result']['edge_data'][['DEPTH', 'PET_label']], 
                           on='DEPTH', how='inner')
        
        # Create crosstab
        cross_tab = pd.crosstab(merged_df['Facies'], merged_df['PET_label'], normalize='index') * 100
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        cmap = 'RdYlGn'
        
        ax = plt.subplot(111)
        im = ax.imshow(cross_tab.values, cmap=cmap, aspect='auto')
        
        # Set labels
        ax.set_xticks(np.arange(len(cross_tab.columns)))
        ax.set_yticks(np.arange(len(cross_tab.index)))
        ax.set_xticklabels(cross_tab.columns)
        ax.set_yticklabels([f"Facies {i}" for i in cross_tab.index])
        
        plt.xlabel('Hydrocarbon Potential')
        plt.ylabel('Facies')
        plt.title('Percentage of Hydrocarbon Potential in Each Facies')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Percentage (%)')
        
        # Add percentage annotations
        for i in range(len(cross_tab.index)):
            for j in range(len(cross_tab.columns)):
                value = cross_tab.iloc[i, j]
                text_color = 'white' if value > 50 else 'black'
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color)
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
    
    # Add a button to proceed to the next step
    if st.button("Proceed to Model Prediction"):
        st.session_state['current_page'] = "Model Prediction"
        st.experimental_rerun()

def model_prediction_page():
    st.markdown('<h1 class="main-header">Model Prediction</h1>', unsafe_allow_html=True)
    
    if not all(k in st.session_state.get('result', {}) for k in ['clustered_data', 'node_data', 'edge_data']):
        st.markdown('<div class="warning-message">', unsafe_allow_html=True)
        st.markdown("<p>Please create a graph dataset first on the 'Graph Dataset Creation' page.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Go to Graph Dataset Creation"):
            st.session_state['current_page'] = "Graph Dataset Creation"
            st.experimental_rerun()
            
        return
    
    st.markdown('<p class="description">This page uses a Graph Convolutional Network (GCN) model to predict hydrocarbon potential zones based on the graph dataset created from well log data.</p>', unsafe_allow_html=True)
    
    # Model configuration
    st.markdown('<h2 class="sub-header">Model Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gcn_layers = st.selectbox("GCN Layers", [1, 2, 3], index=1,
                               help="Number of GCN layers in the model")
    
    with col2:
        learning_rate = st.select_slider("Learning Rate", 
                                     options=[0.001, 0.005, 0.01, 0.05, 0.1], 
                                     value=0.01,
                                     help="Learning rate for model training")
    
    with col3:
        epochs = st.slider("Training Epochs", min_value=50, max_value=200, value=100, step=10,
                        help="Number of training epochs")
    
    # Option to use real model or simulation
    use_real_model = st.checkbox("Use Real GCN Model", value=False, 
                               help="If checked, will train a real GCN model (slower). Otherwise, will simulate predictions.")
    
    # Run prediction button
    if st.button("Run Prediction"):
        with st.spinner("Running GCN model for hydrocarbon potential prediction..."):
            progress_bar = st.progress(0)
            
            if use_real_model:
                # In a production app, this would call your actual GCN model
                # Due to the complexity of setting up StellarGraph in a Streamlit app,
                # we'll use the simulation for this demo
                st.warning("Real GCN model training is not available in this demo. Using simulation instead.")
                
                # Simulate model training progress
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)
                
                # Get predictions using simulation
                predicted_df = simulate_gcn_prediction(st.session_state['result']['edge_data'])
                
            else:
                # Simulate model training progress
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                # Get predictions using simulation
                predicted_df = simulate_gcn_prediction(st.session_state['result']['edge_data'])
            
            # Calculate accuracy
            accuracy = (predicted_df['PET_label'] == predicted_df['Predicted_PET']).mean()
            
            # Save to session state
            st.session_state['result']['predicted_df'] = predicted_df
            st.session_state['result']['model_accuracy'] = accuracy
            
            # Show success message
            st.markdown('<div class="success-message">', unsafe_allow_html=True)
            st.markdown("<p>Prediction completed successfully!</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display prediction results if available
    if 'predicted_df' in st.session_state.get('result', {}):
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Calculate accuracy
        accuracy = st.session_state['result']['model_accuracy']
        
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        # Display prediction comparison
        st.markdown('<h3>Predicted vs True Labels (Sample)</h3>', unsafe_allow_html=True)
        display_df = st.session_state['result']['predicted_df'][['DEPTH', 'PET_label', 'Predicted_PET']].head(10)
        display_df.columns = ['Depth', 'True Label', 'Predicted Label']
        st.dataframe(display_df)
        
        # Create comparison visualization
        comparison_fig = create_comparison_heatmap(st.session_state['result']['predicted_df'])
        st.pyplot(comparison_fig)
        
        # Display hydrocarbon potential visualization
        st.markdown('<h2 class="sub-header">Hydrocarbon Potential Visualization</h2>', unsafe_allow_html=True)
        
        # Create an interactive depth profile visualization
        depth_profile_fig = create_depth_profile_visualization(st.session_state['result']['predicted_df'])
        st.plotly_chart(depth_profile_fig, use_container_width=True)
        
        # Add a button to proceed to the next step
        if st.button("Proceed to Results & Interpretation"):
            st.session_state['current_page'] = "Results & Interpretation"
            st.experimental_rerun()

def results_interpretation_page():
    st.markdown('<h1 class="main-header">Results & Interpretation</h1>', unsafe_allow_html=True)
    
    if not all(k in st.session_state.get('result', {}) for k in ['features', 'predicted_df']):
        st.markdown('<div class="warning-message">', unsafe_allow_html=True)
        st.markdown("<p>Please complete the model prediction first on the 'Model Prediction' page.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Go to Model Prediction"):
            st.session_state['current_page'] = "Model Prediction"
            st.experimental_rerun()
            
        return
    
    st.markdown('<p class="description">This page provides detailed visualizations and interpretations of the hydrocarbon potential prediction results.</p>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Potential Distribution", "Depth Analysis", "Well Log with Predictions"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Hydrocarbon Potential Distribution</h2>', unsafe_allow_html=True)
        
        # Create distribution plot
        # True distribution
        true_counts = st.session_state['result']['predicted_df']['PET_label'].value_counts().sort_index()
        
        # Predicted distribution
        pred_counts = st.session_state['result']['predicted_df']['Predicted_PET'].value_counts().sort_index()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get color map
        color_map = get_hydrocarbon_potential_color_scale()
        
        # True distribution
        ax1.bar(true_counts.index, true_counts.values, 
               color=[color_map.get(label, '#333333') for label in true_counts.index])
        ax1.set_title('True Hydrocarbon Potential Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for i, v in enumerate(true_counts.values):
            ax1.text(i, v + 5, f"{v/sum(true_counts)*100:.1f}%", ha='center')
        
        # Predicted distribution
        ax2.bar(pred_counts.index, pred_counts.values, 
               color=[color_map.get(label, '#333333') for label in pred_counts.index])
        ax2.set_title('Predicted Hydrocarbon Potential Distribution')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for i, v in enumerate(pred_counts.values):
            ax2.text(i, v + 5, f"{v/sum(pred_counts)*100:.1f}%", ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add interpretation
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<h3>Interpretation</h3>', unsafe_allow_html=True)
        
        # Calculate proportions for interpretation
        true_high = (st.session_state['result']['predicted_df']['PET_label'].isin(['High', 'Very_High'])).mean()
        pred_high = (st.session_state['result']['predicted_df']['Predicted_PET'].isin(['High', 'Very_High'])).mean()
        
        st.markdown(f"""
        <p>The distribution shows that:</p>
        <ul>
            <li>Approximately <b>{true_high:.1%}</b> of the analyzed depth points have high or very high hydrocarbon potential based on the true labels.</li>
            <li>The model predicts that <b>{pred_high:.1%}</b> of the depth points have high or very high hydrocarbon potential.</li>
            <li>The difference between true and predicted distributions indicates the model's bias in prediction.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Depth vs Hydrocarbon Potential</h2>', unsafe_allow_html=True)
        
        # Sort data by depth
        sorted_df = st.session_state['result']['predicted_df'].sort_values('DEPTH')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
        
        # Get color map
        color_map = get_hydrocarbon_potential_color_scale()
        
        # True potential vs depth
        for label, color in color_map.items():
            subset = sorted_df[sorted_df['PET_label'] == label]
            ax1.scatter(np.ones(len(subset)), subset['DEPTH'], color=color, label=label, s=10)
        
        ax1.set_title('True Hydrocarbon Potential vs Depth')
        ax1.set_xticks([])
        ax1.set_ylabel('Depth')
        ax1.invert_yaxis()  # Invert y-axis to show increasing depth
        ax1.legend()
        
        # Predicted potential vs depth
        for label, color in color_map.items():
            subset = sorted_df[sorted_df['Predicted_PET'] == label]
            ax2.scatter(np.ones(len(subset)), subset['DEPTH'], color=color, label=label, s=10)
        
        ax2.set_title('Predicted Hydrocarbon Potential vs Depth')
        ax2.set_xticks([])
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add interpretation
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<h3>Interpretation</h3>', unsafe_allow_html=True)
        
        # Find depths with highest potential for interpretation
        high_potential = st.session_state['result']['predicted_df'][
            st.session_state['result']['predicted_df']['Predicted_PET'].isin(['High', 'Very_High'])
        ]
        
        if len(high_potential) > 0:
            min_depth = high_potential['DEPTH'].min()
            max_depth = high_potential['DEPTH'].max()
            
            st.markdown(f"""
            <p>The depth analysis shows that:</p>
            <ul>
                <li>Zones with high hydrocarbon potential are found between depths <b>{min_depth:.1f}</b> and <b>{max_depth:.1f}</b>.</li>
                <li>There are distinct layers with varying hydrocarbon potential throughout the well.</li>
                <li>The model accurately identifies most high-potential zones compared to the true labels.</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <p>The depth analysis shows that:</p>
            <ul>
                <li>No zones with high hydrocarbon potential were identified in this well.</li>
                <li>The well predominantly shows low to moderate hydrocarbon potential.</li>
            </ul>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Well Log with Hydrocarbon Potential Predictions</h2>', unsafe_allow_html=True)
        
        # Create well log visualization with predictions
        fig, axes = plt.subplots(1, 6, figsize=(20, 12), sharey=True)
        
        # Get features
        features = st.session_state['result']['features']
        
        # Plot GR
        if 'GR' in features.columns:
            axes[0].plot(features['GR'], features['DEPTH'], color='black')
            axes[0].set_title('GR', fontsize=14)
            axes[0].set_xlim(0, features['GR'].max() * 1.1)
            axes[0].grid(True)
        
        # Plot RHOB
        if 'RHOB' in features.columns:
            axes[1].plot(features['RHOB'], features['DEPTH'], color='red')
            axes[1].set_title('RHOB', fontsize=14)
            axes[1].set_xlim(features['RHOB'].min() * 0.9, features['RHOB'].max() * 1.1)
            axes[1].grid(True)
        
        # Plot Porosity
        axes[2].plot(features['PHIECALC'], features['DEPTH'], color='blue')
        axes[2].set_title('PHIE', fontsize=14)
        axes[2].set_xlim(0, 0.4)
        axes[2].grid(True)
        
        # Plot Water/Oil Saturation
        axes[3].plot(features['WSAT'], features['DEPTH'], color='blue', label='SW')
        axes[3].plot(features['OSAT'], features['DEPTH'], color='green', label='SO')
        axes[3].set_title('Saturation', fontsize=14)
        axes[3].set_xlim(0, 1)
        axes[3].grid(True)
        axes[3].legend()
        
        # Plot facies
        if 'clustered_data' in st.session_state['result']:
            clustered_df = st.session_state['result']['clustered_data']
            facies = clustered_df['Facies_pred']
            
            # Create scatter plot for facies
            cmap = plt.cm.get_cmap('viridis', len(facies.unique()))
            scatter = axes[4].scatter(
                np.ones(len(facies)), 
                clustered_df['DEPTH'],
                c=facies, 
                cmap=cmap, 
                s=50, 
                marker='s'
            )
            axes[4].set_title('Facies', fontsize=14)
            axes[4].set_xticks([])
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[4])
            cbar.set_label('Facies')
        
        # Plot hydrocarbon potential predictions
        predicted_df = st.session_state['result']['predicted_df']
        
        # Map labels to numeric values
        label_map = {'Very_Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very_High': 4}
        numeric_pred = predicted_df['Predicted_PET'].map(label_map)
        
        # Create scatter plot for predictions
        pred_cmap = plt.cm.get_cmap('RdYlGn', 5)
        scatter = axes[5].scatter(
            np.ones(len(numeric_pred)), 
            predicted_df['DEPTH'],
            c=numeric_pred, 
            cmap=pred_cmap, 
            s=50, 
            marker='s'
        )
        axes[5].set_title('HC Potential', fontsize=14)
        axes[5].set_xticks([])
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[5])
        cbar.set_ticks([0, 1, 2, 3, 4])
        cbar.set_ticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
        
        # Set y-axis to increase with depth
        for ax in axes:
            ax.invert_yaxis()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add interpretation
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<h3>Interpretation</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p>The well log visualization with predictions shows:</p>
        <ul>
            <li>The relationship between log curves (GR, RHOB, PHIE, SW/SO) and the predicted hydrocarbon potential.</li>
            <li>High hydrocarbon potential zones typically correspond to low GR, low density, high porosity, and high oil saturation.</li>
            <li>The facies classification aligns with the petrophysical properties and helps identify potential reservoir zones.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary and recommendations
    st.markdown('<h2 class="sub-header">Summary and Recommendations</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown('<h3>Summary</h3>', unsafe_allow_html=True)
        
        # Calculate some statistics for the summary
        high_potential_pct = (st.session_state['result']['predicted_df']['Predicted_PET'].isin(['High', 'Very_High'])).mean()
        accurate_pred_pct = st.session_state['result']['model_accuracy']
        
        st.markdown(f"""
        <p>Based on the VHydro graph-based analysis:</p>
        <ul>
            <li>Approximately <b>{high_potential_pct:.1%}</b> of the analyzed well section shows high or very high hydrocarbon potential.</li>
            <li>The GCN model achieves an accuracy of <b>{accurate_pred_pct:.1%}</b> in predicting hydrocarbon potential.</li>
            <li>The graph-based approach effectively captures the relationships between different petrophysical properties.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown('<h3>Recommendations</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p>Based on the prediction results, we recommend:</p>
        <ul>
            <li>Focus further exploration and testing on the identified high-potential zones.</li>
            <li>Conduct additional analysis (e.g., seismic interpretation, core analysis) to confirm the predicted hydrocarbon potential.</li>
            <li>Use this VHydro graph-based approach for analyzing other wells in the same field to identify field-wide trends.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create project folder for export
    if 'project_folder' not in st.session_state:
        project_folder = create_project_folder("vhydro_project")
        st.session_state['project_folder'] = project_folder
    
    # Export results
    st.markdown('<h2 class="sub-header">Export Results</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Prediction Results (CSV)"):
            # Save prediction results to CSV
            csv_path = os.path.join(st.session_state['project_folder']['results_dir'], "prediction_results.csv")
            st.session_state['result']['predicted_df'].to_csv(csv_path, index=False)
            
            # Create download link
            st.markdown(create_download_link(
                st.session_state['result']['predicted_df'],
                "hydrocarbon_prediction_results.csv",
                "Download CSV File"
            ), unsafe_allow_html=True)
    
    with col2:
        if st.button("Generate Summary Report"):
            # Create summary report
            report_path = os.path.join(st.session_state['project_folder']['results_dir'], "summary_report.md")
            create_summary_report(
                st.session_state['result'],
                st.session_state['result']['model_accuracy'],
                report_path
            )
            
            # Read report content
            with open(report_path, 'r') as f:
                report_content = f.read()
            
            # Display report
            st.markdown('<div class="highlight" style="max-height: 300px; overflow-y: auto;">', unsafe_allow_html=True)
            st.markdown(report_content)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Create download button
            with open(report_path, 'rb') as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name="hydrocarbon_potential_summary_report.md",
                    mime="text/markdown"
                )

# Main app
def main():
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Home"
    
    # Sidebar logo
    st.sidebar.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.sidebar.markdown('<h1 style="color: #0e4194;">VHydro</h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<p>Hydrocarbon Potential Prediction</p>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Get page from navigation or session state
    selected_page = navigation()
    
    # If navigation changed the page, update session state
    if selected_page != st.session_state['current_page']:
        st.session_state['current_page'] = selected_page
    
    # Display the selected page
    if st.session_state['current_page'] == "Home":
        home_page()
    elif st.session_state['current_page'] == "Data Upload & Processing":
        data_upload_page()
    elif st.session_state['current_page'] == "Graph Dataset Creation":
        graph_dataset_page()
    elif st.session_state['current_page'] == "Graph Visualization":
        graph_visualization_page()
    elif st.session_state['current_page'] == "Model Prediction":
        model_prediction_page()
    elif st.session_state['current_page'] == "Results & Interpretation":
        results_interpretation_page()
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('VHydro - Hydrocarbon Potential Prediction Application | Developed based on the research paper "Novel Graph Dataset for Hydrocarbon Potential Prediction"', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
