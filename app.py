#!/usr/bin/env python3
"""
VHydro - Hydrocarbon Potential Prediction
Main Streamlit application
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import time
from PIL import Image
import plotly.graph_objects as go

# Add the app directory to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'app'))

# Configure page settings
st.set_page_config(
    page_title="VHydro - Hydrocarbon Potential Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import app modules - make imports conditional to handle missing modules gracefully
try:
    from app.data_processing import process_las_file, evaluate_clusters, extract_well_metadata
    from app.data_processing import cluster_data, identify_continuous_segments, generate_node_connections
    from app.data_processing import generate_edge_features, assign_hydrocarbon_potential
    from app.gcn_model import simulate_gcn_prediction
    from app.utils import (create_download_link, save_uploaded_file, create_project_folder, 
                           display_progress, get_hydrocarbon_potential_color_scale, create_summary_report)
    
    # Import visualization modules that might be more problematic
    try:
        from app.network_visualization import (create_network_graph, create_facies_visualization, 
                                              create_pet_distribution_plot, create_feature_importance_plot,
                                              create_3d_graph_visualization, create_comparison_heatmap,
                                              create_depth_profile_visualization)
    except ImportError as viz_error:
        st.warning(f"Some visualization modules could not be loaded. Some features may be limited: {viz_error}")
        
        # Define fallback simple visualization functions
        def create_facies_visualization(cluster_df, n_clusters):
            fig, ax = plt.subplots(figsize=(3, 10))
            cmap = plt.cm.get_cmap('viridis', n_clusters)
            for i in range(n_clusters):
                subset = cluster_df[cluster_df['Facies_pred'] == i]
                ax.scatter(np.ones(len(subset)), subset['DEPTH'], color=cmap(i), label=f'Facies {i}', s=10)
            ax.set_title('Facies Classification')
            ax.set_xticks([])
            ax.set_ylabel('Depth')
            ax.invert_yaxis()
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            return fig
            
        def create_pet_distribution_plot(edge_data):
            pet_counts = edge_data['PET_label'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
            bars = ax.bar(pet_counts.index, pet_counts.values, color=colors)
            ax.set_title('Distribution of Hydrocarbon Potential Classifications')
            ax.set_ylabel('Count')
            ax.set_xlabel('Hydrocarbon Potential')
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig
        
        def create_comparison_heatmap(predicted_df):
            comparison = pd.DataFrame({
                'True': predicted_df['PET_label'],
                'Predicted': predicted_df['Predicted_PET']
            })
            confusion = pd.crosstab(comparison['True'], comparison['Predicted'])
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', linewidths=.5)
            ax.set_title('Confusion Matrix: True vs Predicted Hydrocarbon Potential')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            plt.tight_layout()
            return plt.gcf()
        
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Some functions may not be available.")

# Add custom CSS
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

# Import the main application logic
from deploy_app import (
    navigation, home_page, data_upload_page, graph_dataset_page,
    graph_visualization_page, model_prediction_page, results_interpretation_page
)

# Main function for the app
def main():
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Home"
    
    # Create modules dict with all the imported functions for passing to page functions
    modules = {
        'process_las_file': process_las_file,
        'evaluate_clusters': evaluate_clusters,
        'extract_well_metadata': extract_well_metadata,
        'cluster_data': cluster_data,
        'identify_continuous_segments': identify_continuous_segments,
        'generate_node_connections': generate_node_connections,
        'generate_edge_features': generate_edge_features,
        'assign_hydrocarbon_potential': assign_hydrocarbon_potential,
        'simulate_gcn_prediction': simulate_gcn_prediction,
        'create_download_link': create_download_link,
        'save_uploaded_file': save_uploaded_file,
        'create_project_folder': create_project_folder,
        'display_progress': display_progress,
        'get_hydrocarbon_potential_color_scale': get_hydrocarbon_potential_color_scale,
        'create_summary_report': create_summary_report,
        'create_facies_visualization': create_facies_visualization,
        'create_pet_distribution_plot': create_pet_distribution_plot
    }
    
    # Add the visualization functions if they were imported
    if 'create_network_graph' in globals():
        modules['create_network_graph'] = create_network_graph
        modules['create_3d_graph_visualization'] = create_3d_graph_visualization
        modules['create_feature_importance_plot'] = create_feature_importance_plot
        modules['create_depth_profile_visualization'] = create_depth_profile_visualization
    
    if 'create_comparison_heatmap' in globals():
        modules['create_comparison_heatmap'] = create_comparison_heatmap
    
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
        home_page(modules)
    elif st.session_state['current_page'] == "Data Upload & Processing":
        data_upload_page(modules)
    elif st.session_state['current_page'] == "Graph Dataset Creation":
        graph_dataset_page(modules)
    elif st.session_state['current_page'] == "Graph Visualization":
        graph_visualization_page(modules)
    elif st.session_state['current_page'] == "Model Prediction":
        model_prediction_page(modules)
    elif st.session_state['current_page'] == "Results & Interpretation":
        results_interpretation_page(modules)
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('VHydro - Hydrocarbon Potential Prediction Application | Developed based on the research paper "Novel Graph Dataset for Hydrocarbon Potential Prediction"', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Entry point for the application
if __name__ == "__main__":
    main()
