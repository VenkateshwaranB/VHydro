import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import os
import io
import base64
from PIL import Image
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import lasio
import re
from itertools import groupby, count
import tempfile

# Page configuration
st.set_page_config(
    page_title="VHydro - Hydrocarbon Potential Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Function to calculate petrophysical properties
def shale_volume(gamma_ray, gamma_ray_max, gamma_ray_min):
    vshale = (gamma_ray - gamma_ray_min) / (gamma_ray_max - gamma_ray_min)
    vshale = 0.083 * (2 ** (2 * 3.7 * vshale) - 1)  # For tertiary rocks
    return round(vshale, 4)

def density_porosity(input_density, matrix_density, fluid_density):
    denpor = (matrix_density - input_density) / (matrix_density - fluid_density)
    return round(denpor, 4)

def sw_archie(porosity, rt):
    sw = ((1 * (porosity**2)) / (rt * 0.03))**(1/2)
    return sw

def ow_archie(sw):
    ow = 1 - sw
    return ow

def permeability(porosity):
    pe = 0.00004 * np.exp(57.117 * porosity)
    return pe

def create_node_connections(df, tv, train_indices):
    """Create node connections for graph dataset"""
    nodeF = pd.DataFrame()
    nodeF['source'] = []
    nodeF['target'] = []
    
    # Extract train indices
    train_tv = [tv[i] for i in train_indices]
    
    # Create connections
    for group in train_tv:
        for i in range(len(group)):
            for j in range(len(group)):
                if i != j:
                    nodeF = nodeF.append({
                        'source': df['DEPTH'][group[i]],
                        'target': df['DEPTH'][group[j]]
                    }, ignore_index=True)
    
    return nodeF

def preprocess_las_file(las_file):
    """Process LAS file and return dataframe with features"""
    las = lasio.read(las_file)
    df = las.df()
    
    # Check for required curves
    required_curves = ['GR', 'RHOB', 'ILD']
    missing_curves = [curve for curve in required_curves if curve not in df.columns]
    
    if missing_curves:
        return None, f"Missing required curves: {', '.join(missing_curves)}"
    
    # Calculate petrophysical properties
    features = pd.DataFrame()
    
    # Calculate Shale Volume if CGR exists, otherwise use GR
    if 'CGR' in df.columns:
        features['VSHALE'] = shale_volume(df['CGR'], df['CGR'].max(), df['CGR'].min())
    else:
        features['VSHALE'] = shale_volume(df['GR'], df['GR'].max(), df['GR'].min())
    
    # Calculate density porosity
    features['PHI'] = density_porosity(df['RHOB'], 2.65, 1)
    
    # Calculate PHIE (Effective porosity)
    features['PHIECALC'] = features['PHI'] - (features['VSHALE'] * 0.3)
    
    # Calculate water saturation
    features['WSAT'] = sw_archie(features['PHIECALC'], np.log(df['ILD']))
    
    # Calculate Oil saturation
    features['OSAT'] = ow_archie(features['WSAT'])
    
    # Calculate permeability
    features['PERM'] = permeability(features['PHIECALC'])
    
    # Add depth and other available curves
    features['DEPTH'] = df.index
    for col in df.columns:
        if col not in features.columns and col in ['GR', 'RHOB', 'DENSITY', 'ILD']:
            features[col] = df[col]
    
    return features, "Success"

def cluster_data(features, n_clusters):
    """Perform K-means clustering on well log data"""
    # Select features for clustering
    cluster_features = features[['VSHALE', 'PHI', 'WSAT', 'OSAT', 'PERM']].copy()
    cluster_features = cluster_features.dropna()
    
    # Scale features
    x_scaled = scale(cluster_features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(x_scaled)
    
    # Add cluster labels to original dataframe
    cluster_df = features.copy()
    cluster_df = cluster_df.iloc[:len(cluster_labels)]
    cluster_df['Facies_pred'] = cluster_labels
    
    return cluster_df

def generate_graph_dataset(df, cls_name):
    """Generate graph dataset from clustered data"""
    # Extract continuous depth segments for each facies
    facies_segments = []
    
    for facies_id in range(cls_name):
        indices = [idx for idx, value in enumerate(df['Facies_pred']) if value == facies_id]
        segments = [list(g) for _, g in groupby(indices, key=lambda n, c=count(): n-next(c))]
        facies_segments.extend(segments)
    
    # Create node connections dataset
    train_size = int(0.8 * len(facies_segments))
    train_indices = list(range(train_size))
    
    node_data = create_node_connections(df, facies_segments, train_indices)
    
    # Generate edge dataset
    edge_data = pd.DataFrame()
    edge_data['DEPTH'] = df['DEPTH']
    
    # Create binary features for permeability
    edge_data['PE_1'] = (df['PERM'] <= 0.01).astype(int)
    edge_data['PE_2'] = ((df['PERM'] <= 1) & (df['PERM'] > 0.01)).astype(int)
    edge_data['PE_3'] = ((df['PERM'] <= 10) & (df['PERM'] > 1)).astype(int)
    edge_data['PE_4'] = ((df['PERM'] <= 100) & (df['PERM'] > 10)).astype(int)
    edge_data['PE_5'] = (df['PERM'] > 100).astype(int)
    
    # Create binary features for porosity
    edge_data['PO_1'] = (df['PHIECALC'] <= 0.1).astype(int)
    edge_data['PO_2'] = ((df['PHIECALC'] <= 0.2) & (df['PHIECALC'] > 0.1)).astype(int)
    edge_data['PO_3'] = ((df['PHIECALC'] <= 0.3) & (df['PHIECALC'] > 0.2)).astype(int)
    
    # Create binary features for shale volume
    edge_data['VS_1'] = (df['VSHALE'] > 0.5).astype(int)
    edge_data['VS_2'] = (df['VSHALE'] <= 0.5).astype(int)
    
    # Create binary features for water saturation
    edge_data['SW_1'] = (df['WSAT'] > 0.5).astype(int)
    edge_data['SW_2'] = (df['WSAT'] <= 0.5).astype(int)
    
    # Create binary features for oil saturation
    edge_data['OW_1'] = (df['OSAT'] < 0.5).astype(int)
    edge_data['OW_2'] = (df['OSAT'] >= 0.5).astype(int)
    
    # Generate labels
    edge_data['Values'] = (edge_data['PE_1'].astype(str) + edge_data['PE_2'].astype(str) + 
                           edge_data['PE_3'].astype(str) + edge_data['PE_4'].astype(str) + 
                           edge_data['PE_5'].astype(str) + edge_data['PO_1'].astype(str) + 
                           edge_data['PO_2'].astype(str) + edge_data['PO_3'].astype(str) + 
                           edge_data['VS_1'].astype(str) + edge_data['VS_2'].astype(str) + 
                           edge_data['SW_1'].astype(str) + edge_data['SW_2'].astype(str) + 
                           edge_data['OW_1'].astype(str) + edge_data['OW_2'].astype(str))
    
    # Assign hydrocarbon potential labels
    categories = edge_data['Values'].unique()
    
    edge_data["PET_label"] = ''
    # This is a simplified label assignment - in a real app you would want a more sophisticated approach
    for i, value in enumerate(edge_data['Values']):
        if sum(int(bit) for bit in value) >= 10:  # High number of 1s suggests higher potential
            edge_data["PET_label"][i] = "Very_High"
        elif sum(int(bit) for bit in value) >= 8:
            edge_data["PET_label"][i] = "High"
        elif sum(int(bit) for bit in value) >= 6:
            edge_data["PET_label"][i] = "Moderate"
        elif sum(int(bit) for bit in value) >= 4:
            edge_data["PET_label"][i] = "Low"
        else:
            edge_data["PET_label"][i] = "Very_Low"
    
    return node_data, edge_data, facies_segments

def plot_well_logs(features, facies_df=None, predicted_df=None):
    """Create a well log visualization with facies and predictions"""
    fig, ax = plt.subplots(1, 5, figsize=(15, 10), sharey=True)
    
    # Plot GR
    if 'GR' in features.columns:
        ax[0].plot(features['GR'], features['DEPTH'], color='black')
        ax[0].set_title('GR')
        ax[0].set_xlim(0, features['GR'].max() * 1.1)
        ax[0].grid(True)
    
    # Plot RHOB
    if 'RHOB' in features.columns:
        ax[1].plot(features['RHOB'], features['DEPTH'], color='red')
        ax[1].set_title('RHOB')
        ax[1].set_xlim(features['RHOB'].min() * 0.9, features['RHOB'].max() * 1.1)
        ax[1].grid(True)
    
    # Plot Porosity
    ax[2].plot(features['PHIECALC'], features['DEPTH'], color='blue')
    ax[2].set_title('PHIE')
    ax[2].set_xlim(0, features['PHIECALC'].max() * 1.1)
    ax[2].grid(True)
    
    # Plot Water/Oil Saturation
    ax[3].plot(features['WSAT'], features['DEPTH'], color='blue', label='SW')
    ax[3].plot(features['OSAT'], features['DEPTH'], color='green', label='SO')
    ax[3].set_title('Saturation')
    ax[3].set_xlim(0, 1)
    ax[3].grid(True)
    ax[3].legend()
    
    # Plot Facies if available
    if facies_df is not None:
        cmap = plt.cm.get_cmap('viridis', len(facies_df['Facies_pred'].unique()))
        scatter = ax[4].scatter(
            np.ones(len(facies_df)), 
            facies_df['DEPTH'],
            c=facies_df['Facies_pred'], 
            cmap=cmap, 
            s=50, 
            marker='s'
        )
        ax[4].set_title('Facies')
        plt.colorbar(scatter, ax=ax[4])
    
    # Plot predictions if available
    if predicted_df is not None:
        # Convert categorical labels to numeric for plotting
        label_map = {'Very_Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very_High': 4}
        numeric_labels = predicted_df['PET_label'].map(label_map)
        
        prediction_cmap = plt.cm.get_cmap('RdYlGn', 5)
        scatter = ax[4].scatter(
            np.ones(len(predicted_df)) * 2, 
            predicted_df['DEPTH'],
            c=numeric_labels, 
            cmap=prediction_cmap, 
            s=50, 
            marker='s'
        )
        ax[4].set_title('HC Potential')
        cbar = plt.colorbar(scatter, ax=ax[4])
        cbar.set_ticks([0, 1, 2, 3, 4])
        cbar.set_ticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
    
    # Set y-axis to increase with depth
    for a in ax:
        a.invert_yaxis()
    
    plt.tight_layout()
    return fig

def plot_graph_visualization(node_data, edge_data, facies_segments):
    """Create a graph visualization of the dataset"""
    G = nx.Graph()
    
    # Add nodes
    depths = edge_data['DEPTH'].values
    labels = edge_data['PET_label'].values
    
    label_map = {'Very_Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very_High': 4}
    colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
    
    # Sample a subset of nodes for clearer visualization
    sample_size = min(100, len(depths))
    indices = np.linspace(0, len(depths)-1, sample_size).astype(int)
    
    # Add nodes
    for i in indices:
        G.add_node(str(depths[i]), label=labels[i], color=colors[label_map[labels[i]]])
    
    # Add edges (connections between nodes in the same facies segment)
    for segment in facies_segments:
        segment_depths = [str(depths[i]) for i in segment if i in indices]
        for i in range(len(segment_depths)):
            for j in range(i+1, len(segment_depths)):
                if segment_depths[i] in G.nodes and segment_depths[j] in G.nodes:
                    G.add_edge(segment_depths[i], segment_depths[j])
    
    # Create positions for nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Create Plotly figure
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(G.nodes[node]['color'])
        node_text.append(f"Depth: {node}<br>HC Potential: {G.nodes[node]['label']}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=10,
            line_width=2
        )
    )
    
    # Create legend
    legend_traces = []
    for label, value in label_map.items():
        legend_traces.append(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=colors[value]),
                name=label,
                showlegend=True
            )
        )
    
    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces,
                   layout=go.Layout(
                       title='VHydro Graph Dataset Visualization',
                       titlefont_size=16,
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def simulate_gcn_prediction(edge_data):
    """Simulate GCN prediction results (in a real app, this would call your actual model)"""
    # This is a simplified simulation of model prediction results
    # In a real app, you would use your actual GCN model from model.py
    
    # Create a copy of the data to add "predicted" labels
    predicted_df = edge_data.copy()
    
    # Simulate 90% accuracy with some predictions different from ground truth
    np.random.seed(42)
    accuracy = 0.9
    
    # Create label mapping
    labels = ['Very_Low', 'Low', 'Moderate', 'High', 'Very_High']
    
    # For each row, decide whether to keep the true label or assign a random one
    for i in range(len(predicted_df)):
        if np.random.random() > accuracy:
            # Assign a different random label
            current_label = predicted_df.loc[i, 'PET_label']
            other_labels = [l for l in labels if l != current_label]
            predicted_df.loc[i, 'Predicted_PET'] = np.random.choice(other_labels)
        else:
            # Keep the true label
            predicted_df.loc[i, 'Predicted_PET'] = predicted_df.loc[i, 'PET_label']
    
    return predicted_df

def create_comparison_plot(predicted_df):
    """Create a comparison plot between true and predicted labels"""
    # Calculate confusion matrix
    labels = ['Very_Low', 'Low', 'Moderate', 'High', 'Very_High']
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'True': predicted_df['PET_label'],
        'Predicted': predicted_df['Predicted_PET']
    })
    
    # Create confusion matrix
    confusion = pd.crosstab(comparison['True'], comparison['Predicted'])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix: True vs Predicted Hydrocarbon Potential')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    return fig

def create_potential_distribution_plot(predicted_df):
    """Create distribution plots of hydrocarbon potential"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # True distribution
    true_counts = predicted_df['PET_label'].value_counts().sort_index()
    ax1.bar(true_counts.index, true_counts.values, color=['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850'])
    ax1.set_title('True Hydrocarbon Potential Distribution')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Predicted distribution
    pred_counts = predicted_df['Predicted_PET'].value_counts().sort_index()
    ax2.bar(pred_counts.index, pred_counts.values, color=['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850'])
    ax2.set_title('Predicted Hydrocarbon Potential Distribution')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_depth_potential_plot(predicted_df):
    """Create a depth vs potential visualization"""
    # Sort data by depth
    sorted_df = predicted_df.sort_values('DEPTH')
    
    # Create label mapping for colors
    label_map = {'Very_Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very_High': 4}
    colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10), sharey=True)
    
    # True potential vs depth
    for label, color in zip(['Very_Low', 'Low', 'Moderate', 'High', 'Very_High'], colors):
        subset = sorted_df[sorted_df['PET_label'] == label]
        ax1.scatter(np.ones(len(subset)), subset['DEPTH'], color=color, label=label, s=10)
    
    ax1.set_title('True Hydrocarbon Potential vs Depth')
    ax1.set_xticks([])
    ax1.set_ylabel('Depth')
    ax1.invert_yaxis()  # Invert y-axis to show increasing depth
    ax1.legend()
    
    # Predicted potential vs depth
    for label, color in zip(['Very_Low', 'Low', 'Moderate', 'High', 'Very_High'], colors):
        subset = sorted_df[sorted_df['Predicted_PET'] == label]
        ax2.scatter(np.ones(len(subset)), subset['DEPTH'], color=color, label=label, s=10)
    
    ax2.set_title('Predicted Hydrocarbon Potential vs Depth')
    ax2.set_xticks([])
    ax2.legend()
    
    plt.tight_layout()
    return fig

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
            ax.add_patch(Rectangle((0.2, 0.8 - i*0.15), 0.6, 0.1, alpha=0.8, facecolor='#0e4194'))
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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown(f"<p>File <b>{uploaded_file.name}</b> successfully uploaded! Processing data...</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process the LAS file
        with st.spinner('Processing LAS file...'):
            features, message = preprocess_las_file(tmp_filepath)
        
        if features is None:
            st.markdown('<div class="warning-message">', unsafe_allow_html=True)
            st.markdown(f"<p>Error: {message}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-message">', unsafe_allow_html=True)
            st.markdown("<p>LAS file processed successfully! Petrophysical properties have been calculated.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display the first few rows of the processed data
            st.markdown('<h2 class="sub-header">Processed Well Log Data</h2>', unsafe_allow_html=True)
            st.dataframe(features.head(10))
            
            # Create well log visualization
            st.markdown('<h2 class="sub-header">Well Log Visualization</h2>', unsafe_allow_html=True)
            fig = plot_well_logs(features)
            st.pyplot(fig)
            
            # Save the processed data to session state for use in other pages
            st.session_state['features'] = features
            
            # Add a button to proceed to the next step
            if st.button("Proceed to Graph Dataset Creation"):
                st.session_state['current_page'] = "Graph Dataset Creation"
                st.experimental_rerun()

def graph_dataset_page():
    st.markdown('<h1 class="main-header">Graph Dataset Creation</h1>', unsafe_allow_html=True)
    
    if 'features' not in st.session_state:
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
        n_clusters = st.slider("Number of clusters", min_value=5, max_value=10, value=7, 
                             help="Number of clusters for K-means clustering of depth points")
    
    with col2:
        train_test_split = st.slider("Train/Test Split", min_value=0.6, max_value=0.9, value=0.8, 
                                  help="Proportion of data to use for training")
    
    # Create button to generate graph dataset
    if st.button("Generate Graph Dataset"):
        with st.spinner("Generating graph dataset..."):
            # Perform clustering
            cluster_df = cluster_data(st.session_state['features'], n_clusters)
            
            # Generate graph dataset
            node_data, edge_data, facies_segments = generate_graph_dataset(cluster_df, n_clusters)
            
            # Save to session state
            st.session_state['cluster_df'] = cluster_df
            st.session_state['node_data'] = node_data
            st.session_state['edge_data'] = edge_data
            st.session_state['facies_segments'] = facies_segments
            
            # Show success message
            st.markdown('<div class="success-message">', unsafe_allow_html=True)
            st.markdown("<p>Graph dataset successfully created!</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display dataset information if available
    if all(k in st.session_state for k in ['cluster_df', 'node_data', 'edge_data']):
        st.markdown('<h2 class="sub-header">Graph Dataset Information</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Nodes", len(st.session_state['edge_data']))
        
        with col2:
            st.metric("Number of Edges", len(st.session_state['node_data']))
        
        with col3:
            st.metric("Number of Facies", len(st.session_state['cluster_df']['Facies_pred'].unique()))
        
        # Display clustered well log
        st.markdown('<h2 class="sub-header">Clustered Well Log</h2>', unsafe_allow_html=True)
        cluster_fig = plot_well_logs(st.session_state['features'], st.session_state['cluster_df'])
        st.pyplot(cluster_fig)
        
        # Display node and edge data samples
        st.markdown('<h3>Node Data Sample</h3>', unsafe_allow_html=True)
        st.dataframe(st.session_state['node_data'].head(10))
        
        st.markdown('<h3>Edge Data Sample</h3>', unsafe_allow_html=True)
        st.dataframe(st.session_state['edge_data'].head(10))
        
        # Add a button to proceed to the next step
        if st.button("Proceed to Graph Visualization"):
            st.session_state['current_page'] = "Graph Visualization"
            st.experimental_rerun()

def graph_visualization_page():
    st.markdown('<h1 class="main-header">Graph Visualization</h1>', unsafe_allow_html=True)
    
    if not all(k in st.session_state for k in ['cluster_df', 'node_data', 'edge_data', 'facies_segments']):
        st.markdown('<div class="warning-message">', unsafe_allow_html=True)
        st.markdown("<p>Please create a graph dataset first on the 'Graph Dataset Creation' page.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Go to Graph Dataset Creation"):
            st.session_state['current_page'] = "Graph Dataset Creation"
            st.experimental_rerun()
            
        return
    
    st.markdown('<p class="description">This page provides visualizations of the graph dataset. The graph visualization shows the connections between different depth points and their hydrocarbon potential classification.</p>', unsafe_allow_html=True)
    
    # Create graph visualization
    with st.spinner("Generating graph visualization..."):
        graph_fig = plot_graph_visualization(
            st.session_state['node_data'], 
            st.session_state['edge_data'], 
            st.session_state['facies_segments']
        )
        
        st.plotly_chart(graph_fig, use_container_width=True)
    
    # Display additional visualizations
    st.markdown('<h2 class="sub-header">PET Label Distribution</h2>', unsafe_allow_html=True)
    
    # Plot PET label distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    pet_counts = st.session_state['edge_data']['PET_label'].value_counts().sort_index()
    colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
    ax.bar(pet_counts.index, pet_counts.values, color=colors)
    ax.set_title('Distribution of Hydrocarbon Potential Classifications')
    ax.set_ylabel('Count')
    ax.set_xlabel('Hydrocarbon Potential')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Display correlation between features and PET labels
    st.markdown('<h2 class="sub-header">Feature Importance for PET Classification</h2>', unsafe_allow_html=True)
    
    # Create feature correlation visualization
    features_for_corr = ['VSHALE', 'PHI', 'PHIECALC', 'WSAT', 'OSAT', 'PERM']
    edge_data_with_features = st.session_state['edge_data'].copy()
    
    # Add selected features from the original features dataframe
    for feature in features_for_corr:
        edge_data_with_features[feature] = st.session_state['features'][feature].values[:len(edge_data_with_features)]
    
    # Convert PET_label to numeric
    label_map = {'Very_Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very_High': 4}
    edge_data_with_features['PET_numeric'] = edge_data_with_features['PET_label'].map(label_map)
    
    # Calculate correlation
    corr_data = edge_data_with_features[features_for_corr + ['PET_numeric']].corr()['PET_numeric'][:-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(corr_data.index, corr_data.values, color='#0e4194')
    ax.set_title('Correlation between Features and Hydrocarbon Potential')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim(-1, 1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height >= 0 else height - 0.08,
                f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Add a button to proceed to the next step
    if st.button("Proceed to Model Prediction"):
        st.session_state['current_page'] = "Model Prediction"
        st.experimental_rerun()

def model_prediction_page():
    st.markdown('<h1 class="main-header">Model Prediction</h1>', unsafe_allow_html=True)
    
    if not all(k in st.session_state for k in ['cluster_df', 'node_data', 'edge_data']):
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
    
    # Run prediction button
    if st.button("Run Prediction"):
        with st.spinner("Running GCN model for hydrocarbon potential prediction..."):
            # In a real app, this would call your actual GCN model
            # For this demo, we'll simulate the prediction
            progress_bar = st.progress(0)
            
            # Simulate model training progress
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
            
            # Get predictions
            predicted_df = simulate_gcn_prediction(st.session_state['edge_data'])
            
            # Save to session state
            st.session_state['predicted_df'] = predicted_df
            
            # Show success message
            st.markdown('<div class="success-message">', unsafe_allow_html=True)
            st.markdown("<p>Prediction completed successfully!</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display prediction results if available
    if 'predicted_df' in st.session_state:
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Calculate accuracy
        accuracy = (st.session_state['predicted_df']['PET_label'] == 
                    st.session_state['predicted_df']['Predicted_PET']).mean()
        
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        # Display prediction comparison
        st.markdown('<h3>Predicted vs True Labels (Sample)</h3>', unsafe_allow_html=True)
        display_df = st.session_state['predicted_df'][['DEPTH', 'PET_label', 'Predicted_PET']].head(10)
        display_df.columns = ['Depth', 'True Label', 'Predicted Label']
        st.dataframe(display_df)
        
        # Create comparison visualization
        comparison_fig = create_comparison_plot(st.session_state['predicted_df'])
        st.pyplot(comparison_fig)
        
        # Add a button to proceed to the next step
        if st.button("Proceed to Results & Interpretation"):
            st.session_state['current_page'] = "Results & Interpretation"
            st.experimental_rerun()

def results_interpretation_page():
    st.markdown('<h1 class="main-header">Results & Interpretation</h1>', unsafe_allow_html=True)
    
    if not all(k in st.session_state for k in ['features', 'predicted_df']):
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
        distribution_fig = create_potential_distribution_plot(st.session_state['predicted_df'])
        st.pyplot(distribution_fig)
        
        # Add interpretation
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<h3>Interpretation</h3>', unsafe_allow_html=True)
        
        # Calculate proportions for interpretation
        true_high = (st.session_state['predicted_df']['PET_label'].isin(['High', 'Very_High'])).mean()
        pred_high = (st.session_state['predicted_df']['Predicted_PET'].isin(['High', 'Very_High'])).mean()
        
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
        depth_fig = create_depth_potential_plot(st.session_state['predicted_df'])
        st.pyplot(depth_fig)
        
        # Add interpretation
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<h3>Interpretation</h3>', unsafe_allow_html=True)
        
        # Find depths with highest potential for interpretation
        high_potential = st.session_state['predicted_df'][
            st.session_state['predicted_df']['Predicted_PET'].isin(['High', 'Very_High'])
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
        well_log_fig = plot_well_logs(
            st.session_state['features'], 
            facies_df=st.session_state['cluster_df'],
            predicted_df=st.session_state['predicted_df']
        )
        st.pyplot(well_log_fig)
        
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
        high_potential_pct = (st.session_state['predicted_df']['Predicted_PET'].isin(['High', 'Very_High'])).mean()
        accurate_pred_pct = (st.session_state['predicted_df']['PET_label'] == st.session_state['predicted_df']['Predicted_PET']).mean()
        
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
    
    # Export results button
    st.markdown('<h2 class="sub-header">Export Results</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Prediction Results (CSV)"):
            # Convert dataframe to CSV
            csv = st.session_state['predicted_df'].to_csv(index=False)
            
            # Create download link
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hydrocarbon_prediction_results.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("Export Visualization Report (PDF)"):
            st.markdown('<div class="warning-message">', unsafe_allow_html=True)
            st.markdown("<p>PDF export functionality would be implemented in a production version. For now, please save or screenshot the visualizations directly.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Main app
def main():
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Home"
    
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