import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def create_network_graph(node_data, edge_data, facies_segments, sample_size=100):
    """
    Create an interactive network graph visualization of the VHydro dataset
    
    Parameters:
    -----------
    node_data : pandas.DataFrame
        DataFrame containing node connections with 'source' and 'target' columns
    edge_data : pandas.DataFrame
        DataFrame containing edge data with 'DEPTH' and 'PET_label' columns
    facies_segments : list
        List of lists containing indices grouped by facies
    sample_size : int, optional
        Number of nodes to sample for visualization (default: 100)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive network graph visualization
    """
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Define color mapping for hydrocarbon potential
    label_map = {'Very_Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very_High': 4}
    colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
    
    # Sample nodes for clearer visualization
    depths = edge_data['DEPTH'].values
    labels = edge_data['PET_label'].values
    
    sample_size = min(sample_size, len(depths))
    indices = np.linspace(0, len(depths)-1, sample_size).astype(int)
    
    # Add nodes to the graph
    for i in indices:
        G.add_node(str(depths[i]), 
                   label=labels[i], 
                   color=colors[label_map[labels[i]]], 
                   depth=depths[i])
    
    # Add edges (connections between nodes in the same facies segment)
    edge_count = 0
    for segment in facies_segments:
        segment_depths = [str(depths[i]) for i in segment if i in indices]
        for i in range(len(segment_depths)):
            for j in range(i+1, min(i+5, len(segment_depths))):  # Limit connections for better visualization
                if segment_depths[i] in G.nodes and segment_depths[j] in G.nodes:
                    G.add_edge(segment_depths[i], segment_depths[j])
                    edge_count += 1
                    if edge_count > 500:  # Limit total edges for performance
                        break
            if edge_count > 500:
                break
        if edge_count > 500:
            break
    
    # Create 3D positions for nodes
    pos_3d = nx.spring_layout(G, dim=3, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_z = []
    node_colors = []
    node_text = []
    
    for node in G.nodes():
        x, y, z = pos_3d[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_colors.append(G.nodes[node]['color'])
        node_text.append(f"Depth: {G.nodes[node]['depth']}<br>HC Potential: {G.nodes[node]['label']}")
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=6,
            color=node_colors,
            opacity=0.8,
            line=dict(width=1, color='#555')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Add legend annotations
    annotations = []
    for i, (label, value) in enumerate(label_map.items()):
        annotations.append(
            dict(
                showarrow=False,
                x=0.1,
                y=0.9 - (i * 0.05),
                xref="paper",
                yref="paper",
                text=f"<span style='color:{colors[value]}'>\u25CF</span> {label}",
                font=dict(size=14),
                align="left"
            )
        )
    
    # Update layout
    fig.update_layout(
        title='3D VHydro Graph Dataset Visualization',
        titlefont_size=16,
        showlegend=False,
        annotations=annotations,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
        ),
        margin=dict(r=20, l=10, b=10, t=40)
    )
    
    return fig

def create_comparison_heatmap(predicted_df):
    """
    Create a heatmap comparing true and predicted hydrocarbon potential
    
    Parameters:
    -----------
    predicted_df : pandas.DataFrame
        DataFrame containing 'PET_label' and 'Predicted_PET' columns
        
    Returns:
    --------
    matplotlib.figure.Figure
        Confusion matrix heatmap
    """
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'True': predicted_df['PET_label'],
        'Predicted': predicted_df['Predicted_PET']
    })
    
    # Create confusion matrix
    confusion = pd.crosstab(comparison['True'], comparison['Predicted'])
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    ax = sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', 
                     linewidths=.5, cbar_kws={'label': 'Count'})
    
    # Set labels and title
    ax.set_title('Confusion Matrix: True vs Predicted Hydrocarbon Potential')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Calculate and annotate accuracy
    accuracy = (predicted_df['PET_label'] == predicted_df['Predicted_PET']).mean()
    plt.figtext(0.5, 0.01, f'Overall Accuracy: {accuracy:.2%}', 
                horizontalalignment='center', fontsize=12, 
                bbox=dict(facecolor='#e7f1ff', alpha=0.5))
    
    plt.tight_layout()
    return plt.gcf()

def create_depth_profile_visualization(predicted_df):
    """
    Create a depth profile visualization of hydrocarbon potential
    
    Parameters:
    -----------
    predicted_df : pandas.DataFrame
        DataFrame containing depth and hydrocarbon potential predictions
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive depth profile visualization
    """
    # Sort by depth
    sorted_df = predicted_df.sort_values('DEPTH')
    
    # Define color mapping
    label_map = {'Very_Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very_High': 4}
    color_scale = [[0, '#d73027'], [0.25, '#fc8d59'], [0.5, '#fee090'], 
                   [0.75, '#91cf60'], [1.0, '#1a9850']]
    
    # Create numeric values for labels
    sorted_df['true_numeric'] = sorted_df['PET_label'].map(label_map)
    sorted_df['pred_numeric'] = sorted_df['Predicted_PET'].map(label_map)
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add true potential trace
    fig.add_trace(go.Scatter(
        y=sorted_df['DEPTH'],
        x=sorted_df['true_numeric'],
        mode='markers',
        name='True Potential',
        marker=dict(
            size=8,
            color=sorted_df['true_numeric'],
            colorscale=color_scale,
            showscale=False
        ),
        hovertemplate='Depth: %{y:.2f}<br>True Potential: %{text}<extra></extra>',
        text=sorted_df['PET_label']
    ))
    
    # Add predicted potential trace
    fig.add_trace(go.Scatter(
        y=sorted_df['DEPTH'],
        x=sorted_df['pred_numeric'] + 5,  # Offset for better visibility
        mode='markers',
        name='Predicted Potential',
        marker=dict(
            size=8,
            color=sorted_df['pred_numeric'],
            colorscale=color_scale,
            showscale=False
        ),
        hovertemplate='Depth: %{y:.2f}<br>Predicted Potential: %{text}<extra></extra>',
        text=sorted_df['Predicted_PET']
    ))
    
    # Add legend items for color scale
    for label, value in label_map.items():
        fig.add_trace(go.Scatter(
            y=[None],
            x=[None],
            mode='markers',
            marker=dict(size=10, color=dict(color_scale)[value/4]),
            name=label,
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title='Hydrocarbon Potential vs Depth',
        xaxis=dict(
            title='Hydrocarbon Potential',
            showticklabels=False,
            zeroline=False,
            showgrid=False,
            range=[-1, 10]  # Adjust range to show both true and predicted values
        ),
        yaxis=dict(
            title='Depth',
            autorange='reversed'  # Invert y-axis to show increasing depth
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=800,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Add annotations for true and predicted sections
    fig.add_annotation(
        x=2,
        y=sorted_df['DEPTH'].min(),
        text="True<br>Potential",
        showarrow=False,
        yshift=10
    )
    fig.add_annotation(
        x=7,
        y=sorted_df['DEPTH'].min(),
        text="Predicted<br>Potential",
        showarrow=False,
        yshift=10
    )
    
    return figspace(0, len(depths)-1, sample_size).astype(int)
    
    # Add nodes to the graph
    for i in indices:
        G.add_node(str(depths[i]), 
                   label=labels[i], 
                   color=colors[label_map[labels[i]]], 
                   size=10)
    
    # Add edges (connections between nodes in the same facies segment)
    for segment in facies_segments:
        segment_depths = [str(depths[i]) for i in segment if i in indices]
        for i in range(len(segment_depths)):
            for j in range(i+1, len(segment_depths)):
                if segment_depths[i] in G.nodes and segment_depths[j] in G.nodes:
                    G.add_edge(segment_depths[i], segment_depths[j])
    
    # Create positions for nodes - using spring layout for natural spacing
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge traces
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
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(G.nodes[node]['color'])
        node_sizes.append(G.nodes[node]['size'])
        node_text.append(f"Depth: {node}<br>HC Potential: {G.nodes[node]['label']}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line_width=2
        )
    )
    
    # Create legend traces
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
    
    # Create figure
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

def create_facies_visualization(cluster_df, n_clusters):
    """
    Create a facies visualization of the well
    
    Parameters:
    -----------
    cluster_df : pandas.DataFrame
        DataFrame containing clustered well data with 'DEPTH' and 'Facies_pred' columns
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    matplotlib.figure.Figure
        Facies visualization figure
    """
    # Sort by depth
    sorted_df = cluster_df.sort_values('DEPTH')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3, 10))
    
    # Create colormap for facies
    cmap = plt.cm.get_cmap('viridis', n_clusters)
    
    # Plot facies
    for i in range(n_clusters):
        subset = sorted_df[sorted_df['Facies_pred'] == i]
        ax.scatter(np.ones(len(subset)), subset['DEPTH'], color=cmap(i), label=f'Facies {i}', s=10)
    
    # Set labels and title
    ax.set_title('Facies Classification')
    ax.set_xticks([])
    ax.set_ylabel('Depth')
    ax.invert_yaxis()  # Invert y-axis to show increasing depth
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_pet_distribution_plot(edge_data):
    """
    Create a visualization of PET label distribution
    
    Parameters:
    -----------
    edge_data : pandas.DataFrame
        DataFrame containing edge data with 'PET_label' column
        
    Returns:
    --------
    matplotlib.figure.Figure
        PET distribution figure
    """
    # Count PET labels
    pet_counts = edge_data['PET_label'].value_counts().sort_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each potential level
    colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
    
    # Create bar plot
    bars = ax.bar(pet_counts.index, pet_counts.values, color=colors)
    
    # Add percentage labels on top of bars
    total = pet_counts.sum()
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height/total:.1%}',
                ha='center', va='bottom')
    
    # Set labels and title
    ax.set_title('Distribution of Hydrocarbon Potential Classifications')
    ax.set_ylabel('Count')
    ax.set_xlabel('Hydrocarbon Potential')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def create_feature_importance_plot(edge_data, features_df):
    """
    Create a visualization of feature importance for PET classification
    
    Parameters:
    -----------
    edge_data : pandas.DataFrame
        DataFrame containing edge data with 'PET_label' column
    features_df : pandas.DataFrame
        DataFrame containing feature data
        
    Returns:
    --------
    matplotlib.figure.Figure
        Feature importance figure
    """
    # Features to analyze
    features_for_corr = ['VSHALE', 'PHI', 'PHIECALC', 'WSAT', 'OSAT', 'PERM']
    
    # Create a combined dataframe
    combined_df = edge_data.copy()
    
    # Add selected features
    for feature in features_for_corr:
        combined_df[feature] = features_df[feature].values[:len(combined_df)]
    
    # Convert PET_label to numeric
    label_map = {'Very_Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very_High': 4}
    combined_df['PET_numeric'] = combined_df['PET_label'].map(label_map)
    
    # Calculate correlation with PET_numeric
    corr_data = combined_df[features_for_corr + ['PET_numeric']].corr()['PET_numeric'][:-1]
    
    # Sort by absolute correlation
    corr_data = corr_data.reindex(corr_data.abs().sort_values(ascending=False).index)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    bars = ax.bar(corr_data.index, corr_data.values, color='#0e4194')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height >= 0 else height - 0.08,
                f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Set labels and title
    ax.set_title('Correlation between Features and Hydrocarbon Potential')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_xlabel('Petrophysical Features')
    ax.set_ylim(-1, 1)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_3d_graph_visualization(node_data, edge_data, facies_segments, sample_size=100):
    """
    Create a 3D interactive network graph visualization
    
    Parameters:
    -----------
    node_data : pandas.DataFrame
        DataFrame containing node connections with 'source' and 'target' columns
    edge_data : pandas.DataFrame
        DataFrame containing edge data with 'DEPTH' and 'PET_label' columns
    facies_segments : list
        List of lists containing indices grouped by facies
    sample_size : int, optional
        Number of nodes to sample for visualization (default: 100)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        3D interactive network graph visualization
    """
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Define color mapping for hydrocarbon potential
    label_map = {'Very_Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very_High': 4}
    colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
    
    # Sample nodes for clearer visualization
    depths = edge_data['DEPTH'].values
    labels = edge_data['PET_label'].values
    
    sample_size = min(sample_size, len(depths))
    indices = np.lin