import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import lasio
import re
from itertools import groupby, count

def read_las_file(las_file):
    """
    Read a LAS file and return a lasio LASFile object
    
    Parameters:
    -----------
    las_file : str
        Path to the LAS file
        
    Returns:
    --------
    lasio.LASFile
        LASFile object containing well log data
    str
        Error message or "Success"
    """
    try:
        las = lasio.read(las_file)
        return las, "Success"
    except Exception as e:
        return None, f"Error reading LAS file: {str(e)}"

def las_to_dataframe(las):
    """
    Convert a lasio LASFile object to a pandas DataFrame
    
    Parameters:
    -----------
    las : lasio.LASFile
        LASFile object containing well log data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing well log data
    """
    df = las.df()
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'DEPTH'}, inplace=True)
    return df

def validate_curves(df, required_curves=['GR', 'RHOB', 'ILD']):
    """
    Validate that required curves are present in the DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing well log data
    required_curves : list
        List of required curve names
        
    Returns:
    --------
    bool
        True if all required curves are present, False otherwise
    list
        List of missing curves
    """
    # Check for alternative curve names
    curve_alternatives = {
        'GR': ['GR', 'GRD', 'GAMMA_RAY', 'GRGC'],
        'RHOB': ['RHOB', 'DEN', 'DENSITY', 'RHOZ'],
        'ILD': ['ILD', 'RT', 'RD', 'RESD', 'LLD', 'RT_HRLT']
    }
    
    # Standardize column names (to uppercase)
    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    
    missing_curves = []
    for curve in required_curves:
        alternatives = curve_alternatives.get(curve, [curve])
        if not any(alt in df.columns for alt in alternatives):
            missing_curves.append(curve)
    
    return len(missing_curves) == 0, missing_curves

def standardize_curve_names(df):
    """
    Standardize curve names to common nomenclature
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing well log data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with standardized curve names
    """
    # Define mapping of alternative names to standard names
    name_mapping = {
        'GRD': 'GR', 'GAMMA_RAY': 'GR', 'GRGC': 'GR',
        'DEN': 'RHOB', 'DENSITY': 'RHOB', 'RHOZ': 'RHOB',
        'RT': 'ILD', 'RD': 'ILD', 'RESD': 'ILD', 'LLD': 'ILD', 'RT_HRLT': 'ILD',
        'NPHI': 'NPOR', 'NPOR_LS': 'NPOR', 'NPHI_LS': 'NPOR',
        'DPHI': 'PHI', 'PHI_D': 'PHI', 'PHID': 'PHI'
    }
    
    # Standardize column names (to uppercase)
    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    
    # Rename columns based on the mapping
    for old_name, new_name in name_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    return df

def shale_volume(gamma_ray, gamma_ray_max, gamma_ray_min):
    """
    Calculate shale volume from gamma ray log
    
    Parameters:
    -----------
    gamma_ray : numpy.ndarray
        Gamma ray values
    gamma_ray_max : float
        Maximum gamma ray value (shale baseline)
    gamma_ray_min : float
        Minimum gamma ray value (clean sand baseline)
        
    Returns:
    --------
    numpy.ndarray
        Shale volume values
    """
    # Linear gamma ray index
    vshale = (gamma_ray - gamma_ray_min) / (gamma_ray_max - gamma_ray_min)
    
    # Apply tertiary rocks non-linear correction (Larionov, 1969)
    vshale = 0.083 * (2 ** (2 * 3.7 * vshale) - 1)
    
    # Clip values to [0, 1] range
    vshale = np.clip(vshale, 0, 1)
    
    return vshale

def density_porosity(input_density, matrix_density=2.65, fluid_density=1.0):
    """
    Calculate density porosity from bulk density log
    
    Parameters:
    -----------
    input_density : numpy.ndarray
        Bulk density values
    matrix_density : float
        Matrix density (default: 2.65 g/cc for sandstone)
    fluid_density : float
        Fluid density (default: 1.0 g/cc for water)
        
    Returns:
    --------
    numpy.ndarray
        Density porosity values
    """
    denpor = (matrix_density - input_density) / (matrix_density - fluid_density)
    
    # Clip values to [0, 1] range
    denpor = np.clip(denpor, 0, 1)
    
    return denpor

def effective_porosity(phi, vshale, shale_porosity=0.3):
    """
    Calculate effective porosity from total porosity and shale volume
    
    Parameters:
    -----------
    phi : numpy.ndarray
        Total porosity values
    vshale : numpy.ndarray
        Shale volume values
    shale_porosity : float
        Shale porosity (default: 0.3)
        
    Returns:
    --------
    numpy.ndarray
        Effective porosity values
    """
    phie = phi - (vshale * shale_porosity)
    
    # Clip values to [0, 1] range
    phie = np.clip(phie, 0, 1)
    
    return phie

def sw_archie(porosity, rt, a=1, m=2, n=2, rw=0.03):
    """
    Calculate water saturation using Archie equation
    
    Parameters:
    -----------
    porosity : numpy.ndarray
        Effective porosity values
    rt : numpy.ndarray
        True resistivity values
    a : float
        Tortuosity factor (default: 1)
    m : float
        Cementation exponent (default: 2)
    n : float
        Saturation exponent (default: 2)
    rw : float
        Formation water resistivity (default: 0.03 ohm-m)
        
    Returns:
    --------
    numpy.ndarray
        Water saturation values
    """
    # Apply Archie equation
    sw = ((a * rw) / (porosity**m * rt))**(1/n)
    
    # Clip values to [0, 1] range
    sw = np.clip(sw, 0, 1)
    
    return sw

def ow_archie(sw):
    """
    Calculate oil saturation from water saturation
    
    Parameters:
    -----------
    sw : numpy.ndarray
        Water saturation values
        
    Returns:
    --------
    numpy.ndarray
        Oil saturation values
    """
    ow = 1 - sw
    return ow

def permeability(porosity):
    """
    Calculate permeability from porosity using exponential relation
    
    Parameters:
    -----------
    porosity : numpy.ndarray
        Effective porosity values
        
    Returns:
    --------
    numpy.ndarray
        Permeability values (in mD)
    """
    # Exponential relationship (modified Timur equation)
    perm = 0.00004 * np.exp(57.117 * porosity)
    
    return perm

def calculate_petrophysical_properties(df):
    """
    Calculate petrophysical properties from well log data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing well log data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with calculated petrophysical properties
    """
    # Standardize curve names
    df = standardize_curve_names(df)
    
    # Create a results dataframe
    results = pd.DataFrame()
    results['DEPTH'] = df['DEPTH']
    
    # Calculate shale volume
    if 'CGR' in df.columns:
        results['VSHALE'] = shale_volume(df['CGR'].values, df['CGR'].max(), df['CGR'].min())
    else:
        results['VSHALE'] = shale_volume(df['GR'].values, df['GR'].max(), df['GR'].min())
    
    # Calculate density porosity
    results['PHI'] = density_porosity(df['RHOB'].values)
    
    # Calculate effective porosity
    results['PHIECALC'] = effective_porosity(results['PHI'].values, results['VSHALE'].values)
    
    # Handle resistivity log for saturation calculations
    if 'ILD' in df.columns:
        resistivity = df['ILD'].values
    else:
        # Use the first available resistivity curve
        resistivity_curves = [col for col in df.columns if any(rx in col for rx in ['RT', 'RD', 'RESD', 'ILD'])]
        if resistivity_curves:
            resistivity = df[resistivity_curves[0]].values
        else:
            # No resistivity curve available, use a default value
            resistivity = np.ones(len(df)) * 10
    
    # Apply log transform to resistivity for numerical stability
    log_rt = np.log10(np.maximum(resistivity, 0.001))
    
    # Calculate water saturation
    results['WSAT'] = sw_archie(results['PHIECALC'].values, np.exp(log_rt))
    
    # Calculate oil saturation
    results['OSAT'] = ow_archie(results['WSAT'].values)
    
    # Calculate permeability
    results['PERM'] = permeability(results['PHIECALC'].values)
    
    # Add original curves
    for col in ['GR', 'RHOB', 'ILD']:
        if col in df.columns:
            results[col] = df[col].values
    
    return results

def cluster_data(features, n_clusters=7):
    """
    Perform K-means clustering on well log data
    
    Parameters:
    -----------
    features : pandas.DataFrame
        DataFrame containing feature data
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cluster labels
    sklearn.cluster.KMeans
        Trained KMeans model
    """
    # Select features for clustering
    cluster_features = features[['VSHALE', 'PHI', 'PHIECALC', 'WSAT', 'OSAT', 'PERM']].copy()
    
    # Drop rows with NaN values
    cluster_features = cluster_features.dropna()
    
    # Scale features
    scaled_features = scale(cluster_features)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Create a new DataFrame with cluster labels
    clustered_df = features.copy()
    clustered_df = clustered_df.iloc[:len(cluster_labels)]
    clustered_df['Facies_pred'] = cluster_labels
    
    return clustered_df, kmeans

def find_optimal_clusters(features, max_clusters=10):
    """
    Find the optimal number of clusters using silhouette score
    
    Parameters:
    -----------
    features : pandas.DataFrame
        DataFrame containing feature data
    max_clusters : int
        Maximum number of clusters to evaluate
        
    Returns:
    --------
    dict
        Dictionary with scores for each number of clusters
    int
        Optimal number of clusters
    """
    # Select features for clustering
    cluster_features = features[['VSHALE', 'PHI', 'PHIECALC', 'WSAT', 'OSAT', 'PERM']].copy()
    
    # Drop rows with NaN values
    cluster_features = cluster_features.dropna()
    
    # Scale features
    scaled_features = scale(cluster_features)
    
    # Calculate silhouette scores for different numbers of clusters
    silhouette_scores = {}
    
    for n_clusters in range(5, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        silhouette_scores[n_clusters] = silhouette_avg
        
        print(f"For n_clusters = {n_clusters}, the average silhouette score is {silhouette_avg:.4f}")
    
    # Find the optimal number of clusters
    optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
    
    return silhouette_scores, optimal_clusters

def identify_continuous_segments(depth, facies):
    """
    Identify continuous segments of the same facies
    
    Parameters:
    -----------
    depth : numpy.ndarray
        Depth values
    facies : numpy.ndarray
        Facies labels
        
    Returns:
    --------
    list
        List of lists containing indices of continuous segments
    """
    segments = []
    
    for facies_id in np.unique(facies):
        indices = np.where(facies == facies_id)[0]
        
        # Find continuous segments
        segments_by_facies = []
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            segment = list(map(lambda x: x[1], g))
            segments_by_facies.append(segment)
        
        segments.extend(segments_by_facies)
    
    return segments

def generate_node_connections(df, segments, train_size=0.8):
    """
    Generate node connections for graph dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing well data
    segments : list
        List of lists containing indices of continuous segments
    train_size : float
        Proportion of segments to use for training
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing node connections
    dict
        Dictionary with train, test, and validation indices
    """
    # Split segments into train, test, validation
    n_segments = len(segments)
    n_train = int(n_segments * train_size)
    n_test_val = n_segments - n_train
    n_test = n_test_val // 2
    
    # Shuffle segments
    np.random.seed(42)
    shuffled_indices = np.random.permutation(n_segments)
    
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:n_train+n_test]
    val_indices = shuffled_indices[n_train+n_test:]
    
    # Create train segments
    train_segments = [segments[i] for i in train_indices]
    
    # Create node connections
    node_data = pd.DataFrame(columns=['source', 'target'])
    
    # Create connections for each segment
    for segment in train_segments:
        for i in range(len(segment)):
            for j in range(i+1, len(segment)):
                node_data = node_data.append({
                    'source': df['DEPTH'].iloc[segment[i]],
                    'target': df['DEPTH'].iloc[segment[j]]
                }, ignore_index=True)
    
    split_indices = {
        'train': train_indices,
        'test': test_indices,
        'val': val_indices
    }
    
    return node_data, split_indices

def generate_edge_features(df):
    """
    Generate edge features for graph dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing petrophysical properties
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing edge features
    """
    # Create edge data DataFrame
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
    
    # Create composite feature
    edge_data['Values'] = (edge_data['PE_1'].astype(str) + edge_data['PE_2'].astype(str) + 
                           edge_data['PE_3'].astype(str) + edge_data['PE_4'].astype(str) + 
                           edge_data['PE_5'].astype(str) + edge_data['PO_1'].astype(str) + 
                           edge_data['PO_2'].astype(str) + edge_data['PO_3'].astype(str) + 
                           edge_data['VS_1'].astype(str) + edge_data['VS_2'].astype(str) + 
                           edge_data['SW_1'].astype(str) + edge_data['SW_2'].astype(str) + 
                           edge_data['OW_1'].astype(str) + edge_data['OW_2'].astype(str))
    
    return edge_data

def assign_hydrocarbon_potential(edge_data):
    """
    Assign hydrocarbon potential labels based on petrophysical properties
    
    Parameters:
    -----------
    edge_data : pandas.DataFrame
        DataFrame containing edge features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with hydrocarbon potential labels
    """
    # Create a copy of the DataFrame
    labeled_data = edge_data.copy()
    
    # Get unique value patterns
    unique_values = labeled_data['Values'].unique()
    
    # Rule-based classification for hydrocarbon potential
    labeled_data['PET_label'] = ''
    
    for value in unique_values:
        # Convert binary string to list of integers
        binary_features = [int(bit) for bit in value]
        
        # Calculate feature sums for different categories
        perm_sum = sum(binary_features[0:5])  # PE_1 to PE_5
        por_sum = sum(binary_features[5:8])   # PO_1 to PO_3
        vsh_sum = sum(binary_features[8:10])  # VS_1 to VS_2
        sat_sum = sum(binary_features[10:14]) # SW_1, SW_2, OW_1, OW_2
        
        # Calculate a weighted score for hydrocarbon potential
        # Higher weights for high permeability, high porosity, low shale volume, high oil saturation
        hc_score = (
            perm_sum * 0.3 +          # Permeability contribution
            por_sum * 0.3 +           # Porosity contribution
            (binary_features[9]) * 0.2 +  # Low shale volume (VS_2 = 1 means Vsh <= 0.5)
            (binary_features[13]) * 0.2    # High oil saturation (OW_2 = 1 means So >= 0.5)
        )
        
        # Assign hydrocarbon potential label based on score
        if hc_score >= 0.8:
            label = 'Very_High'
        elif hc_score >= 0.6:
            label = 'High'
        elif hc_score >= 0.4:
            label = 'Moderate'
        elif hc_score >= 0.2:
            label = 'Low'
        else:
            label = 'Very_Low'
        
        # Assign label to all rows with this value pattern
        labeled_data.loc[labeled_data['Values'] == value, 'PET_label'] = label
    
    return labeled_data

def process_las_file(las_file, n_clusters=7):
    """
    Complete processing pipeline for LAS file
    
    Parameters:
    -----------
    las_file : str
        Path to LAS file
    n_clusters : int
        Number of clusters for facies identification
        
    Returns:
    --------
    dict
        Dictionary containing all processed data and results
    str
        Error message or "Success"
    """
    # Read LAS file
    las, message = read_las_file(las_file)
    if las is None:
        return None, message
    
    # Convert to DataFrame
    df_raw = las_to_dataframe(las)
    
    # Validate required curves
    valid, missing_curves = validate_curves(df_raw)
    if not valid:
        return None, f"Missing required curves: {', '.join(missing_curves)}"
    
    # Calculate petrophysical properties
    features = calculate_petrophysical_properties(df_raw)
    
    # Perform clustering
    clustered_df, kmeans_model = cluster_data(features, n_clusters)
    
    # Identify continuous segments
    segments = identify_continuous_segments(clustered_df['DEPTH'].values, 
                                           clustered_df['Facies_pred'].values)
    
    # Generate node connections
    node_data, split_indices = generate_node_connections(clustered_df, segments)
    
    # Generate edge features
    edge_data = generate_edge_features(clustered_df)
    
    # Assign hydrocarbon potential labels
    labeled_data = assign_hydrocarbon_potential(edge_data)
    
    # Create result dictionary
    result = {
        'las': las,
        'raw_data': df_raw,
        'features': features,
        'clustered_data': clustered_df,
        'segments': segments,
        'node_data': node_data,
        'edge_data': labeled_data,
        'split_indices': split_indices,
        'kmeans_model': kmeans_model
    }
    
    return result, "Success"

def evaluate_clusters(features, max_clusters=10):
    """
    Evaluate different numbers of clusters using elbow method and silhouette score
    
    Parameters:
    -----------
    features : pandas.DataFrame
        DataFrame containing feature data
    max_clusters : int
        Maximum number of clusters to evaluate
        
    Returns:
    --------
    tuple
        (wcss_values, silhouette_scores, optimal_clusters)
    matplotlib.figure.Figure
        Evaluation plot
    """
    # Select features for clustering
    cluster_features = features[['VSHALE', 'PHI', 'PHIECALC', 'WSAT', 'OSAT', 'PERM']].copy()
    
    # Drop rows with NaN values
    cluster_features = cluster_features.dropna()
    
    # Scale features
    scaled_features = scale(cluster_features)
    
    # Calculate WCSS and silhouette scores
    wcss_values = []
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss_values.append(kmeans.inertia_)
        
        if n_clusters > 1:
            cluster_labels = kmeans.predict(scaled_features)
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    # Find optimal number of clusters
    # Using the silhouette score (higher is better)
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
    
    # Create evaluation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot WCSS (Elbow method)
    ax1.plot(range(2, max_clusters + 1), wcss_values, 'bo-')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.axvline(x=optimal_clusters, color='r', linestyle='--', 
                label=f'Optimal clusters: {optimal_clusters}')
    ax1.legend()
    
    # Plot Silhouette scores
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, 'go-')
    ax2.set_title('Silhouette Method')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.axvline(x=optimal_clusters, color='r', linestyle='--',
                label=f'Optimal clusters: {optimal_clusters}')
    ax2.legend()
    
    plt.tight_layout()
    
    return (wcss_values, silhouette_scores, optimal_clusters), fig

def extract_well_metadata(las):
    """
    Extract metadata from LAS file
    
    Parameters:
    -----------
    las : lasio.LASFile
        LASFile object containing well log data
        
    Returns:
    --------
    dict
        Dictionary containing well metadata
    """
    metadata = {}
    
    # Try to get basic well information
    try:
        metadata['WELL'] = las.well.WELL.value
    except:
        metadata['WELL'] = 'Unknown'
        
    try:
        metadata['UWI'] = las.well.UWI.value
    except:
        metadata['UWI'] = 'Unknown'
        
    try:
        metadata['FIELD'] = las.well.FLD.value
    except:
        try:
            metadata['FIELD'] = las.well.FIELD.value
        except:
            metadata['FIELD'] = 'Unknown'
    
    # Get depth range
    metadata['STRT'] = las.well.STRT.value
    metadata['STOP'] = las.well.STOP.value
    metadata['STEP'] = las.well.STEP.value
    metadata['DEPTH_UNIT'] = las.well.STRT.unit
    
    # Get available curves
    metadata['CURVES'] = [curve.mnemonic for curve in las.curves]
    
    return metadata