import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import groupby, count
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn import preprocessing, model_selection

# Required for StellarGraph model
import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import EarlyStopping

class VHydro:
    def __init__(self, las_file_path, output_dir):
        """
        Initialize the Well Log Analysis class.
        
        Args:
            las_file_path (str): Path to the LAS file
            output_dir (str): Directory to save output files
        """
        self.las_file_path = las_file_path
        self.output_dir = output_dir
        self.df = None
        self.features = None
        self.x_scaled = None
        self.well_data = None
        self.edge_data = None
        self.clustering_ranges = {}
        self.models = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up colors for visualization
        self.facies_colors = ['gold', 'orange', 'chocolate', 'black', 'slateblue', 
                             'mediumorchid', 'cornflowerblue', 'deepskyblue', 'green']
        
    def load_las_file(self):
        """Load LAS file and convert to dataframe"""
        try:
            from lasio import read as las_read
            las = las_read(self.las_file_path)
            self.df = las.df()
            print(f"Successfully loaded LAS file: {self.las_file_path}")
            print(f"Available columns: {', '.join(self.df.columns)}")
            return True
        except Exception as e:
            print(f"Error loading LAS file: {e}")
            return False
            
    def prepare_features(self, feature_columns=None):
        """
        Prepare features for clustering
        
        Args:
            feature_columns (list): List of column names to use as features.
                                   If None, uses default columns
        """
        if feature_columns is None:
            feature_columns = ['VSHALE', 'PHI', 'SW', 'GR', 'DENSITY']
            # Make sure these columns exist in the dataframe
            available_cols = [col for col in feature_columns if col in self.df.columns]
            if not available_cols:
                print(f"None of the specified columns found. Available columns: {', '.join(self.df.columns)}")
                return False
            feature_columns = available_cols
            
        self.features = self.df[feature_columns].copy()
        self.features = self.features.dropna()
        self.features = self.features.drop_duplicates()
        self.features = self.features.reset_index(drop=True)
        self.x_scaled = scale(self.features)
        
        print(f"Features prepared with columns: {', '.join(feature_columns)}")
        print(f"Feature shape: {self.features.shape}")
        return True
            
    def calculate_petrophysical_properties(self):
        """Calculate petrophysical properties from well logs"""
        # Create a new dataframe for storing calculated properties
        self.well_data = pd.DataFrame()
        
        # Calculate Shale Volume (assuming CGR log is available)
        if 'CGR' in self.df.columns:
            self.well_data['VSHALE'] = self._shale_volume(
                self.df['CGR'], 
                self.df['CGR'].max(), 
                self.df['CGR'].min()
            )
        elif 'GR' in self.df.columns:
            self.well_data['VSHALE'] = self._shale_volume(
                self.df['GR'], 
                self.df['GR'].max(), 
                self.df['GR'].min()
            )
        else:
            print("Warning: Neither CGR nor GR logs found. Cannot calculate VSHALE.")
            
        # Calculate density porosity (if RHOB available)
        if 'RHOB' in self.df.columns:
            self.well_data['PHI'] = self._density_porosity(
                self.df['RHOB'], 2.65, 1
            )
            
            # Calculate effective porosity
            self.well_data['PHIECALC'] = self.well_data['PHI'] - (self.well_data['VSHALE'] * 0.3)
        else:
            print("Warning: RHOB log not found. Cannot calculate porosity.")
            
        # Calculate water saturation (if resistivity log available)
        if 'ILD' in self.df.columns:
            self.well_data['WSAT'] = self._sw_archie(
                self.well_data['PHIECALC'], 
                np.log(self.df['ILD'])
            )
            
            # Calculate oil saturation
            self.well_data['OSAT'] = self._ow_archie(self.well_data['WSAT'])
        else:
            print("Warning: ILD log not found. Cannot calculate water saturation.")
            
        # Calculate permeability
        if 'PHIECALC' in self.well_data.columns:
            self.well_data['PERM'] = self._permeability(self.well_data['PHIECALC'])
            
        print(f"Petrophysical properties calculated: {', '.join(self.well_data.columns)}")
        return True
    
    def _shale_volume(self, gamma_ray, gamma_ray_max, gamma_ray_min):
        """Calculate shale volume from gamma ray log"""
        vshale = (gamma_ray - gamma_ray_min) / (gamma_ray_max - gamma_ray_min)
        # For tertiary rocks
        vshale = 0.083 * (2 ** (2 * 3.7 * vshale) - 1)
        return np.round(vshale, 4)
    
    def _density_porosity(self, input_density, matrix_density, fluid_density):
        """Calculate density porosity"""
        denpor = (matrix_density - input_density) / (matrix_density - fluid_density)
        return np.round(denpor, 4)
    
    def _sw_archie(self, porosity, rt, a=1, m=2, n=2, rw=0.03):
        """Calculate water saturation using Archie's equation"""
        sw = ((a * (porosity**(-m))) / (rt * rw))**(1/n)
        return sw
    
    def _ow_archie(self, sw):
        """Calculate oil saturation"""
        ow = 1 - sw
        return ow
    
    def _permeability(self, porosity):
        """Calculate permeability from porosity"""
        perm = 0.00004 * np.exp(57.117 * porosity)
        return perm
            
    def perform_kmeans_clustering(self, min_clusters=5, max_clusters=10, save_plots=True):
        """
        Perform KMeans clustering to identify different facies
        
        Args:
            min_clusters (int): Minimum number of clusters to try
            max_clusters (int): Maximum number of clusters to try
            save_plots (bool): Whether to save plots
        
        Returns:
            dict: Dictionary with cluster counts and silhouette scores
        """
        if self.x_scaled is None:
            print("Error: Features not prepared. Run prepare_features() first.")
            return False
            
        # Calculate Within-Cluster Sum of Squares (WCSS) for different cluster counts
        wcss = []
        for i in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=10)
            kmeans.fit(self.x_scaled)
            wcss_iter = kmeans.inertia_
            wcss.append(wcss_iter)
            
        # Plot WCSS vs number of clusters (Elbow method)
        if save_plots:
            plt.figure(figsize=(10, 8))
            plt.plot(range(min_clusters, max_clusters + 1), wcss, '*-')
            plt.xlabel('Number of clusters', fontsize=20)
            plt.ylabel('Within-cluster Sum of Squares', fontsize=20)
            plt.savefig(os.path.join(self.output_dir, 'elbow_method.png'))
            plt.close()
            
        # Calculate silhouette scores
        silhouette_scores = {}
        for n_clusters in range(min_clusters, max_clusters + 1):
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(self.x_scaled)
            silhouette_avg = silhouette_score(self.x_scaled, cluster_labels)
            silhouette_scores[n_clusters] = silhouette_avg
            print(f"For n_clusters = {n_clusters}, the average silhouette score is: {silhouette_avg:.4f}")
            
        results = {
            'wcss': wcss,
            'silhouette_scores': silhouette_scores
        }
        
        return results
    
    def create_cluster_dataset(self, n_clusters):
        """
        Create dataset with cluster assignments
        
        Args:
            n_clusters (int): Number of clusters to use
        
        Returns:
            DataFrame: DataFrame with cluster assignments
        """
        # Apply KMeans clustering with the specified number of clusters
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=10)
        kmeans.fit(self.x_scaled)
        
        # Add cluster assignments to the dataframe
        self.df['Facies_pred'] = kmeans.predict(self.x_scaled)
        
        # Save the clustering results
        output_path = os.path.join(self.output_dir, f'facies_for_{n_clusters}.xlsx')
        self.df.to_excel(output_path, index=False)
        print(f"Saved clustering results to {output_path}")
        
        return self.df
    
    def identify_clustering_ranges(self, n_clusters):
        """
        Identify depth ranges for each cluster
        
        Args:
            n_clusters (int): Number of clusters
            
        Returns:
            dict: Dictionary with cluster ranges
        """
        if 'Facies_pred' not in self.df.columns:
            print("Error: Cluster assignments not found. Run create_cluster_dataset() first.")
            return False
            
        # Get the facies predictions as a list
        facies_pred = self.df['Facies_pred'].to_list()
        
        # Store start and end depths for each cluster
        cluster_ranges = {}
        
        # For each facies value
        for facies_value in range(n_clusters):
            # Find indices where this facies occurs
            indices = [idx for idx, value in enumerate(facies_pred) if value == facies_value]
            
            # Group consecutive indices
            grouped_indices = [list(g) for _, g in groupby(indices, key=lambda n, c=count(): n-next(c))]
            
            # Extract start and end depths for each group
            start_depths = []
            end_depths = []
            for group in grouped_indices:
                start_depths.append(self.df['DEPTH'].iloc[group[0]])
                end_depths.append(self.df['DEPTH'].iloc[group[-1]])
            
            cluster_ranges[facies_value] = {
                'start_depths': start_depths,
                'end_depths': end_depths,
                'groups': grouped_indices
            }
        
        # Get all depth ranges in a single list
        all_groups = []
        for facies_value in range(n_clusters):
            all_groups.extend(cluster_ranges[facies_value]['groups'])
        
        # Sort by depth
        all_groups_sorted = sorted(all_groups)
        
        # Extract start and end depths
        start_depths = [self.df['DEPTH'].iloc[group[0]] for group in all_groups_sorted]
        end_depths = [self.df['DEPTH'].iloc[group[-1]] for group in all_groups_sorted]
        
        cluster_ranges['all'] = {
            'start_depths': start_depths,
            'end_depths': end_depths,
            'groups': all_groups_sorted
        }
        
        self.clustering_ranges[n_clusters] = cluster_ranges
        return cluster_ranges
    
    def generate_adjacency_matrix(self, n_clusters, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Generate adjacency matrix for graph neural network
        
        Args:
            n_clusters (int): Number of clusters
            train_ratio (float): Ratio of data for training
            val_ratio (float): Ratio of data for validation
            test_ratio (float): Ratio of data for testing
            
        Returns:
            dict: Dictionary with node and edge data for graph neural network
        """
        if n_clusters not in self.clustering_ranges:
            self.identify_clustering_ranges(n_clusters)
            
        cluster_ranges = self.clustering_ranges[n_clusters]
        all_groups = cluster_ranges['all']['groups']
        
        # Calculate split indices
        total_groups = len(all_groups)
        train_end = int(total_groups * train_ratio)
        val_end = train_end + int(total_groups * val_ratio)
        
        # Split into train, validation, and test sets
        train_groups = all_groups[:train_end]
        val_groups = all_groups[train_end:val_end]
        test_groups = all_groups[val_end:]
        
        # Create node connections
        node_data = self._create_node_connections(all_groups, train_groups, val_groups, test_groups)
        
        # Create edge dataset with petrophysical properties
        edge_data = self._create_edge_dataset()
        
        # Split edge dataset into train, validation, and test sets
        train_edges = self._get_edges_for_groups(edge_data, train_groups)
        val_edges = self._get_edges_for_groups(edge_data, val_groups)
        test_edges = self._get_edges_for_groups(edge_data, test_groups)
        
        # Save node and edge data
        self._save_graph_data(node_data, edge_data, train_edges, val_edges, test_edges, n_clusters)
        
        graph_data = {
            'node_data': node_data,
            'edge_data': edge_data,
            'train_edges': train_edges,
            'val_edges': val_edges,
            'test_edges': test_edges
        }
        
        return graph_data
    
    def _create_node_connections(self, all_groups, train_groups, val_groups, test_groups):
        """Create node connections for graph neural network"""
        # Calculate total number of connections
        total_connections = 0
        for group in all_groups:
            group_size = len(group)
            total_connections += (group_size * (group_size - 1))
            
        # Create dataframe for node connections
        node_data = pd.DataFrame(index=range(total_connections))
        node_data['full'] = ''
        node_data['full1'] = ''
        node_data['train'] = ''
        node_data['train1'] = ''
        node_data['val'] = ''
        node_data['val1'] = ''
        node_data['test'] = ''
        node_data['test1'] = ''
        
        # Create full connections
        k1 = 0
        for group in all_groups:
            for i in range(len(group)):
                node_i = group[i]
                for j in range(len(group)):
                    if node_i != group[j]:
                        node_data.loc[k1, 'full'] = self.df['DEPTH'].iloc[node_i]
                        node_data.loc[k1, 'full1'] = self.df['DEPTH'].iloc[group[j]]
                        k1 += 1
                        
        # Create training connections
        k1 = 0
        for group in train_groups:
            for i in range(len(group)):
                node_i = group[i]
                for j in range(len(group)):
                    if node_i != group[j]:
                        node_data.loc[k1, 'train'] = self.df['DEPTH'].iloc[node_i]
                        node_data.loc[k1, 'train1'] = self.df['DEPTH'].iloc[group[j]]
                        k1 += 1
                        
        # Create validation connections
        k1 = 0
        for group in val_groups:
            for i in range(len(group)):
                node_i = group[i]
                for j in range(len(group)):
                    if node_i != group[j]:
                        node_data.loc[k1, 'val'] = self.df['DEPTH'].iloc[node_i]
                        node_data.loc[k1, 'val1'] = self.df['DEPTH'].iloc[group[j]]
                        k1 += 1
                        
        # Create test connections
        k1 = 0
        for group in test_groups:
            for i in range(len(group)):
                node_i = group[i]
                for j in range(len(group)):
                    if node_i != group[j]:
                        node_data.loc[k1, 'test'] = self.df['DEPTH'].iloc[node_i]
                        node_data.loc[k1, 'test1'] = self.df['DEPTH'].iloc[group[j]]
                        k1 += 1
                        
        return node_data
    
    def _create_edge_dataset(self):
        """Create edge dataset with petrophysical properties"""
        # Create dataframe for edge data
        edge_data = pd.DataFrame()
        edge_data['DEPTH'] = self.df['DEPTH']
        
        # Add permeability classification
        edge_data['PE_1'] = (self.well_data['PERM'] <= 0.01).astype(int)
        edge_data['PE_2'] = ((self.well_data['PERM'] > 0.01) & (self.well_data['PERM'] <= 1)).astype(int)
        edge_data['PE_3'] = ((self.well_data['PERM'] > 1) & (self.well_data['PERM'] <= 10)).astype(int)
        edge_data['PE_4'] = ((self.well_data['PERM'] > 10) & (self.well_data['PERM'] <= 100)).astype(int)
        edge_data['PE_5'] = (self.well_data['PERM'] > 100).astype(int)
        
        # Add porosity classification
        edge_data['PO_1'] = (self.well_data['PHIECALC'] <= 0.1).astype(int)
        edge_data['PO_2'] = ((self.well_data['PHIECALC'] > 0.1) & (self.well_data['PHIECALC'] <= 0.2)).astype(int)
        edge_data['PO_3'] = ((self.well_data['PHIECALC'] > 0.2) & (self.well_data['PHIECALC'] <= 0.3)).astype(int)
        
        # Add shale volume classification
        edge_data['VS_1'] = (self.well_data['VSHALE'] > 0.5).astype(int)
        edge_data['VS_2'] = (self.well_data['VSHALE'] <= 0.5).astype(int)
        
        # Add water saturation classification
        edge_data['SW_1'] = (self.well_data['WSAT'] > 0.5).astype(int)
        edge_data['SW_2'] = (self.well_data['WSAT'] <= 0.5).astype(int)
        
        # Add oil saturation classification
        edge_data['OW_1'] = (self.well_data['OSAT'] < 0.5).astype(int)
        edge_data['OW_2'] = (self.well_data['OSAT'] >= 0.5).astype(int)
        
        # Create categorical values for classification
        edge_data['Values'] = (
            edge_data['PE_1'].astype(str) + edge_data['PE_2'].astype(str) + 
            edge_data['PE_3'].astype(str) + edge_data['PE_4'].astype(str) + 
            edge_data['PE_5'].astype(str) + edge_data['PO_1'].astype(str) + 
            edge_data['PO_2'].astype(str) + edge_data['PO_3'].astype(str) + 
            edge_data['VS_1'].astype(str) + edge_data['VS_2'].astype(str) + 
            edge_data['SW_1'].astype(str) + edge_data['SW_2'].astype(str) + 
            edge_data['OW_1'].astype(str) + edge_data['OW_2'].astype(str)
        )
        
        # Create categorical labels
        edge_data['cate'] = pd.Series(edge_data['Values'], dtype="category")
        unique_categories = edge_data['cate'].unique()
        
        # Assign descriptive labels
        edge_data['RESULTS'] = ''
        for i, value in enumerate(edge_data['Values']):
            if value == unique_categories[0]:
                edge_data.loc[i, 'RESULTS'] = 'Moderate'
            elif value == unique_categories[1]:
                edge_data.loc[i, 'RESULTS'] = 'Moderate'
            elif value == unique_categories[2]:
                edge_data.loc[i, 'RESULTS'] = 'Very_Low'
            elif value == unique_categories[3]:
                edge_data.loc[i, 'RESULTS'] = 'Very_Low'
            elif value == unique_categories[4]:
                edge_data.loc[i, 'RESULTS'] = 'Low'
            elif value == unique_categories[5]:
                edge_data.loc[i, 'RESULTS'] = 'Low'
            elif value == unique_categories[6]:
                edge_data.loc[i, 'RESULTS'] = 'Moderate'
            elif value == unique_categories[7]:
                edge_data.loc[i, 'RESULTS'] = 'High'
            elif value == unique_categories[8]:
                edge_data.loc[i, 'RESULTS'] = 'High'
            else:
                edge_data.loc[i, 'RESULTS'] = 'Unknown'
                
        # Drop intermediate columns
        edge_data = edge_data.drop(columns=['Values', 'cate'])
        
        self.edge_data = edge_data
        return edge_data
    
    def _get_edges_for_groups(self, edge_data, groups):
        """Get edge data for specific groups"""
        edges = []
        for group in groups:
            for node in group:
                edges.append(edge_data.iloc[node])
        return edges
    
    def _save_graph_data(self, node_data, edge_data, train_edges, val_edges, test_edges, n_clusters):
        """Save graph data to files"""
        # Create directory for cluster
        cluster_dir = os.path.join(self.output_dir, str(n_clusters))
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Save full node data
        full_node_data = pd.DataFrame()
        full_node_data['source'] = node_data['full'].apply(lambda x: str(x).replace('.', ''))
        full_node_data['target'] = node_data['full1'].apply(lambda x: str(x).replace('.', ''))
        full_node_data.to_csv(os.path.join(cluster_dir, f'BF_full_node_{n_clusters}.txt'), 
                              sep='\t', index=False, header=False)
        
        # Save train node data
        train_node_data = pd.DataFrame()
        train_node_data['source'] = node_data['train'].apply(lambda x: str(x).replace('.', ''))
        train_node_data['target'] = node_data['train1'].apply(lambda x: str(x).replace('.', ''))
        train_node_data.to_csv(os.path.join(cluster_dir, f'BF_train_node_{n_clusters}.txt'), 
                               sep='\t', index=False, header=False)
        
        # Save validation node data
        val_node_data = pd.DataFrame()
        val_node_data['source'] = node_data['val'].apply(lambda x: str(x).replace('.', ''))
        val_node_data['target'] = node_data['val1'].apply(lambda x: str(x).replace('.', ''))
        val_node_data.to_csv(os.path.join(cluster_dir, f'BF_valid_node_{n_clusters}.txt'), 
                             sep='\t', index=False, header=False)
        
        # Save test node data
        test_node_data = pd.DataFrame()
        test_node_data['source'] = node_data['test'].apply(lambda x: str(x).replace('.', ''))
        test_node_data['target'] = node_data['test1'].apply(lambda x: str(x).replace('.', ''))
        test_node_data.to_csv(os.path.join(cluster_dir, f'BF_test_node_{n_clusters}.txt'), 
                              sep='\t', index=False, header=False)
        
        # Save edge data
        edge_data_for_save = edge_data.copy()
        edge_data_for_save['DEPTH'] = edge_data_for_save['DEPTH'].apply(lambda x: str(x).replace('.', ''))
        edge_data_for_save.to_csv(os.path.join(cluster_dir, f'BF_full_edge_{n_clusters}.txt'), 
                                  sep='\t', index=False, header=False)
        
        # Save train edge data
        train_edge_df = pd.DataFrame(train_edges)
        train_edge_df['DEPTH'] = train_edge_df['DEPTH'].apply(lambda x: str(x).replace('.', ''))
        train_edge_df.to_csv(os.path.join(cluster_dir, f'BF_train_edge_{n_clusters}.txt'), 
                             sep='\t', index=False, header=False)
        
        # Save validation edge data
        val_edge_df = pd.DataFrame(val_edges)
        val_edge_df['DEPTH'] = val_edge_df['DEPTH'].apply(lambda x: str(x).replace('.', ''))
        val_edge_df.to_csv(os.path.join(cluster_dir, f'BF_valid_edge_{n_clusters}.txt'), 
                          sep='\t', index=False, header=False)
        
        # Save test edge data
        test_edge_df = pd.DataFrame(test_edges)
        test_edge_df['DEPTH'] = test_edge_df['DEPTH'].apply(lambda x: str(x).replace('.', ''))
        test_edge_df.to_csv(os.path.join(cluster_dir, f'BF_test_edge_{n_clusters}.txt'), 
                           sep='\t', index=False, header=False)
        
        print(f"Saved graph data for {n_clusters} clusters to {cluster_dir}")
        
    def visualize_loss_accuracy(self, n_clusters_list, run_ids=None):
        """
        Visualize training loss and accuracy for multiple cluster configurations
        
        Args:
            n_clusters_list (list): List of cluster numbers to visualize
            run_ids (dict): Dictionary mapping cluster numbers to run IDs to use (default: use run_id=1)
        """
        if run_ids is None:
            run_ids = {n: 1 for n in n_clusters_list}
            
        # Set up figure
        fig, axes = plt.subplots(2, len(n_clusters_list), figsize=(5*len(n_clusters_list), 10), sharey='row')
        
        # Plot loss and accuracy for each cluster configuration
        for i, n_clusters in enumerate(n_clusters_list):
            if n_clusters not in self.models or run_ids[n_clusters] not in self.models[n_clusters]:
                print(f"Model for {n_clusters} clusters (run {run_ids[n_clusters]}) not found.")
                continue
                
            model_results = self.models[n_clusters][run_ids[n_clusters]]
            history = model_results['history']
            
            # Plot loss
            axes[0, i].plot(history.history['loss'], 'b-', label='train', linewidth=2)
            axes[0, i].plot(history.history['val_loss'], 'r-', label='validation', linewidth=2)
            axes[0, i].fill_between(range(len(history.history['loss'])), 
                                   history.history['loss'], 
                                   history.history['val_loss'], 
                                   color='b', alpha=0.2)
            axes[0, i].set_title(f"Loss (Clusters: {n_clusters})", fontsize=14)
            axes[0, i].set_xlabel("Epoch", fontsize=12)
            if i == 0:
                axes[0, i].set_ylabel("Loss", fontsize=12)
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot accuracy
            axes[1, i].plot(history.history['acc'], 'b-', label='train', linewidth=2)
            axes[1, i].plot(history.history['val_acc'], 'r-', label='validation', linewidth=2)
            axes[1, i].fill_between(range(len(history.history['acc'])), 
                                   history.history['acc'], 
                                   history.history['val_acc'], 
                                   color='b', alpha=0.2)
            axes[1, i].set_title(f"Accuracy (Clusters: {n_clusters})", fontsize=14)
            axes[1, i].set_xlabel("Epoch", fontsize=12)
            if i == 0:
                axes[1, i].set_ylabel("Accuracy", fontsize=12)
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_accuracy_comparison.png'), dpi=300)
        plt.close()
        
    def visualize_facies(self, n_clusters_list=None):
        """
        Visualize facies clusters for multiple cluster configurations
        
        Args:
            n_clusters_list (list): List of cluster numbers to visualize
        """
        if n_clusters_list is None:
            n_clusters_list = sorted([int(d) for d in os.listdir(self.output_dir) 
                                    if os.path.isdir(os.path.join(self.output_dir, d)) 
                                    and d.isdigit()])
            
        # Set up figure
        fig, axes = plt.subplots(1, len(n_clusters_list), figsize=(5*len(n_clusters_list), 15), sharey=True)
        
        # Set up colormap
        cmap = plt.cm.viridis
        
        # Plot facies for each cluster configuration
        for i, n_clusters in enumerate(n_clusters_list):
            # Load facies data
            facies_file = os.path.join(self.output_dir, str(n_clusters), f'facies_for_{n_clusters}.xlsx')
            if not os.path.exists(facies_file):
                print(f"Facies file for {n_clusters} clusters not found.")
                continue
                
            facies_df = pd.read_excel(facies_file)
            
            # Create a 2D array for imshow
            facies_array = np.vstack((facies_df['Facies_pred'].values, facies_df['Facies_pred'].values)).T
            
            # Plot facies
            im = axes[i].imshow(facies_array, aspect='auto', cmap=cmap, 
                              extent=[0, 1, facies_df['DEPTH'].max(), facies_df['DEPTH'].min()])
            axes[i].set_title(f"Cluster {n_clusters}", fontsize=14)
            axes[i].set_xlim(0, 1)
            axes[i].set_xticks([])
            
            if i == 0:
                axes[i].set_ylabel("Depth", fontsize=14)
                
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Facies', fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(self.output_dir, 'facies_comparison.png'), dpi=300)
        plt.close()
        
    def visualize_predicted_results(self, n_clusters_list=None, run_ids=None):
        """
        Visualize predicted results for multiple cluster configurations
        
        Args:
            n_clusters_list (list): List of cluster numbers to visualize
            run_ids (dict): Dictionary mapping cluster numbers to run IDs to use (default: use run_id=1)
        """
        if n_clusters_list is None:
            n_clusters_list = sorted([int(d) for d in os.listdir(self.output_dir) 
                                    if os.path.isdir(os.path.join(self.output_dir, d)) 
                                    and d.isdigit()])
            
        if run_ids is None:
            run_ids = {n: 1 for n in n_clusters_list}
            
        # Load facies data for depth values
        facies_file = os.path.join(self.output_dir, str(n_clusters_list[0]), f'facies_for_{n_clusters_list[0]}.xlsx')
        if not os.path.exists(facies_file):
            print(f"Facies file for {n_clusters_list[0]} clusters not found.")
            return
            
        facies_df = pd.read_excel(facies_file)
        
        # Create dataframe to hold all predictions
        results_df = pd.DataFrame(index=facies_df.index)
        results_df['DEPTH'] = facies_df['DEPTH']
        
        # Add true labels and predictions for each cluster configuration
        true_label_set = set()
        for n_clusters in n_clusters_list:
            results_file = os.path.join(self.output_dir, str(n_clusters), 'results', 
                                      f'Results_{n_clusters}{"" if run_ids[n_clusters] == 1 else "_new" + str(run_ids[n_clusters])}.xlsx')
            
            if not os.path.exists(results_file):
                print(f"Results file for {n_clusters} clusters (run {run_ids[n_clusters]}) not found.")
                continue
                
            cluster_results = pd.read_excel(results_file)
            
            # Add predicted labels
            results_df[f'Pred_{n_clusters}'] = cluster_results['Predicted']
            
            # Add true labels (should be the same for all configurations)
            if 'True_Label' not in results_df.columns:
                results_df['True_Label'] = cluster_results['True']
                
            # Collect unique label values
            true_label_set.update(cluster_results['True'].unique())
            
        # Convert labels to numeric values for visualization
        label_to_num = {label: i for i, label in enumerate(sorted(true_label_set))}
        
        # Convert true labels
        results_df['True_Num'] = results_df['True_Label'].map(label_to_num)
        
        # Convert predicted labels
        for n_clusters in n_clusters_list:
            if f'Pred_{n_clusters}' in results_df.columns:
                results_df[f'Pred_{n_clusters}_Num'] = results_df[f'Pred_{n_clusters}'].map(label_to_num)
            
        # Set up figure
        fig, axes = plt.subplots(1, len(n_clusters_list) + 1, figsize=(5*(len(n_clusters_list) + 1), 15), sharey=True)
        
        # Set up colormap
        cmap = plt.cm.viridis
        
        # Plot true labels
        true_array = np.vstack((results_df['True_Num'].values, results_df['True_Num'].values)).T
        im = axes[0].imshow(true_array, aspect='auto', cmap=cmap, 
                         extent=[0, 1, results_df['DEPTH'].max(), results_df['DEPTH'].min()])
        axes[0].set_title("True Labels", fontsize=14)
        axes[0].set_xlim(0, 1)
        axes[0].set_xticks([])
        axes[0].set_ylabel("Depth", fontsize=14)
        
        # Plot predicted labels for each cluster configuration
        for i, n_clusters in enumerate(n_clusters_list):
            if f'Pred_{n_clusters}_Num' not in results_df.columns:
                continue
                
            pred_array = np.vstack((results_df[f'Pred_{n_clusters}_Num'].values, 
                                  results_df[f'Pred_{n_clusters}_Num'].values)).T
            im = axes[i+1].imshow(pred_array, aspect='auto', cmap=cmap, 
                               extent=[0, 1, results_df['DEPTH'].max(), results_df['DEPTH'].min()])
            axes[i+1].set_title(f"Predicted (Clusters: {n_clusters})", fontsize=14)
            axes[i+1].set_xlim(0, 1)
            axes[i+1].set_xticks([])
            
        # Add colorbar with label mapping
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=list(range(len(label_to_num))))
        cbar.set_ticklabels([label for label, _ in sorted(label_to_num.items(), key=lambda x: x[1])])
        cbar.set_label('Rock Quality', fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(self.output_dir, 'prediction_comparison.png'), dpi=300)
        plt.close()
    
    def run_multiple_models(self, n_clusters_list, num_runs=4):
        """
        Train multiple models for each cluster configuration
        
        Args:
            n_clusters_list (list): List of cluster numbers to use
            num_runs (int): Number of runs for each configuration
            
        Returns:
            dict: Dictionary with best model for each configuration
        """
        best_models = {}
        
        for n_clusters in n_clusters_list:
            best_accuracy = 0
            best_run = None
            
            # Run multiple models
            for run_id in range(1, num_runs + 1):
                print(f"\n=== Training model for {n_clusters} clusters (Run {run_id}/{num_runs}) ===")
                model_results = self.build_gcn_model(n_clusters=n_clusters, run_id=run_id)
                
                # Check if this run has better accuracy
                current_accuracy = model_results['test_metrics'][1]  # Accuracy is usually the second metric
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_run = run_id
            
            best_models[n_clusters] = {
                'run_id': best_run,
                'accuracy': best_accuracy
            }
            
            print(f"\n=== Best model for {n_clusters} clusters: Run {best_run} (Accuracy: {best_accuracy:.4f}) ===")
            
        return best_models
        
    def build_gcn_model(self, n_clusters, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, layer_sizes=None, run_id=1):
        """
        Build and train Graph Convolutional Network model
        
        Args:
            n_clusters (int): Number of clusters
            train_ratio (float): Ratio of data for training
            val_ratio (float): Ratio of data for validation
            test_ratio (float): Ratio of data for testing
            layer_sizes (list): List of layer sizes for GCN
            run_id (int): Run identifier (for multiple runs with the same configuration)
            
        Returns:
            dict: Dictionary with model and evaluation results
        """
        # Set default layer sizes if not provided
        if layer_sizes is None:
            layer_sizes = [16, 16]
            
        # Create directory for results
        result_dir = os.path.join(self.output_dir, str(n_clusters), 'results')
        os.makedirs(result_dir, exist_ok=True)
        
        # Load node and edge data
        cluster_dir = os.path.join(self.output_dir, str(n_clusters))
        
        # Load node connections
        cora_cites_file = os.path.join(cluster_dir, f'BF_full_node_{n_clusters}.txt')
        cora_cites = pd.read_csv(cora_cites_file, sep='\t', header=None, names=['source', 'target'])
        
        # Load edge data
        cora_content_file = os.path.join(cluster_dir, f'BF_full_edge_{n_clusters}.txt')
        feature_names = [f'w{i}' for i in range(14)]
        cora_content = pd.read_csv(
            cora_content_file,
            sep='\t',
            header=None,
            names=['DEPTH', *feature_names, 'RESULTS']
        )
        
        # Set index for content data
        cora_content_indexed = cora_content.set_index('DEPTH')
        
        # Create graph
        features = cora_content_indexed.drop(columns='RESULTS')
        G = StellarGraph({"paper": features}, {"cites": cora_cites})
        
        # Get node subjects
        node_subjects = cora_content_indexed['RESULTS']
        
        # Split data
        train_subjects, test_subjects = model_selection.train_test_split(
            node_subjects, train_size=train_ratio, stratify=node_subjects
        )
        val_subjects, test_subjects = model_selection.train_test_split(
            test_subjects, train_size=val_ratio/(val_ratio+test_ratio), stratify=test_subjects
        )
        
        # Encode targets
        target_encoding = preprocessing.LabelBinarizer()
        train_targets = target_encoding.fit_transform(train_subjects)
        val_targets = target_encoding.transform(val_subjects)
        test_targets = target_encoding.transform(test_subjects)
        
        # Create generator
        generator = FullBatchNodeGenerator(G, method="gcn")
        
        # Create train, validation, and test generators
        train_gen = generator.flow(train_subjects.index, train_targets)
        val_gen = generator.flow(val_subjects.index, val_targets)
        test_gen = generator.flow(test_subjects.index, test_targets)
        
        # Build GCN model
        gcn = GCN(
            layer_sizes=layer_sizes,
            activations=["relu", "relu"],
            generator=generator,
            dropout=0.5
        )
        
        # Create input and output tensors
        x_inp, x_out = gcn.in_out_tensors()
        
        # Add final dense layer
        predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
        
        # Create model
        model = Model(inputs=x_inp, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.01),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )
        
        # Create early stopping callback
        es_callback = EarlyStopping(
            monitor="val_acc",
            patience=50,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            train_gen,
            epochs=200,
            validation_data=val_gen,
            verbose=2,
            shuffle=False,
            callbacks=[es_callback]
        )
        
        # Save training history
        history_df = pd.DataFrame(history.history)
        
        if run_id == 1:
            history_df.to_excel(os.path.join(result_dir, f'History_{n_clusters}.xlsx'))
        else:
            history_df.to_excel(os.path.join(result_dir, f'History_{n_clusters}_new{run_id}.xlsx'))
        
        # Evaluate model on test set
        test_metrics = model.evaluate(test_gen)
        print("\nTest Set Metrics:")
        for name, val in zip(model.metrics_names, test_metrics):
            print(f"\t{name}: {val:.4f}")
            
        # Make predictions for all nodes
        all_nodes = node_subjects.index
        all_gen = generator.flow(all_nodes)
        all_predictions = model.predict(all_gen)
        
        # Convert predictions to labels
        node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
        
        # Create results dataframe
        results_df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
        
        if run_id == 1:
            results_df.to_excel(os.path.join(result_dir, f'Results_{n_clusters}.xlsx'))
        else:
            results_df.to_excel(os.path.join(result_dir, f'Results_{n_clusters}_new{run_id}.xlsx'))
        
        # Calculate confusion matrix and classification report
        cm = confusion_matrix(node_subjects, node_predictions)
        cr = classification_report(node_subjects, node_predictions, output_dict=True)
        
        # Convert classification report to dataframe
        cr_df = pd.DataFrame(cr).transpose()
        
        if run_id == 1:
            cr_df.to_excel(os.path.join(result_dir, f'ClassReport_{n_clusters}.xlsx'))
        else:
            cr_df.to_excel(os.path.join(result_dir, f'ClassReport_{n_clusters}_new{run_id}.xlsx'))
        
        model_results = {
            'model': model,
            'history': history,
            'history_df': history_df,
            'test_metrics': test_metrics,
            'predictions': node_predictions,
            'results_df': results_df,
            'confusion_matrix': cm,
            'classification_report': cr_df
        }
        
        # Store model results
        if n_clusters not in self.models:
            self.models[n_clusters] = {}
        
        self.models[n_clusters][run_id] = model_results
        
        return model_results
        
def main():
    """Main function to demonstrate the workflow"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Well Log Analysis with Graph Convolutional Networks')
    parser.add_argument('--las_file', type=str, required=True, help='Path to LAS file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--min_clusters', type=int, default=5, help='Minimum number of clusters')
    parser.add_argument('--max_clusters', type=int, default=10, help='Maximum number of clusters')
    parser.add_argument('--num_runs', type=int, default=4, help='Number of runs for each cluster configuration')
    parser.add_argument('--feature_columns', type=str, nargs='+', 
                       help='Columns to use as features (space-separated)')
    
    args = parser.parse_args()
    
    # Create Well Log Analysis object
    wla = WellLogAnalysis(args.las_file, args.output_dir)
    
    # Load LAS file
    if not wla.load_las_file():
        print("Failed to load LAS file.")
        return
    
    # Calculate petrophysical properties
    wla.calculate_petrophysical_properties()
    
    # Prepare features
    wla.prepare_features(args.feature_columns)
    
    # Perform KMeans clustering for different cluster counts
    cluster_results = wla.perform_kmeans_clustering(
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters
    )
    
    # Select cluster counts based on silhouette scores
    silhouette_scores = cluster_results['silhouette_scores']
    selected_clusters = sorted(silhouette_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    selected_clusters = [n for n, _ in selected_clusters]
    
    print(f"\nSelected cluster counts: {selected_clusters}")
    
    # Create datasets for selected cluster counts
    for n_clusters in range(args.min_clusters, args.max_clusters + 1):
        print(f"\n=== Creating dataset for {n_clusters} clusters ===")
        
        # Create cluster dataset
        wla.create_cluster_dataset(n_clusters)
        
        # Generate adjacency matrix
        wla.generate_adjacency_matrix(n_clusters)
    
    # Run multiple models for each cluster configuration
    best_models = wla.run_multiple_models(
        n_clusters_list=range(args.min_clusters, args.max_clusters + 1),
        num_runs=args.num_runs
    )
    
    # Get best run IDs for each cluster configuration
    best_run_ids = {n: info['run_id'] for n, info in best_models.items()}
    
    # Visualize results
    wla.visualize_loss_accuracy(
        n_clusters_list=range(args.min_clusters, args.max_clusters + 1),
        run_ids=best_run_ids
    )
    
    wla.visualize_facies(
        n_clusters_list=range(args.min_clusters, args.max_clusters + 1)
    )
    
    wla.visualize_predicted_results(
        n_clusters_list=range(args.min_clusters, args.max_clusters + 1),
        run_ids=best_run_ids
    )
    
    print("\nAnalysis complete. Results saved to:", args.output_dir)

if __name__ == "__main__":
    main()