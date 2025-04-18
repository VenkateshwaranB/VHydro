import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

# Configure the page
st.set_page_config(
    page_title="VHydro - Hydrocarbon Quality Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

# Simple login functionality
def show_login_page():
    st.title("Login to VHydro")
    
    # Demo credentials
    demo_users = {
        "user@example.com": "Password123",
        "admin@vhydro.com": "Admin123"
    }
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if email in demo_users and demo_users[email] == password:
                st.session_state['logged_in'] = True
                st.session_state['email'] = email
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid email or password")
    
    # Demo credentials hint
    st.info("Demo credentials: user@example.com / Password123")

# Sidebar navigation - using standard Streamlit components
def sidebar_navigation():
    st.sidebar.title("VHydro")
    st.sidebar.subheader("Navigation")
    
    # User info if logged in
    if st.session_state.get('logged_in', False):
        st.sidebar.info(f"Logged in as: {st.session_state.get('email', 'User')}")
        if st.sidebar.button("Logout"):
            for key in ['logged_in', 'email']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Navigation buttons
    pages = {
        "Home": "🏠",
        "Dataset Preparation": "📊",
        "Model Workflow": "🔄",
        "Analysis Tool": "🔍",
        "Results Visualization": "📈"
    }
    
    # Create navigation buttons with icons
    for page, icon in pages.items():
        # Highlight current page
        if st.session_state['current_page'] == page:
            button_label = f"**{icon} {page}**"
        else:
            button_label = f"{icon} {page}"
            
        if st.sidebar.button(button_label, key=f"nav_{page}"):
            st.session_state['current_page'] = page
            st.rerun()
    
    # Configuration section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Configuration")
    
    min_clusters = st.sidebar.slider("Min Clusters", 2, 15, 5)
    max_clusters = st.sidebar.slider("Max Clusters", min_clusters, 15, 10)
    
    with st.sidebar.expander("Advanced Settings"):
        train_ratio = st.slider("Training Ratio", 0.5, 0.9, 0.8, 0.05)
        validation_ratio = st.slider("Validation Ratio", 0.05, 0.3, 0.1, 0.05)
        test_ratio = st.slider("Test Ratio", 0.05, 0.3, 0.1, 0.05)
        
        # Adjust ratios if they don't sum to 1
        total = train_ratio + validation_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            st.warning(f"Ratios sum to {total:.2f}, should be 1.0")
            
        hidden_channels = st.number_input("Hidden Channels", 4, 64, 16, 4)
        num_runs = st.number_input("Number of Runs", 1, 10, 4, 1)
    
    return {
        "min_clusters": min_clusters,
        "max_clusters": max_clusters,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "hidden_channels": hidden_channels,
        "num_runs": num_runs
    }

# Home page
def home_page():
    st.title("VHydro - Hydrocarbon Quality Prediction")
    st.subheader("Advanced Graph Convolutional Network for Petrophysical Analysis")
    
    # About section
    st.markdown("### About VHydro")
    st.info("""
    **VHydro** is an advanced tool for hydrocarbon quality prediction using well log data.
    It combines traditional petrophysical analysis with machine learning techniques
    to provide accurate predictions of reservoir quality.
    
    The tool uses Graph Convolutional Networks (GCN) to model the complex relationships
    between different petrophysical properties and depth values, enabling more accurate
    classification of hydrocarbon potential zones.
    """)
    
    # Key features
    st.markdown("### Key Features")
    
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
    
    # Getting started
    st.markdown("### Getting Started")
    st.write("Use the sidebar to navigate between different sections of the application.")
    
    if st.button("Start Analysis", key="start_btn"):
        st.session_state['current_page'] = "Dataset Preparation"
        st.rerun()

# Dataset preparation page
def dataset_preparation_page():
    st.title("Dataset Preparation")
    st.write("Upload your LAS file containing well log data.")
    
    uploaded_file = st.file_uploader("Upload LAS file", type=["las"])
    use_sample = st.checkbox("Use sample data instead", value=False)
    
    # Parameters section
    st.markdown("### Calculation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fluid Parameters")
        matrix_density = st.number_input("Matrix Density (g/cc)", min_value=2.0, max_value=3.0, value=2.65, step=0.01)
        fluid_density = st.number_input("Fluid Density (g/cc)", min_value=0.5, max_value=1.5, value=1.0, step=0.01)
    
    with col2:
        st.subheader("Archie Parameters")
        a_param = st.number_input("Tortuosity Factor (a)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        m_param = st.number_input("Cementation Exponent (m)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        n_param = st.number_input("Saturation Exponent (n)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    
    # Feature selection
    st.markdown("### Feature Selection")
    
    # Available logs
    logs = {
        "GR": "Gamma Ray",
        "RHOB": "Bulk Density",
        "NPHI": "Neutron Porosity",
        "RT": "Resistivity",
        "DT": "Sonic Travel Time",
        "VSHALE": "Calculated Shale Volume",
        "PHI": "Calculated Porosity",
        "SW": "Calculated Water Saturation"
    }
    
    # Organize checkboxes in columns
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    
    selected_logs = []
    for i, (log_code, log_name) in enumerate(logs.items()):
        with columns[i % 4]:
            if st.checkbox(f"{log_code}: {log_name}", value=log_code in ["GR", "RHOB", "VSHALE", "PHI", "SW"]):
                selected_logs.append(log_code)
    
    # Process button
    if (uploaded_file or use_sample) and st.button("Process Data"):
        with st.spinner('Processing data...'):
            # Simulate processing with a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Show success message
            st.success("Data processed successfully!")
            
            # Show sample data
            st.markdown("### Processed Data Preview")
            
            # Create sample dataframe
            sample_data = {
                "DEPTH": [1000.0, 1000.5, 1001.0, 1001.5, 1002.0],
                "GR": [75.2, 78.5, 80.1, 76.3, 72.8],
                "RHOB": [2.45, 2.48, 2.52, 2.47, 2.44],
                "VSHALE": [0.35, 0.38, 0.42, 0.36, 0.32],
                "PHI": [0.18, 0.16, 0.14, 0.17, 0.19],
                "SW": [0.45, 0.48, 0.52, 0.47, 0.43]
            }
            
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df)
            
            # Continue button
            if st.button("Continue to Model Workflow"):
                st.session_state['current_page'] = "Model Workflow"
                st.rerun()
    else:
        st.info("Please upload a LAS file or use the sample data to continue.")

# Model workflow page
def model_workflow_page():
    st.title("Model Workflow")
    
    st.info("""
    **Graph Convolutional Network (GCN) Workflow**
    
    This page outlines the workflow for creating a Graph Convolutional Network model for
    hydrocarbon quality prediction. The process includes dataset creation, graph construction,
    model training, and evaluation.
    """)
    
    # Workflow steps
    steps = [
        {"title": "Dataset Preparation", "status": "completed", "desc": "Process well log data and calculate petrophysical properties"},
        {"title": "K-means Clustering", "status": "active", "desc": "Perform K-means clustering to identify facies"},
        {"title": "Graph Construction", "status": "pending", "desc": "Create nodes and edges for the graph dataset"},
        {"title": "GCN Training", "status": "pending", "desc": "Train the Graph Convolutional Network model"},
        {"title": "Model Evaluation", "status": "pending", "desc": "Evaluate model performance and visualize results"}
    ]
    
    # Display workflow steps
    for i, step in enumerate(steps):
        status_color = "green" if step["status"] == "completed" else "blue" if step["status"] == "active" else "gray"
        
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown(f"**{i+1}.**")
        with col2:
            st.markdown(f"**{step['title']}** - *{step['status']}*")
            st.markdown(f"{step['desc']}")
        
        if i < len(steps) - 1:  # Don't add space after the last step
            st.markdown("---")
    
    # K-means Clustering section (active step)
    st.markdown("### K-means Clustering for Facies Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=3, max_value=15, value=7, key="kmeans_clusters")
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=42, key="kmeans_random")
    
    with col2:
        scaling_method = st.radio("Feature Scaling Method", 
                               ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                               horizontal=True)
        
        optimize_silhouette = st.checkbox("Automatically determine optimal cluster count", value=True)
    
    # Run clustering button
    if st.button("Run Clustering"):
        with st.spinner('Performing clustering...'):
            # Simulate clustering with a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Show clustering results
            st.success("Clustering completed successfully!")
            
            # Show some results
            st.markdown("### Silhouette Score Analysis")
            
            # Create a simple chart
            fig, ax = plt.subplots(figsize=(8, 5))
            
            cluster_range = range(3, 11)
            silhouette_scores = [0.42, 0.48, 0.52, 0.58, 0.55, 0.51, 0.47, 0.43]
            
            ax.plot(cluster_range, silhouette_scores, 'o-', linewidth=2, color='blue')
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("Silhouette Score")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Find the best score
            best_idx = silhouette_scores.index(max(silhouette_scores))
            best_n = cluster_range[best_idx]
            ax.axvline(x=best_n, color='red', linestyle='--', alpha=0.7)
            ax.text(best_n + 0.1, max(silhouette_scores) - 0.02, 
                   f"Optimal: {best_n} clusters", 
                   color='red')
            
            st.pyplot(fig)
            
            # Continue button
            if st.button("Continue to Graph Construction"):
                st.info("Next: Graph Construction")

# Analysis tool page
def analysis_tool_page(config):
    # Check if user is logged in
    if not st.session_state.get('logged_in', False):
        show_login_page()
        return
    
    st.title("Analysis Tool")
    st.write("Analyze your well log data using the VHydro Graph Convolutional Network model.")
    
    # Create tabs for different analysis steps
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Process", "Graph Creation", "Model Training", "Results"])
    
    with tab1:
        st.header("Data Upload and Processing")
        
        uploaded_file = st.file_uploader("Upload LAS file", type=["las"], key="analysis_uploader")
        use_sample = st.checkbox("Use sample data instead", value=not uploaded_file, key="analysis_sample")
        
        # Process button
        if (uploaded_file or use_sample) and st.button("Process Data", key="analysis_process"):
            with st.spinner('Processing well log data...'):
                # Simulate processing with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Show success message
                st.success("Data processed successfully!")
    
    with tab2:
        st.header("Graph Dataset Creation")
        
        # Show options for graph creation
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=5, max_value=10, value=7, 
                                key="gcn_clusters")
            train_test_split = st.slider("Train/Test Split", min_value=0.6, max_value=0.9, value=0.8,
                                      key="gcn_split")
        
        with col2:
            connection_strategy = st.radio("Node Connection Strategy",
                                       ["K-Nearest Neighbors", "Distance Threshold", "Facies-based"],
                                       index=2,
                                       key="gcn_connection")
            
            connection_param = st.slider("Connection Parameter", 
                                      min_value=1, max_value=10, value=5,
                                      key="gcn_param")
        
        # Create graph button
        if st.button("Generate Graph Dataset", key="graph_generate"):
            with st.spinner('Creating graph dataset...'):
                # Simulate processing with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Success message
                st.success("Graph dataset created successfully!")
                
                # Show statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nodes", "458")
                col2.metric("Edges", "1,245")
                col3.metric("Facies", "7")
                col4.metric("Quality Classes", "4")
    
    with tab3:
        st.header("GCN Model Training")
        
        # Model configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            hidden_channels = st.slider("Hidden Channels", min_value=8, max_value=64, value=16, step=8,
                                     key="gcn_channels")
            learning_rate = st.select_slider("Learning Rate", 
                                         options=[0.001, 0.005, 0.01, 0.05, 0.1], 
                                         value=0.01,
                                         key="gcn_lr")
        
        with col2:
            epochs = st.slider("Training Epochs", min_value=50, max_value=500, value=200, step=50,
                            key="gcn_epochs")
            dropout = st.slider("Dropout Rate", min_value=0.1, max_value=0.7, value=0.5, step=0.1,
                             key="gcn_dropout")
        
        # Train model button
        if st.button("Train GCN Model", key="model_train"):
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
        st.header("Results Analysis")
        
        # Sample visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ["Very Low", "Low", "Moderate", "High"]
        values = [22, 35, 30, 13]
        
        ax.bar(categories, values, color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"])
        ax.set_ylabel('Percentage')
        ax.set_title('Distribution of Hydrocarbon Potential')
        
        st.pyplot(fig)
        
        # Export options
        st.subheader("Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="Download CSV",
                data="sample,data",
                file_name="hydrocarbon_predictions.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="Download Report",
                data="sample report",
                file_name="hydrocarbon_report.pdf",
                mime="application/pdf"
            )
        
        with col3:
            st.download_button(
                label="Download Visualization",
                data="sample image",
                file_name="hydrocarbon_visualization.png",
                mime="image/png"
            )

# Results visualization page
def results_visualization_page():
    # Check if user is logged in
    if not st.session_state.get('logged_in', False):
        show_login_page()
        return
    
    st.title("Results Visualization")
    st.write("Interactive visualization of hydrocarbon potential prediction results.")
    
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
    
    # Visualization controls
    st.sidebar.subheader("Visualization Controls")
    
    # Depth range slider
    depth_min = float(df["DEPTH"].min())
    depth_max = float(df["DEPTH"].max())
    
    depth_range = st.sidebar.slider(
        "Depth Range",
        min_value=depth_min,
        max_value=depth_max,
        value=(depth_min, depth_min + 50),
        step=5.0
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
        fig, axes = plt.subplots(1, 5, figsize=(15, 8), sharey=True)
        
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
        
        # Plot Quality
        cmap = mcolors.ListedColormap(["#e74c3c", "#f39c12", "#3498db", "#2ecc71"])
        quality_2d = np.vstack((filtered_data["QUALITY_NUMERIC"], filtered_data["QUALITY_NUMERIC"])).T
        im = axes[4].imshow(quality_2d, aspect='auto', cmap=cmap,
                          extent=[0, 1, filtered_data["DEPTH"].max(), filtered_data["DEPTH"].min()])
        axes[4].set_title('HC Potential')
        axes[4].set_xticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[4], ticks=[0.125, 0.375, 0.625, 0.875])
        cbar.set_ticklabels(["Very Low", "Low", "Moderate", "High"])
        
        plt.tight_layout()
        st.pyplot(fig)
        
    elif viz_type == "Cross-Plot":
        st.subheader("Cross-Plot Analysis")
        
        # Select properties for cross-plot
        x_property = st.selectbox(
            "X-Axis Property",
            ["PHI", "GR", "RHOB", "SW"],
            index=0
        )
        
        y_property = st.selectbox(
            "Y-Axis Property",
            ["PHI", "GR", "RHOB", "SW"],
            index=1
        )
        
        # Create cross-plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot with color based on quality
        scatter = ax.scatter(
            filtered_data[x_property],
            filtered_data[y_property],
            c=filtered_data["QUALITY_NUMERIC"],
            cmap=mcolors.ListedColormap(["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]),
            s=50,
            alpha=0.7
        )
        
        # Set labels and title
        ax.set_xlabel(x_property)
        ax.set_ylabel(y_property)
        ax.set_title(f'{x_property} vs {y_property} Cross-Plot')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, ticks=[0.125, 0.375, 0.625, 0.875])
        cbar.set_ticklabels(["Very Low", "Low", "Moderate", "High"])
        cbar.set_label('Hydrocarbon Potential')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    elif viz_type == "Statistical Summary":
        st.subheader("Statistical Summary")
        
        # Display statistics
        st.dataframe(filtered_data.describe())
        
        # Quality distribution
        st.subheader("Hydrocarbon Potential Distribution")
        
        quality_counts = filtered_data["QUALITY"].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        ax.bar(
            quality_counts.index,
            quality_counts.values,
            color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
        )
        
        ax.set_ylabel('Count')
        ax.set_title('Hydrocarbon Potential Distribution')
        
        # Add count labels
        for i, count in enumerate(quality_counts.values):
            ax.text(i, count + 1, str(count), ha='center')
        
        st.pyplot(fig)
        
        # Show key metrics
        st.subheader("Key Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate metrics
        high_quality_count = filtered_data[filtered_data["QUALITY"].isin(["High", "Moderate"])].shape[0]
        total_count = filtered_data.shape[0]
        quality_ratio = high_quality_count / total_count
        
        col1.metric("Quality Ratio (High+Moderate)", f"{quality_ratio:.1%}")
        col2.metric("Net Pay", f"{high_quality_count * 0.5:.1f}m")
        
        avg_porosity = filtered_data[filtered_data["QUALITY"].isin(["High", "Moderate"])]["PHI"].mean()
        col3.metric("Average Porosity (High+Moderate)", f"{avg_porosity:.1%}")

# Main function
def main():
    # Get config from sidebar
    config = sidebar_navigation()
    
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

if __name__ == "__main__":
    main()
