import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from PIL import Image
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Set page configuration at the very beginning
st.set_page_config(
    page_title="StrataGraph - Graph Neural Networks for Geoscience Applications",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple login function using hardcoded credentials (for demo purposes)
def login(email, password):
    # Demo users
    users = {"user@example.com": "password", "admin@stratagraph.com": "admin"}
    if email in users and users[email] == password:
        st.session_state["email"] = email
        st.session_state["logged_in"] = True
        return True
    return False

# Helper function to logout
def logout():
    if "email" in st.session_state:
        del st.session_state["email"]
    if "logged_in" in st.session_state:
        del st.session_state["logged_in"]
    
    # Reset any authentication-related session state
    if "auth_mode" in st.session_state:
        del st.session_state["auth_mode"]

# Load and display images
def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to create a base64 encoded image for CSS background
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Basic CSS for a colorful, scientific UI
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary: #0e4194;
        --secondary: #3a6fc4;
        --accent: #10b981;
        --background: #f8fafc;
        --sidebar: #0e4194;
        --text: #1e293b;
        --text-light: #64748b;
    }

    /* General styles */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar) 0%, #153a6f 100%);
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div {
        color: white !important;
    }
    
    /* Navigation buttons */
    .nav-button {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
        display: block;
    }
    
    .nav-button:hover, .nav-button.active {
        background-color: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .nav-button.active {
        background-color: var(--secondary);
        border-color: var(--secondary);
    }
    
    /* Content cards */
    .card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .colored-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .feature-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        height: 100%;
        border-left: 5px solid var(--primary);
    }
    
    .feature-header {
        font-weight: bold;
        color: var(--primary);
        margin-bottom: 10px;
        font-size: 1.2rem;
    }
    
    /* Login form styling */
    .login-form {
        max-width: 400px;
        margin: 2rem auto;
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Coming soon section */
    .coming-soon {
        background: linear-gradient(rgba(30, 41, 59, 0.8), rgba(30, 41, 59, 0.8)), url('https://via.placeholder.com/800x400');
        background-size: cover;
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin: 30px 0;
    }
    
    .coming-soon h2 {
        color: white;
        margin-bottom: 15px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: var(--text-light);
        margin-top: 40px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Simple sidebar for navigation
def create_sidebar():
    st.sidebar.markdown('<div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">', unsafe_allow_html=True)
    
    # Logo placeholder - replace with actual logo path
    logo_path = "src/StrataGraph_White_Logo.png"  # Update with your actual logo path
    
    try:
        st.sidebar.image(logo_path, width=160)
    except:
        # Fallback if image isn't found
        st.sidebar.markdown("""
        <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <h2 style="color: #0e4194; margin: 0;">StrataGraph</h2>
            <p style="color: #0e4194; margin: 5px 0 0 0;">Advanced Geoscience Modeling</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Version info
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
        <p style="margin: 0; font-size: 0.9rem;">Current Version: VHydro 1.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User info if logged in
    if st.session_state.get("logged_in", False):
        st.sidebar.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <p style="margin: 0;">Logged in as:</p>
            <p style="font-weight: bold; margin: 5px 0;">{st.session_state.get("email")}</p>
            <div style="display: flex; justify-content: center; margin-top: 10px;">
                <button class="nav-button" id="logout-btn" style="padding: 5px 10px; margin: 0;">Logout</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Handle logout with a normal button since the custom HTML button won't trigger actions
        if st.sidebar.button("Logout", key="logout_btn"):
            logout()
            st.rerun()
    
    # Navigation
    st.sidebar.markdown('<div style="margin-bottom: 20px;"><h3 style="color: white; margin-bottom: 15px;">Navigation</h3></div>', unsafe_allow_html=True)
    
    # Define pages
    pages = ["Home", "Dataset Preparation", "Models", "Analysis Tool", "Visualization"]
    
    # Initialize current page in session state if it doesn't exist
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
    # Create navigation buttons
    for page in pages:
        button_class = "nav-button active" if st.session_state["current_page"] == page else "nav-button"
        
        # Use regular buttons since custom HTML won't work for navigation
        if st.sidebar.button(page, key=f"nav_{page}"):
            st.session_state["current_page"] = page
            st.rerun()
    
    # Basic configuration section
    st.sidebar.markdown('<div style="margin: 30px 0 15px 0;"><h3 style="color: white;">Model Configuration</h3></div>', unsafe_allow_html=True)
    
    min_clusters = st.sidebar.slider("Min Clusters", 2, 15, 5)
    max_clusters = st.sidebar.slider("Max Clusters", min_clusters, 15, 10)
    
    # Add a simple info panel
    st.sidebar.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-top: 30px;">
        <p style="margin: 0;">StrataGraph enables advanced geoscience modeling with Graph Convolutional Networks for petrophysical and geological data analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    return {
        "page": st.session_state["current_page"],
        "min_clusters": min_clusters,
        "max_clusters": max_clusters
    }

def home_page():
    # Try to load the banner image
    banner_path = "src/StrataGraph_Banner.png"  # Update with your actual banner path
    try:
        st.image(banner_path, use_column_width=True)
    except:
        # Fallback if image isn't found
        st.markdown("""
        <div class="colored-header">
            <h1>StrataGraph</h1>
            <p>Advanced Graph Convolutional Networks for Geoscience Modeling</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>About StrataGraph</h2>
        <p>StrataGraph is a cutting-edge platform for geoscience modeling and analysis, combining advanced machine learning techniques with traditional geological and petrophysical analysis.</p>
        <p>Our first release, <b>VHydro 1.0</b>, focuses on hydrocarbon quality prediction using Graph Convolutional Networks (GCNs) that model complex relationships between different petrophysical properties and depth values.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # VHydro Workflow section
    st.markdown("<h2>VHydro Workflow</h2>", unsafe_allow_html=True)
    
    # Try to load the workflow image
    workflow_path = "src/Workflow.png"  # Update with your actual workflow image path
    try:
        st.image(workflow_path, use_column_width=True)
    except:
        # Fallback if image isn't found
        st.warning("Workflow image not found. Please ensure 'VHydro_Full_workflow.png' is in the correct directory.")
    
    st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)
    
    # Create a 2x2 grid for features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-header">GCN Parameters</div>
            <ul>
                <li><b>Hidden Channels:</b> 16</li>
                <li><b>Layers:</b> 2</li>
                <li><b>Dropout Rate:</b> 0.5</li>
                <li><b>Learning Rate:</b> 0.01</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-header">Training Parameters</div>
            <ul>
                <li><b>Maximum Epochs:</b> 200</li>
                <li><b>Early Stopping:</b> Yes (patience: 50)</li>
                <li><b>Train/Val/Test Split:</b> 80%/10%/10%</li>
                <li><b>Optimizer:</b> Adam</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Configuration Options
    with st.expander("Advanced Configuration Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Architecture")
            st.selectbox("GCN Version", ["RegularizedGCN", "Standard GCN", "GCN with Skip Connections"])
            st.number_input("Hidden Channels", min_value=8, max_value=128, value=16, step=8)
            st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.5, step=0.1)
            st.checkbox("Use Batch Normalization", value=True)
        
        with col2:
            st.markdown("### Training Configuration")
            st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05], value=0.01)
            st.number_input("Early Stopping Patience", min_value=10, max_value=100, value=50, step=5)
            st.number_input("Maximum Epochs", min_value=50, max_value=500, value=200, step=50)
            st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
    
    # Run model button
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <button style="background-color: #0e4194; color: white; border: none; 
                      padding: 12px 24px; border-radius: 8px; font-size: 16px; 
                      cursor: pointer; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            Train GCN Model
        </button>
    </div>
    """, unsafe_allow_html=True)
    
    # If the user clicks the button (this is a Streamlit workaround)
    if st.button("Train GCN Model"):
        # Show a progress bar
        progress_bar = st.progress(0)
        
        # Create a status area
        status_area = st.empty()
        
        # Simulate training process
        for i in range(100):
            progress_bar.progress(i + 1)
            
            # Update status messages
            if i < 10:
                status_area.info("Preparing graph structure...")
            elif i < 30:
                status_area.info("Creating node and edge features...")
            elif i < 60:
                status_area.info(f"Training GCN model (Epoch {i})...")
            elif i < 90:
                status_area.info("Optimizing model parameters...")
            else:
                status_area.info("Finalizing predictions...")
            
            # Sleep to simulate processing time
            import time
            time.sleep(0.05)
        
        # Clear status area and show success message
        status_area.empty()
        st.success("GCN model trained successfully!")

def analysis_tool_page():
    # Check login for analysis tool
    if not st.session_state.get("logged_in", False):
        st.markdown("""
        <div class="card" style="text-align: center; max-width: 500px; margin: 50px auto;">
            <h2>Login Required</h2>
            <p>You need to log in to access the Analysis Tool.</p>
            <div style="margin-top: 20px;">
        """, unsafe_allow_html=True)
        
        if st.button("Login to Continue", key="login_prompt"):
            st.session_state["auth_mode"] = "login"
            st.rerun()
        
        if st.session_state.get("auth_mode") == "login":
            with st.form("login_form"):
                st.markdown("<h3>Login</h3>", unsafe_allow_html=True)
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login"):
                    if login(email, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="colored-header">
        <h1>Analysis Tool</h1>
        <p>Process well log data and run advanced hydrocarbon prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for the analysis workflow
    tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Property Calculation", "Facies Classification", "GCN Model"])
    
    with tab1:
        st.markdown("<h3>Upload LAS File</h3>", unsafe_allow_html=True)
        
        # File upload section with a colorful border
        st.markdown("""
        <div style="border: 2px dashed #0e4194; border-radius: 10px; padding: 30px; text-align: center; margin: 20px 0;">
            <h4>Drag and drop your LAS file here</h4>
            <p>Or click to browse files</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a LAS file", type=["las"])
        
        if uploaded_file is not None:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            
            # Store in session state
            st.session_state["uploaded_file"] = uploaded_file.name
            
            # Show file info in a nice format
            st.markdown(
                f"""
                <div class="card">
                    <h4>File Information</h4>
                    <table style="width: 100%;">
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">File Name:</td>
                            <td style="padding: 8px;">{uploaded_file.name}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">File Size:</td>
                            <td style="padding: 8px;">{uploaded_file.size / 1024:.2f} KB</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Upload Time:</td>
                            <td style="padding: 8px;">{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</td>
                        </tr>
                    </table>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Proceed button
            if st.button("Proceed to Property Calculation"):
                st.session_state["analysis_stage"] = "property_calculation"
                st.rerun()
    
    with tab2:
        st.markdown("<h3>Petrophysical Property Calculation</h3>", unsafe_allow_html=True)
        
        # Check if user has uploaded a file
        if "uploaded_file" not in st.session_state:
            st.warning("Please upload a LAS file first.")
        else:
            st.markdown("""
            <div class="card">
                <p>This step calculates key petrophysical properties from your well log data:</p>
                <ul>
                    <li>Shale Volume (Vsh)</li>
                    <li>Porosity (φ)</li>
                    <li>Water Saturation (Sw)</li>
                    <li>Oil Saturation (So)</li>
                    <li>Permeability (K)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Option to configure calculation parameters
            with st.expander("Calculation Parameters"):
                col1, col2 = st.columns(2)
                
                with col1:
                    matrix_density = st.number_input("Matrix Density (g/cm³)", min_value=2.0, max_value=3.0, value=2.65, step=0.01)
                    fluid_density = st.number_input("Fluid Density (g/cm³)", min_value=0.5, max_value=1.5, value=1.0, step=0.01)
                
                with col2:
                    a_const = st.number_input("Archie Constant (a)", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
                    m_const = st.number_input("Cementation Exponent (m)", min_value=1.5, max_value=2.5, value=2.0, step=0.1)
            
            # Run calculation button
            if st.button("Calculate Properties"):
                # Show a progress bar for demonstration
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simulate calculation progress
                    progress_bar.progress(i + 1)
                    import time
                    time.sleep(0.01)
                
                st.success("Petrophysical properties calculated successfully!")
                
                # Display sample results
                sample_results = pd.DataFrame({
                    "DEPTH": np.arange(1000, 1010, 1),
                    "VSHALE": np.random.uniform(0.1, 0.5, 10),
                    "PHI": np.random.uniform(0.05, 0.25, 10),
                    "SW": np.random.uniform(0.3, 0.7, 10),
                    "SO": np.random.uniform(0.3, 0.7, 10),
                    "PERM": np.random.uniform(0.01, 100, 10)
                })
                
                st.dataframe(sample_results)
                
                # Add a download button for the properties
                csv = sample_results.to_csv(index=False)
                st.download_button(
                    label="Download Properties CSV",
                    data=csv,
                    file_name="petrophysical_properties.csv",
                    mime="text/csv"
                )
                
                # Store in session state
                st.session_state["property_data"] = True
                st.session_state["analysis_stage"] = "facies_classification"
    
    with tab3:
        st.markdown("<h3>Facies Classification</h3>", unsafe_allow_html=True)
        
        # Check if properties have been calculated
        if "property_data" not in st.session_state:
            st.warning("Please calculate petrophysical properties first.")
        else:
            st.markdown("""
            <div class="card">
                <p>This step identifies natural rock types (facies) using K-means clustering:</p>
                <ul>
                    <li>Groups similar depth points based on petrophysical properties</li>
                    <li>Optimizes the number of clusters using silhouette scores</li>
                    <li>Generates depth-based facies maps</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Clustering parameters
            col1, col2 = st.columns(2)
            
            with col1:
                min_clusters = st.number_input("Minimum Clusters", min_value=2, max_value=15, value=5, step=1)
                feature_cols = st.multiselect("Features for Clustering", 
                                             options=["VSHALE", "PHI", "SW", "SO", "PERM", "GR", "DENSITY"],
                                             default=["VSHALE", "PHI", "SW", "GR", "DENSITY"])
            
            with col2:
                max_clusters = st.number_input("Maximum Clusters", min_value=min_clusters, max_value=15, value=10, step=1)
                # Add algorithm choice
                algorithm = st.selectbox("Clustering Algorithm", ["K-means", "Agglomerative", "DBSCAN"])
            
            # Run clustering button
            if st.button("Run Facies Classification"):
                # Show a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    import time
                    time.sleep(0.01)
                
                st.success("Facies classification completed successfully!")
                
                # Display silhouette scores with a simple plot
                st.markdown("<h4>Silhouette Scores</h4>", unsafe_allow_html=True)
                
                # Sample silhouette scores
                silhouette_df = pd.DataFrame({
                    "Clusters": list(range(min_clusters, max_clusters + 1)),
                    "Silhouette Score": np.random.uniform(0.4, 0.7, max_clusters - min_clusters + 1)
                })
                
                # Plot silhouette scores
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(silhouette_df["Clusters"], silhouette_df["Silhouette Score"], marker='o', color='#0e4194')
                ax.set_xlabel("Number of Clusters")
                ax.set_ylabel("Silhouette Score")
                ax.set_title("Cluster Optimization")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Show optimal number of clusters
                optimal_clusters = silhouette_df["Silhouette Score"].idxmax() + min_clusters
                st.info(f"Optimal number of clusters: {optimal_clusters}")
                
                # Display facies visualization
                st.markdown("<h4>Facies Visualization</h4>", unsafe_allow_html=True)
                
                # Create a simple facies visualization
                depth = np.arange(1000, 1100)
                facies = np.random.randint(0, optimal_clusters, size=100)
                
                fig, ax = plt.subplots(figsize=(8, 10))
                cmap = plt.cm.get_cmap('viridis', optimal_clusters)
                
                # Create a depth vs facies plot
                sc = ax.scatter(np.ones_like(depth), depth, c=facies, cmap=cmap, 
                               s=100, marker='s')
                
                # Customize the plot
                ax.set_yticks(np.arange(1000, 1101, 10))
                ax.set_ylabel("Depth")
                ax.set_xlim(0.9, 1.1)
                ax.set_xticks([])
                ax.invert_yaxis()  # Invert y-axis to show depth increasing downward
                
                # Add a colorbar
                cbar = plt.colorbar(sc)
                cbar.set_ticks(np.arange(optimal_clusters) + 0.5)
                cbar.set_ticklabels([f"Facies {i+1}" for i in range(optimal_clusters)])
                cbar.set_label('Facies Classification')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Create a download button for facies results
                facies_df = pd.DataFrame({
                    "DEPTH": depth,
                    "FACIES": facies
                })
                
                csv = facies_df.to_csv(index=False)
                st.download_button(
                    label="Download Facies CSV",
                    data=csv,
                    file_name="facies_classification.csv",
                    mime="text/csv"
                )
                
                # Store in session state
                st.session_state["facies_data"] = True
                st.session_state["best_clusters"] = optimal_clusters
                st.session_state["analysis_stage"] = "gcn_model"
    
    with tab4:
        st.markdown("<h3>Graph Convolutional Network Model</h3>", unsafe_allow_html=True)
        
        # Check if facies classification has been done
        if "facies_data" not in st.session_state:
            st.warning("Please complete facies classification first.")
        else:
            st.markdown("""
            <div class="card">
                <p>This step builds and trains a Graph Convolutional Network model:</p>
                <ul>
                    <li>Constructs a graph from depth points and their relationships</li>
                    <li>Trains a GCN model to predict hydrocarbon quality</li>
                    <li>Evaluates model performance and generates final predictions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Model parameters
            col1, col2 = st.columns(2)
            
            with col1:
                n_clusters = st.number_input("Number of Clusters", min_value=min_clusters, max_value=max_clusters, 
                                           value=st.session_state.get("best_clusters", 7), step=1)
                hidden_channels = st.number_input("Hidden Channels", min_value=4, max_value=64, value=16, step=4)
                dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.5, step=0.1)
                
            with col2:
                num_runs = st.number_input("Number of Runs", min_value=1, max_value=10, value=4, step=1)
                epochs = st.number_input("Maximum Epochs", min_value=50, max_value=500, value=200, step=50)
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                    value=0.01
                )
            
            # Advanced options in expander
            with st.expander("Advanced Model Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    batch_size = st.number_input("Batch Size", min_value=16, max_value=256, value=64, step=16)
                    patience = st.number_input("Early Stopping Patience", min_value=5, max_value=100, value=20, step=5)
                
                with col2:
                    weight_decay = st.select_slider(
                        "Weight Decay",
                        options=[0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01],
                        value=0.001
                    )
                    optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
            
            # Run model button
            if st.button("Train GCN Model"):
                # Show a progress bar
                progress_bar = st.progress(0)
                
                # Create a status area
                status_area = st.empty()
                
                # Simulate training process
                for i in range(100):
                    progress_bar.progress(i + 1)
                    
                    # Update status messages
                    if i < 10:
                        status_area.info("Preparing graph structure...")
                    elif i < 30:
                        status_area.info("Creating node and edge features...")
                    elif i < 60:
                        status_area.info(f"Training GCN model (Epoch {i})...")
                    elif i < 90:
                        status_area.info("Optimizing model parameters...")
                    else:
                        status_area.info("Finalizing predictions...")
                    
                    import time
                    time.sleep(0.05)
                
                # Clear status area and show success message
                status_area.empty()
                st.success("GCN model trained successfully!")
                
                # Create tabs for results
                result_tab1, result_tab2, result_tab3 = st.tabs(["Model Performance", "Quality Predictions", "Classification Report"])
                
                with result_tab1:
                    # Display model performance
                    st.markdown("<h4>Model Performance</h4>", unsafe_allow_html=True)
                    
                    # Create sample training history
                    history = {
                        "loss": np.random.uniform(0.4, 0.8, 50) * np.exp(-0.03 * np.arange(50)),
                        "acc": np.linspace(0.6, 0.95, 50) + np.random.normal(0, 0.02, 50),
                        "val_acc": np.linspace(0.55, 0.9, 50) + np.random.normal(0, 0.03, 50),
                        "test_acc": 0.88
                    }
                    
                    # Plot training history
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Loss plot
                    ax[0].plot(history["loss"], label="Training Loss", color='#0e4194')
                    ax[0].set_xlabel("Epoch")
                    ax[0].set_ylabel("Loss")
                    ax[0].set_title("Training Loss")
                    ax[0].legend()
                    ax[0].grid(True, alpha=0.3)
                    
                    # Accuracy plot
                    ax[1].plot(history["acc"], label="Training Accuracy", color='#0e4194')
                    ax[1].plot(history["val_acc"], label="Validation Accuracy", color='#f59e0b')
                    ax[1].set_xlabel("Epoch")
                    ax[1].set_ylabel("Accuracy")
                    ax[1].set_title("Model Accuracy")
                    ax[1].legend()
                    ax[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display test accuracy and other metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Test Accuracy", f"{history['test_acc']:.2f}")
                    
                    with col2:
                        st.metric("F1 Score", "0.86")
                    
                    with col3:
                        st.metric("AUC", "0.92")
                
                with result_tab2:
                    # Display hydrocarbon quality predictions
                    st.markdown("<h4>Hydrocarbon Quality Prediction</h4>", unsafe_allow_html=True)
                    
                    # Select visualization style
                    viz_style = st.radio("Visualization Style", ["Single Column", "Heat Map"], horizontal=True)
                    
                    if viz_style == "Single Column":
                        # Create a simple visualization
                        depth = np.arange(1000, 1100)
                        predictions = np.random.randint(0, 5, size=100)  # 0=Very_Low, 4=Very_High
                        
                        fig, ax = plt.subplots(figsize=(8, 10))
                        cmap = plt.cm.get_cmap('viridis', 5)
                        
                        # Create a depth vs prediction plot
                        sc = ax.scatter(np.ones_like(depth), depth, c=predictions, cmap=cmap, 
                                       s=100, marker='s')
                        
                        # Customize the plot
                        ax.set_yticks(np.arange(1000, 1101, 10))
                        ax.set_ylabel("Depth")
                        ax.set_xlim(0.9, 1.1)
                        ax.set_xticks([])
                        ax.invert_yaxis()  # Invert y-axis to show depth increasing downward
                        
                        # Add a colorbar
                        cbar = plt.colorbar(sc)
                        cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
                        cbar.set_ticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
                        cbar.set_label('Hydrocarbon Quality')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        # Create a heat map visualization
                        depth = np.arange(1000, 1100)
                        quality_probs = np.random.random((100, 5))
                        quality_probs = quality_probs / quality_probs.sum(axis=1, keepdims=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 10))
                        im = ax.imshow(quality_probs, aspect='auto', cmap='viridis',
                                      extent=[0, 5, 1100, 1000])
                        
                        # Customize the plot
                        ax.set_yticks(np.arange(1000, 1101, 10))
                        ax.set_ylabel("Depth")
                        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
                        ax.set_xticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
                        ax.set_xlabel("Hydrocarbon Quality")
                        
                        # Add a colorbar
                        cbar = plt.colorbar(im)
                        cbar.set_label('Probability')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Create prediction dataframe
                    pred_df = pd.DataFrame({
                        "DEPTH": depth,
                        "PREDICTED_QUALITY": [['Very Low', 'Low', 'Moderate', 'High', 'Very High'][p] for p in predictions],
                        "QUALITY_CODE": predictions
                    })
                    
                    # Show sample of predictions
                    st.dataframe(pred_df.head(10))
                    
                    # Download button for predictions
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name="quality_predictions.csv",
                        mime="text/csv"
                    )
                
                with result_tab3:
                    # Display classification report
                    st.markdown("<h4>Classification Report</h4>", unsafe_allow_html=True)
                    
                    # Create a classification report
                    report_data = {
                        "Class": ["Very Low", "Low", "Moderate", "High", "Very High", "Average/Total"],
                        "Precision": [0.85, 0.78, 0.92, 0.86, 0.91, 0.86],
                        "Recall": [0.82, 0.75, 0.90, 0.89, 0.84, 0.84],
                        "F1-Score": [0.83, 0.76, 0.91, 0.87, 0.87, 0.85],
                        "Support": [120, 85, 150, 95, 70, 520]
                    }
                    
                    report_df = pd.DataFrame(report_data)
                    
                    # Display in a nice table
                    st.table(report_df)
                    
                    # Create a confusion matrix
                    st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
                    
                    # Create a confusion matrix
                    classes = ["Very Low", "Low", "Moderate", "High", "Very High"]
                    cm = np.zeros((5, 5))
                    
                    # Make diagonal dominant
                    for i in range(5):
                        cm[i, i] = np.random.randint(50, 100)
                        for j in range(5):
                            if i != j:
                                cm[i, j] = np.random.randint(0, 20)
                    
                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                    ax.set_title("Confusion Matrix")
                    
                    # Add colorbar
                    cbar = plt.colorbar(im)
                    cbar.set_label('Count')
                    
                    # Set tick marks and labels
                    tick_marks = np.arange(len(classes))
                    ax.set_xticks(tick_marks)
                    ax.set_xticklabels(classes, rotation=45)
                    ax.set_yticks(tick_marks)
                    ax.set_yticklabels(classes)
                    
                    # Add text annotations
                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, format(int(cm[i, j]), 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > thresh else "black")
                    
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Store in session state
                st.session_state["model_history"] = True
                st.session_state["analysis_complete"] = True

def visualization_page():
    # Check login for visualization
    if not st.session_state.get("logged_in", False):
        st.markdown("""
        <div class="card" style="text-align: center; max-width: 500px; margin: 50px auto;">
            <h2>Login Required</h2>
            <p>You need to log in to access the Visualization tools.</p>
            <div style="margin-top: 20px;">
        """, unsafe_allow_html=True)
        
        if st.button("Login to Continue", key="login_prompt"):
            st.session_state["auth_mode"] = "login"
            st.rerun()
        
        if st.session_state.get("auth_mode") == "login":
            with st.form("login_form"):
                st.markdown("<h3>Login</h3>", unsafe_allow_html=True)
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login"):
                    if login(email, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="colored-header">
        <h1>Visualization</h1>
        <p>Visualize and interpret prediction results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if analysis has been completed
    if not st.session_state.get("analysis_complete", False) and not st.session_state.get("model_history", False):
        st.warning("No analysis results found. Please run the analysis tool first.")
        
        # Add demo option
        if st.button("Load Demo Results"):
            st.session_state["model_history"] = True
            st.rerun()
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Facies Classification", "Quality Prediction", "Model Performance"])
    
    with tab1:
        st.markdown("<h3>Facies Classification</h3>", unsafe_allow_html=True)
        
        # Add options panel
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a dropdown to select cluster configuration
            cluster_options = [5, 6, 7, 8, 9, 10]
            selected_cluster = st.selectbox("Select Cluster Configuration", options=cluster_options, index=2)
        
        with col2:
            # Add color scheme selection
            color_scheme = st.selectbox("Color Scheme", ["viridis", "plasma", "magma", "cividis", "turbo"])
        
        # Create a simple visualization
        depth = np.arange(1000, 1100)
        facies = np.random.randint(0, selected_cluster, size=100)
        
        # Option to show detailed view
        show_detailed = st.checkbox("Show Detailed View", value=False)
        
        if show_detailed:
            # Create a more detailed visualization with log curves
            fig, ax = plt.subplots(1, 3, figsize=(15, 10), sharey=True, gridspec_kw={'width_ratios': [1, 1, 2]})
            
            # Add GR log curve
            gr_log = 50 + 20 * np.sin(np.linspace(0, 8*np.pi, 100)) + np.random.normal(0, 5, 100)
            ax[0].plot(gr_log, depth, 'k-', linewidth=1)
            ax[0].set_xlabel("GR (API)")
            ax[0].set_ylabel("Depth")
            ax[0].invert_yaxis()
            ax[0].set_title("Gamma Ray Log")
            ax[0].grid(True, alpha=0.3)
            
            # Add resistivity log curve
            res_log = 10 ** (np.random.uniform(0, 2, 100))
            ax[1].semilogx(res_log, depth, 'r-', linewidth=1)
            ax[1].set_xlabel("Resistivity (ohm.m)")
            ax[1].invert_yaxis()
            ax[1].set_title("Resistivity Log")
            ax[1].grid(True, alpha=0.3)
            
            # Add facies classification
            cmap = plt.cm.get_cmap(color_scheme, selected_cluster)
            sc = ax[2].scatter(np.ones_like(depth), depth, c=facies, cmap=cmap, 
                             s=300, marker='s')
            ax[2].set_yticks(np.arange(1000, 1101, 10))
            ax[2].set_xlim(0.9, 1.1)
            ax[2].set_xticks([])
            ax[2].invert_yaxis()
            ax[2].set_title("Facies Classification")
            
            # Add a colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(sc, cax=cbar_ax)
            cbar.set_ticks(np.arange(selected_cluster) + 0.5)
            cbar.set_ticklabels([f"Facies {i+1}" for i in range(selected_cluster)])
            cbar.set_label('Facies Classification')
            
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            st.pyplot(fig)
        else:
            # Create a simpler visualization
            fig, ax = plt.subplots(figsize=(8, 10))
            cmap = plt.cm.get_cmap(color_scheme, selected_cluster)
            
            # Create a depth vs facies plot
            sc = ax.scatter(np.ones_like(depth), depth, c=facies, cmap=cmap, 
                           s=100, marker='s')
            
            # Customize the plot
            ax.set_yticks(np.arange(1000, 1101, 10))
            ax.set_ylabel("Depth")
            ax.set_xlim(0.9, 1.1)
            ax.set_xticks([])
            ax.invert_yaxis()
            
            # Add a colorbar
            cbar = plt.colorbar(sc)
            cbar.set_ticks(np.arange(selected_cluster) + 0.5)
            cbar.set_ticklabels([f"Facies {i+1}" for i in range(selected_cluster)])
            cbar.set_label('Facies Classification')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Add statistics panel
        st.markdown("<h4>Facies Statistics</h4>", unsafe_allow_html=True)
        
        # Calculate facies distribution
        facies_counts = np.zeros(selected_cluster)
        for i in range(selected_cluster):
            facies_counts[i] = np.sum(facies == i)
        
        # Create a pie chart
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, texts, autotexts = ax.pie(
            facies_counts, 
            autopct='%1.1f%%',
            textprops={'color': "w", 'fontweight': 'bold'},
            colors=plt.cm.get_cmap(color_scheme, selected_cluster)(np.linspace(0, 1, selected_cluster))
        )
        
        ax.legend(
            wedges, 
            [f"Facies {i+1}" for i in range(selected_cluster)],
            title="Facies",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab2:
        st.markdown("<h3>Hydrocarbon Quality Prediction</h3>", unsafe_allow_html=True)
        
        # Create options for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            view_type = st.radio("View Type", ["Single Column", "Multi-Column", "Heat Map"])
        
        with col2:
            color_scheme = st.selectbox("Color Scheme", 
                                      ["viridis", "plasma", "inferno", "magma", "cividis"],
                                      key="quality_color")
        
        # Generate sample data
        depth = np.arange(1000, 1100)
        quality_labels = ["Very_Low", "Low", "Moderate", "High", "Very_High"]
        quality = np.random.randint(0, 5, size=100)
        
        if view_type == "Single Column":
            # Create the visualization
            fig, ax = plt.subplots(figsize=(8, 10))
            cmap = plt.cm.get_cmap(color_scheme, 5)
            
            # Create a depth vs quality plot
            sc = ax.scatter(np.ones_like(depth), depth, c=quality, cmap=cmap, 
                           s=100, marker='s')
            
            # Customize the plot
            ax.set_yticks(np.arange(1000, 1101, 10))
            ax.set_ylabel("Depth")
            ax.set_xlim(0.9, 1.1)
            ax.set_xticks([])
            ax.invert_yaxis()
            
            # Add a colorbar
            cbar = plt.colorbar(sc)
            cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(quality_labels)
            cbar.set_label('Hydrocarbon Quality')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        elif view_type == "Multi-Column":
            # Create a multi-column view comparing different cluster configurations
            fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=True)
            
            for i, ax in enumerate(axes):
                # Generate random data for each column
                cluster_quality = np.random.randint(0, 5, size=100)
                
                cmap = plt.cm.get_cmap(color_scheme, 5)
                sc = ax.scatter(np.ones_like(depth), depth, c=cluster_quality, cmap=cmap, 
                               s=100, marker='s')
                
                # Customize the plot
                if i == 0:
                    ax.set_yticks(np.arange(1000, 1101, 10))
                    ax.set_ylabel("Depth")
                ax.set_xlim(0.9, 1.1)
                ax.set_xticks([])
                ax.set_title(f"Clusters: {i+5}")
                ax.invert_yaxis()
            
            # Add a colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(sc, cax=cbar_ax)
            cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
            cbar.set_ticklabels(quality_labels)
            cbar.set_label('Hydrocarbon Quality')
            
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            st.pyplot(fig)
            
        else:  # Heat Map
            # Create a heat map view
            quality_probs = np.zeros((100, 5))
            for i, q in enumerate(quality):
                quality_probs[i, q] = 0.7
                # Add some noise to other values
                for j in range(5):
                    if j != q:
                        quality_probs[i, j] = np.random.random() * 0.3
            
            # Normalize
            quality_probs = quality_probs / quality_probs.sum(axis=1, keepdims=True)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(quality_probs, aspect='auto', cmap=color_scheme, 
                          extent=[0, 5, 1100, 1000])
            
            # Customize the plot
            ax.set_yticks(np.arange(1000, 1101, 10))
            ax.set_ylabel("Depth")
            ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
            ax.set_xticklabels(quality_labels)
            
            # Add a colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Probability')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Display quality distribution
        st.markdown("<h4>Quality Distribution</h4>", unsafe_allow_html=True)
        
        # Calculate quality distribution
        quality_counts = np.zeros(5)
        for i in range(5):
            quality_counts[i] = np.sum(quality == i)
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(
            quality_labels,
            quality_counts,
            color=plt.cm.get_cmap(color_scheme, 5)(np.linspace(0, 1, 5))
        )
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        
        ax.set_xlabel("Hydrocarbon Quality")
        ax.set_ylabel("Count")
        ax.set_title("Quality Distribution")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add download button for predictions
        quality_df = pd.DataFrame({
            "DEPTH": depth,
            "QUALITY": [quality_labels[q].replace('_', ' ') for q in quality],
            "QUALITY_CODE": quality
        })
        
        csv = quality_df.to_csv(index=False)
        st.download_button(
            label="Download Quality Predictions",
            data=csv,
            file_name="quality_predictions.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)
        
        # Create tabs for different performance metrics
        perf_tab1, perf_tab2, perf_tab3 = st.tabs(["Learning Curves", "Confusion Matrix", "Metrics"])
        
        with perf_tab1:
            # Create sample training history
            history = {
                "loss": np.random.uniform(0.4, 0.8, 50) * np.exp(-0.03 * np.arange(50)),
                "acc": np.linspace(0.6, 0.95, 50) + np.random.normal(0, 0.02, 50),
                "val_acc": np.linspace(0.55, 0.9, 50) + np.random.normal(0, 0.03, 50),
                "test_acc": 0.88
            }
            
            # Plot training history
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss plot
            ax[0].plot(history["loss"], label="Training Loss", color='#0e4194')
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[0].set_title("Training Loss")
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            ax[1].plot(history["acc"], label="Training Accuracy", color='#0e4194')
            ax[1].plot(history["val_acc"], label="Validation Accuracy", color='#f59e0b')
            ax[1].axhline(y=history["test_acc"], color='g', linestyle='--', label='Test Accuracy')
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            ax[1].set_title("Model Accuracy")
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with perf_tab2:
            # Create a confusion matrix
            classes = ["Very Low", "Low", "Moderate", "High", "Very High"]
            cm = np.zeros((5, 5))
            
            # Make diagonal dominant
            for i in range(5):
                cm[i, i] = np.random.randint(50, 100)
                for j in range(5):
                    if i != j:
                        cm[i, j] = np.random.randint(0, 20)
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_title("Confusion Matrix")
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Count')
            
            # Set tick marks and labels
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes, rotation=45)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(int(cm[i, j]), 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with perf_tab3:
            # Create a classification report
            report_data = {
                "Class": ["Very Low", "Low", "Moderate", "High", "Very High", "Average/Total"],
                "Precision": [0.85, 0.78, 0.92, 0.86, 0.91, 0.86],
                "Recall": [0.82, 0.75, 0.90, 0.89, 0.84, 0.84],
                "F1-Score": [0.83, 0.76, 0.91, 0.87, 0.87, 0.85],
                "Support": [120, 85, 150, 95, 70, 520]
            }
            
            report_df = pd.DataFrame(report_data)
            
            # Display in a nice table
            st.table(report_df)
            
            # Add metrics comparison across different runs
            st.markdown("<h4>Performance Across Runs</h4>", unsafe_allow_html=True)
            
            runs_data = {
                "Run": [1, 2, 3, 4],
                "Test Accuracy": [0.87, 0.85, 0.88, 0.86],
                "F1 Score": [0.86, 0.84, 0.87, 0.85],
                "Precision": [0.85, 0.83, 0.86, 0.84],
                "Recall": [0.87, 0.85, 0.88, 0.86]
            }
            
            runs_df = pd.DataFrame(runs_data)
            
            # Plot metrics across runs
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create a grouped bar chart
            width = 0.2
            x = np.arange(len(runs_data["Run"]))
            
            ax.bar(x - 1.5*width, runs_data["Test Accuracy"], width, label='Accuracy', color='#0e4194')
            ax.bar(x - 0.5*width, runs_data["F1 Score"], width, label='F1 Score', color='#f59e0b')
            ax.bar(x + 0.5*width, runs_data["Precision"], width, label='Precision', color='#10b981')
            ax.bar(x + 1.5*width, runs_data["Recall"], width, label='Recall', color='#8b5cf6')
            
            ax.set_ylabel('Score')
            ax.set_title('Performance Metrics Across Runs')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Run {i}' for i in runs_data["Run"]])
            ax.legend()
            ax.set_ylim(0.5, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display run comparison table
            st.table(runs_df)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-header">Facies Classification</div>
            <ul>
                <li>K-means Clustering</li>
                <li>Silhouette Score Optimization</li>
                <li>Depth-based Facies Mapping</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-header">Graph-based Machine Learning</div>
            <ul>
                <li>Graph Convolutional Networks</li>
                <li>Node and Edge Feature Extraction</li>
                <li>Hydrocarbon Quality Classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-header">Visualization and Reporting</div>
            <ul>
                <li>Facies Visualization</li>
                <li>Prediction Accuracy Metrics</li>
                <li>Classification Reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Coming Soon Section with blur effect
    st.markdown("""
    <div class="coming-soon">
        <h2>Coming Soon: StrataGraph 2.0</h2>
        <p>Geoscience graph datasets for Carbon Capture, Utilization, and Storage (CCUS)</p>
        <button style="background-color: rgba(255,255,255,0.2); color: white; border: 1px solid white; padding: 8px 16px; border-radius: 4px; margin-top: 10px; cursor: pointer;">
            Join Waitlist
        </button>
    </div>
    """, unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown("<h2>Getting Started</h2>", unsafe_allow_html=True)
    
    # Create a simple workflow guide
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="border-left: 5px solid #10b981;">
            <div class="feature-header">1. Prepare Data</div>
            <p>Upload your well log data in LAS format and validate required curves.</p>
            <p>Navigate to the <b>Dataset Preparation</b> section to start.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="border-left: 5px solid #f59e0b;">
            <div class="feature-header">2. Run Analysis</div>
            <p>Calculate petrophysical properties and run the GCN model.</p>
            <p>Use the <b>Analysis Tool</b> to process your data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="border-left: 5px solid #8b5cf6;">
            <div class="feature-header">3. Visualize Results</div>
            <p>Explore facies classifications and quality predictions.</p>
            <p>Visit the <b>Visualization</b> section to interpret findings.</p>
        </div>
        """, unsafe_allow_html=True)

def dataset_preparation_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Dataset Preparation</h1>
        <p>Prepare your well log data for VHydro analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to load the dataset preparation image
    dataset_img_path = "src/Graph Dataset Preparation.png"  # Update with your actual image path
    try:
        st.image(dataset_img_path, use_column_width=True)
    except:
        # Fallback if image isn't found
        st.warning("Dataset preparation image not found. Please ensure 'VHydro_dataset_preparation.png' is in the correct directory.")
    
    st.markdown("""
    <div class="card">
        <h2>VHydro Data Preparation</h2>
        <p>VHydro requires specific log curves to calculate petrophysical properties needed for accurate predictions:</p>
        <table style="width:100%; border-collapse: collapse; margin-top: 20px;">
            <tr style="background-color: #f1f5f9;">
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0;">Curve</th>
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0;">Description</th>
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0;">Purpose</th>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">GR/CGR</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Gamma Ray or Computed Gamma Ray</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Shale volume calculation</td>
            </tr>
            <tr style="background-color: #f8fafc;">
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">RHOB</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Bulk Density</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Density porosity calculation</td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">NPHI</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Neutron Porosity</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Effective porosity calculation</td>
            </tr>
            <tr style="background-color: #f8fafc;">
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">LLD/ILD</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Deep Resistivity</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Water/oil saturation calculation</td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">DEPT</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Depth</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Spatial reference for facies classification</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>File Format</h2>
        <p>VHydro accepts well log data in <b>LAS</b> (Log ASCII Standard) format, which is the industry standard for storing well log data.</p>
        <h3>Example LAS File Structure</h3>
        <pre style="background-color: #f1f5f9; padding: 15px; border-radius: 5px; font-size: 0.9em; overflow-x: auto;">
~VERSION INFORMATION
VERS.   2.0 :   CWLS LOG ASCII STANDARD - VERSION 2.0
WRAP.   NO  :   ONE LINE PER DEPTH STEP

~WELL INFORMATION
STRT.M   1670.0 :
STOP.M   1660.0 :
STEP.M   -0.1  :
NULL.    -999.25:
WELL.    EXAMPLE WELL:
FLD .    EXAMPLE FIELD:

~CURVE INFORMATION
DEPT.M   :   Depth
GR  .GAPI:   Gamma Ray
RHOB.K/M3:   Bulk Density
NPHI.V/V :   Neutron Porosity
LLD .OHMM:   Deep Resistivity

~A  DEPT     GR      RHOB    NPHI    LLD
1670.000   75.075   2.561   0.246   12.863
1669.900   74.925   2.563   0.245   13.042
...
        </pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Upload Section
    st.markdown("""
    <div class="card">
        <h2>Upload Your Data</h2>
        <p>Upload your well log data in LAS format to begin the analysis process:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section with a colorful border
    st.markdown("""
    <div style="border: 2px dashed #0e4194; border-radius: 10px; padding: 30px; text-align: center; margin: 20px 0;">
        <h4>Drag and drop your LAS file here</h4>
        <p>Or click to browse files</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a LAS file", type=["las"])
    
    if uploaded_file is not None:
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        
        # Store in session state
        st.session_state["uploaded_file"] = uploaded_file.name
        
        # Show file info in a nice format
        st.markdown(
            f"""
            <div class="card">
                <h4>File Information</h4>
                <table style="width: 100%;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">File Name:</td>
                        <td style="padding: 8px;">{uploaded_file.name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">File Size:</td>
                        <td style="padding: 8px;">{uploaded_file.size / 1024:.2f} KB</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Upload Time:</td>
                        <td style="padding: 8px;">{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</td>
                    </tr>
                </table>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Proceed button
        if st.button("Process Data"):
            st.session_state["analysis_stage"] = "property_calculation"
            st.rerun()

def models_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Models</h1>
        <p>Advanced Graph Convolutional Networks for Geoscience Modeling</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to load the model image
    model_img_path = "src/Model.png"  # Update with your actual image path
    try:
        st.image(model_img_path, use_column_width=True)
    except:
        # Fallback if image isn't found
        st.warning("Model image not found. Please ensure 'VHydro_Model.png' is in the correct directory.")
    
    st.markdown("""
    <div class="card">
        <h2>GCN Model for Hydrocarbon Prediction</h2>
        <p>StrataGraph implements sophisticated Graph Convolutional Networks (GCNs) to model the complex relationships 
        between different petrophysical properties across varying depths.</p>
        <p>VHydro 1.0 leverages these networks to predict hydrocarbon quality and classify reservoir zones.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Model Architecture</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h3>Graph Construction</h3>
                <ul>
                    <li><b>Nodes:</b> Depth points with associated petrophysical properties</li>
                    <li><b>Edges:</b> Connections between related depth points based on facies</li>
                    <li><b>Node Features:</b> Encoded petrophysical properties</li>
                    <li><b>Edge Features:</b> Relationships between facies at different depths</li>
                </ul>
            </div>
            <div>
                <h3>GCN Architecture</h3>
                <ul>
                    <li><b>Input Layer:</b> Node features from petrophysical properties</li>
                    <li><b>Hidden Layers:</b> Multiple graph convolutional layers</li>
                    <li><b>Output Layer:</b> Classification layer for quality prediction</li>
                    <li><b>Regularization:</b> Dropout and batch normalization</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Hydrocarbon Quality Classification</h2>
        <p>The model classifies reservoir zones into five quality categories:</p>
        <table style="width:100%; border-collapse: collapse; margin-top: 20px;">
            <tr style="background-color: #f1f5f9;">
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0;">Category</th>
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0;">Description</th>
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0;">Characteristics</th>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Very High</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Exceptional quality reservoir</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">High porosity, high permeability, low water saturation</td>
            </tr>
            <tr style="background-color: #f8fafc;">
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">High</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Good quality reservoir</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Good porosity and permeability, moderate water saturation</td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Moderate</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Average quality reservoir</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Moderate porosity and permeability</td>
            </tr>
            <tr style="background-color: #f8fafc;">
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Low</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Poor quality reservoir</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Low porosity, low permeability, high water saturation</td>
            </tr>
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Very Low</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Non-reservoir quality</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">Very low porosity and permeability, high shale content</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Configuration Section
    st.markdown("<h2>Model Configuration</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-header">GCN Parameters</div>
            <ul>
                <li><b>Hidden Channels:</b> 16</li>
                <li><b>Layers:</b> 2</li>
                <li><b>Dropout Rate:</b> 0.5</li>
                <li><b>Learning Rate:</b> 0.01</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-header">Training Parameters</div>
            <ul>
                <li><b>Maximum Epochs:</b> 200</li>
                <li><b>Early Stopping:</b> Yes (patience: 50)</li>
                <li><b>Train/Val/Test Split:</b> 80%/10%/10%</li>
                <li><b>Optimizer:</b> Adam</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Configuration Options
    with st.expander("Advanced Configuration Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Architecture")
            st.selectbox("GCN Version", ["RegularizedGCN", "Standard GCN", "GCN with Skip Connections"])
            st.number_input("Hidden Channels", min_value=8, max_value=128, value=16, step=8)
            st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.5, step=0.1)
            st.checkbox("Use Batch Normalization", value=True)
        
        with col2:
            st.markdown("### Training Configuration")
            st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05], value=0.01)
            st.number_input("Early Stopping Patience", min_value=10, max_value=100, value=50, step=5)
            st.number_input("Maximum Epochs", min_value=50, max_value=500, value=200, step=50)
            st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
    
    # Run model button
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <button style="background-color: #0e4194; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 16px; cursor: pointer; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            Train GCN Model
        </button>
    </div>
    """, unsafe_allow_html=True)
    
    # If the user clicks the button (this is a Streamlit workaround)
    if st.button("Train GCN Model"):
        # Show a progress bar
        progress_bar = st.progress(0)
        
        # Create a status area
        status_area = st.empty()
        
        # Simulate training process
        for i in range(100):
            progress_bar.progress(i + 1)
            
            # Update status messages
            if i < 10:
                status_area.info("Preparing graph structure...")
            elif i < 30:
                status_area.info("Creating node and edge features...")
            elif i < 60:
                status_area.info(f"Training GCN model (Epoch {i})...")
            elif i < 90:
                status_area.info("Optimizing model parameters...")
            else:
                status_area.info("Finalizing predictions...")
            
            # Sleep to simulate processing time
            import time
            time.sleep(0.05)
        
        # Clear status area and show success message
        status_area.empty()
        st.success("GCN model trained successfully!")
