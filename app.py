import streamlit as st
import os
import numpy as np
import pandas as pd
import time
from PIL import Image
import base64
import logging

# Configure matplotlib for faster loading
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/matplotlib_cache'
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration at the very beginning
st.set_page_config(
    page_title="StrataGraph",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache expensive operations
@st.cache_data
def load_css():
    css = """
    /* Minimal core styles only */
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }
    h1, h2, h3, h4, h5, h6 { color: #0e4194; }
    .colored-header { background: linear-gradient(90deg, #0e4194 0%, #3a6fc4 100%); color: white; 
                     padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
    .card { border-radius: 10px; padding: 20px; margin-bottom: 20px; background: white; 
           box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); }
    .feature-card { border-radius: 10px; padding: 20px; margin-bottom: 20px; background: white; 
                   box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); height: 100%; 
                   border-left: 5px solid #0e4194; }
    .feature-header { font-weight: bold; color: #0e4194; margin-bottom: 10px; font-size: 1.2rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0e4194 0%, #153a6f 100%); }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] div { color: white !important; }
    .user-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; 
                margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Simple login function using hardcoded credentials
@st.cache_data
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
    for key in ["email", "logged_in", "auth_mode"]:
        if key in st.session_state:
            del st.session_state[key]

# Preload and cache images
@st.cache_data
def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {e}")
        return None

# Create sidebar for navigation
def create_sidebar():
    st.sidebar.markdown('<div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">', unsafe_allow_html=True)
    
    # Logo placeholder
    logo_path = "src/StrataGraph_White_Logo.png"
    try:
        st.sidebar.image(logo_path, width=160)
    except:
        st.sidebar.markdown("""
        <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <h2 style="color: #0e4194; margin: 0;">StrataGraph</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Version info
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
        <p style="margin: 0; font-size: 0.9rem;">VHydro 1.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User info if logged in
    if st.session_state.get("logged_in", False):
        st.sidebar.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <p style="margin: 0;">Logged in as: {st.session_state.get("email")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("Logout", key="logout_btn"):
            logout()
            st.rerun()
    
    # Navigation
    st.sidebar.markdown('<h3 style="color: white; margin-bottom: 15px;">Navigation</h3>', unsafe_allow_html=True)
    
    # Define pages
    pages = ["Home", "Dataset Preparation", "Models", "Analysis Tool", "Visualization"]
    
    # Initialize current page in session state if it doesn't exist
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
    # Create navigation buttons
    for page in pages:
        # Use different key for each button
        if st.sidebar.button(page, key=f"nav_{page}"):
            st.session_state["current_page"] = page
            st.rerun()
    
    # Basic configuration section
    st.sidebar.markdown('<h3 style="color: white;">Model Configuration</h3>', unsafe_allow_html=True)
    
    min_clusters = st.sidebar.slider("Min Clusters", 2, 15, 5)
    max_clusters = st.sidebar.slider("Max Clusters", min_clusters, 15, 10)
    
    return {
        "page": st.session_state["current_page"],
        "min_clusters": min_clusters,
        "max_clusters": max_clusters
    }

def home_page():
    # Try to load the banner image
    banner_path = "src/StrataGraph_Banner.png"
    try:
        st.image(banner_path, use_column_width=True)
    except:
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
    workflow_path = "src/Workflow.png"
    try:
        st.image(workflow_path, use_column_width=True)
    except:
        st.warning("Workflow image not found.")
    
    st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)
    
    # Create a 2-column layout for features
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
                <li><b>Early Stopping:</b> Yes</li>
                <li><b>Train/Val/Test Split:</b> 80%/10%/10%</li>
                <li><b>Optimizer:</b> Adam</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Configuration Options
    if st.checkbox("Show Advanced Configuration Options"):
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
    
    # Train model button - simplified
    if st.button("Train GCN Model", key="train_model_btn"):
        with st.spinner("Training GCN model..."):
            progress_bar = st.progress(0)
            status_area = st.empty()
            
            # Simulate training process with fewer updates
            for i in range(0, 101, 20):
                progress_bar.progress(i)
                
                if i == 0:
                    status_area.info("Preparing graph structure...")
                elif i == 20:
                    status_area.info("Creating features...")
                elif i == 40:
                    status_area.info("Training model...")
                elif i == 60:
                    status_area.info("Optimizing parameters...")
                elif i == 80:
                    status_area.info("Finalizing predictions...")
                
                # Use shorter sleep time
                time.sleep(0.1)
            
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
        </div>
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
        
        return
    
    st.markdown("""
    <div class="colored-header">
        <h1>Analysis Tool</h1>
        <p>Process well log data and run advanced hydrocarbon prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs but load content conditionally
    tab_titles = ["Data Upload", "Property Calculation", "Facies Classification", "GCN Model"]
    tabs = st.tabs(tab_titles)
    
    # Only load the content of the selected tab
    active_tab = 0
    if "active_tab" in st.session_state:
        active_tab = st.session_state["active_tab"]
    
    with tabs[active_tab]:
        if active_tab == 0:  # Data Upload
            data_upload_tab()
        elif active_tab == 1:  # Property Calculation
            property_calculation_tab()
        elif active_tab == 2:  # Facies Classification
            facies_classification_tab()
        elif active_tab == 3:  # GCN Model
            gcn_model_tab()

# Individual tab content functions
def data_upload_tab():
    st.markdown("<h3>Upload LAS File</h3>", unsafe_allow_html=True)
    
    # File upload component
    uploaded_file = st.file_uploader("Choose a LAS file", type=["las"])
    
    if uploaded_file is not None:
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        
        # Store in session state
        st.session_state["uploaded_file"] = uploaded_file.name
        
        # Show file info in a table
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
            st.session_state["active_tab"] = 1  # Switch to property calculation tab
            st.rerun()

def property_calculation_tab():
    st.markdown("<h3>Petrophysical Property Calculation</h3>", unsafe_allow_html=True)
    
    # Check if user has uploaded a file
    if "uploaded_file" not in st.session_state:
        st.warning("Please upload a LAS file first.")
        
        # Button to go back to upload tab
        if st.button("Go to Data Upload"):
            st.session_state["active_tab"] = 0
            st.rerun()
        return
    
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
    if st.checkbox("Show Calculation Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            matrix_density = st.number_input("Matrix Density (g/cm³)", min_value=2.0, max_value=3.0, value=2.65, step=0.01)
            fluid_density = st.number_input("Fluid Density (g/cm³)", min_value=0.5, max_value=1.5, value=1.0, step=0.01)
        
        with col2:
            a_const = st.number_input("Archie Constant (a)", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
            m_const = st.number_input("Cementation Exponent (m)", min_value=1.5, max_value=2.5, value=2.0, step=0.1)
    
    # Run calculation button
    if st.button("Calculate Properties"):
        with st.spinner("Calculating petrophysical properties..."):
            # Simulate calculation progress with fewer updates
            progress_bar = st.progress(0)
            for i in range(0, 101, 25):
                progress_bar.progress(i)
                time.sleep(0.1)
        
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
        
        # Button to proceed to next step
        if st.button("Proceed to Facies Classification"):
            st.session_state["active_tab"] = 2
            st.rerun()

def facies_classification_tab():
    st.markdown("<h3>Facies Classification</h3>", unsafe_allow_html=True)
    
    # Check if properties have been calculated
    if "property_data" not in st.session_state:
        st.warning("Please calculate petrophysical properties first.")
        
        # Button to go back to property calculation tab
        if st.button("Go to Property Calculation"):
            st.session_state["active_tab"] = 1
            st.rerun()
        return
    
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
        algorithm = st.selectbox("Clustering Algorithm", ["K-means", "Agglomerative", "DBSCAN"])
    
    # Run clustering button
    if st.button("Run Facies Classification"):
        with st.spinner("Running facies classification..."):
            # Simpler progress indicator
            progress_bar = st.progress(0)
            for i in range(0, 101, 20):
                progress_bar.progress(i)
                time.sleep(0.1)
        
        st.success("Facies classification completed successfully!")
        
        # Generate simulated silhouette scores
        silhouette_scores = {
            i: np.random.uniform(0.4, 0.7) for i in range(min_clusters, max_clusters + 1)
        }
        
        # Find optimal clusters
        optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
        st.info(f"Optimal number of clusters: {optimal_clusters}")
        
        # Button to show visualization (lazy loading of heavy content)
        if st.checkbox("Show Facies Visualization"):
            # Simple HTML/CSS based visualization instead of matplotlib for better performance
            st.markdown(f"""
            <div style="width:100%; height:400px; background: linear-gradient(180deg, #0e4194 0%, #3a6fc4 100%); 
                        border-radius:10px; color:white; text-align:center; padding:20px;">
                <h4>Facies Visualization (Cluster {optimal_clusters})</h4>
                <p>Interactive visualization would appear here in the full version.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a simple facies dataset for download
        facies_df = pd.DataFrame({
            "DEPTH": np.arange(1000, 1100),
            "FACIES": np.random.randint(0, optimal_clusters, size=100)
        })
        
        # Show sample data
        st.dataframe(facies_df.head())
        
        # Download button
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
        
        # Button to proceed to next step
        if st.button("Proceed to GCN Model"):
            st.session_state["active_tab"] = 3
            st.rerun()

def gcn_model_tab():
    st.markdown("<h3>Graph Convolutional Network Model</h3>", unsafe_allow_html=True)
    
    # Check if facies classification has been done
    if "facies_data" not in st.session_state:
        st.warning("Please complete facies classification first.")
        
        # Button to go back to facies classification tab
        if st.button("Go to Facies Classification"):
            st.session_state["active_tab"] = 2
            st.rerun()
        return
    
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
        n_clusters = st.number_input("Number of Clusters", 
                                   min_value=2, 
                                   max_value=15, 
                                   value=int(st.session_state.get("best_clusters", 7)), 
                                   step=1)
        hidden_channels = st.number_input("Hidden Channels", min_value=4, max_value=64, value=16, step=4)
        
    with col2:
        num_runs = st.number_input("Number of Runs", min_value=1, max_value=10, value=4, step=1)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            value=0.01
        )
    
    # Advanced parameters in toggle
    if st.checkbox("Show Advanced Model Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.5, step=0.1)
            epochs = st.number_input("Maximum Epochs", min_value=50, max_value=500, value=200, step=50)
        
        with col2:
            patience = st.number_input("Early Stopping Patience", min_value=5, max_value=100, value=20, step=5)
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
    
    # Run model button
    if st.button("Train GCN Model", key="train_gcn_btn"):
        with st.spinner("Training GCN model..."):
            # Simpler progress indicator with fewer updates
            progress_bar = st.progress(0)
            status_area = st.empty()
            
            for i in range(0, 101, 25):
                progress_bar.progress(i)
                
                # Update status with fewer messages
                if i == 0:
                    status_area.info("Preparing graph structure...")
                elif i == 25:
                    status_area.info("Creating features...")
                elif i == 50:
                    status_area.info("Training model...")
                elif i == 75:
                    status_area.info("Finalizing predictions...")
                
                time.sleep(0.1)
        
        # Clear status area and show success message
        status_area.empty()
        st.success("GCN model trained successfully!")
        
        # Instead of creating tabs that load everything at once,
        # use expanders to lazy load content
        with st.expander("Model Performance"):
            # Create basic metrics with less visualization
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Test Accuracy", "0.88")
            with col2: st.metric("F1 Score", "0.86")
            with col3: st.metric("AUC", "0.92")
            
            # Option to view detailed plots
            if st.checkbox("Show Learning Curves"):
                # Use simple HTML/CSS representation instead of matplotlib
                st.markdown("""
                <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                    <div style="width: 48%; background-color: #f1f5f9; padding: 15px; border-radius: 10px;">
                        <h4 style="text-align:center;">Training Loss</h4>
                        <div style="height: 150px; background: linear-gradient(to bottom right, #0e4194, #3a6fc4); border-radius: 5px;"></div>
                    </div>
                    <div style="width: 48%; background-color: #f1f5f9; padding: 15px; border-radius: 10px;">
                        <h4 style="text-align:center;">Model Accuracy</h4>
                        <div style="height: 150px; background: linear-gradient(to top right, #10b981, #3a6fc4); border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("Quality Predictions"):
            # Create a simple prediction table
            predictions = pd.DataFrame({
                "DEPTH": np.arange(1000, 1010),
                "PREDICTED_QUALITY": np.random.choice(
                    ["Very Low", "Low", "Moderate", "High", "Very High"], 
                    size=10
                )
            })
            
            st.dataframe(predictions)
            
            # Download button
            csv = predictions.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="quality_predictions.csv",
                mime="text/csv"
            )
            
            # Visualization toggle to lazy load heavy content
            if st.checkbox("Show Quality Visualization"):
                # A very simplified visualization
                st.markdown("""
                <div style="width:100%; height:300px; background: linear-gradient(90deg, #ff0000, #ffaa00, #00ff00, #0000ff); 
                            border-radius:10px; text-align:center; padding:20px; color:white;">
                    <h4>Hydrocarbon Quality Visualization</h4>
                    <p>From Very Low (left) to Very High (right)</p>
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("Classification Report"):
            # Simple classification report
            report_data = {
                "Class": ["Very Low", "Low", "Moderate", "High", "Very High", "Average"],
                "Precision": [0.85, 0.78, 0.92, 0.86, 0.91, 0.86],
                "Recall": [0.82, 0.75, 0.90, 0.89, 0.84, 0.84],
                "F1-Score": [0.83, 0.76, 0.91, 0.87, 0.87, 0.85]
            }
            
            # Create and display dataframe
            report_df = pd.DataFrame(report_data)
            st.table(report_df)
        
        # Store in session state
        st.session_state["model_history"] = True
        st.session_state["analysis_complete"] = True

def dataset_preparation_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Dataset Preparation</h1>
        <p>Prepare your well log data for VHydro analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to load the dataset preparation image (only if explicitly requested)
    if st.checkbox("Show Dataset Preparation Diagram"):
        dataset_img_path = "src/Graph Dataset Preparation.png"
        try:
            st.image(dataset_img_path, use_column_width=True)
        except:
            st.warning("Dataset preparation image not found.")
    
    st.markdown("""
    <div class="card">
        <h2>VHydro Data Preparation</h2>
        <p>VHydro requires specific log curves to calculate petrophysical properties needed for accurate predictions:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use tabs to organize content but load lazily
    tab1, tab2 = st.tabs(["Required Curves", "Upload Data"])
    
    with tab1:
        # Simple table instead of complex HTML
        curves = [
            ["GR/CGR", "Gamma Ray", "Shale volume calculation"],
            ["RHOB", "Bulk Density", "Density porosity calculation"],
            ["NPHI", "Neutron Porosity", "Effective porosity calculation"],
            ["LLD/ILD", "Deep Resistivity", "Water/oil saturation calculation"],
            ["DEPT", "Depth", "Spatial reference for facies"]
        ]
        
        df = pd.DataFrame(curves, columns=["Curve", "Description", "Purpose"])
        st.table(df)
    
    with tab2:
        # File upload component
        uploaded_file = st.file_uploader("Choose a LAS file", type=["las"])
        
        if uploaded_file is not None:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            
            # Store in session state
            st.session_state["uploaded_file"] = uploaded_file.name
            
            # Show file info
            file_info = {
                "File Name": uploaded_file.name,
                "File Size": f"{uploaded_file.size / 1024:.2f} KB",
                "Upload Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Use dataframe for display instead of HTML
            st.dataframe(pd.DataFrame(list(file_info.items()), 
                        columns=["Property", "Value"]))
            
            # Proceed button
            if st.button("Process Data"):
                with st.spinner("Processing data..."):
                    # Simulate processing with minimal updates
                    progress_bar = st.progress(0)
                    for i in range(0, 101, 25):
                        progress_bar.progress(i)
                        time.sleep(0.1)
                
                st.success("Data processing complete!")
                st.session_state["analysis_stage"] = "property_calculation"

def models_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Models</h1>
        <p>Advanced Graph Convolutional Networks for Geoscience Modeling</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model image only if requested
    if st.checkbox("Show Model Diagram"):
        model_img_path = "src/Model.png"
        try:
            st.image(model_img_path, use_column_width=True)
        except:
            st.warning("Model image not found.")
    
    st.markdown("""
    <div class="card">
        <h2>GCN Model for Hydrocarbon Prediction</h2>
        <p>StrataGraph implements sophisticated Graph Convolutional Networks (GCNs) to model the complex relationships 
        between different petrophysical properties across varying depths.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use tabs for content organization but load lazily
    tab1, tab2, tab3 = st.tabs(["Model Architecture", "Training Parameters", "Quality Classification"])
    
    with tab1:
        # Model architecture description using columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-header">Graph Construction</div>
                <ul>
                    <li><b>Nodes:</b> Depth points with properties</li>
                    <li><b>Edges:</b> Connections between related depths</li>
                    <li><b>Node Features:</b> Encoded properties</li>
                    <li><b>Edge Features:</b> Relationships between facies</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-header">GCN Architecture</div>
                <ul>
                    <li><b>Input Layer:</b> Node features</li>
                    <li><b>Hidden Layers:</b> Graph convolutional layers</li>
                    <li><b>Output Layer:</b> Classification layer</li>
                    <li><b>Regularization:</b> Dropout and normalization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        # Training parameters using simple metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Hidden Channels", "16")
        with col2: st.metric("Layers", "2")
        with col3: st.metric("Dropout Rate", "0.5")
        with col4: st.metric("Learning Rate", "0.01")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Max Epochs", "200")
        with col2: st.metric("Early Stopping", "Yes")
        with col3: st.metric("Train Split", "80%")
        with col4: st.metric("Optimizer", "Adam")
    
    with tab3:
        # Quality classification 
        quality_data = [
            ["Very High", "Exceptional quality", "High porosity, high permeability"],
            ["High", "Good quality", "Good porosity and permeability"],
            ["Moderate", "Average quality", "Moderate porosity and permeability"],
            ["Low", "Poor quality", "Low porosity, high water saturation"],
            ["Very Low", "Non-reservoir", "Very low porosity, high shale content"]
        ]
        
        df = pd.DataFrame(quality_data, columns=["Category", "Description", "Characteristics"])
        st.table(df)
    
    # Model training button
    if st.button("Train GCN Model", key="train_model_models_page"):
        with st.spinner("Training GCN model..."):
            # Simple progress indicator
            progress_bar = st.progress(0)
            for i in range(0, 101, 25):
                progress_bar.progress(i)
                time.sleep(0.1)
            
            st.success("GCN model trained successfully!")
            
            # Show sample results if requested
            if st.checkbox("Show Sample Results"):
                st.metric("Test Accuracy", "88%")
                st.dataframe(pd.DataFrame({
                    "Depth": np.arange(1000, 1005),
                    "Predicted Quality": ["High", "Moderate", "Very High", "Low", "Moderate"]
                }))

def visualization_page():
    # Check login for visualization
    if not st.session_state.get("logged_in", False):
        st.markdown("""
        <div class="card" style="text-align: center; max-width: 500px; margin: 50px auto;">
            <h2>Login Required</h2>
            <p>You need to log in to access the Visualization tools.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Login to Continue"):
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
    
    # Instead of loading all tabs at once, use radio buttons to show one at a time
    viz_type = st.radio(
        "Select Visualization", 
        ["Facies Classification", "Quality Prediction", "Model Performance"],
        horizontal=True
    )
    
    if viz_type == "Facies Classification":
        show_facies_visualization()
    elif viz_type == "Quality Prediction":
        show_quality_visualization()
    else:  # Model Performance
        show_performance_visualization()

# Split visualization into separate functions for lazy loading
def show_facies_visualization():
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
    
    # Generate demo data
    depth = np.arange(1000, 1100)
    facies = np.random.randint(0, selected_cluster, size=100)
    
    # Use HTML/CSS for basic visualization instead of matplotlib
    # Create a simplified visualization
    if st.checkbox("Show Facies Visualization"):
        # Generate colors from the selected colormap
        import matplotlib.cm as cm
        color_map = cm.get_cmap(color_scheme, selected_cluster)
        colors = []
        for i in range(selected_cluster):
            rgba = color_map(i)
            colors.append(f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})")
        
        # Create a simple HTML/CSS visualization
        html = """
        <div style="display: flex; height: 400px; width: 100%; border: 1px solid #ddd; border-radius: 5px;">
        """
        
        for i in range(selected_cluster):
            html += f"""
            <div style="flex: 1; background-color: {colors[i]}; display: flex; 
                       justify-content: center; align-items: center; color: white; font-weight: bold;">
                Facies {i+1}
            </div>
            """
        
        html += "</div>"
        
        st.markdown(html, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("<h4>Facies Statistics</h4>", unsafe_allow_html=True)
    
    # Calculate facies distribution
    facies_counts = np.zeros(selected_cluster)
    for i in range(selected_cluster):
        facies_counts[i] = np.sum(facies == i)
    
    # Create a simple bar chart using HTML/CSS
    st.bar_chart(pd.DataFrame({
        'Facies': [f"Facies {i+1}" for i in range(selected_cluster)],
        'Count': facies_counts
    }).set_index('Facies'))

def show_quality_visualization():
    st.markdown("<h3>Hydrocarbon Quality Prediction</h3>", unsafe_allow_html=True)
    
    # Generate demo data
    depth = np.arange(1000, 1100)
    quality_labels = ["Very_Low", "Low", "Moderate", "High", "Very_High"]
    quality = np.random.randint(0, 5, size=100)
    
    # Create quality distribution dataframe
    quality_counts = [np.sum(quality == i) for i in range(5)]
    quality_df = pd.DataFrame({
        'Quality': quality_labels,
        'Count': quality_counts
    })
    
    # Display as a bar chart
    st.bar_chart(quality_df.set_index('Quality'))
    
    # Show prediction table
    if st.checkbox("Show Prediction Table"):
        predictions = pd.DataFrame({
            "DEPTH": depth[:10],  # Show just first 10 rows
            "QUALITY": [quality_labels[q] for q in quality[:10]],
            "QUALITY_CODE": quality[:10]
        })
        
        st.dataframe(predictions)
        
        # Download button
        csv = predictions.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="quality_predictions.csv",
            mime="text/csv"
        )

def show_performance_visualization():
    st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)
    
    # Use selectbox instead of tabs to load one view at a time
    metric_type = st.selectbox(
        "Select Metric", 
        ["Learning Curves", "Confusion Matrix", "Classification Report"]
    )
    
    if metric_type == "Learning Curves":
        # Create simple metrics
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Final Accuracy", "88%")
        with col2: st.metric("Final Loss", "0.32")
        with col3: st.metric("Training Time", "2.4 min")
        
        # Show a simplified chart
        chart_data = pd.DataFrame({
            'Training Accuracy': np.linspace(0.6, 0.95, 20) + np.random.normal(0, 0.02, 20),
            'Validation Accuracy': np.linspace(0.55, 0.9, 20) + np.random.normal(0, 0.03, 20)
        })
        st.line_chart(chart_data)
        
    elif metric_type == "Confusion Matrix":
        # Create simplified confusion matrix
        classes = ["Very Low", "Low", "Moderate", "High", "Very High"]
        cm_data = np.zeros((5, 5))
        
        # Make diagonal dominant
        for i in range(5):
            cm_data[i, i] = np.random.randint(50, 100)
            for j in range(5):
                if i != j:
                    cm_data[i, j] = np.random.randint(0, 20)
        
        # Show as dataframe
        cm_df = pd.DataFrame(cm_data, 
                           index=classes,
                           columns=classes)
        st.table(cm_df)
        
    else:  # Classification Report
        # Create report dataframe
        report_data = {
            "Class": ["Very Low", "Low", "Moderate", "High", "Very High", "Average"],
            "Precision": [0.85, 0.78, 0.92, 0.86, 0.91, 0.86],
            "Recall": [0.82, 0.75, 0.90, 0.89, 0.84, 0.84],
            "F1-Score": [0.83, 0.76, 0.91, 0.87, 0.87, 0.85],
            "Support": [120, 85, 150, 95, 70, 520]
        }
        
        report_df = pd.DataFrame(report_data)
        st.table(report_df)

# The main function to run the app
def main():
    # Load CSS
    load_css()
    
    # Create sidebar
    sidebar_options = create_sidebar()
    
    # Render the appropriate page based on navigation
    page = sidebar_options["page"]
    
    if page == "Home":
        home_page()
    elif page == "Dataset Preparation":
        dataset_preparation_page()
    elif page == "Models":
        models_page()
    elif page == "Analysis Tool":
        analysis_tool_page()
    elif page == "Visualization":
        visualization_page()

if __name__ == "__main__":
    main()
