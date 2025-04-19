import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from PIL import Image
from io import BytesIO
import logging

# Configure logging - Keep this simple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Set page configuration at the very beginning
st.set_page_config(
    page_title="VHydro - Hydrocarbon Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define simple auth functions - no imports required
def login(email, password):
    # Demo users
    users = {"user@example.com": "password", "admin@vhydro.com": "admin"}
    if email in users and users[email] == password:
        st.session_state["email"] = email
        st.session_state["logged_in"] = True
        return True
    return False

def logout():
    if "email" in st.session_state:
        del st.session_state["email"]
    if "logged_in" in st.session_state:
        del st.session_state["logged_in"]

# Basic CSS for a colorful, scientific UI
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary: #0066cc;
        --secondary: #6d28d9;
        --accent: #10b981;
        --background: #f8fafc;
        --sidebar: #1e293b;
        --text: #1e293b;
        --text-light: #64748b;
    }

    /* General styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar) 0%, #2d3748 100%);
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
        background-color: var(--primary);
        border-color: var(--primary);
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

# Simple function to display images
def display_image(image_path, caption="", width=None):
    try:
        image = Image.open(image_path)
        if width:
            w, h = image.size
            ratio = width / w
            image = image.resize((width, int(h * ratio)))
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        st.image(buffered, caption=caption, use_column_width=width is None)
    except Exception as e:
        st.error(f"Could not display image: {e}")

# Simple sidebar for navigation
def create_sidebar():
    st.sidebar.markdown('<div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">', unsafe_allow_html=True)
    # Logo - simple text as fallback
    st.sidebar.markdown("""
    <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
        <h2 style="color: #0066cc; margin: 0;">VHydro</h2>
        <p style="color: #0066cc; margin: 5px 0 0 0;">Hydrocarbon Quality Prediction</p>
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
    pages = ["Home", "Dataset Preparation", "Model Workflow", "Analysis Tool", "Results Visualization"]
    
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
        <p style="margin: 0;">VHydro predicts hydrocarbon quality zones using petrophysical properties and Graph Convolutional Networks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    return {
        "page": st.session_state["current_page"],
        "min_clusters": min_clusters,
        "max_clusters": max_clusters
    }

def home_page():
    st.markdown("""
    <div class="colored-header">
        <h1>VHydro - Hydrocarbon Quality Prediction</h1>
        <p>Advanced Graph Convolutional Network for Petrophysical Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>About VHydro</h2>
        <p>VHydro is an advanced tool for hydrocarbon quality prediction using well log data. It combines traditional petrophysical analysis with modern machine learning techniques to provide accurate predictions of reservoir quality.</p>
        <p>The tool uses Graph Convolutional Networks (GCN) to model the complex relationships between different petrophysical properties and depth values, enabling more accurate classification of hydrocarbon potential zones.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)
    
    # Create a 2x2 grid for features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-header">Petrophysical Property Calculation</div>
            <ul>
                <li>Shale Volume</li>
                <li>Porosity</li>
                <li>Water/Oil Saturation</li>
                <li>Permeability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
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
    
    st.markdown("<h2>Getting Started</h2>", unsafe_allow_html=True)
    
    # Create a simple workflow guide
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="border-left: 5px solid #10b981;">
            <div class="feature-header">1. Prepare Data</div>
            <p>Upload your well log data in LAS format and validate required curves.</p>
            <p>Navigate to the <b>Dataset Preparation</b> section to understand data requirements.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="border-left: 5px solid #f59e0b;">
            <div class="feature-header">2. Run Analysis</div>
            <p>Calculate petrophysical properties and run the GCN model.</p>
            <p>Use the <b>Analysis Tool</b> to process your data and generate predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="border-left: 5px solid #8b5cf6;">
            <div class="feature-header">3. Visualize Results</div>
            <p>Explore facies classifications and hydrocarbon quality predictions.</p>
            <p>Visit the <b>Results Visualization</b> section to interpret findings.</p>
        </div>
        """, unsafe_allow_html=True)

def dataset_preparation_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Dataset Preparation</h1>
        <p>Prepare your well log data for hydrocarbon quality prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Required Log Curves</h2>
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
    
    # Sample data download section
    st.markdown("""
    <div class="card">
        <h2>Sample Data</h2>
        <p>You can use the following sample data to test the VHydro workflow:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="border-left: 5px solid #f59e0b;">
            <div class="feature-header">Sample Well #1</div>
            <p>Sandstone reservoir with moderate hydrocarbon potential</p>
            <p>Contains all required log curves</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mock download button
        st.download_button(
            label="Download Sample #1", 
            data="Sample data placeholder", 
            file_name="sample_well_1.las",
            mime="text/plain"
        )
        
    with col2:
        st.markdown("""
        <div class="feature-card" style="border-left: 5px solid #8b5cf6;">
            <div class="feature-header">Sample Well #2</div>
            <p>Carbonate reservoir with high hydrocarbon potential</p>
            <p>Contains all required log curves plus additional data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mock download button
        st.download_button(
            label="Download Sample #2", 
            data="Sample data placeholder", 
            file_name="sample_well_2.las",
            mime="text/plain"
        )

def model_workflow_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Model Workflow</h1>
        <p>Understanding the VHydro prediction process</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Workflow Overview</h2>
        <p>VHydro follows a multi-stage workflow to predict hydrocarbon quality from well log data:</p>
        <ol>
            <li><b>Data Loading and Validation:</b> Import LAS file and validate required curves</li>
            <li><b>Petrophysical Property Calculation:</b> Calculate key reservoir properties</li>
            <li><b>Facies Classification:</b> Group similar rock types using K-means clustering</li>
            <li><b>Graph Construction:</b> Create node connections based on depth relationships</li>
            <li><b>GCN Model Training:</b> Train the Graph Convolutional Network model</li>
            <li><b>Hydrocarbon Quality Prediction:</b> Generate predictions for each depth point</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Petrophysical Property Calculation</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h3>Shale Volume (Vsh)</h3>
                <p>Calculated from Gamma Ray logs:</p>
                <pre style="background-color: #f1f5f9; padding: 10px; border-radius: 5px;">Vsh = (GR - GRmin) / (GRmax - GRmin)</pre>
                
                <h3>Density Porosity (φd)</h3>
                <p>Calculated from Bulk Density logs:</p>
                <pre style="background-color: #f1f5f9; padding: 10px; border-radius: 5px;">φd = (ρmatrix - ρbulk) / (ρmatrix - ρfluid)</pre>
                
                <h3>Effective Porosity (φeff)</h3>
                <p>Corrected for shale content:</p>
                <pre style="background-color: #f1f5f9; padding: 10px; border-radius: 5px;">φeff = φd - (Vsh * 0.3)</pre>
            </div>
            <div>
                <h3>Water Saturation (Sw)</h3>
                <p>Calculated using Archie's equation:</p>
                <pre style="background-color: #f1f5f9; padding: 10px; border-radius: 5px;">Sw = ((a * (φeff^-m)) / (Rt * Rw))^(1/n)</pre>
                
                <h3>Oil Saturation (So)</h3>
                <p>Calculated as:</p>
                <pre style="background-color: #f1f5f9; padding: 10px; border-radius: 5px;">So = 1 - Sw</pre>
                
                <h3>Permeability (K)</h3>
                <p>Estimated using porosity-based correlation:</p>
                <pre style="background-color: #f1f5f9; padding: 10px; border-radius: 5px;">K = 0.00004 * exp(57.117 * φeff)</pre>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Graph Convolutional Network Model</h2>
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
                    <li><b>Hidden Layers:</b> Graph convolutional layers</li>
                    <li><b>Output Layer:</b> Classification layer for quality prediction</li>
                    <li><b>Regularization:</b> Dropout and batch normalization</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple workflow diagram with matplotlib
    st.markdown("<h2>Workflow Diagram</h2>", unsafe_allow_html=True)
    
    # Create a simple workflow diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Disable axis
    ax.axis('off')
    
    # Create boxes for steps
    steps = [
        "Data Loading", "Petrophysical\nCalculation", 
        "Facies\nClassification", "Graph\nConstruction", 
        "GCN Model\nTraining", "Quality\nPrediction"
    ]
    
    colors = [
        "#3b82f6", "#8b5cf6", "#ec4899", 
        "#f59e0b", "#10b981", "#0ea5e9"
    ]
    
    # Position boxes
    for i, (step, color) in enumerate(zip(steps, colors)):
        x = 0.1 + i * 0.15
        ax.add_patch(plt.Rectangle((x, 0.4), 0.12, 0.2, 
                                  fill=True, color=color, alpha=0.7))
        ax.text(x + 0.06, 0.5, step, ha='center', va='center', 
               color='white', fontweight='bold')
        
        # Add arrow if not the last step
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.13, 0.5), xytext=(x + 0.15, 0.5),
                       arrowprops=dict(arrowstyle="->", lw=2, color="#64748b"))
    
    st.pyplot(fig)

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
        <p>Upload data and run the VHydro analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for the analysis workflow
    tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Property Calculation", "Facies Classification", "GCN Model"])
    
    with tab1:
        st.markdown("<h3>Upload LAS File</h3>", unsafe_allow_html=True)
        
        # File upload section with a colorful border
        st.markdown("""
        <div style="border: 2px dashed #0066cc; border-radius: 10px; padding: 30px; text-align: center; margin: 20px 0;">
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
            st.markdown("""
            <div class="card">
                <h4>File Information</h4>
                <table style="width: 100%;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">File Name:</td>
                        <td style="padding: 8px;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">File Size:</td>
                        <td style="padding: 8px;">{:.2f} KB</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Upload Time:</td>
                        <td style="padding: 8px;">{}</td>
                    </tr>
                </table>
            </div>
            """.format(
                uploaded_file.name,
                uploaded_file.size / 1024,
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
