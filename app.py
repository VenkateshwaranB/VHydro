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
    /* Main styling */
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
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #0e4194 0%, #153a6f 100%); 
    }
    
    /* Sidebar header/logo area */
    .sidebar-logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem 0;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sidebar-logo-container img {
        max-width: 80%;
        margin-bottom: 1rem;
    }
    
    .sidebar-logo-container h1 {
        color: white !important;
        font-size: 1.8rem;
        margin: 0;
        padding: 0;
    }
    
    .version-tag {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 8px 0;
    }
    
    /* Navigation menu styling */
    .nav-container {
        color: white;
        margin-top: 2rem;
    }
    
    .nav-section {
        margin-bottom: 1rem;
    }
    
    .section-title {
        color: white !important;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        padding-left: 1rem;
    }
    
    /* Coming soon tag */
    .coming-soon-tag {
        background-color: rgba(255, 152, 0, 0.2);
        color: rgb(255, 152, 0);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-left: 8px;
        vertical-align: middle;
    }
    
    /* Coming soon section */
    .coming-soon-section {
        background: linear-gradient(rgba(30, 41, 59, 0.8), rgba(30, 41, 59, 0.8)), url('https://placehold.co/600x400');
        background-size: cover;
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin: 30px 0;
        filter: blur(0px); /* Container not blurred */
    }
    
    .coming-soon-section h2 {
        color: white;
        margin-bottom: 15px;
    }
    
    .coming-soon-section .content {
        filter: blur(3px); /* Content inside is blurred */
        pointer-events: none;
    }
    
    .footer-text {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.8rem;
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    /* Streamlit element styling overrides for white text */
    div[data-baseweb="select"] > div,
    div[class*="stRadio"] label,
    .stButton button,
    .stButton button:hover,
    .custom-radio,
    .custom-radio:hover,
    .custom-radio.selected,
    .version-section h4,
    .version-item {
        color: white !important;
    }
    
    /* Specific overrides where white might not be the default */
    div[data-baseweb="select"] > div {
        background-color: #0e4194; /* Ensure background is dark for white text */
    }
    
    .stButton button {
        background-color: #0e4194;
    }
    
    .stButton button:hover {
        background-color: #3a6fc4;
    }
    
    /* Custom radio buttons */
    .custom-radio {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    .custom-radio:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    .custom-radio.selected {
        background-color: rgba(255, 255, 255, 0.3);
        border-left: 3px solid white;
    }
    
    /* Version section */
    .version-section {
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    
    .version-section h4 {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .version-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
    }
    
    .version-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 10px;
    }
    
    .active-version {
        background-color: #4CAF50;
    }
    
    .coming-version {
        background-color: #FFA500;
    }
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

# Create a simplified sidebar navigation system
def create_sidebar():
    # Logo and title section
    # Function to load and encode image
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode()
        return f"data:image/png;base64,{b64_data}"
    
    # Path to your image
    image_base64 = get_base64_image("src/StrataGraph_White_Logo.png")
    
    # Inject image via HTML in the sidebar
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="{image_base64}" alt="StrataGraph Logo" style="width: 80%; max-width: 250px; margin-bottom: 10px;"/>
            <h1 style="font-size: 24px;">StrataGraph</h1>
            <div style="font-size: 14px; color: gray;">VHydro 1.0</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize current page if not exists
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
     # Initialize VHydro subpage state
    if "vhydro_subpage" not in st.session_state:
        st.session_state["vhydro_subpage"] = None
    
    # Simple navigation using radio buttons instead of custom HTML/buttons
    st.sidebar.markdown('<div class="nav-container">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    
    # Main navigation as a simple radio
    main_pages = ["Home", "VHydro", "CO2 Storage Applications", "Help and Contact", "About Us"]
    selected_main = st.sidebar.radio("", main_pages, index=main_pages.index(st.session_state["current_page"]) 
                                     if st.session_state["current_page"] in main_pages else 0,
                                     label_visibility="collapsed")
    
    # If VHydro is selected, show sub-pages using buttons and session state
    if selected_main == "VHydro":
        st.sidebar.markdown('<div style="margin-left: 1.5rem;">', unsafe_allow_html=True)
        vhydro_pages = {
            "VHydro Overview": "VHydro Overview",
            "Data Preparation": "Data Preparation",
            "Petrophysical Properties": "Petrophysical Properties",
            "Facies Classification": "Facies Classification",
            "Hydrocarbon Potential Using GCN": "Hydrocarbon Potential Using GCN"
        }
    
        for page_name, page_value in vhydro_pages.items():
            if st.sidebar.button(page_name, key=page_name):
                st.session_state["current_page"] = page_value
                st.session_state["vhydro_subpage"] = page_value  # Track selected VHydro subpage
                st.rerun()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Update session state with selected main page
    if selected_main != "VHydro" and selected_main != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_main
        st.session_state["vhydro_subpage"] = None  # Clear VHydro subpage
        st.rerun()
    
    # Version selection section
    st.sidebar.markdown(
        """
        <div class="version-section">
            <h4>Versions</h4>
            <div class="version-item">
                <div class="version-indicator active-version"></div>
                VHydro 1.0 (Current)
            </div>
            <div class="version-item">
                <div class="version-indicator coming-version"></div>
                CO2 Storage 2.0 (Coming Soon)
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Only show model configuration in analysis pages
    if st.session_state["current_page"] == "Facies Classification":
        st.sidebar.markdown('<div class="section-title">Analysis Parameters</div>', unsafe_allow_html=True)
        min_clusters = st.sidebar.slider("Min Clusters", 2, 15, 5)
        max_clusters = st.sidebar.slider("Max Clusters", min_clusters, 15, 10)
    else:
        min_clusters = 5
        max_clusters = 10
    
    # Footer
    st.sidebar.markdown(
        """
        <div class="footer-text">
            © 2025 StrataGraph. All rights reserved.<br>
            Version 1.0.0
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    return {
        "page": st.session_state["current_page"],
        "min_clusters": min_clusters,
        "max_clusters": max_clusters
    }

def home_page():
    # Try to load the banner image
    banner_path = "src/StrataGraph_Banner.png"
    try:
        st.image(banner_path, use_container_width=True)
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
    </div>
    """, unsafe_allow_html=True)
    
    # First section: VHydro
    st.markdown("""
    <div class="card">
        <h2>StrataGraph 1.0 - VHydro</h2>
        <p>Our first release focuses on hydrocarbon quality prediction using Graph Convolutional Networks (GCNs) that model complex relationships between different petrophysical properties and depth values.</p>
        <p>VHydro 1.0 enables accurate prediction of hydrocarbon zones using a graph-based approach that captures the spatial relationships between well log measurements.</p>
        <p>This approach was introduced in our paper: <a href="https://link.springer.com/article/10.1007/s11053-024-10311-x" target="_blank">Hydrocarbon Potential Prediction Using Novel Graph Dataset</a>, which combines petrophysical and facies features to classify potential zones using GCN.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # VHydro Workflow section
    st.markdown("<h2>VHydro Workflow</h2>", unsafe_allow_html=True)
    
    # Try to load the workflow image
    workflow_path = "src/Workflow.png"
    try:
        st.image(workflow_path, use_container_width=True)
    except:
        st.warning("Workflow image not found.")
    
    # Button to explore VHydro - simplified single button
    if st.button("Explore VHydro Analysis Tool", key="explore_vhydro_btn"):
        st.session_state["current_page"] = "VHydro Overview"
        st.rerun()

    # Second section: CO2 Storage (Coming Soon) - Now directly below VHydro
    st.markdown("""
    <div class="coming-soon-section">
        <h2>StrataGraph 2.0 - CO2 Storage Potential Analysis</h2>
        <div class="content">
            <p>Advanced carbon capture utilization and storage (CCUS) modules powered by Graph Neural Networks.</p>
            <ul>
                <li>CO2 storage capacity prediction</li>
                <li>Caprock integrity analysis using geomechanical properties</li>
                <li>Built upon VHydro reservoir identification techniques</li>
                <li>Long-term storage monitoring</li>
            </ul>
        </div>
        <h3>Coming Soon</h3>
    </div>
    """, unsafe_allow_html=True)

def vhydro_overview_page():
    st.markdown("""
    <div class="colored-header">
        <h1>VHydro</h1>
        <p>Hydrocarbon Quality Prediction Using Graph Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>About VHydro</h2>
        <p>VHydro is StrataGraph's flagship module for predicting hydrocarbon quality zones using Graph Convolutional Networks (GCNs).</p>
        <p>This advanced approach models the complex relationships between petrophysical properties and depth to provide more accurate predictions than traditional methods.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add tabs for the different sections
    tab1, tab2, tab3 = st.tabs(["Overview", "Workflow", "Technical Details"])
    
    with tab1:
        st.markdown("""
        <h3>VHydro Features</h3>
        <ul>
            <li>Graph-based representation of well log data</li>
            <li>Advanced facies classification using K-means clustering</li>
            <li>GCN-based quality prediction model</li>
            <li>Interactive visualization of results</li>
        </ul>
        """, unsafe_allow_html=True)
        
    with tab2:
        # Try to load the workflow image
        workflow_path = "src/Workflow.png"
        try:
            st.image(workflow_path, use_container_width=True)
        except:
            st.warning("Workflow image not found.")
            
        st.markdown("""
        <h3>VHydro Workflow</h3>
        <ol>
            <li><strong>Data Preparation</strong>: Upload and process well log data</li>
            <li><strong>Petrophysical Analysis</strong>: Calculate key properties like porosity, permeability</li>
            <li><strong>Facies Classification</strong>: Group similar depth points using K-means clustering</li>
            <li><strong>Graph Construction</strong>: Create a graph representation of the well data</li>
            <li><strong>GCN Training</strong>: Train the model to predict hydrocarbon quality</li>
            <li><strong>Visualization</strong>: Interpret and visualize results</li>
        </ol>
        """, unsafe_allow_html=True)
        
    with tab3:
        st.markdown("""
        <h3>Technical Details</h3>
        <p>VHydro utilizes a Graph Convolutional Network (GCN) model to predict hydrocarbon quality zones. The model is trained on a dataset of well log measurements, including petrophysical properties and facies classifications.</p>
        <p>The GCN model is able to capture the spatial relationships between well log measurements, which allows it to make more accurate predictions than traditional methods.</p>
        """, unsafe_allow_html=True)

def data_preparation_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Data Preparation</h1>
        <p>Prepare and upload well log data for VHydro analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Data Upload</h2>
        <p>Upload well log data in LAS or CSV format.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add file uploader
    uploaded_file = st.file_uploader("Upload well log data", type=["las", "csv"])
    
    if uploaded_file is not None:
        # Read the data into a Pandas DataFrame
        try:
            df = pd.read_csv(uploaded_file)  # Assuming CSV format
        except:
            st.error("Failed to read the CSV file. Please ensure it is properly formatted.")
            return
        
        # Display the DataFrame
        st.dataframe(df)
        
        # Data cleaning options
        st.markdown("""
        <div class="card">
            <h2>Data Cleaning Options</h2>
            <p>Clean and preprocess the data before analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Option to remove NaN values
        if st.checkbox("Remove NaN values"):
            df = df.dropna()
            st.success("NaN values removed from the data.")
        
        # Option to select columns
        st.markdown("Select columns to use for analysis:")
        selected_columns = st.multiselect("Columns", df.columns)
        
        # Display selected columns
        if selected_columns:
            st.dataframe(df[selected_columns])
        
        # Store the DataFrame in session state
        st.session_state["data"] = df[selected_columns]

def petrophysical_properties_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Petrophysical Properties</h1>
        <p>Calculate petrophysical properties from well log data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Calculate Properties</h2>
        <p>Calculate petrophysical properties such as porosity, permeability, and water saturation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if data is loaded
    if "data" not in st.session_state:
        st.warning("Please upload data on the Data Preparation page first.")
        return
    
    # Get the DataFrame from session state
    df = st.session_state["data"]
    
    # Add input fields for petrophysical properties
    st.markdown("""
    <div class="card">
        <h2>Input Parameters</h2>
        <p>Enter the parameters for calculating petrophysical properties.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input parameters for porosity calculation
    porosity_method = st.selectbox("Porosity Calculation Method", ["Density Log", "Sonic Log"])
    
    # Display different input fields based on the selected method
    if porosity_method == "Density Log":
        matrix_density = st.number_input("Matrix Density (g/cm³)", value=2.65)
        fluid_density = st.number_input("Fluid Density (g/cm³)", value=1.0)
        density_log_column = st.selectbox("Density Log Column", df.columns)
        
        # Calculate porosity using the density log
        df["Porosity"] = (matrix_density - df[density_log_column]) / (matrix_density - fluid_density)
        
    elif porosity_method == "Sonic Log":
        matrix_transit_time = st.number_input("Matrix Transit Time (µs/ft)", value=55.5)
        fluid_transit_time = st.number_input("Fluid Transit Time (µs/ft)", value=189.0)
        sonic_log_column = st.selectbox("Sonic Log Column", df.columns)
        
        # Calculate porosity using the sonic log
        df["Porosity"] = (df[sonic_log_column] - matrix_transit_time) / (fluid_transit_time - matrix_transit_time)
    
    # Display the DataFrame with calculated porosity
    st.dataframe(df)

def facies_classification_page(min_clusters=5, max_clusters=10):
    st.markdown("""
    <div class="colored-header">
        <h1>Facies Classification</h1>
        <p>Classify facies using K-means clustering.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login form (moved to the top)
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        with st.form("login"):
            st.subheader("Login to StrataGraph")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        
        if submitted:
            if login(email, password):
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Incorrect email or password")
        return  # Exit the function if not logged in
    
    st.markdown("""
    <div class="card">
        <h2>K-means Clustering</h2>
        <p>Group similar depth points into facies using K-means clustering.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if data is loaded
    if "data" not in st.session_state:
        st.warning("Please upload data on the Data Preparation page first.")
        return
    
    # Get the DataFrame from session state
    df = st.session_state["data"]
    
    # Select columns for clustering
    st.markdown("Select columns to use for clustering:")
    clustering_columns = st.multiselect("Columns", df.columns)
    
    # Run K-means clustering
    if clustering_columns:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[clustering_columns])
        
        # Run K-means clustering
        kmeans = KMeans(n_clusters=min_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        
        # Add the cluster labels to the DataFrame
        df["Facies"] = kmeans.labels_
        
        # Display the DataFrame with cluster labels
        st.dataframe(df)
        
        # Plot the clusters
        fig, ax = plt.subplots()
        scatter = ax.scatter(df[clustering_columns[0]], df[clustering_columns[1]], c=df["Facies"])
        ax.set_xlabel(clustering_columns[0])
        ax.set_ylabel(clustering_columns[1])
        ax.legend(*scatter.legend_elements(), title="Facies")
        st.pyplot(fig)
    
    # Add a logout button
    if st.button("Logout"):
        logout()
        st.rerun()

def hydrocarbon_potential_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Hydrocarbon Potential</h1>
        <p>Predict hydrocarbon potential using GCN.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>GCN Prediction</h2>
        <p>Predict hydrocarbon potential using GCN based on petrophysical properties and facies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if data is loaded
    if "data" not in st.session_state:
        st.warning("Please upload data on the Data Preparation page first.")
        return
    
    # Get the DataFrame from session state
    df = st.session_state["data"]
    
    st.write("Hydrocarbon Potential Prediction page content goes here.")

def help_and_contact_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Help and Contact</h1>
        <p>Get help and contact us for support.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Contact Information</h2>
        <p>Contact us for any questions or support.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add contact information
    st.markdown("""
    <p>Email: support@stratagraph.com</p>
    <p>Phone: 1-800-STRATAGRAPH</p>
    """, unsafe_allow_html=True)

def about_us_page():
    st.markdown("""
    <div class="colored-header">
        <h1>About Us</h1>
        <p>Learn more about StrataGraph.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Our Mission</h2>
        <p>Our mission is to provide cutting-edge solutions for geoscience modeling and analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add more information about the company
    st.markdown("""
    <p>StrataGraph is a leading provider of geoscience modeling and analysis solutions. We are committed to providing our customers with the best possible tools and services.</p>
    """, unsafe_allow_html=True)

def co2_storage_page():
    st.markdown("""
    <div class="colored-header">
        <h1>CO2 Storage</h1>
        <p>Carbon Capture and Storage Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h2>CO2 Storage Potential Analysis</h2>
        <p>Analyze potential sites for carbon capture and storage using advanced GNN models.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("CO2 Storage page content goes here.")

def main():
    load_css()
    sidebar_data = create_sidebar()
    page = sidebar_data["page"]
    min_clusters = sidebar_data["min_clusters"]
    max_clusters = sidebar_data["max_clusters"]
    
    if page == "Home":
        home_page()
    elif page == "VHydro Overview":
        vhydro_overview_page()
    elif page == "Data Preparation":
        data_preparation_page()
    elif page == "Petrophysical Properties":
        petrophysical_properties_page()
    elif page == "Facies Classification":
        facies_classification_page(min_clusters, max_clusters)
    elif page == "Hydrocarbon Potential Using GCN":
        hydrocarbon_potential_page()
    elif page == "Help and Contact":
        help_and_contact_page()
    elif page == "About Us":
        about_us_page()
    elif page == "CO2 Storage Applications":
        co2_storage_page()

if __name__ == "__main__":
    main()
