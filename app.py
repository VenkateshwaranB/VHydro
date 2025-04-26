import streamlit as st
import os
import numpy as np
import pandas as pd
import time
from PIL import Image
import base64
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="StrataGraph",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
def load_css():
    st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }
    h1, h2, h3, h4, h5, h6 { color: #0e4194; }
    .colored-header { 
        background: linear-gradient(90deg, #0e4194 0%, #3a6fc4 100%); 
        color: white; 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px; 
        text-align: center; 
    }
    .card { 
        border-radius: 10px; 
        padding: 20px; 
        margin-bottom: 20px; 
        background: white; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); 
    }
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #0e4194 0%, #153a6f 100%); 
    }
    .sidebar-title {
        color: white !important;
        text-align: center;
        margin-bottom: 10px;
    }
    .coming-soon-section {
        background: rgba(30, 41, 59, 0.8);
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin: 30px 0;
    }
    .coming-soon-section h2 {
        color: white;
        margin-bottom: 15px;
    }
    .coming-soon-section .content {
        filter: blur(3px);
        pointer-events: none;
    }
    .footer-text {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.8rem;
        text-align: center;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Create sidebar
def create_sidebar():
    st.sidebar.markdown('<h1 class="sidebar-title">StrataGraph</h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<p style="color: white; text-align: center; margin-bottom: 20px;">VHydro 1.0</p>', unsafe_allow_html=True)
    
    # Initialize current page if not exists
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    
    # Main navigation
    st.sidebar.markdown("### Navigation", unsafe_allow_html=True)
    
    main_pages = ["Home", "VHydro", "CO2 Storage Applications", "Help and Contact", "About Us"]
    selected_main = st.sidebar.radio(
        "Main Pages", 
        main_pages, 
        index=main_pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in main_pages else 0
    )
    
    # VHydro subpages
    sub_page = None
    if selected_main == "VHydro":
        st.sidebar.markdown("#### VHydro Sections", unsafe_allow_html=True)
        vhydro_pages = [
            "VHydro Overview", 
            "Data Preparation", 
            "Petrophysical Properties", 
            "Facies Classification", 
            "Hydrocarbon Potential Using GCN"
        ]
        
        # Find the index of the current page in vhydro_pages if it exists
        current_index = 0
        if st.session_state["current_page"] in vhydro_pages:
            current_index = vhydro_pages.index(st.session_state["current_page"])
        
        sub_page = st.sidebar.radio(
            "VHydro Pages", 
            vhydro_pages, 
            index=current_index
        )
    
    # Version information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Versions")
    st.sidebar.markdown("✅ VHydro 1.0 (Current)")
    st.sidebar.markdown("🔶 CO2 Storage 2.0 (Coming Soon)")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="footer-text">© 2025 StrataGraph. All rights reserved.</div>', unsafe_allow_html=True)
    
    # Update session state based on navigation
    if sub_page and sub_page != st.session_state["current_page"]:
        st.session_state["current_page"] = sub_page
        st.rerun()
    elif selected_main != "VHydro" and selected_main != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_main
        st.rerun()
    
    return {
        "page": st.session_state["current_page"]
    }

def home_page():
    st.markdown("""
    <div class="colored-header">
        <h1>StrataGraph</h1>
        <p>Subsurface strata properties represented as a graph dataset for deep learning applications</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>About StrataGraph</h2>
        <p>StrataGraph is a cutting-edge platform for geoscience modeling and analysis, combining advanced machine learning techniques with traditional geological and petrophysical analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # VHydro section
    st.markdown("""
    <div class="card">
        <h2>StrataGraph 1.0 - VHydro</h2>
        <p>Our first release focuses on hydrocarbon quality prediction using Graph Convolutional Networks (GCNs) that model complex relationships between different petrophysical properties and depth values.</p>
        <p>This approach was introduced in our paper: <a href="https://link.springer.com/article/10.1007/s11053-024-10311-x" target="_blank">Hydrocarbon Potential Prediction Using Novel Graph Dataset</a>, which combines petrophysical and facies features to classify potential zones using GCN.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Button to explore VHydro
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Explore VHydro Analysis Tool", use_container_width=True):
            st.session_state["current_page"] = "VHydro Overview"
            st.rerun()

    # CO2 Storage section
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
    
    # Workflow tabs
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
        <h3>Technical Implementation</h3>
        <p>VHydro uses PyTorch Geometric and StellarGraph frameworks to implement Graph Convolutional Networks tailored for geoscience applications.</p>
        """, unsafe_allow_html=True)
    
    # Button to start analysis
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Begin Data Preparation", use_container_width=True):
            st.session_state["current_page"] = "Data Preparation"
            st.rerun()

def data_preparation_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Data Preparation</h1>
        <p>Prepare your well log data for VHydro analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>VHydro Data Preparation</h2>
        <p>VHydro requires specific log curves to calculate petrophysical properties.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload component
    uploaded_file = st.file_uploader("Choose a LAS file", type=["las"])
    
    if uploaded_file is not None:
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        
        # Store in session state
        st.session_state["uploaded_file"] = uploaded_file.name
        
        # Process data button
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                # Simulate processing with minimal updates
                progress_bar = st.progress(0)
                for i in range(0, 101, 25):
                    progress_bar.progress(i)
                    time.sleep(0.1)
            
            st.success("Data processing complete!")
            st.session_state["property_data"] = True
            
            # Next step button
            if st.button("Proceed to Petrophysical Properties"):
                st.session_state["current_page"] = "Petrophysical Properties"
                st.rerun()

def petrophysical_properties_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Petrophysical Properties</h1>
        <p>Calculate key reservoir properties from well log data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user has uploaded a file
    if "uploaded_file" not in st.session_state:
        st.warning("Please upload a LAS file first.")
        
        # Button to go back to upload tab
        if st.button("Go to Data Preparation"):
            st.session_state["current_page"] = "Data Preparation"
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
    
    # Calculate properties button
    if st.button("Calculate Properties"):
        with st.spinner("Calculating petrophysical properties..."):
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
        
        # Store in session state
        st.session_state["property_data"] = True
        
        # Next step button
        if st.button("Proceed to Facies Classification"):
            st.session_state["current_page"] = "Facies Classification"
            st.rerun()

def facies_classification_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Facies Classification</h1>
        <p>Identify geological facies using machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if properties have been calculated
    if "property_data" not in st.session_state:
        st.warning("Please calculate petrophysical properties first.")
        
        # Button to go back to property calculation
        if st.button("Go to Petrophysical Properties"):
            st.session_state["current_page"] = "Petrophysical Properties"
            st.rerun()
        return
    
    st.markdown("""
    <div class="card">
        <p>This step identifies natural rock types (facies) using K-means clustering.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clustering parameters
    min_clusters = st.number_input("Minimum Clusters", min_value=2, max_value=15, value=5, step=1)
    max_clusters = st.number_input("Maximum Clusters", min_value=min_clusters, max_value=15, value=10, step=1)
    
    # Run clustering button
    if st.button("Run Facies Classification"):
        with st.spinner("Running facies classification..."):
            progress_bar = st.progress(0)
            for i in range(0, 101, 20):
                progress_bar.progress(i)
                time.sleep(0.1)
        
        st.success("Facies classification completed successfully!")
        
        # Generate simulated silhouette scores
        optimal_clusters = 7
        st.info(f"Optimal number of clusters: {optimal_clusters}")
        
        # Create a simple facies dataset 
        facies_df = pd.DataFrame({
            "DEPTH": np.arange(1000, 1010),
            "FACIES": np.random.randint(0, optimal_clusters, size=10)
        })
        
        st.dataframe(facies_df)
        
        # Store in session state
        st.session_state["facies_data"] = True
        st.session_state["best_clusters"] = optimal_clusters
        
        # Next step button
        if st.button("Proceed to Hydrocarbon Potential Prediction"):
            st.session_state["current_page"] = "Hydrocarbon Potential Using GCN"
            st.rerun()

def hydrocarbon_prediction_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Hydrocarbon Potential Prediction</h1>
        <p>Graph Convolutional Network (GCN) Model for Quality Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if facies classification has been done
    if "facies_data" not in st.session_state:
        st.warning("Please complete facies classification first.")
        
        # Button to go back to facies classification
        if st.button("Go to Facies Classification"):
            st.session_state["current_page"] = "Facies Classification"
            st.rerun()
        return
    
    st.markdown("""
    <div class="card">
        <p>This step builds and trains a Graph Convolutional Network model to predict hydrocarbon quality.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model parameters
    n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=15, value=7, step=1)
    
    # Train model button
    if st.button("Train GCN Model"):
        with st.spinner("Training GCN model..."):
            progress_bar = st.progress(0)
            for i in range(0, 101, 20):
                progress_bar.progress(i)
                time.sleep(0.1)
        
        st.success("GCN model trained successfully!")
        
        # Show results
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Test Accuracy", "0.88")
        with col2: st.metric("F1 Score", "0.86")
        with col3: st.metric("AUC", "0.92")
        
        # Quality predictions
        st.subheader("Hydrocarbon Quality Predictions")
        predictions = pd.DataFrame({
            "DEPTH": np.arange(1000, 1010),
            "PREDICTED_QUALITY": np.random.choice(
                ["Very Low", "Low", "Moderate", "High", "Very High"], 
                size=10
            )
        })
        
        st.dataframe(predictions)
        
        # Store in session state
        st.session_state["model_complete"] = True

def co2_storage_page():
    st.markdown("""
    <div class="colored-header">
        <h1>CO2 Storage Applications</h1>
        <p>Carbon Capture, Utilization, and Storage (CCUS) Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Blurred coming soon section
    st.markdown("""
    <div class="coming-soon-section">
        <h2>CO2 Storage Potential Analysis</h2>
        <div class="content">
            <p>Advanced carbon capture utilization and storage (CCUS) modules powered by Graph Neural Networks.</p>
            <ul>
                <li>CO2 storage capacity prediction</li>
                <li>Caprock integrity analysis using geomechanical properties</li>
                <li>Built upon VHydro reservoir identification techniques</li>
                <li>Long-term storage monitoring</li>
            </ul>
        </div>
        <h3>Coming Soon in StrataGraph 2.0</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature preview
    st.markdown("""
    <div class="card">
        <h2>Upcoming Features</h2>
        <p>StrataGraph 2.0 will build upon the graph-based reservoir characterization from VHydro 1.0 to assess CO2 storage potential.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sign up form
    st.markdown("""
    <div class="card">
        <h2>Stay Updated</h2>
        <p>Sign up to receive updates when CO2 Storage Applications becomes available.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Name")
        st.text_input("Email")
    with col2:
        st.text_input("Organization")
        st.selectbox("Area of Interest", ["Carbon Storage", "Hydrocarbon Production", "Research", "Other"])
    
    if st.button("Notify Me"):
        st.success("Thank you for your interest! We'll keep you updated.")

def help_contact_page():
    st.markdown("""
    <div class="colored-header">
        <h1>Help and Contact</h1>
        <p>Get support and connect with our team</p>
    </div>
    """, unsafe_allow_html=True)
    
    # FAQ section
    st.markdown("""
    <div class="card">
        <h2>Frequently Asked Questions</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("What file formats are supported?"):
        st.markdown("""
        Currently, StrataGraph supports the following file formats:
        - LAS (Log ASCII Standard) for well log data
        - CSV for tabular data
        - Excel spreadsheets for tabular data
        """)
    
    # Contact form
    st.markdown("""
    <div class="card">
        <h2>Contact Us</h2>
        <p>Have questions or need assistance? Reach out to our team.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Name")
        st.text_input("Email")
    with col2:
        st.text_input("Subject")
        st.selectbox("Category", ["Technical Support", "Feature Request", "Billing", "Other"])
    
    st.text_area("Message", height=150)
    
    if st.button("Send Message"):
        st.success("Your message has been sent. We'll get back to you soon!")

def about_us_page():
    st.markdown("""
    <div class="colored-header">
        <h1>About Us</h1>
        <p>Learn about StrataGraph and our mission</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Our Mission</h2>
        <p>StrataGraph is committed to revolutionizing geoscience analysis through advanced machine learning techniques. Our mission is to provide geoscientists and engineers with powerful, intuitive tools that transform complex subsurface data into actionable insights.</p>
        
        <h2>Research</h2>
        <p>Our technology is built on peer-reviewed research:</p>
        <ul>
            <li><a href="https://link.springer.com/article/10.1007/s11053-024-10311-x" target="_blank">Hydrocarbon Potential Prediction Using Novel Graph Dataset</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact information
    st.markdown("""
    <div class="card">
        <h2>Contact Information</h2>
        <p><strong>Email:</strong> info@stratagraph.ai</p>
        <p><strong>Address:</strong> 123 Innovation Way, Houston, TX 77002</p>
    </div>
    """, unsafe_allow_html=True)

# Main app function
def main():
    # Load CSS
    load_css()
    
    # Create sidebar
    sidebar_options = create_sidebar()
    
    # Render the appropriate page based on navigation
    page = sidebar_options["page"]
    
    if page == "Home":
        home_page()
    elif page == "VHydro Overview":
        vhydro_overview_page()
    elif page == "Data Preparation":
        data_preparation_page()
    elif page == "Petrophysical Properties":
        petrophysical_properties_page()
    elif page == "Facies Classification":
        facies_classification_page()
    elif page == "Hydrocarbon Potential Using GCN":
        hydrocarbon_prediction_page()
    elif page == "VHydro":
        vhydro_overview_page()
    elif page == "CO2 Storage Applications":
        co2_storage_page()
    elif page == "Help and Contact":
        help_contact_page()
    elif page == "About Us":
        about_us_page()
    else:
        home_page()  # Default to home page

if __name__ == "__main__":
    main()
