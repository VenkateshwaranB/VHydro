import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import tempfile
import json
import time
from datetime import datetime

def create_download_link(df, filename, text):
    """
    Create a download link for a pandas DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to download
    filename : str
        Name of the download file
    text : str
        Text to display for the download link
        
    Returns:
    --------
    str
        HTML link for downloading the DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" target="_blank">{text}</a>'
    return href

def create_figure_download_link(fig, filename, text):
    """
    Create a download link for a matplotlib figure
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to download
    filename : str
        Name of the download file
    text : str
        Text to display for the download link
        
    Returns:
    --------
    str
        HTML link for downloading the figure
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" target="_blank">{text}</a>'
    return href

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary directory
    
    Parameters:
    -----------
    uploaded_file : streamlit.UploadedFile
        Uploaded file object
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        # Write the contents of the uploaded file to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        # Return the path to the temporary file
        return tmp_file.name

def create_project_folder(project_name, base_dir=None):
    """
    Create a project folder structure
    
    Parameters:
    -----------
    project_name : str
        Name of the project
    base_dir : str, optional
        Base directory for the project
        
    Returns:
    --------
    dict
        Dictionary containing paths to the project folders
    """
    # Create timestamp for unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set base directory
    if base_dir is None:
        base_dir = tempfile.gettempdir()
    
    # Create project directory
    project_dir = os.path.join(base_dir, f"{project_name}_{timestamp}")
    os.makedirs(project_dir, exist_ok=True)
    
    # Create subdirectories
    data_dir = os.path.join(project_dir, "data")
    results_dir = os.path.join(project_dir, "results")
    models_dir = os.path.join(project_dir, "models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Return paths dictionary
    paths = {
        'project_dir': project_dir,
        'data_dir': data_dir,
        'results_dir': results_dir,
        'models_dir': models_dir
    }
    
    return paths

def save_project_metadata(project_dir, metadata):
    """
    Save project metadata to a JSON file
    
    Parameters:
    -----------
    project_dir : str
        Path to the project directory
    metadata : dict
        Dictionary containing project metadata
    """
    # Create metadata file path
    metadata_file = os.path.join(project_dir, "metadata.json")
    
    # Add timestamp
    metadata['timestamp'] = datetime.now().isoformat()
    
    # Save metadata to file
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_project_metadata(project_dir):
    """
    Load project metadata from a JSON file
    
    Parameters:
    -----------
    project_dir : str
        Path to the project directory
        
    Returns:
    --------
    dict
        Dictionary containing project metadata
    """
    # Create metadata file path
    metadata_file = os.path.join(project_dir, "metadata.json")
    
    # Load metadata from file
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    except:
        return None

def format_time(seconds):
    """
    Format time in seconds to a human-readable string
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
        
    Returns:
    --------
    str
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def display_progress(progress_bar, progress, message=None):
    """
    Display progress with a progress bar and message
    
    Parameters:
    -----------
    progress_bar : streamlit.ProgressBar
        Streamlit progress bar object
    progress : float
        Progress value (0-100)
    message : str, optional
        Message to display
    """
    # Update progress bar
    progress_bar.progress(progress / 100)
    
    # Display message
    if message:
        st.write(message)
    
    # Add small delay for visual feedback
    time.sleep(0.01)

def create_export_zip(project_dir, filename="project_export.zip"):
    """
    Create a ZIP file containing the project files
    
    Parameters:
    -----------
    project_dir : str
        Path to the project directory
    filename : str
        Name of the ZIP file
        
    Returns:
    --------
    str
        Path to the created ZIP file
    """
    import shutil
    
    # Create ZIP file path
    zip_path = os.path.join(tempfile.gettempdir(), filename)
    
    # Create ZIP file
    shutil.make_archive(
        os.path.splitext(zip_path)[0],  # Path without extension
        'zip',                          # Format
        project_dir                     # Directory to zip
    )
    
    return zip_path

def create_download_button_for_file(file_path, button_text, mime_type="application/zip"):
    """
    Create a Streamlit download button for a file
    
    Parameters:
    -----------
    file_path : str
        Path to the file to download
    button_text : str
        Text to display on the button
    mime_type : str
        MIME type of the file
        
    Returns:
    --------
    bool
        True if the button was clicked, False otherwise
    """
    # Read file
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    # Create download button
    return st.download_button(
        label=button_text,
        data=file_bytes,
        file_name=os.path.basename(file_path),
        mime=mime_type
    )

def get_hydrocarbon_potential_color_scale():
    """
    Get a color scale for hydrocarbon potential labels
    
    Returns:
    --------
    dict
        Dictionary mapping labels to colors
    """
    return {
        'Very_Low': '#d73027',
        'Low': '#fc8d59',
        'Moderate': '#fee090',
        'High': '#91cf60',
        'Very_High': '#1a9850'
    }

def format_number(value, precision=2):
    """
    Format a number with specified precision
    
    Parameters:
    -----------
    value : float
        Number to format
    precision : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted number
    """
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return value

def create_summary_report(result_dict, model_accuracy, output_path):
    """
    Create a summary report of the analysis
    
    Parameters:
    -----------
    result_dict : dict
        Dictionary containing analysis results
    model_accuracy : float
        Model accuracy
    output_path : str
        Path to save the report
        
    Returns:
    --------
    str
        Path to the created report
    """
    # Extract data
    features = result_dict['features']
    labeled_data = result_dict['edge_data']
    
    # Calculate statistics
    hc_potential_counts = labeled_data['PET_label'].value_counts().to_dict()
    hc_potential_percent = {k: v/len(labeled_data)*100 for k, v in hc_potential_counts.items()}
    
    # Extract minimum and maximum depth
    min_depth = features['DEPTH'].min()
    max_depth = features['DEPTH'].max()
    
    # Find zones with high potential
    high_potential = labeled_data[labeled_data['PET_label'].isin(['High', 'Very_High'])]
    high_potential_intervals = []
    
    if len(high_potential) > 0:
        # Group consecutive depths
        high_potential_sorted = high_potential.sort_values('DEPTH')
        high_potential_depths = high_potential_sorted['DEPTH'].values
        
        # Find breaks in consecutive depths
        breaks = np.where(np.diff(high_potential_depths) > 1)[0] + 1
        segments = np.split(high_potential_depths, breaks)
        
        # Format intervals
        for segment in segments:
            if len(segment) > 0:
                interval = {
                    'start': float(segment[0]),
                    'end': float(segment[-1]),
                    'thickness': float(segment[-1] - segment[0])
                }
                high_potential_intervals.append(interval)
    
    # Create report content
    report = f"""
    # Hydrocarbon Potential Analysis Summary Report
    
    ## Well Information
    
    - Depth Range: {format_number(min_depth)} - {format_number(max_depth)} m
    - Analyzed Interval: {format_number(max_depth - min_depth)} m
    
    ## Hydrocarbon Potential Distribution
    
    | Potential | Count | Percentage |
    |-----------|-------|------------|
    """
    
    for label in ['Very_High', 'High', 'Moderate', 'Low', 'Very_Low']:
        if label in hc_potential_counts:
            report += f"| {label} | {hc_potential_counts[label]} | {format_number(hc_potential_percent[label])}% |\n"
    
    report += f"""
    ## Model Performance
    
    - Model Accuracy: {format_number(model_accuracy*100)}%
    
    ## High Potential Zones
    
    """
    
    if high_potential_intervals:
        report += "| Start Depth | End Depth | Thickness |\n"
        report += "|-------------|-----------|----------|\n"
        
        for interval in high_potential_intervals:
            report += f"| {format_number(interval['start'])} | {format_number(interval['end'])} | {format_number(interval['thickness'])} |\n"
    else:
        report += "No high potential zones identified.\n"
    
    report += """
    ## Recommendations
    
    Based on the analysis:
    
    1. Focus further exploration and testing on the identified high-potential zones.
    2. Conduct additional analysis (e.g., seismic interpretation, core analysis) to confirm the predicted hydrocarbon potential.
    3. Use this VHydro graph-based approach for analyzing other wells in the same field to identify field-wide trends.
    
    ## Notes
    
    This report was generated using the VHydro Graph Convolutional Network approach for hydrocarbon potential prediction.
    """
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write(report)
    
    return output_path

def show_colorbar_legend():
    """
    Show a colorbar legend for hydrocarbon potential
    """
    # Get color scale
    color_scale = get_hydrocarbon_potential_color_scale()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.set_title('Hydrocarbon Potential Legend')
    
    # Hide axes
    ax.axis('off')
    
    # Create color patches
    labels = ['Very_Low', 'Low', 'Moderate', 'High', 'Very_High']
    colors = [color_scale[label] for label in labels]
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = i / len(labels)
        rect = plt.Rectangle((x, 0), 1/len(labels), 1, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + 0.5/len(labels), 0.5, label, 
                ha='center', va='center', fontsize=10, 
                color='black' if label == 'Moderate' else 'white')
    
    return fig