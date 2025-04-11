# VHydro - Hydrocarbon Quality Prediction

VHydro is an advanced tool for predicting hydrocarbon quality from well log data using Graph Convolutional Networks (GCN). This application provides a full workflow from data loading and petrophysical property calculation to facies classification and hydrocarbon quality prediction.

![VHydro Logo](/src/VHydro_Logo.png)

## Features

- **Petrophysical Property Calculation**
  - Shale Volume
  - Porosity
  - Water/Oil Saturation
  - Permeability

- **Facies Classification**
  - K-means Clustering
  - Silhouette Score Optimization
  - Depth-based Facies Mapping

- **Graph-based Machine Learning**
  - Graph Convolutional Networks
  - Node and Edge Feature Extraction
  - Hydrocarbon Quality Classification

- **Visualization and Reporting**
  - Facies Visualization
  - Prediction Accuracy Metrics
  - Classification Reports

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/vhydro.git
   cd vhydro
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501).

3. Use the sidebar to navigate between different sections of the application.

## File Structure

```
vhydro/
├── app.py                          # Main Streamlit application
├── VHydro_final.py                 # VHydro class implementation
├── requirements.txt                # Required Python packages
├── logo.png                        # VHydro logo
├── workflow.png                    # Overall workflow diagram
├── dataset_preparation_workflow.png # Dataset preparation workflow
└── model_workflow.png              # Model workflow diagram
```

## Required Input Data

VHydro accepts well log data in LAS (Log ASCII Standard) format. The LAS file should contain depth information and relevant log curves, including:

- Gamma Ray (GR)
- Resistivity (LLD)
- Density (RHOB)
- Neutron Porosity (NPHI)

## Workflow

1. **Data Loading and Validation**
   - Load LAS file
   - Validate required curves
   - Handle missing or invalid data

2. **Petrophysical Property Calculation**
   - Calculate Shale Volume, Porosity, Water/Oil Saturation, and Permeability
   - Apply industry-standard equations and parameters

3. **Facies Classification**
   - Apply K-means clustering to identify natural groupings in the data
   - Determine optimal number of clusters using silhouette scores
   - Generate depth-based facies maps

4. **Graph Construction**
   - Create depth nodes and PET (Petrophysical Entity) nodes
   - Establish connections between related nodes
   - Generate adjacency matrices for GCN input

5. **GCN Model Training**
   - Train Graph Convolutional Network models
   - Optimize hyperparameters
   - Evaluate model performance

6. **Hydrocarbon Quality Prediction**
   - Classify depth points into quality categories
   - Identify high-potential zones
   - Generate detailed visualization and reports

## Output

VHydro generates various outputs, including:

- Excel files with petrophysical properties
- Facies classification results
- GCN model training history
- Hydrocarbon quality predictions
- Visualizations of facies, model performance, and quality predictions

## Deployment on Streamlit Cloud

1. Create a GitHub repository with your VHydro application code.
2. Make sure your repository includes:
   - app.py (the main Streamlit application)
   - VHydro_final.py
   - requirements.txt
   - All necessary image files (logo.png, workflow.png, etc.)
3. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.
4. Deploy your app by selecting your repository and specifying the main file (app.py).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use VHydro in your research, please cite:

```
@article{vhydro2023,
  title={VHydro: Graph Convolutional Networks for Hydrocarbon Quality Prediction from Well Log Data},
  author={Your Name},
  journal={Journal Name},
  year={2023}
}
```

## Contact

For any questions or feedback, please contact [your.email@example.com](mailto:your.email@example.com).
