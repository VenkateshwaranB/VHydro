# VHydro - Hydrocarbon Potential Prediction

![VHydro Banner](https://via.placeholder.com/1200x300/0e4194/ffffff?text=VHydro+-+Hydrocarbon+Potential+Prediction)

A Streamlit application for predicting hydrocarbon potential zones in wells using Graph Convolutional Networks (GCN) and petrophysical analysis.

## Overview

VHydro is a novel approach that uses Graph Convolutional Networks to analyze well log data and predict hydrocarbon potential zones. Based on the research paper ["Novel Graph Dataset for Hydrocarbon Potential Prediction"](https://link.springer.com/article/10.1007/s11053-024-10311-x), this application provides an intuitive interface for:

- Processing well log data from LAS files
- Creating graph datasets with nodes representing depth points
- Visualizing the graph data and well logs
- Predicting hydrocarbon potential using GCN
- Analyzing and interpreting prediction results

## Features

- **LAS File Processing**: Upload and process LAS files to extract well log curves
- **Petrophysical Analysis**: Calculate properties like porosity, saturation, and permeability
- **Graph Dataset Creation**: Convert well log data into graph format for GCN analysis
- **Interactive Visualizations**: Explore well logs, graph structures, and prediction results
- **GCN Model Integration**: Train and deploy Graph Convolutional Networks for prediction
- **Result Interpretation**: Analyze hydrocarbon potential distribution and depth profiles

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/vhydro.git
   cd vhydro
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Home Page**: Get an overview of the application and methodology
2. **Data Upload**: Upload your LAS file containing well log data
3. **Graph Dataset**: Create a graph dataset from the processed well log data
4. **Graph Visualization**: Explore the graph structure and petrophysical properties
5. **Model Prediction**: Run the GCN model to predict hydrocarbon potential
6. **Results**: Analyze and interpret the prediction results

For detailed instructions, refer to the [Tutorial](TUTORIAL.md).

## Example Data

The `examples` directory contains sample LAS files that can be used to test the application:

- `example1.las`: A well with known hydrocarbon zones
- `example2.las`: A dry well for comparison

## Directory Structure

```
vhydro/
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── README.md             # This file
├── TUTORIAL.md           # User tutorial
├── TECHNICAL.md          # Technical documentation
├── examples/             # Example LAS files
└── src/                  # Source code
    ├── data_processing.py    # Data processing functions
    ├── graph_generation.py   # Graph dataset generation
    ├── gcn_model.py          # GCN model implementation
    └── visualization.py      # Visualization functions
```

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## Technical Documentation

For detailed technical information, refer to the [Technical Documentation](TECHNICAL.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [StellarGraph](https://github.com/stellargraph/stellargraph) for the GCN implementation
- [Streamlit](https://streamlit.io/) for the web application framework
- [LASio](https://lasio.readthedocs.io/) for LAS file parsing
- [Original research paper](https://link.springer.com/article/10.1007/s11053-024-10311-x) on novel graph dataset for hydrocarbon potential prediction

## Citation

If you use VHydro in your research, please cite the original paper:

```
@article{author2024novel,
  title={Novel Graph Dataset for Hydrocarbon Potential Prediction},
  author={Author, A.},
  journal={Natural Resources Research},
  year={2024},
  publisher={Springer}
}
```

## Contact

For questions and support, please open an issue on GitHub or contact the project maintainers at [email@example.com](mailto:email@example.com).