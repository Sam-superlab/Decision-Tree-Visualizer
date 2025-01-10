# Decision Tree Visualizer

An interactive web application that visualizes the decision tree learning process, helping users understand how decision trees are constructed step by step.

## Features

- **Interactive Visualization**: Watch the decision tree grow in real-time with step-by-step visualization
- **Multiple Datasets**: Choose from various datasets including:
  - Moon-shaped data
  - Circular data
  - Linearly separable data
- **Adjustable Parameters**:
  - Maximum tree depth
  - Minimum samples per split
  - Split criterion (Gini Index or Entropy)
- **Dual View Display**:
  - Tree Structure View: Shows the hierarchical structure of the decision tree
  - Feature Space View: Displays data points and decision boundaries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Decision-Tree-Visualizer.git
cd Decision-Tree-Visualizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## Usage

1. Select a dataset from the dropdown menu
2. Adjust the parameters:
   - Max Depth: Controls the maximum depth of the tree
   - Min Samples: Sets the minimum number of samples required for splitting
   - Split Criterion: Choose between Gini Index or Entropy
3. Click "Train Decision Tree" to start the visualization
4. Use the control buttons to:
   - Step through the tree construction process
   - Play/pause automatic progression
   - Navigate between steps

## Technical Details

- **Backend**: Python Flask
- **Frontend**: HTML, JavaScript, Bootstrap
- **Visualization**: Plotly.js
- **Machine Learning**: Custom implementation of decision tree algorithm
- **Data Processing**: NumPy, scikit-learn for dataset generation

## Requirements

- Python 3.7+
- Flask
- NumPy
- scikit-learn
- matplotlib
- networkx
- plotly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

