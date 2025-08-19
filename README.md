# SIAL: Spiking Neural Networks for Pattern Detection

## Overview
SIAL represents my research project conducted during a two-month internship at CNRS (Centre National de la Recherche Scientifique). The project focuses on biologically plausible spiking neural networks (SNNs) for pattern detection, with a special emphasis on orca whale sound recognition.

## Repository Structure

### Network Architectures

- **convolutionalApproach/**  
  A convolutional spiking neural network architecture specialized in detecting orca whale sounds in audio data.

- **Dumbo/**  
  Contains `DUMBO.py`: A baseline spiking neural network with fixed weights and delays without learning capabilities. This serves as a control for comparing the effectiveness of various learning approaches.

- **just_stdp/**  
  Contains `stdp.py`: Implementation of a spiking neural network using only weight-based Spike-Timing Dependent Plasticity (STDP) with fixed delays. This represents the most common approach in the field and is not convolutional.

- **just_delay/**  
  Implementation of a spiking neural network where weight STDP is disabled to isolate and study the impact of delay learning. Based on the 2020 research paper "Bio-plausible delay learning in detecting spatio-temporal patterns." Includes:
  - `SIAL.py`: Core implementation
  - `delay_learning.py`: Delay learning algorithm
  - `run_delay_learning_experiment.py`: Experiment runner
  - `test_spikes.py`: Testing utilities
  - `visualisation_and_metrics.py`: Visualization tools and performance metrics

- **stdp&delay/**  
  A hybrid approach combining both delay learning and weight STDP for optimal pattern recognition performance.

### Analysis and Results

- **manual_test/**  
  Contains `specialized_neurons.py` and visualization outputs: A hand-tuned network with perfectly adjusted delays for specific patterns, achieving 100% accuracy. This validates the feasibility of the approach and serves as a theoretical upper bound for performance.

- **results/**  
  Experimental outcomes including accuracy measurements and visualizations:
  - `delay_learning_results.png`: Performance metrics for delay learning
  - `membrane_potential_coincident_arrival.png`: Visualization of neuron behavior during coincident spike arrival
  - `membrane_potential_separated_arrival.png`: Visualization of neuron behavior during temporally separated spike arrival

### Documentation

- **Final_Report/**  
  Contains `research_report.md`: Comprehensive documentation of findings, methodologies, experiments, and results from the internship.

- **Papers&References/**  
  Collection of key research papers and academic references that informed this project.

## Key Innovations

This project explores several novel approaches to pattern recognition in spiking neural networks:

1. **Delay Learning**: Implementation and analysis of biologically plausible delay adaptation mechanisms
2. **Convolutional Architectures**: Application of convolutional principles to spiking neural networks
3. **Hybrid Learning**: Combination of weight STDP and delay learning for enhanced pattern detection

## Technologies

- Python-based implementation
- Specialized simulation of spiking neural dynamics
- Custom visualization tools for neural activity analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/zaynab-nachabe/SIAL.git
cd SIAL

# Create and activate a virtual environment (recommended)
python -m venv sial_env
source sial_env/bin/activate  # On Windows: sial_env\Scripts\activate

# Install dependencies
pip install numpy matplotlib scipy librosa  # Add any other dependencies your project uses
```

## Usage Examples

### Running the Delay Learning Experiment
```python
# Navigate to the appropriate directory
cd just_delay

# Run the experiment with default parameters
python run_delay_learning_experiment.py

# For custom parameters
python run_delay_learning_experiment.py --patterns 3 --neurons 10 --learning_rate 0.01
```

### Testing with Fixed Delays (DUMBO)
```python
# Navigate to Dumbo directory
cd Dumbo

# Run the baseline network
python DUMBO.py
```

### Visualizing Results
```python
# From the main directory
python just_delay/visualisation_and_metrics.py --results_dir results/
```

### Example: Creating a Custom Spiking Pattern
```python
# Sample code to create a custom spiking pattern
import numpy as np
from just_delay.SIAL import create_pattern

# Create a simple pattern with 5 input neurons and 3 output neurons
input_spikes = [
    [10, 20, 30],  # Neuron 1 spikes at times 10, 20, 30 ms
    [15, 25, 35],  # Neuron 2 spikes at times 15, 25, 35 ms
    [5, 40],       # Neuron 3 spikes at times 5, 40 ms
    [22, 44],      # Neuron 4 spikes at times 22, 44 ms
    [18, 36, 54]   # Neuron 5 spikes at times 18, 36, 54 ms
]

# Test the pattern with the network
test_pattern(input_spikes)
```

## Reproducing Experiments

To reproduce the results presented in the Final Report:

1. **Baseline Performance**:
   ```bash
   cd Dumbo
   python DUMBO.py --save_results
   ```

2. **STDP-only Network**:
   ```bash
   cd just_stdp
   python stdp.py --epochs 100 --save_results
   ```

3. **Delay Learning Network**:
   ```bash
   cd just_delay
   python run_delay_learning_experiment.py --full --save_results
   ```

4. **Combined Approach**:
   ```bash
   # When implemented
   cd stdp&delay
   python combined_learning.py --epochs 100 --save_results
   ```
