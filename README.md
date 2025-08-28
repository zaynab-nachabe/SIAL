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
pip install numpy matplotlib scipy librosa pyNN
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

## Reproducing Experiments

To reproduce the results presented in the Final Report:

1. **Baseline Performance**:
   ```bash
   cd Dumbo
   python DUMBO.py --save_results
   ```

2. **STDP-only Network**:
   ```bash
   python stdp.py
   ```

3. **Delay Learning Network**:
   ```bash
   cd just_delay
   python just_delay.py 
   ```

4. **Combined Approach**:
   ```bash
   python stdp_and_delay.py 
   ```
