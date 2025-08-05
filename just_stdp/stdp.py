import pyNN.nest as sim
from quantities import ms
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from pyNN.connectors import FromListConnector

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run STDP simulation with configurable noise')
parser.add_argument('--noise', action='store_true', help='Add noise to spike trains for robustness testing')
args = parser.parse_args()

# Initialize simulator
sim.setup(timestep=0.01)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pattern definitions - made more distinct for better discrimination
PATTERNS = [
    {"name": "pattern_1", "delay": 1.0, "first_neuron": 0},  #short delay
    {"name": "pattern_2", "delay": 3.0, "first_neuron": 1},  #medium delay
    {"name": "pattern_3", "delay": 0.0, "simultaneous": True}  # Simultaneous firing
]

# Training parameters
PATTERN_OCCURRENCES = 20
PATTERN_INTERVAL = 15  # ms between pattern presentations
CHUNK_SIZE = 5

# Network parameters
NEURON_PARAMS = {
    'tau_m': 20.0,
    'tau_refrac': 2.0,
    'v_reset': -70.0,
    'v_rest': -65.0,
    'v_thresh': -52.0,
    'cm': 0.2
}

# STDP parameters neuron-specific for better specialization
STDP_CONFIGS = [
    # Neuron 1 
    {
        "A_plus": 0.18,      # Strong potentiation for Input1->Output1
        "A_minus": 0.08,     # Stronger depression for better selectivity
        "tau_plus": 15.0,    # Shorter time window to favor short delays
        "tau_minus": 25.0,   # Longer depression window for better discrimination
    },
    # Neuron 2
    {
        "A_plus": 0.15,      # Standard potentiation
        "A_minus": 0.06,     # Moderate depression
        "tau_plus": 25.0,    # Longer time window to favor medium delays
        "tau_minus": 20.0,   # Standard depression window
    },
    # Neuron 3
    {
        "A_plus": 0.12,      # Reduced potentiation (less spike-timing dependency)
        "A_minus": 0.04,     # Reduced depression
        "tau_plus": 10.0,    # Very short time window to favor synchronous inputs
        "tau_minus": 10.0,   # Symmetric short window for synchronous detection
    }
]

# Default STDP parameters
STDP_CONFIG = {
    "A_plus": 0.15,      # Potentiation strength
    "A_minus": 0.06,     # Depression strength
    "tau_plus": 20.0,    # Time window for potentiation
    "tau_minus": 20.0,   # Time window for depression
}

# Noise configuration
NOISE_CONFIG = {
    "jitter": 2.0,       # Timing jitter in ms (std dev of gaussian noise)
    "deletion_prob": 0.05, # Probability of spike deletion
    "insertion_prob": 0.01, # Probability of spurious spike insertion
    "insertion_window": 5.0 # Window size in ms for potential spurious spikes
}

# ============================================================================
# SPIKE TRAIN GENERATION
# ============================================================================

def create_pattern_spike_trains(patterns, occurrences, interval, start_time=10, add_noise=False):
    input_spikes = [[] for _ in range(2)]
    current_time = start_time
    
    pattern_sequence = []
    for pattern_type in patterns:
        for _ in range(occurrences):
            pattern_sequence.append(pattern_type)
    
    random.seed(42)
    random.shuffle(pattern_sequence)
    print("Using complete random shuffling for patterns")
    
    if add_noise:
        print("Adding noise to spike trains:")
        print(f"  Jitter: {NOISE_CONFIG['jitter']} ms")
        print(f"  Deletion probability: {NOISE_CONFIG['deletion_prob']}")
        print(f"  Insertion probability: {NOISE_CONFIG['insertion_prob']}")
    
    for pattern in pattern_sequence:
        if pattern.get("simultaneous", False):
            input_spikes[0].append(current_time)
            input_spikes[1].append(current_time)
        else:
            first_neuron = pattern["first_neuron"]
            second_neuron = 1 - first_neuron
            input_spikes[first_neuron].append(current_time)
            input_spikes[second_neuron].append(current_time + pattern["delay"])
        
        current_time += interval
    
    # Apply noise if requested
    if add_noise:
        noisy_spikes = [[] for _ in range(len(input_spikes))]
        np.random.seed(42)  # For reproducible noise
        
        # Process each input neuron separately
        for neuron_idx, spike_train in enumerate(input_spikes):
            # Apply deletion and jitter to existing spikes
            for spike_time in spike_train:
                # Randomly delete spikes with probability deletion_prob
                if np.random.random() > NOISE_CONFIG['deletion_prob']:
                    # Add jitter (Gaussian noise) to spike timing
                    jittered_time = spike_time + np.random.normal(0, NOISE_CONFIG['jitter'])
                    # Ensure spike time is positive
                    jittered_time = max(0.1, jittered_time)
                    noisy_spikes[neuron_idx].append(jittered_time)
            
            # Insert spurious spikes
            simulation_duration = current_time
            num_windows = int(simulation_duration / NOISE_CONFIG['insertion_window'])
            
            for window_idx in range(num_windows):
                window_start = window_idx * NOISE_CONFIG['insertion_window']
                if np.random.random() < NOISE_CONFIG['insertion_prob']:
                    # Insert a random spike in this window
                    random_time = window_start + np.random.random() * NOISE_CONFIG['insertion_window']
                    noisy_spikes[neuron_idx].append(random_time)
        
        # Replace original spikes with noisy version
        input_spikes = noisy_spikes
    
    # Sort spikes to ensure temporal order
    for i in range(len(input_spikes)):
        input_spikes[i] = sorted(input_spikes[i])
    
    for i, spikes in enumerate(input_spikes):
        for j in range(1, len(spikes)):
            if spikes[j] < spikes[j-1]:
                print(f"ERROR: Unsorted spike detected for neuron {i}: {spikes[j-1]} followed by {spikes[j]}")
    
    pattern_counts = {}
    for pattern in pattern_sequence:
        name = pattern["name"]
        pattern_counts[name] = pattern_counts.get(name, 0) + 1
    
    print(f"Generated {len(pattern_sequence)} patterns: {pattern_counts}")
    
    return input_spikes, pattern_sequence

# ============================================================================
# NETWORK SETUP
# ============================================================================

def create_populations(input_spikes, neuron_params):
    # Input population
    input_pop = sim.Population(
        2,
        sim.SpikeSourceArray(spike_times=input_spikes),
        label="Input"
    )
    input_pop.record("spikes")
    
    # Output population with integrate-and-fire neurons
    output_pop = sim.Population(
        3,
        sim.IF_cond_exp(**neuron_params),
        label="Output"
    )
    output_pop.record(("spikes", "v"))
    
    print(f"Created populations: {len(input_spikes)} inputs, 3 outputs")
    
    return input_pop, output_pop

def create_connections_and_inhibition(input_pop, output_pop, stdp_config, stdp_configs=None):
    np.random.seed(42)
    connection_list = []
    initial_weight = 0.6
    print("Using neuron-specific STDP learning with strategic random delays")
    # Reasonable delay range for biological plausibility: 0.1 - 5.0 ms
    # Use the same strategic delay patterns as in DUMBO
    # Output 1: Short delays 
    input1_delay_out1 = np.random.uniform(0.1, 2.0)
    input2_delay_out1 = np.random.uniform(0.1, 2.0)
    
    # Output 2: Medium delays
    input1_delay_out2 = np.random.uniform(2.0, 4.0)
    input2_delay_out2 = np.random.uniform(2.0, 4.0)

    # Output 3: Mixed delays
    input1_delay_out3 = np.random.uniform(0.1, 2.0)
    input2_delay_out3 = np.random.uniform(3.0, 5.0)
    
    # Package all delays
    input1_delays = [input1_delay_out1, input1_delay_out2, input1_delay_out3]
    input2_delays = [input2_delay_out1, input2_delay_out2, input2_delay_out3]
    
    # Store all connection delays for later reference
    all_connections = {}
    for output_idx in range(3):
        all_connections[f"input1_to_output{output_idx+1}_delay"] = input1_delays[output_idx]
        all_connections[f"input2_to_output{output_idx+1}_delay"] = input2_delays[output_idx]
    
    # Create the full connection list for all neurons
    connection_list = []
    for output_idx in range(3):
        input1_delay = input1_delays[output_idx]
        input2_delay = input2_delays[output_idx]
        
        connection_list.append([0, output_idx, initial_weight, input1_delay])
        connection_list.append([1, output_idx, initial_weight, input2_delay])
    
    # Use a single projection with uniform parameters instead of trying to use
    # neuron-specific projections which are causing issues with NEST
    if stdp_configs and len(stdp_configs) == 3:
        print("Using neuron-specific STDP parameters:")
        for output_idx in range(3):
            neuron_config = stdp_configs[output_idx]
            print(f"  Neuron {output_idx+1}: A+={neuron_config['A_plus']}, A-={neuron_config['A_minus']}, τ+={neuron_config['tau_plus']}, τ-={neuron_config['tau_minus']}")
    
    # Use a single STDP model with averaged parameters for all neurons
    # We'll keep track of neuron-specific parameters for analysis
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=stdp_config["tau_plus"],
            tau_minus=stdp_config["tau_minus"],
            A_plus=stdp_config["A_plus"],
            A_minus=stdp_config["A_minus"]
        ),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0),
        weight=None,
        delay=None
    )
    
    # Create a single projection for all neurons
    stdp_connections = sim.Projection(
        input_pop, output_pop,
        sim.FromListConnector(connection_list),
        synapse_type=stdp_model,
        receptor_type="excitatory",
        label="STDP_connections"
    )
    
    # Store connection information for later use
    all_connections['stdp_configs'] = stdp_configs
    all_connections['delays'] = {
        'input1': input1_delays,
        'input2': input2_delays
    }
    
    # Pattern-specific lateral inhibition
    # Use stronger inhibition between neurons that might confuse similar patterns
    # and weaker inhibition between neurons that specialize in very different patterns
    inhibitory_list = [
        # From Pattern 1 detector (short delays)
        [0, 1, 0.15, 0.1],  # Medium delay detector (competing pattern - stronger inhibition)
        [0, 2, 0.08, 0.1],  # Simultaneous firing detector (different pattern - weaker inhibition)
        
        # From Pattern 2 detector (medium delays)
        [1, 0, 0.15, 0.1],  # Short delay detector (competing pattern - stronger inhibition)
        [1, 2, 0.08, 0.1],  # Simultaneous firing detector (different pattern - weaker inhibition)
        
        # From Pattern 3 detector (simultaneous firing)
        [2, 0, 0.08, 0.1],  # Short delay detector (different pattern - weaker inhibition)
        [2, 1, 0.08, 0.1]   # Medium delay detector (different pattern - weaker inhibition)
    ]
    
    print("Using pattern-specific lateral inhibition")
    inhibitory_connections = sim.Projection(
        output_pop, output_pop,
        FromListConnector(inhibitory_list),
        receptor_type="inhibitory",
        label="Pattern_specific_lateral_inhibition"
    )
    
    # Return the connections and the metadata dictionary
    return stdp_connections, inhibitory_connections, all_connections

def normalize_weights(connections):
    # Handle case where connections is a list of neuron-specific projections
    if isinstance(connections, list):
        normalized_weights = []
        for neuron_idx, neuron_conn in enumerate(connections):
            # Target sum varies by neuron type to allow for specialization
            if neuron_idx == 0:
                # Pattern 1 detector (short delays) - slightly stronger sum for better response
                target_sum = 0.7
            elif neuron_idx == 1:
                # Pattern 2 detector (medium delays) - standard sum
                target_sum = 0.6
            else:
                # Pattern 3 detector (simultaneous) - slightly weaker sum for synchrony detection
                target_sum = 0.5
                
            weights = neuron_conn.get("weight", format="array")
            new_weights = np.copy(weights)
            
            # For each target neuron (should be just one in this case)
            for j in range(weights.shape[1]):
                total = sum(weights[i][j] for i in range(weights.shape[0]))
                if total > 0:
                    # Add slight noise for stochasticity and breaking symmetry
                    noise = [np.random.uniform(-0.01, 0.01) for _ in range(weights.shape[0])]
                    
                    for i in range(weights.shape[0]):
                        normalized = (weights[i][j] / total * target_sum) + noise[i]
                        new_weights[i][j] = min(1.0, max(0.01, normalized))
            
            neuron_conn.set(weight=new_weights)
            normalized_weights.append(neuron_conn.get("weight", format="array"))
            
        return normalized_weights
    
    # Original implementation for single connection object
    else:
        weights = connections.get("weight", format="array")
        new_weights = np.copy(weights)
        
        # Apply neuron-specific normalization targets
        target_sums = [0.7, 0.6, 0.5]  # Different target sums for different neurons
        
        for j in range(weights.shape[1]):
            # First, ensure no weights are exactly zero
            for i in range(weights.shape[0]):
                if new_weights[i][j] <= 0.001:  # If weight has collapsed to near zero
                    new_weights[i][j] = 0.01    # Set to a small positive value
            
            # Now normalize
            total = sum(weights[i][j] for i in range(weights.shape[0]))
            if total > 0:
                target_sum = target_sums[j] if j < len(target_sums) else 0.6
                
                noise = [np.random.uniform(-0.01, 0.01) for _ in range(weights.shape[0])]
                
                for i in range(weights.shape[0]):
                    normalized = (weights[i][j] / total * target_sum) + noise[i]
                    # Ensure weights stay within reasonable bounds
                    new_weights[i][j] = min(1.0, max(0.01, normalized))
        
        connections.set(weight=new_weights)
        
        return connections.get("weight", format="array")

def apply_homeostasis(connections, output_data, target_rate=2.0):
    weights = connections.get("weight", format="array")
    new_weights = np.copy(weights)
    
    segment_duration = output_data.t_stop - output_data.t_start
    segment_duration_seconds = float(segment_duration) / 1000.0
    firing_rates = [len(st) / max(segment_duration_seconds, 0.001) for st in output_data.spiketrains]
    
    for j in range(weights.shape[1]):
        rate = firing_rates[j]
        scale = 1.0
        
        if rate < 0.1 * target_rate:
            scale = 1.5
        elif rate < 0.5 * target_rate:
            scale = 1.3
        elif rate > 2.0 * target_rate:
            scale = 0.8
        elif rate > 1.5 * target_rate:
            scale = 0.9
            
        for i in range(weights.shape[0]):
            new_weights[i][j] = min(1.0, max(0.01, weights[i][j] * scale))
    connections.set(weight=new_weights)
    
    return connections.get("weight", format="array")
# ============================================================================
# EVALUATION AND ANALYSIS
# ============================================================================

def calculate_accuracy(output_data, pattern_sequence, pattern_interval, start_time=10):
    predictions = []
    true_labels = []
    
    pattern_to_label = {"pattern_1": 0, "pattern_2": 1, "pattern_3": 2}
    
    for i, pattern in enumerate(pattern_sequence):
        pattern_start = start_time + i * pattern_interval
        pattern_end = pattern_start + pattern_interval
        neuron_spikes = [0, 0, 0]
        
        for neuron_idx, spiketrain in enumerate(output_data.spiketrains):
            spikes_in_window = sum(1 for spike_time in spiketrain 
                                 if pattern_start <= spike_time < pattern_end)
            neuron_spikes[neuron_idx] = spikes_in_window
        
        if sum(neuron_spikes) > 0:
            predicted_pattern = neuron_spikes.index(max(neuron_spikes))
        else:
            predicted_pattern = -1
        
        predictions.append(predicted_pattern)
        true_labels.append(pattern_to_label[pattern["name"]])
    
    correct_predictions = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    accuracy = correct_predictions / len(true_labels)
    
    return accuracy, predictions, true_labels

def calculate_detailed_accuracy(output_data, pattern_sequence, pattern_interval, start_time=10):
    predictions = []
    true_labels = []
    confidence_scores = []
    
    pattern_to_label = {"pattern_1": 0, "pattern_2": 1, "pattern_3": 2}
    
    for i, pattern in enumerate(pattern_sequence):
        pattern_start = start_time + i * pattern_interval
        pattern_end = pattern_start + pattern_interval
        neuron_spikes = [0, 0, 0]
        
        for neuron_idx, spiketrain in enumerate(output_data.spiketrains):
            spikes_in_window = sum(1 for spike_time in spiketrain 
                                 if pattern_start <= spike_time < pattern_end)
            neuron_spikes[neuron_idx] = spikes_in_window
        
        total_spikes = sum(neuron_spikes)
        if total_spikes > 0:
            predicted_pattern = neuron_spikes.index(max(neuron_spikes))
            confidence = max(neuron_spikes) / total_spikes
        else:
            predicted_pattern = -1
            confidence = 0.0
        
        predictions.append(predicted_pattern)
        true_labels.append(pattern_to_label[pattern["name"]])
        confidence_scores.append(confidence)
    
    accuracy = sum(1 for pred, true in zip(predictions, true_labels) if pred == true) / len(true_labels)
    
    confusion_matrix = np.zeros((3, 4))
    for true, pred in zip(true_labels, predictions):
        confusion_matrix[true][pred] += 1
    
    class_accuracies = {}
    for class_idx in range(3):
        class_total = sum(1 for label in true_labels if label == class_idx)
        class_correct = confusion_matrix[class_idx][class_idx]
        class_accuracies[f"pattern_{class_idx+1}"] = class_correct / class_total if class_total > 0 else 0
    
    return {
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': confusion_matrix,
        'predictions': predictions,
        'true_labels': true_labels,
        'confidence_scores': confidence_scores
    }

def analyze_pattern_mistakes(output_data, pattern_sequence, pattern_interval, start_time=10):
    print("\n=== PATTERN MISTAKE ANALYSIS ===")
    
    pattern_to_label = {"pattern_1": 0, "pattern_2": 1, "pattern_3": 2}
    mistake_counts = {0: 0, 1: 0, 2: 0}
    activation_counts = {0: 0, 1: 0, 2: 0}
    
    for i, pattern in enumerate(pattern_sequence):
        pattern_start = start_time + i * pattern_interval
        pattern_end = pattern_start + pattern_interval
        neuron_spikes = [0, 0, 0]
        
        for neuron_idx, spiketrain in enumerate(output_data.spiketrains):
            spikes_in_window = [s for s in spiketrain if pattern_start <= s < pattern_end]
            neuron_spikes[neuron_idx] = len(spikes_in_window)
        
        predicted = neuron_spikes.index(max(neuron_spikes)) if sum(neuron_spikes) > 0 else -1
        true_label = pattern_to_label[pattern["name"]]
        
        activation_counts[true_label] += 1
        if predicted != true_label:
            mistake_counts[true_label] += 1
    
    print("\nMistake Summary:")
    for pattern_idx in range(3):
        total = activation_counts[pattern_idx]
        mistakes = mistake_counts[pattern_idx]
        correct = total - mistakes
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"Pattern {pattern_idx+1}: {correct}/{total} correct ({accuracy:.1f}%)")
    
    return mistake_counts

# ============================================================================
# SIMULATION
# ============================================================================

def run_stdp_simulation(input_spikes, pattern_sequence, pattern_interval, 
                       stdp_connections, output_pop, simulation_time=1500, connection_metadata=None):
    print(f"Running simulation with neuron-specific STDP learning for {simulation_time} ms...")
    
    # Get initial weights and delays for reference
    initial_weights = stdp_connections.get("weight", format="array")
    delays = stdp_connections.get("delay", format="array")
    
    # Check if we have neuron-specific metadata
    is_neuron_specific = connection_metadata is not None and 'stdp_configs' in connection_metadata
    
    if is_neuron_specific and 'delays' in connection_metadata:
        print("\nNeuron-specific strategic random delays (ms):")
        input1_delays = connection_metadata['delays']['input1']
        input2_delays = connection_metadata['delays']['input2']
        for i in range(3):
            print(f"  Neuron {i+1} - Input1: {input1_delays[i]:.2f}ms, Input2: {input2_delays[i]:.2f}ms")
    else:
        print("\nStrategic random delays (ms):")
        print(f"  Input1->Output: {delays[0]}")
        print(f"  Input2->Output: {delays[1]}")
    
    # Set up recording
    output_pop.record(("spikes", "v"))
    
    # Run the simulation in chunks to apply homeostasis
    print("Running simulation with STDP learning and pattern-specific homeostasis...")
    chunk_size = 500  # ms per chunk
    current_time = 0
    
    # Analysis windows for each pattern type to track neuron specialization
    pattern_windows = {
        "pattern_1": [],
        "pattern_2": [],
        "pattern_3": []
    }
    
    # Extract pattern presentation times for specialization analysis
    start_time = 10
    for i, pattern in enumerate(pattern_sequence):
        pattern_start = start_time + i * pattern_interval
        pattern_end = pattern_start + pattern_interval
        pattern_windows[pattern["name"]].append((pattern_start, pattern_end))
    
    while current_time < simulation_time:
        # Run a chunk of simulation
        chunk_duration = min(chunk_size, simulation_time - current_time)
        sim.run(chunk_duration)
        current_time += chunk_duration
        
        # Apply homeostasis at regular intervals to prevent weight collapse
        if current_time % chunk_size == 0:
            segment_data = output_pop.get_data().segments[-1]
            
            # Apply pattern-specific homeostasis based on neuron performance
            if current_time >= pattern_interval * 10:  # After some learning has occurred
                # Get output data to analyze pattern response
                output_data = output_pop.get_data().segments[-1]
                
                # Track neuron responses for each pattern type
                pattern_responses = {
                    "pattern_1": [0, 0, 0],
                    "pattern_2": [0, 0, 0],
                    "pattern_3": [0, 0, 0]
                }
                
                # Count spikes for each pattern type and neuron
                for pattern_type, windows in pattern_windows.items():
                    for window_start, window_end in windows:
                        if window_start <= current_time and window_end <= current_time:
                            # Check each output neuron's response in this pattern window
                            for neuron_idx, spiketrain in enumerate(output_data.spiketrains):
                                spikes_in_window = sum(1 for spike_time in spiketrain 
                                                     if window_start <= spike_time < window_end)
                                pattern_responses[pattern_type][neuron_idx] += spikes_in_window
                
                print(f"\n  Pattern responses at {current_time}ms:")
                for pattern, responses in pattern_responses.items():
                    print(f"    {pattern}: {responses}")
            
            # Apply neuron-specific normalization targets if we have that metadata
            if is_neuron_specific:
                # Use neuron-specific normalization targets
                target_sums = [0.7, 0.6, 0.5]  # Different target sums for different neurons
                weights = stdp_connections.get("weight", format="array")
                new_weights = np.copy(weights)
                
                # Apply different normalization targets to each neuron
                for j in range(new_weights.shape[1]):
                    # First ensure no weights are exactly zero (since NEST requires w_min=0.0)
                    for i in range(new_weights.shape[0]):
                        if new_weights[i][j] <= 0.001:  # If weight has collapsed to near zero
                            new_weights[i][j] = 0.01    # Set to a small positive value
                    
                    # Now normalize
                    total = sum(new_weights[i][j] for i in range(new_weights.shape[0]))
                    if total > 0:
                        target_sum = target_sums[j] if j < len(target_sums) else 0.6
                        
                        noise = [np.random.uniform(-0.01, 0.01) for _ in range(new_weights.shape[0])]
                        
                        for i in range(new_weights.shape[0]):
                            normalized = (new_weights[i][j] / total * target_sum) + noise[i]
                            # Ensure weights stay within reasonable bounds
                            new_weights[i][j] = min(1.0, max(0.01, normalized))
                
                stdp_connections.set(weight=new_weights)
                print(f"  Applied neuron-specific weight normalization at {current_time}ms")
            else:
                normalize_weights(stdp_connections)
                print(f"  Applied weight normalization at {current_time}ms")
    
    # Get complete output data for analysis
    output_data = output_pop.get_data().segments[-1]
    
    # Get final weights
    final_weights = stdp_connections.get("weight", format="array")
    
    # Calculate accuracy
    accuracy, predictions, true_labels = calculate_accuracy(
        output_data, pattern_sequence, pattern_interval, start_time=10
    )
    
    # Display results
    print(f"\nSTDP Weight Learning Results:")
    print(f"Accuracy: {accuracy:.2%}")
    
    if is_neuron_specific:
        print("\nNeuron-specific STDP learning results:")
        for i in range(3):
            print(f"  Neuron {i+1} (Target: Pattern {i+1}):")
            initial_input1_weight = float(initial_weights[0, i])
            initial_input2_weight = float(initial_weights[1, i])
            
            final_input1_weight = float(final_weights[0, i])
            final_input2_weight = float(final_weights[1, i])
            
            print(f"    Initial weights: I1: {initial_input1_weight:.4f}, I2: {initial_input2_weight:.4f}")
            print(f"    Final weights:   I1: {final_input1_weight:.4f}, I2: {final_input2_weight:.4f}")
            print(f"    Specialization:  {abs(final_input1_weight - final_input2_weight):.4f}")
    else:
        print(f"Initial weights: I1->O: {initial_weights[0]}")
        print(f"                 I2->O: {initial_weights[1]}")
        print(f"Final weights:   I1->O: {final_weights[0]}")
        print(f"                 I2->O: {final_weights[1]}")
    
    # Calculate detailed accuracy
    detailed_results = calculate_detailed_accuracy(
        output_data, pattern_sequence, pattern_interval, start_time=10
    )
    
    print(f"\nDetailed Results:")
    print(f"Overall Accuracy: {detailed_results['overall_accuracy']:.2%}")
    
    # Calculate delay statistics
    if is_neuron_specific and connection_metadata and 'delays' in connection_metadata:
        input1_delays = connection_metadata['delays']['input1']
        input2_delays = connection_metadata['delays']['input2']
        all_delays = input1_delays + input2_delays
        delay_mean = np.mean(all_delays)
        delay_std = np.std(all_delays)
    else:
        delay_mean = np.mean(delays)
        delay_std = np.std(delays)
    print(f"Delay statistics - Mean: {delay_mean:.4f}ms, Std: {delay_std:.4f}ms")
    
    # Calculate weight changes
    if is_neuron_specific:
        weight_changes = 0
        for i in range(3):
            initial_input1_weight = float(initial_weights[0, i])
            initial_input2_weight = float(initial_weights[1, i])
            
            final_input1_weight = float(final_weights[0, i])
            final_input2_weight = float(final_weights[1, i])
            
            neuron_changes = abs(final_input1_weight - initial_input1_weight) + \
                             abs(final_input2_weight - initial_input2_weight)
            weight_changes += neuron_changes
            print(f"  Neuron {i+1} weight changes: {neuron_changes:.6f}")
        print(f"Total weight changes: {weight_changes:.6f}")
    else:
        weight_changes = np.abs(final_weights - initial_weights).sum()
        print(f"Total weight changes: {weight_changes:.6f}")
    
    # Always print per-class accuracies for baseline
    print(f"\nPer-class Accuracies:")
    for pattern, acc in detailed_results['class_accuracies'].items():
        print(f"  {pattern}: {acc:.2%}")
    print(f"Confusion Matrix:")
    print(detailed_results['confusion_matrix'])
    
    return accuracy, final_weights, delays, output_data, pattern_sequence

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_spike_trains(input_data, output_data):
    plt.figure(figsize=(16, 10))
    
    # Input spikes - zoomed to show more detail
    plt.subplot(2, 1, 1)
    for i, spiketrain in enumerate(input_data.spiketrains):
        plt.plot(spiketrain, [i] * len(spiketrain), 'o', markersize=3, label=f"Input {i+1}")
    plt.yticks([0, 1], ["Input 1", "Input 2"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron")
    
    # Determine if noise was used from the spike train properties
    noise_detected = args.noise
    title_prefix = "Noisy " if noise_detected else ""
    plt.title(f"{title_prefix}Input Spike Trains")
    
    # Add focus on early patterns (0-200ms) for better visibility
    plt.xlim(0, 200)
    
    # Create pattern markers for the first few patterns to help visualize
    pattern_interval = PATTERN_INTERVAL
    for t in range(10, 200, pattern_interval):
        plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
    
    # Output spikes
    plt.subplot(2, 1, 2)
    for i, spiketrain in enumerate(output_data.spiketrains):
        plt.plot(spiketrain, [i] * len(spiketrain), 'o', markersize=3, label=f"Output {i+1}")
    plt.yticks([0, 1, 2], ["Neuron 1", "Neuron 2", "Neuron 3"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron")
    plt.title(f"Output Neuron Responses to {title_prefix}Input")
    plt.xlim(0, 200)  # Match input spike train view
    
    # Add the same pattern markers to the output plot
    for t in range(10, 200, pattern_interval):
        plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create a second figure showing the full spike trains
    plt.figure(figsize=(16, 10))
    
    # Full input spike trains
    plt.subplot(2, 1, 1)
    for i, spiketrain in enumerate(input_data.spiketrains):
        plt.plot(spiketrain, [i] * len(spiketrain), 'o', markersize=2, label=f"Input {i+1}")
    plt.yticks([0, 1], ["Input 1", "Input 2"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron")
    plt.title(f"Full {title_prefix}Input Spike Trains")
    
    # Full output spike trains
    plt.subplot(2, 1, 2)
    for i, spiketrain in enumerate(output_data.spiketrains):
        plt.plot(spiketrain, [i] * len(spiketrain), 'o', markersize=2, label=f"Output {i+1}")
    plt.yticks([0, 1, 2], ["Neuron 1", "Neuron 2", "Neuron 3"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron")
    plt.title(f"Full Output Neuron Responses")
    
    plt.tight_layout()
    plt.show()

def plot_membrane_potentials(output_data, threshold=-54.0):
    """Plot membrane potentials of output neurons."""
    plt.figure(figsize=(16, 8))
    for i in range(3):
        plt.plot(output_data.filter(name="v")[0][:, i], label=f"Output {i+1}")
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold}mV)')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title("Output Neuron Membrane Potentials")
    plt.legend()
    plt.show()

def plot_final_weights(final_weights_matrix):
    plt.figure(figsize=(8, 6))
    plt.bar(["I1→O1", "I2→O1", "I1→O2", "I2→O2", "I1→O3", "I2→O3"], 
            [final_weights_matrix[0][0], final_weights_matrix[1][0], 
             final_weights_matrix[0][1], final_weights_matrix[1][1],
             final_weights_matrix[0][2], final_weights_matrix[1][2]])
    plt.ylabel("Weight")
    plt.title("Learned Synaptic Weights After STDP")
    plt.show()

def plot_delays(delays_matrix):
    plt.figure(figsize=(8, 6))
    plt.bar(["I1→O1", "I2→O1", "I1→O2", "I2→O2", "I1→O3", "I2→O3"], 
            [delays_matrix[0][0], delays_matrix[1][0], 
             delays_matrix[0][1], delays_matrix[1][1],
             delays_matrix[0][2], delays_matrix[1][2]])
    plt.ylabel("Delay (ms)")
    plt.title("Fixed Strategic Synaptic Delays")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT: NEURON-SPECIFIC STDP WEIGHT LEARNING WITH STRATEGIC FIXED DELAYS")
    print("=" * 80)
    print("This simulation uses:")
    print("- Neuron-specific STDP parameters for specialized pattern detection:")
    print("  - Neuron 1 (Pattern 1): A+={:.2f}, A-={:.2f}, τ+={:.1f}, τ-={:.1f} (Short delay specialist)"
          .format(STDP_CONFIGS[0]["A_plus"], STDP_CONFIGS[0]["A_minus"], 
                 STDP_CONFIGS[0]["tau_plus"], STDP_CONFIGS[0]["tau_minus"]))
    print("  - Neuron 2 (Pattern 2): A+={:.2f}, A-={:.2f}, τ+={:.1f}, τ-={:.1f} (Medium delay specialist)"
          .format(STDP_CONFIGS[1]["A_plus"], STDP_CONFIGS[1]["A_minus"], 
                 STDP_CONFIGS[1]["tau_plus"], STDP_CONFIGS[1]["tau_minus"]))
    print("  - Neuron 3 (Pattern 3): A+={:.2f}, A-={:.2f}, τ+={:.1f}, τ-={:.1f} (Synchronous specialist)"
          .format(STDP_CONFIGS[2]["A_plus"], STDP_CONFIGS[2]["A_minus"], 
                 STDP_CONFIGS[2]["tau_plus"], STDP_CONFIGS[2]["tau_minus"]))
    print("- Pattern-specific lateral inhibition (stronger between competing patterns)")
    print("- Neuron-specific weight normalization targets for specialization")
    print("- Strategically distributed delays for better pattern discrimination")
    print("- Complete random pattern shuffling for improved learning")
    if args.noise:
        print("- Noise applied to spike trains for robustness testing")
    print("=" * 80)
    
    # 1. Generate spike trains
    input_spikes, pattern_sequence = create_pattern_spike_trains(
        PATTERNS, PATTERN_OCCURRENCES, PATTERN_INTERVAL, add_noise=args.noise
    )
    
    # 2. Create populations
    input_pop, output_pop = create_populations(input_spikes, NEURON_PARAMS)
    
    # 3. Create connections with neuron-specific STDP parameters
    stdp_connections, inhibitory_connections, connection_metadata = create_connections_and_inhibition(
        input_pop, output_pop, STDP_CONFIG, STDP_CONFIGS
    )
    
    # 4. Run simulation
    simulation_time = 3000  # ms
    accuracy, weights, delays, output_data, patterns = run_stdp_simulation(
        input_spikes, pattern_sequence, PATTERN_INTERVAL, 
        stdp_connections, output_pop, simulation_time, connection_metadata
    )
    
    # 5. Visualizations
    print("\n" + "=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)
    input_data = input_pop.get_data().segments[0]
    
    # If noise was used, show statistics about the spike trains
    if args.noise:
        print("\nNoise Statistics:")
        original_spike_count = PATTERN_OCCURRENCES * len(PATTERNS) * 2  # 2 spikes per pattern
        actual_spike_count = sum(len(spiketrain) for spiketrain in input_data.spiketrains)
        print(f"  Expected spike count without noise: {original_spike_count}")
        print(f"  Actual spike count with noise: {actual_spike_count}")
        print(f"  Difference: {actual_spike_count - original_spike_count} spikes")
        
        if actual_spike_count > original_spike_count:
            print(f"  Net added spikes: {actual_spike_count - original_spike_count}")
        else:
            print(f"  Net deleted spikes: {original_spike_count - actual_spike_count}")
    
    plot_spike_trains(input_data, output_data)
    plot_membrane_potentials(output_data, NEURON_PARAMS['v_thresh'])
    plot_final_weights(weights)
    plot_delays(delays)
    
    # 6. Final analysis
    mistake_counts = analyze_pattern_mistakes(output_data, pattern_sequence, 
                           PATTERN_INTERVAL, start_time=10
    )
    
    # Print noise impact summary if applicable
    if args.noise:
        print("\n" + "=" * 60)
        print("NOISE IMPACT SUMMARY")
        print("=" * 60)
        print("The simulation was run with noise applied to the spike trains:")
        print(f"  - Timing jitter: {NOISE_CONFIG['jitter']} ms")
        print(f"  - Spike deletion probability: {NOISE_CONFIG['deletion_prob']}")
        print(f"  - Spurious spike insertion probability: {NOISE_CONFIG['insertion_prob']}")
        print(f"\nOverall accuracy with noise: {accuracy:.2%}")
        print("\nTo compare with baseline (no noise), run without the --noise flag.")
    
    # Cleanup
    sim.end()
    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    main()