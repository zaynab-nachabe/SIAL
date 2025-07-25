import pyNN.nest as sim
from quantities import ms
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from pyNN.connectors import FromListConnector

sim.setup(timestep=0.01)

# ============================================================================
# CONFIGURATION
# ============================================================================

PATTERNS = [
    {"name": "pattern_1", "delay": 1.0, "first_neuron": 0},  # Short delay
    {"name": "pattern_2", "delay": 3.0, "first_neuron": 1},  # Medium delay
    {"name": "pattern_3", "delay": 0.0, "simultaneous": True}  #simultaneous firing
]

PATTERN_OCCURRENCES = 30
PATTERN_INTERVAL = 30 #ms between pattern presentations
CHUNK_SIZE = 5

NOISE_CONFIG = {
    "enabled": False,
    "jitter_std": 1.0,       # Standard deviation of temporal jitter (ms)
    "missing_prob": 0.0,     # Probability of a spike being deleted
    "extra_prob": 0.0,       # Probability of adding an extra spike
    "extra_sigma": 5.0       # Temporal spread of extra spikes (ms)
}

NEURON_PARAMS = {
    'tau_m': 10.0,
    'tau_refrac': 1.0,
    'v_reset': -70.0,
    'v_rest': -65.0,
    'v_thresh': -55.0,
    'cm': 0.2,
    'tau_syn_E': 2.0,
    'tau_syn_I': 5.0
}

STDP_CONFIG = {
    "A_plus": 0.2,
    "A_minus": 0.1,
    "Aplus": 0.2,
    "Aminus": 0.1,
    "tau_plus": 15.0,
    "tau_minus": 30.0,
}

# ============================================================================
# SPIKE TRAIN GENERATION
# ============================================================================

def create_pattern_spike_trains(patterns, occurrences, interval, start_time=10, noise_config=None):
    """
    Generate spike trains for input neurons based on pattern definitions.
    
    Parameters:
    - patterns: List of pattern dictionaries
    - occurrences: Number of times each pattern should occur
    - interval: Time between pattern presentations (ms)
    - start_time: Starting time for the first pattern (ms)
    - noise_config: Dictionary with noise parameters (jitter_std, missing_prob, etc.)
    
    Returns:
    - input_spikes: List of spike times for each input neuron
    - pattern_sequence: Sequence of patterns presented
    """
    input_spikes = [[] for _ in range(2)]
    current_time = start_time
    
    # Use global noise config if none provided
    if noise_config is None:
        noise_config = NOISE_CONFIG
    
    pattern_sequence = []
    for pattern_type in patterns:
        for _ in range(occurrences):
            pattern_sequence.append(pattern_type)
    
    random.seed(42)
    for i in range(0, len(pattern_sequence), CHUNK_SIZE):
        chunk = pattern_sequence[i:i+CHUNK_SIZE]
        random.shuffle(chunk)
        pattern_sequence[i:i+CHUNK_SIZE] = chunk
    
    for pattern in pattern_sequence:
        if pattern.get("simultaneous", False):
            time1 = float(current_time)
            time2 = float(current_time)
            
            input_spikes[0].append(time1)
            input_spikes[1].append(time2)
        else:
            first_neuron = pattern["first_neuron"]
            second_neuron = 1 - first_neuron
            time1 = float(current_time)
            time2 = float(current_time + pattern["delay"])
            input_spikes[first_neuron].append(time1)
            input_spikes[second_neuron].append(time2)
        
        current_time += interval
    
    # Apply noise if enabled
    if noise_config.get("enabled", False):
        jitter_std = noise_config.get("jitter_std", 0.0)
        if jitter_std > 0:
            for neuron_idx in range(len(input_spikes)):
                for i in range(len(input_spikes[neuron_idx])):
                    jitter = np.random.normal(0, jitter_std)
                    input_spikes[neuron_idx][i] += jitter
        missing_prob = noise_config.get("missing_prob", 0.0)
        if missing_prob > 0:
            for neuron_idx in range(len(input_spikes)):
                keep_mask = np.random.random(len(input_spikes[neuron_idx])) > missing_prob
                input_spikes[neuron_idx] = [spike for i, spike in enumerate(input_spikes[neuron_idx]) if keep_mask[i]]
        
        extra_prob = noise_config.get("extra_prob", 0.0)
        extra_sigma = noise_config.get("extra_sigma", 5.0)
        if extra_prob > 0:
            n_patterns = len(pattern_sequence)
            for neuron_idx in range(len(input_spikes)):
                for i in range(n_patterns):
                    if np.random.random() < extra_prob:
                        pattern_time = start_time + i * interval
                        extra_time = pattern_time + np.random.normal(interval/2, extra_sigma)
                        if extra_time > 0:
                            input_spikes[neuron_idx].append(float(extra_time))
        
        print("Noise applied to spike trains:")
        print("  Jitter: {}ms std dev".format(jitter_std))
        print("  Missing spikes: {:.1%} probability".format(missing_prob))
        print("  Extra spikes: {:.1%} probability".format(extra_prob))
    

    for i in range(len(input_spikes)):
        input_spikes[i] = sorted(input_spikes[i])
    
    for i, spikes in enumerate(input_spikes):
        for j in range(1, len(spikes)):
            if spikes[j] < spikes[j-1]:
                print("ERROR: Unsorted spike detected for neuron {}: {} followed by {}".format(i, spikes[j-1], spikes[j]))
    
    pattern_counts = {}
    for pattern in pattern_sequence:
        name = pattern["name"]
        pattern_counts[name] = pattern_counts.get(name, 0) + 1
    
    print("Generated {} patterns: {}".format(len(pattern_sequence), pattern_counts))
    
    return input_spikes, pattern_sequence

# ============================================================================
# NETWORK SETUP
# ============================================================================

def create_populations(input_spikes, neuron_params):
    print("\nDEBUG: Creating input population with spike times:")
    for i, spikes in enumerate(input_spikes):
        print("  Input {}: {} spikes, first 3: {}".format(i, len(spikes), spikes[:3] if spikes else 'None'))
        
    print("\nDEBUG: Pattern timings (first 5 patterns):")
    pattern_times = []
    for i in range(0, min(5, len(input_spikes[0]))):
        pattern_times.append(input_spikes[0][i])
    print(f"  Pattern times: {pattern_times}")
    
    input_pop = sim.Population(
        2,
        sim.SpikeSourceArray(spike_times=input_spikes),
        label="Input"
    )
    input_pop.record("spikes")
    
    output_pop = sim.Population(
        3,
        sim.IF_cond_exp(**neuron_params),
        label="Output"
    )
    output_pop.record(("spikes", "v"))
    
    print("Created populations: {} inputs, 3 outputs".format(len(input_spikes)))
    
    return input_pop, output_pop

def create_connections_and_inhibition(input_pop, output_pop, stdp_config, args=None):
    np.random.seed(42)
    connection_list = []
    
    # Use specified weight value if fixed weights, otherwise use base weight with variations
    if args and args.fixed_weights:
        base_weight = args.weight_value
        weight_variations = [[0, 0], [0, 0], [0, 0]]
    else:
        base_weight = 2.0
        # Small random variations to break symmetry
        weight_variations = [
            [0.2, -0.1],  # Input1 stronger for Output1
            [-0.1, 0.2],  # Input2 stronger for Output2
            [0.1, 0.1]    # Balanced for Output3
        ]
    
    # Reasonable delay range for biological plausibility: 0.1 - 5.0 ms
    min_delay = 0.1
    max_delay = 5.0
    
    # Generate random delays
    input1_delay_out1 = np.random.uniform(min_delay, max_delay)
    input2_delay_out1 = np.random.uniform(min_delay, max_delay)
    
    input1_delay_out2 = np.random.uniform(min_delay, max_delay)
    input2_delay_out2 = np.random.uniform(min_delay, max_delay)
    
    input1_delay_out3 = np.random.uniform(min_delay, max_delay)
    input2_delay_out3 = np.random.uniform(min_delay, max_delay)
    
    # Package all delays
    input1_delays = [input1_delay_out1, input1_delay_out2, input1_delay_out3]
    input2_delays = [input2_delay_out1, input2_delay_out2, input2_delay_out3]
    
    for output_idx in range(3):
        input1_delay = input1_delays[output_idx]
        input2_delay = input2_delays[output_idx]
        
        input1_weight = base_weight + weight_variations[output_idx][0]
        input2_weight = base_weight + weight_variations[output_idx][1]
        
        print(f"  Output {output_idx+1}: Input1 delay={input1_delay:.2f}ms, Input2 delay={input2_delay:.2f}ms")
        print(f"  Output {output_idx+1}: Input1 weight={input1_weight:.2f}, Input2 weight={input2_weight:.2f}")
        
        connection_list.append((0, output_idx, input1_weight, input1_delay))
        connection_list.append((1, output_idx, input2_weight, input2_delay))
    
    # Create appropriate synapse type based on args
    if args and args.fixed_weights:
        synapse_type = sim.StaticSynapse()
        print(f"Using static synapses with fixed weight value: {args.weight_value}")
    else:
        # Use STDP for weight learning
        try:
            synapse_type = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(
                    tau_plus=stdp_config["tau_plus"],
                    tau_minus=stdp_config["tau_minus"],
                    A_plus=stdp_config["A_plus"],
                    A_minus=stdp_config["A_minus"]
                ),
                weight_dependence=sim.AdditiveWeightDependence(
                    w_min=0.0,
                    w_max=5.0
                )
            )
            print("Using STDP with: A_plus={}, A_minus={}, tau_plus={}, tau_minus={}".format(
                stdp_config["A_plus"], stdp_config["A_minus"], 
                stdp_config["tau_plus"], stdp_config["tau_minus"]
            ))
        except Exception as e:
            print("Error with standard STDP configuration: {}".format(e))
    
    # Create main connections
    connections_label = "Static_connections" if (args and args.fixed_weights) else "STDP_connections"
    stdp_connections = sim.Projection(
        input_pop, output_pop,
        sim.FromListConnector(connection_list),
        synapse_type=synapse_type,
        receptor_type="excitatory",
        label=connections_label
    )
    
    # Lateral inhibition
    inhibition_weight = 0.15
    inhibitory_list = [
        (0, 1, inhibition_weight, 0.1), (0, 2, inhibition_weight, 0.1),
        (1, 0, inhibition_weight, 0.1), (1, 2, inhibition_weight, 0.1),
        (2, 0, inhibition_weight, 0.1), (2, 1, inhibition_weight, 0.1)
    ]
    
    inhibitory_connections = sim.Projection(
        output_pop, output_pop,
        sim.FromListConnector(inhibitory_list),
        synapse_type=sim.StaticSynapse(),
        receptor_type="inhibitory",
        label="Lateral_inhibition"
    )
    
    return stdp_connections, inhibitory_connections

#not used
def normalize_weights(connections):
    weights = connections.get("weight", format="array")
    new_weights = np.copy(weights)
    
    for j in range(weights.shape[1]):
        total = sum(weights[i][j] for i in range(weights.shape[0]))
        if total > 0:
            target_sum = 0.6
            noise = [np.random.uniform(-0.01, 0.01) for _ in range(weights.shape[0])]
            
            for i in range(weights.shape[0]):
                normalized = (weights[i][j] / total * target_sum) + noise[i]
                new_weights[i][j] = min(1.0, max(0.01, normalized))
    
    connections.set(weight=new_weights)
    
    return connections.get("weight", format="array")
#not used
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
from delay_learning import apply_delay_learning
from visualize_spike_timing import plot_spike_timing_analysis

def run_simulation_with_delay_learning(input_spikes, pattern_sequence, pattern_interval,
                                      connections, input_pop, output_pop, simulation_time=1500,
                                      chunk_size=100, learning_enabled=True,
                                      B_plus=0.1, B_minus=0.1, sigma_plus=10.0, sigma_minus=10.0,
                                      weight_scale=1.0):
    """
    Run a simulation with delay learning in chunks to allow for updating delays between chunks.
    
    Parameters:
    - input_pop: The input population object (needed for recording spikes)
    - output_pop: The output population object
    - learning_enabled: If True, apply delay learning. If False, just run baseline.
    - B_plus, B_minus: Parameters controlling the magnitude of delay changes
    - sigma_plus, sigma_minus: Time constants for the delay learning rule
    """
    if learning_enabled:
        print(f"Running simulation with delay learning for {simulation_time} ms...")
    else:
        print(f"Running baseline simulation with fixed weights for {simulation_time} ms...")
    
    # Debug: Print when patterns should occur
    print("\nDEBUG: Expected pattern times within simulation duration:")
    pattern_times = []
    for i, pattern in enumerate(pattern_sequence):
        time = 10 + i * pattern_interval
        if time < simulation_time:
            pattern_times.append((time, pattern["name"]))
    
    for time, pattern in pattern_times[:10]:  # Show first 10 patterns
        print(f"  {pattern} at {time} ms")
    
    # Extract weights and delays from connection objects
    try:
        # Get the initial weights and delays directly from the connection object
        weights = connections.get("weight", format="array")
        initial_delays = connections.get("delay", format="array")
        
    except Exception as e:
        print(f"ERROR extracting weights/delays: {e}")
        # Fallback values if we can't read from the connection
        if isinstance(connections, list) and len(connections) > 0:
            # Try to access the first connection in the list
            try:
                weights = connections[0].get("weight", format="array")
                initial_delays = connections[0].get("delay", format="array")
            except:
                weights = [[0.6, 0.6, 0.6], [0.6, 0.6, 0.6]]
                initial_delays = [[1.0, 3.0, 0.5], [2.0, 3.0, 3.0]]
        else:
            weights = [[0.6, 0.6, 0.6], [0.6, 0.6, 0.6]]
            initial_delays = [[1.0, 3.0, 0.5], [2.0, 3.0, 3.0]]
    
    print("Fixed weights:")
    print(f"  Input1->Output: {weights[0]}")
    print(f"  Input2->Output: {weights[1]}")
    
    print("\nInitial delays (ms):")
    print(f"  Input1->Output: {initial_delays[0]}")
    print(f"  Input2->Output: {initial_delays[1]}")
    
    # Set up recording for both input and output populations
    output_pop.record(("spikes", "v"))
    input_pop.record("spikes")
    
    if learning_enabled:
        # Run the simulation in chunks to allow for delay updates
        current_time = 0
        
        # For tracking spike data across chunks
        all_input_data = None
        all_output_data = None
        
        while current_time < simulation_time:
            # Reset recording before each chunk
            output_pop.record(None)
            output_pop.record(("spikes", "v"))
            
            # Reset recording for input neurons
            input_pop.record(None)
            input_pop.record("spikes")
            
            # Run a chunk of simulation
            chunk_duration = min(chunk_size, simulation_time - current_time)
            print(f"Running chunk from {current_time} to {current_time + chunk_duration} ms...")
            sim.run(chunk_duration)
            current_time += chunk_duration
            
            # Get the output spikes for this chunk only
            output_data = output_pop.get_data().segments[-1]
            
            # Instead of relying on the recording, use the original input spike times
            # that fall within the current chunk's time window
            input_spike_times = []
            for neuron_idx, spike_train in enumerate(input_spikes):
                # Filter spikes that fall within this chunk's time window
                chunk_spikes = [t for t in spike_train 
                               if current_time - chunk_duration <= t < current_time]
                input_spike_times.append(chunk_spikes)
                print(f"Input neuron {neuron_idx} has {len(chunk_spikes)} spikes in this chunk")
                if chunk_spikes:
                    print(f"  Spike times: {chunk_spikes}")
            
            # Also get recorded spikes (for debugging)
            input_data = input_pop.get_data().segments[-1]
            print(f"Recorded input spikes: {input_data.spiketrains}")
            
            # Process output spike trains directly from the recorded data
            output_spike_times = []
            print(f"Raw output spike train data: {output_data.spiketrains}")
            print(f"Number of output spiketrains: {len(output_data.spiketrains)}")
            
            for i, st in enumerate(output_data.spiketrains):
                try:
                    spike_list = [float(t) for t in st]
                    print(f"Output neuron {i} has {len(spike_list)} spikes")
                    if spike_list:
                        print(f"  Spike times: {spike_list}")
                    output_spike_times.append(spike_list)
                except Exception as e:
                    print(f"Error converting output spike train {i}: {e}")
                    output_spike_times.append([])
            
            print(f"Processed {sum(len(st) for st in input_spike_times)} input spikes")
            print(f"Processed {sum(len(st) for st in output_spike_times)} output spikes")
            
            # Visualize spike timing for this chunk (before learning)
            current_delays = connections.get("delay", format="array")
            
            # Only visualize if we have actual spikes
            if sum(len(st) for st in input_spike_times) > 0 and sum(len(st) for st in output_spike_times) > 0:
                print("\nVisualizing spike timing for this chunk:")
                try:
                    plot_spike_timing_analysis(
                        input_spike_times, output_spike_times, current_delays, 
                        chunk_start_time=current_time-chunk_duration,
                        chunk_end_time=current_time
                    )
                except Exception as e:
                    print(f"Warning: Could not visualize spike timing: {e}")
            
            # Apply delay learning if there are input spikes to learn from (even without output spikes)
            if sum(len(st) for st in input_spike_times) > 0:
                try:
                    from delay_learning import apply_delay_learning
                    
                    # If there are no output spikes, we can't apply standard delay learning,
                    # but we can still make some adjustments based on input spikes
                    if sum(len(st) for st in output_spike_times) == 0:
                        print("No output spikes detected. There may be an issue with the network parameters.")
                        print("Proceeding without delay updates for this chunk.")
                        
                        # Debug - let's directly check the membrane potential to see if neurons are close to spiking
                        try:
                            v_traces = output_data.analogsignals
                            if len(v_traces) > 0:
                                for i, v in enumerate(v_traces):
                                    print(f"Output neuron {i} membrane potential range: {min(v)} to {max(v)}")
                                    print(f"  Distance to threshold: {-55.0 - min(v)}")
                        except Exception as e:
                            print(f"Error accessing membrane potentials: {e}")
                    else:
                        # Apply the delay learning rule
                        new_delays = apply_delay_learning(
                            connections, input_spike_times, output_spike_times,
                            learning_rate=0.05, min_delay=0.1, max_delay=10.0,
                            B_plus=B_plus, B_minus=B_minus, 
                            sigma_plus=sigma_plus, sigma_minus=sigma_minus,
                            window_size=20.0
                        )
                        
                        # Print the updated delays
                        print(f"\nUpdated delays after chunk at {current_time} ms:")
                        print(f"  Input->Output delays: {new_delays}")
                except Exception as e:
                    print(f"Error applying delay learning: {e}")
    else:
        print("Running simulation with fixed weights and initial random delays...")
        sim.run(simulation_time)
    
    # Get final recording of output
    output_pop.record(None)
    output_pop.record(("spikes", "v"))
    output_pop.record(("spikes", "v"))
    input_pop.record(None)
    input_pop.record("spikes")
    
    # Run a final evaluation simulation
    print("\nRunning final evaluation with learned delays...")
    sim.run(simulation_time)
    
    # Get complete output and input data for analysis
    output_data = output_pop.get_data().segments[-1]
    input_data_final = input_pop.get_data().segments[-1]
    
    # Get final delays after learning
    final_delays = connections.get("delay", format="array")
    
    # Show final delay changes
    print("\nDelay changes from learning:")
    for i in range(final_delays.shape[0]):
        for j in range(final_delays.shape[1]):
            change = final_delays[i][j] - initial_delays[i][j]
            print(f"  Input{i+1}->Output{j+1}: {initial_delays[i][j]:.2f} -> {final_delays[i][j]:.2f} (Δ{change:+.2f})")
    
    # Calculate accuracy
    try:
        accuracy, predictions, true_labels = calculate_accuracy(
            output_data, pattern_sequence, pattern_interval, start_time=10
        )
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        accuracy = 0.0
        predictions = []
        true_labels = []
    
    # Display results
    if learning_enabled:
        print(f"\nDelay Learning Results:")
    else:
        print(f"\nRandom Delay Baseline Results:")
    
    print(f"Accuracy: {accuracy:.2%}")
    
    # Print weights and delays in a format that works regardless of shape
    print(f"Weights: {weights}")
    print(f"Initial delays: {initial_delays}")
    print(f"Final delays: {final_delays}")
    
    # Calculate detailed accuracy
    try:
        detailed_results = calculate_detailed_accuracy(
            output_data, pattern_sequence, pattern_interval, start_time=10
        )
        
        print(f"\nDetailed Results:")
        print(f"Overall Accuracy: {detailed_results['overall_accuracy']:.2%}")
        
        # Calculate delay statistics and changes
        try:
            delay_mean_initial = np.mean(initial_delays)
            delay_std_initial = np.std(initial_delays)
            delay_mean_final = np.mean(final_delays)
            delay_std_final = np.std(final_delays)
            delay_changes = np.abs(np.array(final_delays) - np.array(initial_delays)).sum()
        except Exception as e:
            print(f"Warning: Could not calculate delay statistics: {e}")
            delay_mean_initial = delay_std_initial = delay_mean_final = delay_std_final = delay_changes = 0.0
    except Exception as e:
        print(f"Error calculating detailed results: {e}")
        detailed_results = None
        delay_mean_initial = delay_std_initial = delay_mean_final = delay_std_final = delay_changes = 0.0
    
    print(f"Initial delay statistics - Mean: {delay_mean_initial:.4f}ms, Std: {delay_std_initial:.4f}ms")
    print(f"Final delay statistics   - Mean: {delay_mean_final:.4f}ms, Std: {delay_std_final:.4f}ms")
    print(f"Total delay changes: {delay_changes:.6f}ms")
    
    if detailed_results:
        print(f"\nPer-class Accuracies:")
        for pattern, acc in detailed_results['class_accuracies'].items():
            print(f"  {pattern}: {acc:.2%}")
        print(f"Confusion Matrix:")
        print(detailed_results['confusion_matrix'])
    
    return accuracy, weights, final_delays, output_data, pattern_sequence

def run_stdp_simulation(input_spikes, pattern_sequence, pattern_interval, 
                       connections, output_pop, simulation_time=1500):
    print(f"Running baseline simulation with STDP weight learning for {simulation_time} ms...")
    
    #get initial weights and delays
    initial_weights = connections.get("weight", format="array")
    delays = connections.get("delay", format="array")
    
    print("Initial weights:")
    print(f"  Input1->Output: {initial_weights[0]}")
    print(f"  Input2->Output: {initial_weights[1]}")
    
    print("\nRandom delays (ms):")
    print(f"  Input1->Output: {delays[0]}")
    print(f"  Input2->Output: {delays[1]}")
    
    # Set up recording
    output_pop.record(("spikes", "v"))
    
    # Run the simulation with STDP learning enabled
    print("Running simulation with STDP learning and random delays...")
    sim.run(simulation_time)
    
    # Get complete output data for analysis
    output_data = output_pop.get_data().segments[-1]
    
    #get final weights after STDP learning
    final_weights = connections.get("weight", format="array")
    
    # Calculate accuracy
    accuracy, predictions, true_labels = calculate_accuracy(
        output_data, pattern_sequence, pattern_interval, start_time=10
    )
    
    # Display results
    print(f"\nSTDP Weight Learning Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Initial weights: I1->O: {initial_weights[0]}")
    print(f"                 I2->O: {initial_weights[1]}")
    print(f"Final weights:   I1->O: {final_weights[0]}")
    print(f"                 I2->O: {final_weights[1]}")
    print(f"Random delays:   I1->O: {delays[0]}")
    print(f"                 I2->O: {delays[1]}")
    
    # Calculate detailed accuracy
    detailed_results = calculate_detailed_accuracy(
        output_data, pattern_sequence, pattern_interval, start_time=10
    )
    
    print(f"\nDetailed Results:")
    print(f"Overall Accuracy: {detailed_results['overall_accuracy']:.2%}")
    
    # Calculate delay statistics
    delay_mean = np.mean(delays)
    delay_std = np.std(delays)
    print(f"Delay statistics - Mean: {delay_mean:.4f}ms, Std: {delay_std:.4f}ms")
    
    # Always print per-class accuracies for baseline
    print(f"\nPer-class Accuracies:")
    for pattern, acc in detailed_results['class_accuracies'].items():
        print(f"  {pattern}: {acc:.2%}")
    print(f"Confusion Matrix:")
    print(detailed_results['confusion_matrix'])
    
    # Calculate weight changes
    weight_changes = np.abs(final_weights - initial_weights).sum()
    print(f"Total weight changes: {weight_changes:.6f}")
    
    return accuracy, final_weights, delays, output_data, pattern_sequence

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_spike_trains(input_data, output_data):
    plt.figure(figsize=(12, 8))
    
    # Input spikes
    plt.subplot(2, 1, 1)
    for i, spiketrain in enumerate(input_data.spiketrains):
        plt.plot(spiketrain, [i] * len(spiketrain), 'o', markersize=3, label=f"Input {i+1}")
    plt.yticks([0, 1], ["Input 1", "Input 2"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron")
    plt.title("Input Spike Trains")

    # Output spikes
    plt.subplot(2, 1, 2)
    for i, spiketrain in enumerate(output_data.spiketrains):
        plt.plot(spiketrain, [i] * len(spiketrain), 'o', markersize=3, label=f"Output {i+1}")
    plt.yticks([0, 1, 2], ["Neuron 1", "Neuron 2", "Neuron 3"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron")
    plt.title("Output Spike Trains")
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
    plt.title("Output Neuron Membrane Potentials with STDP Learning")
    plt.legend()
    plt.show()

def plot_final_weights(final_weights_matrix, title="Synaptic Weights after STDP Learning"):
    plt.figure(figsize=(8, 6))
    plt.bar(["I1→O1", "I2→O1", "I1→O2", "I2→O2", "I1→O3", "I2→O3"], 
            [final_weights_matrix[0][0], final_weights_matrix[1][0], 
             final_weights_matrix[0][1], final_weights_matrix[1][1],
             final_weights_matrix[0][2], final_weights_matrix[1][2]])
    plt.ylabel("Weight")
    plt.title(title)
    plt.show()
    
def plot_delays(delays_matrix, title="Synaptic Delays"):
    """Plot the synaptic delays as a heatmap."""
    plt.figure(figsize=(10, 6))
    plt.imshow(delays_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Delay (ms)')
    plt.xlabel('Output Neuron')
    plt.ylabel('Input Neuron')
    plt.title(title)
    plt.xticks(range(delays_matrix.shape[1]), ['Out 1', 'Out 2', 'Out 3'])
    plt.yticks(range(delays_matrix.shape[0]), ['In 1', 'In 2'])
    # Add text annotations for delay values
    for i in range(delays_matrix.shape[0]):
        for j in range(delays_matrix.shape[1]):
            plt.text(j, i, f"{delays_matrix[i, j]:.2f}", 
                     ha='center', va='center', 
                     color='white' if delays_matrix[i, j] > 2.5 else 'black')
    plt.show()

def plot_delay_changes(initial_delays, final_delays):
    """Plot the changes in synaptic delays after learning."""
    plt.figure(figsize=(12, 12))
    
    # First, plot the bar chart comparison
    plt.subplot(2, 1, 1)
    connection_labels = ["I1→O1", "I2→O1", "I1→O2", "I2→O2", "I1→O3", "I2→O3"]
    initial_values = [initial_delays[0][0], initial_delays[1][0], 
                     initial_delays[0][1], initial_delays[1][1],
                     initial_delays[0][2], initial_delays[1][2]]
    final_values = [final_delays[0][0], final_delays[1][0], 
                   final_delays[0][1], final_delays[1][1],
                   final_delays[0][2], final_delays[1][2]]
    
    x = np.arange(len(connection_labels))
    width = 0.35
    
    plt.bar(x - width/2, initial_values, width, label='Initial Delays')
    plt.bar(x + width/2, final_values, width, label='Final Delays')
    
    plt.ylabel('Delay (ms)')
    plt.title('Synaptic Delay Changes After Learning')
    plt.xticks(x, connection_labels)
    plt.legend()
    
    for i in range(len(connection_labels)):
        change = final_values[i] - initial_values[i]
        change_text = f"{change:+.2f}"
        plt.annotate(change_text, 
                    xy=(i, max(initial_values[i], final_values[i]) + 0.1), 
                    ha='center')
    
    # Then, plot the heatmap of delay changes
    plt.subplot(2, 1, 2)
    delay_changes = final_delays - initial_delays
    plt.imshow(delay_changes, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Delay Change (ms)')
    plt.xlabel('Output Neuron')
    plt.ylabel('Input Neuron')
    plt.title('Synaptic Delay Changes (Final - Initial)')
    plt.xticks(range(final_delays.shape[1]), ['Out 1', 'Out 2', 'Out 3'])
    plt.yticks(range(final_delays.shape[0]), ['In 1', 'In 2'])
    for i in range(final_delays.shape[0]):
        for j in range(final_delays.shape[1]):
            plt.text(j, i, f"{delay_changes[i, j]:+.2f}", 
                     ha='center', va='center', 
                     color='white' if abs(delay_changes[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run spiking neural network with configurable learning mechanisms')
    
    # Noise parameters
    noise_group = parser.add_argument_group('Noise Parameters')
    noise_group.add_argument('--noise', action='store_true', help='Enable noise in spike trains')
    noise_group.add_argument('--jitter', type=float, default=1.0, help='Standard deviation of temporal jitter (ms)')
    noise_group.add_argument('--missing', type=float, default=0.1, help='Probability of missing spikes (0-1)')
    noise_group.add_argument('--extra', type=float, default=0.1, help='Probability of extra spikes (0-1)')
    
    # Learning mode parameters
    learning_group = parser.add_argument_group('Learning Parameters')
    learning_group.add_argument('--fixed-weights', action='store_true', 
                                help='Use fixed weights (disable STDP weight learning)')
    learning_group.add_argument('--weight-value', type=float, default=2.0,
                                help='Value for fixed weights when --fixed-weights is used')
    learning_group.add_argument('--learn-delays', action='store_true',
                                help='Enable delay learning (otherwise use fixed random delays)')
    
    # Delay learning parameters
    delay_group = parser.add_argument_group('Delay Learning Parameters (when --learn-delays is used)')
    delay_group.add_argument('--B-plus', type=float, default=0.1,
                           help='Magnitude of delay increase for causal spike pairs')
    delay_group.add_argument('--B-minus', type=float, default=0.1,
                           help='Magnitude of delay decrease for anti-causal spike pairs')
    delay_group.add_argument('--sigma-plus', type=float, default=10.0,
                           help='Time constant for causal spike pairs')
    delay_group.add_argument('--sigma-minus', type=float, default=10.0,
                           help='Time constant for anti-causal spike pairs')
    delay_group.add_argument('--chunk-size', type=int, default=100,
                           help='Simulation chunk size for delay learning (ms)')
    
    # General simulation parameters
    parser.add_argument('--sim-time', type=int, default=1000, help='Simulation time (ms)')
    
    args = parser.parse_args()
    
    noise_config = NOISE_CONFIG.copy()
    noise_config['enabled'] = args.noise
    noise_config['jitter_std'] = args.jitter
    noise_config['missing_prob'] = args.missing
    noise_config['extra_prob'] = args.extra
    
    # Determine the simulation title and configuration based on arguments
    if args.fixed_weights and not args.learn_delays:
        experiment_title = "FIXED WEIGHTS AND RANDOM DELAYS"
    elif not args.fixed_weights and not args.learn_delays:
        experiment_title = "STDP WEIGHT LEARNING WITH RANDOM DELAYS"
    elif args.fixed_weights and args.learn_delays:
        experiment_title = "FIXED WEIGHTS WITH LEARNABLE DELAYS"
    else:  # not args.fixed_weights and args.learn_delays
        experiment_title = "COMBINED WEIGHT AND DELAY LEARNING"
        
    print("=" * 60)
    print(f"EXPERIMENT: {experiment_title}")
    print("=" * 60)
    print("This simulation uses:")
    
    # Print weight learning configuration
    if args.fixed_weights:
        print(f"- Fixed weights ({args.weight_value}) for all connections")
        print("- StaticSynapse (no weight learning)")
    else:
        print("- STDP for weight learning")
        print("- Weight modification parameters: A_plus=0.2, A_minus=0.1, tau_plus=15.0, tau_minus=30.0")
    
    print("- Lateral inhibition (0.15)")
    
    # Print delay learning configuration
    if args.learn_delays:
        print("- Delay learning enabled with parameters:")
        print(f"  * B_plus={args.B_plus}, B_minus={args.B_minus}")
        print(f"  * sigma_plus={args.sigma_plus}, sigma_minus={args.sigma_minus}")
        print(f"  * Simulation in chunks of {args.chunk_size}ms for incremental updates")
    else:
        print("- Random delays within biologically plausible range:")
        print("  * All connections: Random delays between 0.1-5.0 ms")
        
    # Print noise configuration if enabled
    if args.noise:
        print("- Noise in spike trains:")
        print(f"  * Temporal jitter: {args.jitter}ms std dev")
        print(f"  * Missing spikes: {args.missing:.1%} probability")
        print(f"  * Extra spikes: {args.extra:.1%} probability")
    
    # 1. Generate spike trains
    input_spikes, pattern_sequence = create_pattern_spike_trains(
        PATTERNS, PATTERN_OCCURRENCES, PATTERN_INTERVAL,
        noise_config=noise_config
    )
    
    # 2. Create populations
    input_pop, output_pop = create_populations(input_spikes, NEURON_PARAMS)
    
    # 3. Create connections
    stdp_connections, inhibitory_connections = create_connections_and_inhibition(
        input_pop, output_pop, STDP_CONFIG, args
    )
    
    # 4. Run simulation based on selected mode
    simulation_time = args.sim_time
    
    if args.learn_delays:
        print("\nRunning simulation with delay learning...")
        
        # Instead of using the chunked simulation which has issues,
        # let's perform a single-step simulation but manually update delays
        # based on our delay learning rule from delay_learning.py
        
        # Set up recording
        input_pop.record("spikes")
        output_pop.record(("spikes", "v"))
        
        # Extract initial weights and delays for reference
        initial_weights = stdp_connections.get("weight", format="array")
        initial_delays = stdp_connections.get("delay", format="array")
        
        print(f"Initial weights: {initial_weights}")
        print(f"Initial delays: {initial_delays}")
        
        # Run the full simulation first to get spike data
        print(f"Running full simulation for {simulation_time} ms to get spike data...")
        sim.run(simulation_time)
        
        # Get the spike data
        input_data = input_pop.get_data().segments[-1]
        output_data = output_pop.get_data().segments[-1]
        
        # Extract spike times as simple lists
        input_spike_times = []
        for st in input_data.spiketrains:
            input_spike_times.append([float(t) for t in st])
            
        output_spike_times = []
        for st in output_data.spiketrains:
            output_spike_times.append([float(t) for t in st])
            
        # Print spike counts
        print(f"Input spikes: {[len(spikes) for spikes in input_spike_times]}")
        print(f"Output spikes: {[len(spikes) for spikes in output_spike_times]}")
        
        # Apply the delay learning rule manually
        from delay_learning import apply_delay_learning
        
        print("\nApplying delay learning rule...")
        try:
            final_delays = apply_delay_learning(
                stdp_connections, input_spike_times, output_spike_times,
                learning_rate=0.05, min_delay=0.1, max_delay=10.0,
                B_plus=args.B_plus, B_minus=args.B_minus, 
                sigma_plus=args.sigma_plus, sigma_minus=args.sigma_minus,
                window_size=20.0
            )
            
            print(f"Updated delays after learning:")
            print(f"Initial: {initial_delays}")
            print(f"Final: {final_delays}")
            
            # Calculate accuracy
            accuracy, predictions, true_labels = calculate_accuracy(
                output_data, pattern_sequence, PATTERN_INTERVAL, start_time=10
            )
            
            weights = initial_weights  # We're using fixed weights
            delays = final_delays
            
            # We already have output_data and pattern_sequence
            
        except Exception as e:
            print(f"Error applying delay learning: {e}")
            accuracy = 0.0
            weights = initial_weights
            delays = initial_delays
    else:
        # Use regular STDP or static weights simulation
        accuracy, weights, delays, output_data, patterns = run_stdp_simulation(
            input_spikes, pattern_sequence, PATTERN_INTERVAL,
            stdp_connections, output_pop, simulation_time
        )
    
    # 5. Visualizations
    print("\n" + "=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)
    input_data = input_pop.get_data().segments[0]
    plot_spike_trains(input_data, output_data)
    plot_membrane_potentials(output_data, NEURON_PARAMS['v_thresh'])
    plot_final_weights(weights, title="Synaptic Weights after STDP Learning")
    plot_delays(delays)
    
    # 6. Final analysis
    analyze_pattern_mistakes(output_data, pattern_sequence, 
                           PATTERN_INTERVAL, start_time=10
    )
    
    sim.end()
    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    main()