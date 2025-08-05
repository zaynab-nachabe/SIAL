import pyNN.nest as sim
from quantities import ms
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from pyNN.connectors import FromListConnector

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run DUMBO SNN simulation with configurable shuffling and noise')
parser.add_argument('--shuffle', action='store_true', help='Use complete random shuffling instead of chunk-based')
parser.add_argument('--noise', action='store_true', help='Add noise to spike trains for robustness testing')
args = parser.parse_args()

# Initialize simulator
sim.setup(timestep=0.01)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pattern definitions
PATTERNS = [
    {"name": "pattern_1", "delay": 1.0, "first_neuron": 0},  # Short delay
    {"name": "pattern_2", "delay": 3.0, "first_neuron": 1},  # Medium delay
    {"name": "pattern_3", "delay": 0.0, "simultaneous": True}  # Simultaneous firing
]

# Training parameters
PATTERN_OCCURRENCES = 20
PATTERN_INTERVAL = 15  # ms between pattern presentations
CHUNK_SIZE = 5  # For semi-structured pattern shuffling

# Network parameters
NEURON_PARAMS = {
    'tau_m': 20.0,
    'tau_refrac': 2.0,
    'v_reset': -70.0,
    'v_rest': -65.0,
    'v_thresh': -52.0,
    'cm': 0.2
}

# STDP parameters
STDP_CONFIG = {
    "A_plus": 0.2,
    "A_minus": 0.1,
    "tau_plus": 15.0,
    "tau_minus": 30.0,
}

# Noise configuration for spike trains
NOISE_CONFIG = {
    "jitter": 2.0,       # Timing jitter in ms (std dev of gaussian noise)
    "deletion_prob": 0.05, # Probability of spike deletion
    "insertion_prob": 0.01, # Probability of spurious spike insertion
    "insertion_window": 5.0 # Window size in ms for potential spurious spikes
}

# ============================================================================
# SPIKE TRAIN GENERATION
# ============================================================================

def create_pattern_spike_trains(patterns, occurrences, interval, start_time=10, complete_shuffle=False, add_noise=False):
    input_spikes = [[] for _ in range(2)]
    current_time = start_time
    
    pattern_sequence = []
    for pattern_type in patterns:
        for _ in range(occurrences):
            pattern_sequence.append(pattern_type)
    
    random.seed(42)
    if complete_shuffle:
        random.shuffle(pattern_sequence)
        print("Using complete random shuffling for patterns")
    else:
        for i in range(0, len(pattern_sequence), CHUNK_SIZE):
            chunk = pattern_sequence[i:i+CHUNK_SIZE]
            random.shuffle(chunk)
            pattern_sequence[i:i+CHUNK_SIZE] = chunk
        print(f"Using semi-structured shuffling (chunks of {CHUNK_SIZE} patterns)")
    
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

def create_connections_and_inhibition(input_pop, output_pop, stdp_config):
    np.random.seed(42)
    connection_list = []
    
    # Fixed weights for baseline comparison
    fixed_weight = 0.6
    
    print("Using fixed weights and random delays:")
    
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
    
    for output_idx in range(3):
        input1_delay = input1_delays[output_idx]
        input2_delay = input2_delays[output_idx]
        
        print(f"  Output {output_idx+1}: Input1 delay={input1_delay:.2f}ms, Input2 delay={input2_delay:.2f}ms")
        
        connection_list.append([0, output_idx, fixed_weight, input1_delay])
        connection_list.append([1, output_idx, fixed_weight, input2_delay])
    
    # Using StaticSynapse instead of STDP for baseline experiment
    static_model = sim.StaticSynapse()
    
    stdp_connections = sim.Projection(
        input_pop, output_pop,
        sim.FromListConnector(connection_list),
        synapse_type=static_model,
        receptor_type="excitatory",
        label="Static_connections"
    )
    
    # Lateral inhibition
    inhibitory_list = [
        [0, 1, 0.1, 0.1], [0, 2, 0.1, 0.1],
        [1, 0, 0.1, 0.1], [1, 2, 0.1, 0.1],
        [2, 0, 0.1, 0.1], [2, 1, 0.1, 0.1]
    ]
    
    inhibitory_connections = sim.Projection(
        output_pop, output_pop,
        FromListConnector(inhibitory_list),
        receptor_type="inhibitory",
        label="Lateral_inhibition"
    )    
    return stdp_connections, inhibitory_connections

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
    
    # Print summary analysis
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
                       connections, output_pop, simulation_time=1500):
    print(f"Running baseline simulation with fixed weights for {simulation_time} ms...")
    
    weights = connections.get("weight", format="array")
    delays = connections.get("delay", format="array")
    
    print("Fixed weights:")
    print(f"  Input1->Output: {weights[0]}")
    print(f"  Input2->Output: {weights[1]}")
    
    print("\nRandom delays (ms):")
    print(f"  Input1->Output: {delays[0]}")
    print(f"  Input2->Output: {delays[1]}")
    
    # Set up recording
    output_pop.record(("spikes", "v"))
    
    # Run the simulation with fixed weights
    print("Running simulation with fixed weights and random delays...")
    sim.run(simulation_time)
    
    # Get complete output data for analysis
    output_data = output_pop.get_data().segments[-1]
    
    # Calculate accuracy
    accuracy, predictions, true_labels = calculate_accuracy(
        output_data, pattern_sequence, pattern_interval, start_time=10
    )
    
    # Display results
    print(f"\nRandom Delay Baseline Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Fixed weights: I1->O: {weights[0]}")
    print(f"               I2->O: {weights[1]}")
    print(f"Random delays: I1->O: {delays[0]}")
    print(f"               I2->O: {delays[1]}")
    
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
    
    return accuracy, weights, delays, output_data, pattern_sequence

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_spike_trains(input_data, output_data):
    pattern_interval = PATTERN_INTERVAL
    
    def create_focused_plot(start_time, end_time, title_suffix=""):
        plt.figure(figsize=(24, 10))
        
        # Input spikes
        plt.subplot(2, 1, 1)
        for i, spiketrain in enumerate(input_data.spiketrains):
            visible_spikes = [spike for spike in spiketrain if start_time <= spike <= end_time]
            plt.plot(visible_spikes, [i] * len(visible_spikes), 'o', markersize=5, label=f"Input {i+1}")
            
            pattern_times = range(10, int(end_time), pattern_interval)
            for t in pattern_times:
                if start_time <= t <= end_time:
                    plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
        
        plt.yticks([0, 1], ["Input 1", "Input 2"])
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron")
        plt.xlim(start_time, end_time)
        plt.title(f"Input Spike Trains {title_suffix}")
        plt.grid(True, axis='x', alpha=0.3)

        # Output spikes
        plt.subplot(2, 1, 2)
        for i, spiketrain in enumerate(output_data.spiketrains):
            visible_spikes = [spike for spike in spiketrain if start_time <= spike <= end_time]
            plt.plot(visible_spikes, [i] * len(visible_spikes), 'o', markersize=5, label=f"Output {i+1}")
            
            for t in pattern_times:
                if start_time <= t <= end_time:
                    plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
        
        plt.yticks([0, 1, 2], ["Neuron 1", "Neuron 2", "Neuron 3"])
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron")
        plt.xlim(start_time, end_time)
        plt.title(f"Output Spike Trains {title_suffix}")
        plt.grid(True, axis='x', alpha=0.3)
        plt.figtext(0.5, 0.01, 
                  "Pattern 1: Input 1 fires, then Input 2 after 1ms delay\n" +
                  "Pattern 2: Input 2 fires, then Input 1 after 3ms delay\n" +
                  "Pattern 3: Both inputs fire simultaneously",
                  ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Create three focused views: early, middle, and late patterns
    create_focused_plot(10, 100, "(Early Patterns: 10-100ms)")
    create_focused_plot(200, 290, "(Mid-Simulation Patterns: 200-290ms)")
    plt.show()

def plot_membrane_potentials(output_data, threshold=-54.0):
    plt.figure(figsize=(24, 8))
    
    v_data = output_data.filter(name="v")[0]
    sampling_interval = 0.1
    time_points = np.arange(0, v_data.shape[0] * sampling_interval, sampling_interval)
    
    for i in range(3):
        plt.plot(time_points, v_data[:, i], label=f"Output {i+1}")
    
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold}mV)')
    
    # Focus on a 100ms window where we can see oscillations clearly
    start_time = 200
    window_size = 100
    plt.xlim(start_time, start_time + window_size)
    
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title(f"Output Neuron Membrane Potentials (Showing {start_time}-{start_time+window_size}ms)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.figure(figsize=(24, 8))
    start_time2 = 2300
    for i in range(3):
        plt.plot(time_points, v_data[:, i], label=f"Output {i+1}")
    
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold}mV)')
    plt.xlim(start_time2, start_time2 + window_size)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title(f"Output Neuron Membrane Potentials (Showing {start_time2}-{start_time2+window_size}ms)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_final_weights(final_weights_matrix):
    plt.figure(figsize=(8, 6))
    plt.bar(["I1→O1", "I2→O1", "I1→O2", "I2→O2", "I1→O3", "I2→O3"], 
            [final_weights_matrix[0][0], final_weights_matrix[1][0], 
             final_weights_matrix[0][1], final_weights_matrix[1][1],
             final_weights_matrix[0][2], final_weights_matrix[1][2]])
    plt.ylabel("Weight")
    plt.title("Fixed Synaptic Weights")
    plt.show()
    
def plot_delays(delays_matrix):
    plt.figure(figsize=(8, 6))
    plt.bar(["I1→O1", "I2→O1", "I1→O2", "I2→O2", "I1→O3", "I2→O3"], 
            [delays_matrix[0][0], delays_matrix[1][0], 
             delays_matrix[0][1], delays_matrix[1][1],
             delays_matrix[0][2], delays_matrix[1][2]])
    plt.ylabel("Delay (ms)")
    plt.title("Random Synaptic Delays")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("BASELINE EXPERIMENT: FIXED WEIGHTS AND STRATEGIC RANDOM DELAYS")
    print("=" * 60)
    print("This simulation uses:")
    print("- Fixed weights (0.6) for all connections")
    print("- StaticSynapse (no learning)")
    print("- Reduced lateral inhibition (0.1)")
    print("- Strategically distributed delays for better pattern discrimination:")
    print("  * Output 1: Short delays (0.1-2.0 ms)")
    print("  * Output 2: Medium delays (2.0-4.0 ms)")
    print("  * Output 3: Mixed delays (0.1-2.0 and 3.0-5.0 ms)")
    print(f"- Pattern shuffling: {'Complete random' if args.shuffle else f'Semi-structured (chunks of {CHUNK_SIZE})'}")
    if args.noise:
        print("- Noise applied to spike trains for robustness testing")
    print("=" * 60)
    
    # 1. Generate spike trains
    input_spikes, pattern_sequence = create_pattern_spike_trains(
        PATTERNS, PATTERN_OCCURRENCES, PATTERN_INTERVAL, 
        complete_shuffle=args.shuffle, add_noise=args.noise
    )
    
    # 2. Create populations
    input_pop, output_pop = create_populations(input_spikes, NEURON_PARAMS)
    
    # 3. Create connections
    stdp_connections, inhibitory_connections = create_connections_and_inhibition(
        input_pop, output_pop, STDP_CONFIG
    )
    
    # 4. Run simulation
    simulation_time = 3000  # ms
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
    plot_final_weights(weights)
    plot_delays(delays)
    
    # 6. Final analysis
    analyze_pattern_mistakes(output_data, pattern_sequence, 
                           PATTERN_INTERVAL, start_time=10
    )
    
    # Cleanup
    sim.end()
    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    main()