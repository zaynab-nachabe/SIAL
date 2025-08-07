import pyNN.nest as sim
import numpy as np
import matplotlib.pyplot as plt
import random
import os
sim.setup(timestep=0.01)

def delay_learning_rule(pre_time, post_time, current_delay, 
                       B_plus=3.0, B_minus=2.5, sigma_plus=2.5, sigma_minus=2.5):
    delta_t = post_time - pre_time - current_delay
    if delta_t >= 0:
        return -B_minus * np.exp(-delta_t / sigma_minus)
    else:
        return B_plus * np.exp(delta_t / sigma_plus)


def analyze_pattern_responses(output_spiketrains, pattern_times, window=20):
    pattern0_responses = [0, 0]
    pattern1_responses = [0, 0]
    
    for time, pattern in pattern_times:
        for j, spiketrain in enumerate(output_spiketrains):
            response = any((time <= t <= time + window) for t in spiketrain)
            if pattern == 0 and response:
                pattern0_responses[j] += 1
            elif pattern == 1 and response:
                pattern1_responses[j] += 1
    
    pattern0_count = sum(1 for _, p in pattern_times if p == 0)
    pattern1_count = sum(1 for _, p in pattern_times if p == 1)
    
    # Calculate response rates
    pattern0_rates = [resp/pattern0_count if pattern0_count > 0 else 0 for resp in pattern0_responses]
    pattern1_rates = [resp/pattern1_count if pattern1_count > 0 else 0 for resp in pattern1_responses]
    
    # Calculate specialization score (higher is better)
    spec_score1 = (pattern0_rates[0] * (1 - pattern1_rates[0]) * 
                  (1 - pattern0_rates[1]) * pattern1_rates[1])
    spec_score2 = (pattern1_rates[0] * (1 - pattern0_rates[0]) * 
                  (1 - pattern1_rates[1]) * pattern0_rates[1])
    
    spec_score = max(spec_score1, spec_score2)
    
    return {
        'pattern0_responses': pattern0_responses,
        'pattern1_responses': pattern1_responses,
        'pattern0_rates': pattern0_rates,
        'pattern1_rates': pattern1_rates,
        'specialization_score': spec_score
    }


def update_neuron_activity_history(neuron_activity_history, pattern_response_history, 
                              output_spiketrains, pattern_times):
    """Update the history of neuron activity and pattern responses.
    
    Returns:
        Tuple of (pattern0_responses, pattern1_responses) - arrays tracking response counts.
    """
    pattern0_responses = [0, 0]
    pattern1_responses = [0, 0]
    window = 20
    
    # Calculate pattern-specific responses
    for time, pattern in pattern_times:
        for j, spiketrain in enumerate(output_spiketrains):
            response = any((time <= t <= time + window) for t in spiketrain)
            if pattern == 0 and response:
                pattern0_responses[j] += 1
            elif pattern == 1 and response:
                pattern1_responses[j] += 1
    
    # Update neuron activity history
    pattern0_count = sum(1 for _, p in pattern_times if p == 0)
    pattern1_count = sum(1 for _, p in pattern_times if p == 1)
    
    for j in range(2):
        # Add normalized spike count (activity level)
        total_patterns = len(pattern_times)
        if total_patterns > 0:
            neuron_activity_history[j].append(len(output_spiketrains[j]) / total_patterns)
        else:
            neuron_activity_history[j].append(0)
        
        # Add normalized pattern responses (1.0 = responded to all presentations of this pattern)
        if pattern0_count > 0:
            pattern_response_history[0][j].append(pattern0_responses[j] / pattern0_count)
        else:
            pattern_response_history[0][j].append(0)
            
        if pattern1_count > 0:
            pattern_response_history[1][j].append(pattern1_responses[j] / pattern1_count)
        else:
            pattern_response_history[1][j].append(0)
    
    return pattern0_responses, pattern1_responses


def print_debug_spike_pairs(all_pairs, current_delays, learning_rate):
    """Print detailed debug information about spike pairs and their delay updates."""
    print("  Spike pairs and their individual updates:")
    for i in range(2):
        for j in range(2):
            pairs = all_pairs.get((i,j), [])
            if pairs:
                print(f"    input{i} to output{j} ({len(pairs)} pairs):")
                total_delta = 0
                for pre_time, post_time in pairs:
                    delta = delay_learning_rule(pre_time, post_time, current_delays[i][j])
                    final_delta = delta * learning_rate
                    total_delta += final_delta
                    
                    print(f"      pre: {pre_time:.2f}ms, post: {post_time:.2f}ms, " +
                          f"delta: {delta:.4f}ms, " +
                          f"final: {final_delta:.4f}ms")
                
                print(f"      Total change for input{i}→output{j}: {total_delta:.4f}ms")


def print_delay_changes(current_delays, delay_history):
    """Print changes in delays from the previous iteration."""
    print("  Delay changes in this iteration (ms):")
    print(f"    input0→output0: {current_delays[0][0] - delay_history['input0_output0'][-2]:.4f}")
    print(f"    input0→output1: {current_delays[0][1] - delay_history['input0_output1'][-2]:.4f}")
    print(f"    input1→output0: {current_delays[1][0] - delay_history['input1_output0'][-2]:.4f}")
    print(f"    input1→output1: {current_delays[1][1] - delay_history['input1_output1'][-2]:.4f}")


def print_final_results(initial_delays, current_delays, delay_history, final_input_spiketrains, final_output_spiketrains, final_pattern_times):
    """Print the final results of the simulation."""
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print("Initial delays (ms):")
    print("  input0 to output0: {:.2f}, input0 to output1: {:.2f}".format(
        initial_delays[0][0], initial_delays[0][1]))
    print("  input1 to output0: {:.2f}, input1 to output1: {:.2f}".format(
        initial_delays[1][0], initial_delays[1][1]))
    
    print("\nFinal delays (ms):")
    print("  input0 to output0: {:.2f}, input0 to output1: {:.2f}".format(
        current_delays[0][0], current_delays[0][1]))
    print("  input1 to output0: {:.2f}, input1 to output1: {:.2f}".format(
        current_delays[1][0], current_delays[1][1]))
    
    print("\nChanges (ms):")
    print("  input0 to output0: {:+.2f}, input0 to output1: {:+.2f}".format(
        current_delays[0][0] - initial_delays[0][0],
        current_delays[0][1] - initial_delays[0][1]))
    print("  input1 to output0: {:+.2f}, input1 to output1: {:+.2f}".format(
        current_delays[1][0] - initial_delays[1][0],
        current_delays[1][1] - initial_delays[1][1]))
    
    print("\nIterations: {}".format(len(delay_history['input0_output0'])))
    print("Final output spikes: {}".format([len(st) for st in final_output_spiketrains]))
    
    # Print spike times
    print("\nINPUT SPIKE TIMES:")
    for i, spiketrain in enumerate(final_input_spiketrains):
        print(f"  Input {i} spikes: {[float(t) for t in spiketrain]}")
    
    print("\nOUTPUT SPIKE TIMES:")
    for i, spiketrain in enumerate(final_output_spiketrains):
        print(f"  Output {i} spikes: {[float(t) for t in spiketrain]}")
    
    # Analyze pattern responses
    print("\nPATTERN RESPONSES:")
    for i, (time, pattern) in enumerate(final_pattern_times):
        print(f"  Pattern {pattern} at time {time}ms:")
        for j, spiketrain in enumerate(final_output_spiketrains):
            response_spikes = [t for t in spiketrain if time <= t <= time + 20]
            if response_spikes:
                print(f"    Output {j} responded with {len(response_spikes)} spikes at times: {[float(t) for t in response_spikes]}")
            else:
                print(f"    Output {j} did not respond")


def print_specialization_analysis(output_spiketrains, pattern_times):
    """Print and return specialization analysis based on pattern responses."""
    results = analyze_pattern_responses(output_spiketrains, pattern_times)
    
    print("\nSPECIALIZATION ANALYSIS:")
    print("  Output 0 response rates: Pattern 0: {:.0%}, Pattern 1: {:.0%}".format(
        results['pattern0_rates'][0], results['pattern1_rates'][0]))
    print("  Output 1 response rates: Pattern 0: {:.0%}, Pattern 1: {:.0%}".format(
        results['pattern0_rates'][1], results['pattern1_rates'][1]))
    
    print("\nSPECIALIZATION SCORE: {:.2%}".format(results['specialization_score']))
    print("  (1.00 = perfect specialization, 0.00 = no specialization)")
    
    return results


def initialize_simulation_parameters():
    """Initialize all simulation parameters and return them as a dictionary."""
    params = {
        'simulation_time': 600.0,  # Longer simulation to accommodate longer pattern interval
        'num_presentations': 5,
        'pattern_interval': 60,  # Further increased pattern interval
        'learning_rate': 0.8,  # Increased to amplify learning effects
        'c_threshold': 1.0,  # Lower threshold to continue learning longer
        'homeostasis_learning_rate': 0.4,  # Increased from 0.3
        'target_firing_rate': 15,
        'max_iterations': 70,
        'history_window': 5,  # Number of past iterations to consider for specialization tracking
        'inhibition_strength': 6.0,  # Strong PyNN inhibition
    }
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize with extreme specialization bias
    # Output 0 should only respond to pattern 0 (input0 fires first)
    # Output 1 should only respond to pattern 1 (input1 fires first)
    params['initial_delays'] = np.array([
        [1.0, 20.0],  # input0 to output0 (shortest possible), output1 (longest possible)
        [20.0, 1.0]   # input1 to output0 (longest possible), output1 (shortest possible)
    ])

    # Fixed weights with extreme bias to enforce specialization
    params['initial_weights'] = np.array([
        [0.04, 0.04],  # input0 to output0 (very strong), output1 (very weak)
        [0.04, 0.04]   # input1 to output0 (very weak), output1 (very strong)
    ])
    
    return params

def check_stop_condition(delays_to_neuron, c=0.8):
    return np.min(delays_to_neuron) <= c

def find_spike_pairs(pre_spikes, post_spikes, current_delay, window=20.0):
    pairs = []
    for pre_time in pre_spikes:
        arrival_time = pre_time + current_delay
        valid_posts = [t for t in post_spikes if abs(t - arrival_time) <= window]
        if valid_posts:
            closest_post = min(valid_posts, key=lambda t: abs(t - arrival_time))
            pairs.append((pre_time, closest_post))
    return pairs

def get_neuron_params():
    NEURON_PARAMS = {
        'tau_m': 20,          # Increased membrane time constant
        'tau_refrac': 3.0,    # Increased refractory period
        'v_reset': -70.0,     # typically near resting potential
        'v_rest': -65.0,      # typical range -60 to -70mV
        'v_thresh': -52.5,    # typical range -55 to -50mV
        'cm': 0.25,           # scaled for single compartment
        'tau_syn_E': 5.0,     # AMPA receptors ~5ms
        'tau_syn_I': 10.0,    # GABA receptors ~10ms
        'e_rev_E': 0.0,       # Excitatory reversal potential (mV)
        'e_rev_I': -80.0      # Inhibitory reversal potential (mV)
    }
    return NEURON_PARAMS

def get_input_patterns():
    base_patterns = {
        0: {'input0': 4.0, 'input1': 28.0},  # input0 fires first, input1 follows 24ms later (further increased)
        1: {'input1': 4.0, 'input0': 28.0}   # input1 fires first, input0 follows 24ms later (further increased)
    }
    return base_patterns

def generate_dataset(num_presentations=5, pattern_interval=30):
    base_patterns = get_input_patterns()
    
    pattern_sequence = []
    for _ in range(num_presentations):
        pattern_sequence.extend([0, 1])
    
    random.shuffle(pattern_sequence)
    
    input0_spikes = []
    input1_spikes = []
    pattern_start_times = []
    pattern_labels = []
    
    current_time = 0.0
    
    for pattern_id in pattern_sequence:
        pattern = base_patterns[pattern_id]
        
        pattern_start_times.append(current_time)
        pattern_labels.append(pattern_id)
        
        if 'input0' in pattern:
            input0_spikes.append(current_time + pattern['input0'])
        if 'input1' in pattern:
            input1_spikes.append(current_time + pattern['input1'])
        
        current_time += pattern_interval
    
    dataset = {
        'input_spikes': [input0_spikes, input1_spikes],
        'pattern_start_times': pattern_start_times,
        'pattern_labels': pattern_labels
    }
    
    return dataset


#neuron parameters are defined
#the dataset is generated
#now we need to make the input and output populations
#connect them (excitatory and inhibitory connections)
# run the simulation with fixed weights
#make a function to collect simulation data (calculate the delay updates)
#start another simulation with the updated delays
#keep doing this until the stop condition is met.
#finally plot the results, calculate accuracy

def create_populations(num_presentations=5, pattern_interval=30):
    neuron_params = get_neuron_params()
    dataset = generate_dataset(num_presentations=num_presentations, pattern_interval=pattern_interval)
    spike_times = dataset['input_spikes']

    # Create standard input population (input0, input1)
    input_pop = sim.Population(
        2,
        sim.SpikeSourceArray(spike_times=spike_times),
        label="Input"
    )
    input_pop.record("spikes")

    # Create output population
    output_pop = sim.Population(
        2,
        sim.IF_cond_exp(**neuron_params),
        label="Output"
    )
    output_pop.record(("spikes", "v"))

    return input_pop, output_pop

def create_connections(input_pop, output_pop, current_delays, current_weights, inhibition_weight=3.0):
    """Create excitatory and inhibitory connections between input and output populations."""
    #create 4 excitatory connections
    # Connect input 0 to output 0
    conn00 = sim.Projection(
        input_pop[0:1], output_pop[0:1],
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=current_weights[0][0], delay=current_delays[0][0]),
        receptor_type="excitatory"
    )

    # Connect input 1 to output 0
    conn10 = sim.Projection(
        input_pop[1:2], output_pop[0:1],
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=current_weights[1][0], delay=current_delays[1][0]),
        receptor_type="excitatory"
    )

    # Connect input 0 to output 1
    conn01 = sim.Projection(
        input_pop[0:1], output_pop[1:2],
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=current_weights[0][1], delay=current_delays[0][1]),
        receptor_type="excitatory"
    )
    
    # Connect input 1 to output 1
    conn11 = sim.Projection(
        input_pop[1:2], output_pop[1:2],
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=current_weights[1][1], delay=current_delays[1][1]),
        receptor_type="excitatory"
    )
    
    # Use configurable inhibition weight to help with specialization
    print(f"Using inhibition weight of {inhibition_weight} to promote neuron specialization")
    inhibition = sim.Projection(
        output_pop, output_pop,
        sim.AllToAllConnector(allow_self_connections=False),
        synapse_type=sim.StaticSynapse(weight=inhibition_weight, delay=0.1),
        receptor_type="inhibitory"
    )
    connections = [conn00, conn10, conn01, conn11]
    return connections, inhibition

def determine_inhibition_strategy(pattern0_responses, pattern1_responses, output_spike_counts,
                           current_weights, inhibition_strength):
    """Determine the inhibition strategy based on neuron responses.
    
    Returns:
        Tuple of (inhibition_weight, current_weights)
    """
    # Use strong PyNN inhibition by default
    inhibition_weight = 6.0
    
    # Check if one neuron is completely silent
    if output_spike_counts[0] == 0 or output_spike_counts[1] == 0:
        silent_neuron = 0 if output_spike_counts[0] == 0 else 1
        
        print(f"  Output{silent_neuron} is completely silent! Boosting it and reducing inhibition.")
        
        # Increase the weight of connections to the silent neuron for its preferred input
        # The preferred input for output0 is input0, and for output1 is input1
        current_weights[silent_neuron][silent_neuron] *= 1.2  # Increase weight by 20%
        current_weights[silent_neuron][silent_neuron] = min(current_weights[silent_neuron][silent_neuron], 2.0)  # Cap at 2.0
        
        # Reduce PyNN inhibition to allow the silent neuron to recover
        inhibition_weight = 4.0
        
        return inhibition_weight, current_weights
    
    # If both neurons are firing, use standard PyNN inhibition
    return inhibition_weight, current_weights

def collect_simulation_data(input_spiketrains, output_spiketrains, current_delays, window=15.0):
    # Instead of collecting and averaging updates, we'll return all individual spike pairs
    # This allows for more precise per-spike-pair updates
    all_pairs = {}
    all_pair_counts = np.zeros_like(current_delays, dtype=int)
    
    for i in range(2):
        input_spikes = [float(t) for t in input_spiketrains[i]]
        
        for j in range(2):
            output_spikes = [float(t) for t in output_spiketrains[j]]
            
            pairs = find_spike_pairs(input_spikes, output_spikes, current_delays[i][j], window=window)
            all_pair_counts[i][j] = len(pairs)
            
            # Store all spike pairs for this connection
            all_pairs[(i,j)] = pairs
    
    return all_pairs, all_pair_counts

def print_delay_changes(current_delays, delay_history):
    """Print information about delay changes in the current iteration."""
    print("  Delay changes in this iteration (ms):")
    print(f"    input0→output0: {current_delays[0][0] - delay_history['input0_output0'][-2]:.4f}")
    print(f"    input0→output1: {current_delays[0][1] - delay_history['input0_output1'][-2]:.4f}")
    print(f"    input1→output0: {current_delays[1][0] - delay_history['input1_output0'][-2]:.4f}")
    print(f"    input1→output1: {current_delays[1][1] - delay_history['input1_output1'][-2]:.4f}")

def print_final_results(initial_delays, current_delays, delay_history, 
                       input_spiketrains, output_spiketrains, pattern_times):
    """Print final results of the simulation."""
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print("Initial delays (ms):")
    print("  input0 to output0: {:.2f}, input0 to output1: {:.2f}".format(
        initial_delays[0][0], initial_delays[0][1]))
    print("  input1 to output0: {:.2f}, input1 to output1: {:.2f}".format(
        initial_delays[1][0], initial_delays[1][1]))
    
    print("\nFinal delays (ms):")
    print("  input0 to output0: {:.2f}, input0 to output1: {:.2f}".format(
        current_delays[0][0], current_delays[0][1]))
    print("  input1 to output0: {:.2f}, input1 to output1: {:.2f}".format(
        current_delays[1][0], current_delays[1][1]))
    
    print("\nChanges (ms):")
    print("  input0 to output0: {:+.2f}, input0 to output1: {:+.2f}".format(
        current_delays[0][0] - initial_delays[0][0],
        current_delays[0][1] - initial_delays[0][1]))
    print("  input1 to output0: {:+.2f}, input1 to output1: {:+.2f}".format(
        current_delays[1][0] - initial_delays[1][0],
        current_delays[1][1] - initial_delays[1][1]))
    
    print("\nIterations: {}".format(len(delay_history['input0_output0'])))
    print("Final output spikes: {}".format([len(st) for st in output_spiketrains]))
    
    # Print spike times
    print("\nINPUT SPIKE TIMES:")
    for i, spiketrain in enumerate(input_spiketrains):
        print(f"  Input {i} spikes: {[float(t) for t in spiketrain]}")
    
    print("\nOUTPUT SPIKE TIMES:")
    for i, spiketrain in enumerate(output_spiketrains):
        print(f"  Output {i} spikes: {[float(t) for t in spiketrain]}")
    
    # Analyze pattern responses
    print("\nPATTERN RESPONSES:")
    for i, (time, pattern) in enumerate(pattern_times):
        print(f"  Pattern {pattern} at time {time}ms:")
        for j, spiketrain in enumerate(output_spiketrains):
            response_spikes = [t for t in spiketrain if time <= t <= time + 20]
            if response_spikes:
                print(f"    Output {j} responded with {len(response_spikes)} spikes at times: {[float(t) for t in response_spikes]}")
            else:
                print(f"    Output {j} did not respond")

def analyze_pattern_responses(output_spiketrains, pattern_times):
    """Analyze pattern-specific responses of output neurons.
    
    Returns:
        Tuple of (pattern0_rates, pattern1_rates, specialization_score)
    """
    pattern0_count = sum(1 for _, p in pattern_times if p == 0)
    pattern1_count = sum(1 for _, p in pattern_times if p == 1)
    
    # Count responses per pattern per neuron
    pattern0_responses = [0, 0]
    pattern1_responses = [0, 0]
    window = 20
    
    for time, pattern in pattern_times:
        for j, spiketrain in enumerate(output_spiketrains):
            response = any((time <= t <= time + window) for t in spiketrain)
            if pattern == 0 and response:
                pattern0_responses[j] += 1
            elif pattern == 1 and response:
                pattern1_responses[j] += 1
    
    # Calculate response rates
    pattern0_rates = [resp/pattern0_count if pattern0_count > 0 else 0 for resp in pattern0_responses]
    pattern1_rates = [resp/pattern1_count if pattern1_count > 0 else 0 for resp in pattern1_responses]
    
    # Calculate specialization score (higher is better)
    # Perfect specialization: neuron 0 responds only to pattern 0, neuron 1 only to pattern 1
    # or vice versa
    spec_score1 = (pattern0_rates[0] * (1 - pattern1_rates[0]) * 
                  (1 - pattern0_rates[1]) * pattern1_rates[1])
    # Alternative specialization: neuron 0 responds only to pattern 1, neuron 1 only to pattern 0
    spec_score2 = (pattern1_rates[0] * (1 - pattern0_rates[0]) * 
                  (1 - pattern1_rates[1]) * pattern0_rates[1])
    
    spec_score = max(spec_score1, spec_score2)
    
    return pattern0_rates, pattern1_rates, spec_score

def print_specialization_analysis(output_spiketrains, pattern_times):
    """Print analysis of neuron specialization."""
    pattern0_rates, pattern1_rates, spec_score = analyze_pattern_responses(output_spiketrains, pattern_times)
    
    print("\nSPECIALIZATION ANALYSIS:")
    print("  Output 0 response rates: Pattern 0: {:.0%}, Pattern 1: {:.0%}".format(
        pattern0_rates[0], pattern1_rates[0]))
    print("  Output 1 response rates: Pattern 0: {:.0%}, Pattern 1: {:.0%}".format(
        pattern0_rates[1], pattern1_rates[1]))
    
    print("\nSPECIALIZATION SCORE: {:.2%}".format(spec_score))
    print("  (1.00 = perfect specialization, 0.00 = no specialization)")

#return delta_delays the delays that should be applied in the next simulation.
def apply_delay_learning(current_delays, all_pairs, learning_rate, output_spike_counts, 
                         target_firing_rate=15, min_delay=1.0, max_delay=20.0, 
                         homeostasis_learning_rate=0.3):
    """Apply delay learning rule to the connections based on spike pairs."""
    new_delays = current_delays.copy()
    
    # Apply delay updates selectively based on the stop condition
    for j in range(2):  # For each output neuron
        # Check if learning should stop for this output neuron
        delays_to_neuron_j = current_delays[:, j]
        if check_stop_condition(delays_to_neuron_j):
            # Skip updates for all connections to this output neuron
            continue
        
        # Apply individual spike-pair updates to all connections for this output neuron
        for i in range(2):  # For each input neuron
            pairs = all_pairs.get((i,j), [])
            for pre_time, post_time in pairs:
                # Calculate the basic delay update
                delta_delay = delay_learning_rule(pre_time, post_time, current_delays[i][j])
                
                # Apply the learning rate and update the delay
                new_delays[i][j] += delta_delay * learning_rate
        
        # Apply homeostasis based on firing rate
        observed_firing_rate = output_spike_counts[j]
        # Calculate homeostasis factor K
        K = (target_firing_rate - observed_firing_rate) / target_firing_rate
        
        # Special case: if neuron is completely silent, apply aggressive homeostasis
        if observed_firing_rate == 0:
            print(f"  SILENT NEURON: Output {j} is not firing. Applying aggressive homeostasis.")
            # For output 0, decrease delay from input 0 (its preferred input)
            # For output 1, decrease delay from input 1 (its preferred input)
            preferred_input = j  # Each output neuron should prefer the input with same index
            new_delays[preferred_input][j] = max(min_delay, new_delays[preferred_input][j] - 3.0)
            print(f"    Aggressively decreased delay for input{preferred_input} to output{j} to {new_delays[preferred_input][j]:.2f}ms")
            
            # Also increase delay for non-preferred input to further discourage it
            non_preferred_input = 1 - j  # opposite of preferred input
            new_delays[non_preferred_input][j] = min(max_delay, new_delays[non_preferred_input][j] + 2.0)
            print(f"    Increased delay for input{non_preferred_input} to output{j} to {new_delays[non_preferred_input][j]:.2f}ms")
        else:
            # Apply normal homeostasis to both inputs, but in opposite directions to enforce specialization
            preferred_input = j
            non_preferred_input = 1 - j
            
            # Decrease delay for preferred input (to encourage firing)
            new_delays[preferred_input][j] -= homeostasis_learning_rate * K
            
            # For the non-preferred input, we do the opposite - increase delay when firing rate is low
            # and decrease when firing rate is high (opposite of normal homeostasis)
            non_preferred_K = -K  # Opposite direction
            new_delays[non_preferred_input][j] -= homeostasis_learning_rate * non_preferred_K * 2.0  # Stronger effect
    
    # Ensure delays stay within bounds
    new_delays = np.clip(new_delays, min_delay, max_delay)
    
    return new_delays

def run_simulation(current_delays, current_weights, simulation_time, inhibition_weight=0.0, num_presentations=5, pattern_interval=30):
    sim.setup(timestep=0.01)

    # Create populations and generate dataset
    input_pop, output_pop = create_populations(num_presentations, pattern_interval)
    
    # Get the pattern times from the dataset for analysis
    dataset = generate_dataset(num_presentations, pattern_interval)
    pattern_times = list(zip(dataset['pattern_start_times'], dataset['pattern_labels']))

    # Create connections
    connections, inhibition = create_connections(input_pop, output_pop, current_delays, current_weights, inhibition_weight)

    # Run simulation
    sim.run(simulation_time)
    
    # Get data
    input_data = input_pop.get_data().segments[-1]
    output_data = output_pop.get_data().segments[-1]
    
    input_spiketrains = input_data.spiketrains
    output_spiketrains = output_data.spiketrains

    # End simulation
    sim.end()
    
    return input_spiketrains, output_spiketrains, output_data, pattern_times


def initialize_simulation_parameters():
    """Initialize all simulation parameters and return them as a dictionary."""
    params = {
        'simulation_time': 600.0,  # Longer simulation to accommodate longer pattern interval
        'num_presentations': 5,
        'pattern_interval': 60,  # Further increased pattern interval
        'learning_rate': 0.8,  # Increased to amplify learning effects
        'c_threshold': 1.0,  # Lower threshold to continue learning longer
        'homeostasis_learning_rate': 0.4,  # Increased from 0.3
        'target_firing_rate': 15,
        'max_iterations': 70,
        'history_window': 5,  # Number of past iterations to consider for specialization tracking
        'inhibition_strength': 6.0,  # Strong PyNN inhibition (no custom inhibition)
    }
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize with extreme specialization bias
    # Output 0 should only respond to pattern 0 (input0 fires first)
    # Output 1 should only respond to pattern 1 (input1 fires first)
    params['initial_delays'] = np.array([
        [1.0, 20.0],  # input0 to output0 (shortest possible), output1 (longest possible)
        [20.0, 1.0]   # input1 to output0 (longest possible), output1 (shortest possible)
    ])

    # Fixed weights with extreme bias to enforce specialization
    params['initial_weights'] = np.array([
        [0.15, 0.001],  # input0 to output0 (very strong), output1 (very weak)
        [0.001, 0.15]   # input1 to output0 (very weak), output1 (very strong)
    ])
    
    return params


def update_neuron_activity_history(neuron_activity_history, pattern_response_history, 
                                   output_spiketrains, pattern_times):
    """Update the history of neuron activity and pattern responses."""
    pattern0_responses = [0, 0]
    pattern1_responses = [0, 0]
    window = 20
    
    # Count pattern responses
    for time, pattern in pattern_times:
        for j, spiketrain in enumerate(output_spiketrains):
            response = any((time <= t <= time + window) for t in spiketrain)
            if pattern == 0 and response:
                pattern0_responses[j] += 1
            elif pattern == 1 and response:
                pattern1_responses[j] += 1
    
    pattern0_count = sum(1 for _, p in pattern_times if p == 0)
    pattern1_count = sum(1 for _, p in pattern_times if p == 1)
    
    # Update history
    for j in range(2):
        # Add normalized spike count (activity level)
        total_patterns = len(pattern_times)
        if total_patterns > 0:
            neuron_activity_history[j].append(len(output_spiketrains[j]) / total_patterns)
        else:
            neuron_activity_history[j].append(0)
        
        # Add normalized pattern responses (1.0 = responded to all presentations of this pattern)
        if pattern0_count > 0:
            pattern_response_history[0][j].append(pattern0_responses[j] / pattern0_count)
        else:
            pattern_response_history[0][j].append(0)
            
        if pattern1_count > 0:
            pattern_response_history[1][j].append(pattern1_responses[j] / pattern1_count)
        else:
            pattern_response_history[1][j].append(0)
    
    return pattern0_responses, pattern1_responses


def determine_inhibition_strategy(pattern0_responses, pattern1_responses, output_spike_counts,
                           current_weights, inhibition_strength):
    """Determine the inhibition strategy based on neuron responses.
    
    Returns:
        Tuple of (inhibition_weight, current_weights)
    """
    # Use strong PyNN inhibition by default
    inhibition_weight = 6.0
    
    # Check if one neuron is completely silent
    if output_spike_counts[0] == 0 or output_spike_counts[1] == 0:
        silent_neuron = 0 if output_spike_counts[0] == 0 else 1
        
        print(f"  Output{silent_neuron} is completely silent! Boosting it and reducing inhibition.")
        
        # Increase the weight of connections to the silent neuron for its preferred input
        # The preferred input for output0 is input0, and for output1 is input1
        current_weights[silent_neuron][silent_neuron] *= 1.2  # Increase weight by 20%
        current_weights[silent_neuron][silent_neuron] = min(current_weights[silent_neuron][silent_neuron], 2.0)  # Cap at 2.0
        
        # Reduce PyNN inhibition to allow the silent neuron to recover
        inhibition_weight = 4.0
    
    # Use stronger inhibition if both neurons are responding to the same pattern
    elif (pattern0_responses[0] > 0 and pattern0_responses[1] > 0) or (pattern1_responses[0] > 0 and pattern1_responses[1] > 0):
        print("  Both neurons responding to the same pattern. Using strong inhibition to force competition.")
        inhibition_weight = 6.0
    
    print(f"  Using PyNN inhibition weight: {inhibition_weight:.1f}")
    return inhibition_weight, current_weights


def main():
    """Main function to run the simulation and learning process."""
    # Initialize simulation parameters
    params = initialize_simulation_parameters()
    
    # Unpack parameters for clarity
    simulation_time = params['simulation_time']
    num_presentations = params['num_presentations']
    pattern_interval = params['pattern_interval']
    learning_rate = params['learning_rate']
    c_threshold = params['c_threshold']
    homeostasis_learning_rate = params['homeostasis_learning_rate']
    target_firing_rate = params['target_firing_rate']
    max_iterations = params['max_iterations']
    history_window = params['history_window']
    inhibition_strength = params['inhibition_strength']
    initial_delays = params['initial_delays']
    initial_weights = params['initial_weights']
    
    # Run initial simulation with strong inhibition
    print(f"\nRunning initial simulation for {simulation_time}ms...")
    inhibition_weight = 6.0  # Start with strong PyNN inhibition (more biologically plausible)
    input_spiketrains, output_spiketrains, output_data, pattern_times = run_simulation(
        initial_delays, initial_weights, simulation_time, inhibition_weight, num_presentations, pattern_interval)
    
    print("Initial simulation results:")
    print(f"  Input spikes: {[len(st) for st in input_spiketrains]}")
    print(f"  Output spikes: {[len(st) for st in output_spiketrains]}")
    
    if sum(len(st) for st in output_spiketrains) == 0:
        print("ERROR: No output spikes in initial simulation. Please check neuron parameters or weights.")
        return
        print("ERROR: No output spikes in initial simulation. Please check neuron parameters or weights.")
        return
    
    # Initialize current state
    current_delays = initial_delays.copy()
    current_weights = initial_weights.copy()
    
    # Initialize history tracking
    delay_history = {
        'input0_output0': [current_delays[0][0]],
        'input0_output1': [current_delays[0][1]],
        'input1_output0': [current_delays[1][0]],
        'input1_output1': [current_delays[1][1]]
    }
    
    # Initialize neuron activity history for custom inhibition
    neuron_activity_history = {0: [], 1: []}
    pattern_response_history = {0: {0: [], 1: []}, 1: {0: [], 1: []}}
    
    # Save final results
    final_input_spiketrains = input_spiketrains
    final_output_spiketrains = output_spiketrains
    final_output_data = output_data
    final_pattern_times = pattern_times
    
    print("\nStarting delay learning process...")
    
    # Main learning loop
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration+1}:")
        print("  Current delays (ms):")
        print(f"    input0 to output0: {current_delays[0][0]:.2f}, input0 to output1: {current_delays[0][1]:.2f}")
        print(f"    input1 to output0: {current_delays[1][0]:.2f}, input1 to output1: {current_delays[1][1]:.2f}")
        
        # Check stop condition
        stop_learning = True
        for j in range(2):
            delays_to_neuron_j = current_delays[:, j]
            if check_stop_condition(delays_to_neuron_j, c_threshold):
                print(f"  STOP CONDITION: Delays to output {j} have reached threshold {c_threshold}")
                print(f"  Delays to output {j}: {delays_to_neuron_j}")
            else:
                stop_learning = False
        
        if stop_learning:
            print("  All output neurons have reached stop condition. Stopping learning.")
            break
        
        # Get output spike counts
        output_spike_counts = [len(st) for st in output_spiketrains]
        
        # Update neuron activity history
        pattern0_responses, pattern1_responses = update_neuron_activity_history(
            neuron_activity_history, pattern_response_history, output_spiketrains, pattern_times)
        
        # Determine inhibition strategy (strong PyNN inhibition only)
        inhibition_weight, current_weights = determine_inhibition_strategy(
            pattern0_responses, pattern1_responses, output_spike_counts, 
            current_weights, inhibition_strength)
        
        # Collect simulation data
        all_pairs, pair_counts = collect_simulation_data(input_spiketrains, output_spiketrains, current_delays)
        
        if np.any(pair_counts > 0):
            # Print debug information about spike pairs
            print_debug_spike_pairs(all_pairs, current_delays, learning_rate)
            
            # Apply delay learning rule
            current_delays = apply_delay_learning(
                current_delays, all_pairs, learning_rate, output_spike_counts,
                target_firing_rate=target_firing_rate, homeostasis_learning_rate=homeostasis_learning_rate
            )
            
            # Update delay history
            delay_history['input0_output0'].append(current_delays[0][0])
            delay_history['input0_output1'].append(current_delays[0][1])
            delay_history['input1_output0'].append(current_delays[1][0])
            delay_history['input1_output1'].append(current_delays[1][1])
            
            # Print delay changes
            print_delay_changes(current_delays, delay_history)
            
            # Run simulation with updated delays
            input_spiketrains, output_spiketrains, output_data, pattern_times = run_simulation(
                current_delays, current_weights, simulation_time, inhibition_weight, 
                num_presentations, pattern_interval)
            
            # Update final results
            final_input_spiketrains = input_spiketrains
            final_output_spiketrains = output_spiketrains
            final_output_data = output_data
            final_pattern_times = pattern_times
            
            print(f"  New simulation output spikes: {[len(st) for st in output_spiketrains]}")
            print(f"  Homeostasis effect: Target firing rate = {target_firing_rate}, Observed firing rates = {[len(st) for st in output_spiketrains]}")
        else:
            print("  No spike pairs found for updates, stopping learning")
            break
    
    # Print final results
    print_final_results(initial_delays, current_delays, delay_history, 
                       final_input_spiketrains, final_output_spiketrains, final_pattern_times)
    
    # Visualize results
    visualize_results(delay_history, final_input_spiketrains, final_output_spiketrains, 
                     final_output_data, final_pattern_times, c_threshold, simulation_time,
                     neuron_activity_history, pattern_response_history)
    
    # Print specialization analysis
    print_specialization_analysis(final_output_spiketrains, final_pattern_times)

def visualize_results(delay_history, input_spiketrains, output_spiketrains, 
                     output_data, pattern_times, c_threshold, simulation_time=300.0,
                     neuron_activity_history=None, pattern_response_history=None):
    plt.figure(figsize=(15, 15))  # Made figure taller for the additional plot
    
    plt.subplot(4, 2, 1)  # Changed to 4 rows instead of 3
    plt.plot(delay_history['input0_output0'], 'r-o', label='input0 to output0')
    plt.plot(delay_history['input0_output1'], 'g-o', label='input0 to output1')
    plt.plot(delay_history['input1_output0'], 'b-o', label='input1 to output0')
    plt.plot(delay_history['input1_output1'], 'c-o', label='input1 to output1')
    plt.axhline(y=c_threshold, color='r', linestyle='--', label='Stop threshold')
    plt.xlabel('Iteration')
    plt.ylabel('Delay (ms)')
    plt.title('Delay Learning Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add custom inhibition history plot if available
    if neuron_activity_history and len(neuron_activity_history[0]) > 0:
        plt.subplot(4, 2, 2)
        iterations = range(len(neuron_activity_history[0]))
        plt.plot(iterations, neuron_activity_history[0], 'b-o', label='Output 0 Activity')
        plt.plot(iterations, neuron_activity_history[1], 'r-o', label='Output 1 Activity')
        
        # Also plot pattern response history if available
        if pattern_response_history:
            plt.plot(iterations, pattern_response_history[0][0], 'b--', label='Output 0 → Pattern 0')
            plt.plot(iterations, pattern_response_history[1][0], 'b:', label='Output 0 → Pattern 1')
            plt.plot(iterations, pattern_response_history[0][1], 'r--', label='Output 1 → Pattern 0')
            plt.plot(iterations, pattern_response_history[1][1], 'r:', label='Output 1 → Pattern 1')
        
        plt.xlabel('Iteration')
        plt.ylabel('Activity / Response Rate')
        plt.title('Neuron Activity and Pattern Response History')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    ax1 = plt.subplot(4, 2, 3)
    for i, spiketrain in enumerate(input_spiketrains):
        if len(spiketrain) > 0:
            plt.scatter(spiketrain, np.full(len(spiketrain), i), 
                       marker='|', s=100, label='Input {}'.format(i+1))
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Neuron')
    plt.title('Input Spike Trains')
    plt.legend()
    plt.grid(True, alpha=0.3)
   
    ymin, ymax = ax1.get_ylim()
    for time, pattern in pattern_times:
        color = 'blue' if pattern == 1 else 'green'
        plt.axvline(x=time, color=color, alpha=0.3, linestyle='--')
    
    ax2 = plt.subplot(4, 2, 4)
    for i, spiketrain in enumerate(output_spiketrains):
        if len(spiketrain) > 0:
            plt.scatter(spiketrain, np.full(len(spiketrain), i), 
                       marker='|', s=100, label='Output {}'.format(i+1))
    plt.xlabel('Time (ms)')
    plt.ylabel('Output Neuron')
    plt.title('Output Spike Trains')
    plt.legend()
    plt.grid(True, alpha=0.3)
   
    ymin, ymax = ax2.get_ylim()
    for time, pattern in pattern_times:
        color = 'blue' if pattern == 1 else 'green'
        plt.axvline(x=time, color=color, alpha=0.3, linestyle='--')
   
    plt.subplot(4, 2, (5, 6))
    # Get membrane potentials for both output neurons
    analog_signals = output_data.analogsignals
    
    if len(analog_signals) > 0:
        # The first analog signal contains membrane potentials for all neurons
        signal = analog_signals[0]
        
        # Plot the membrane potential of output neuron 0 in blue
        plt.plot(signal.times, signal[:, 0], 'b-', linewidth=1.0, label='Output 0')
        
        # Plot the membrane potential of output neuron 1 in red
        if signal.shape[1] > 1:  # Make sure there's data for the second neuron
            plt.plot(signal.times, signal[:, 1], 'r-', linewidth=1.0, label='Output 1')
    
    neuron_params = get_neuron_params()
    plt.axhline(y=neuron_params['v_thresh'], color='k', linestyle='--', 
                label='Threshold ({:.0f}mV)'.format(neuron_params['v_thresh']))
    
    # Add markers for resting potential
    plt.axhline(y=neuron_params['v_rest'], color='gray', linestyle=':', 
                label='Rest ({:.0f}mV)'.format(neuron_params['v_rest']))
    
    # Improve visibility
    plt.grid(True, alpha=0.3)
    plt.xlim(0, simulation_time)
    plt.legend(loc='upper right')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Output Neuron Membrane Potentials')
    plt.legend()
    plt.grid(True, alpha=0.3)
 
    plt.subplot(4, 2, (7, 8))
    p1_responses = [0, 0] 
    p2_responses = [0, 0] 
 
    window = 20 
    
    for time, pattern in pattern_times:
        for j, spiketrain in enumerate(output_spiketrains):
            response = any((time <= t <= time + window) for t in spiketrain)
            if pattern == 1 and response:
                p1_responses[j] += 1
            elif pattern == 0 and response:
                p2_responses[j] += 1
    
    p1_count = sum(1 for _, p in pattern_times if p == 1)
    p2_count = sum(1 for _, p in pattern_times if p == 0)
    
    labels = ['Pattern 1', 'Pattern 0']
    output0_responses = [p1_responses[0]/max(1, p1_count), p2_responses[0]/max(1, p2_count)]
    output1_responses = [p1_responses[1]/max(1, p1_count), p2_responses[1]/max(1, p2_count)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, output0_responses, width, label='Output 0', color='blue')
    plt.bar(x + width/2, output1_responses, width, label='Output 1', color='red')
    
    plt.ylabel('Response Rate')
    plt.title('Pattern Recognition Performance')
    plt.xticks(x, labels)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(output0_responses):
        plt.text(i - width/2, v + 0.05, '{:.0%}'.format(v), ha='center')
    for i, v in enumerate(output1_responses):
        plt.text(i + width/2, v + 0.05, '{:.0%}'.format(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('delay_learning_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()