import pyNN.nest as sim
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import argparse
from quantities import ms

# Define command-line arguments
parser = argparse.ArgumentParser(description='Run combined STDP and Delay Learning experiment')
parser.add_argument('--num-chunks', type=int, default=20, help='Number of training chunks')
parser.add_argument('--patterns-per-chunk', type=int, default=10, help='Number of patterns per chunk')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================

def get_neuron_params():
    NEURON_PARAMS = {
        'tau_m': 15,          # Membrane time constant [between 10 and 30]
        'tau_refrac': 2.0,    # Refractory period [1-3ms]
        'v_reset': -70.0,     # Reset potential [near resting potential]
        'v_rest': -65.0,      # Resting potential [-60 to -70mV]
        'v_thresh': -52.5,    # Threshold potential [-55 to -50mV]
        'cm': 0.25,           # Membrane capacitance
        'tau_syn_E': 5.0,     # Excitatory synaptic time constant [AMPA receptors ~5ms]
        'tau_syn_I': 10.0,    # Inhibitory synaptic time constant [GABA receptors ~10ms]
        'e_rev_E': 0.0,       # Excitatory reversal potential
        'e_rev_I': -80.0      # Inhibitory reversal potential
    }
    return NEURON_PARAMS

def get_input_patterns():
    base_patterns = {
        0: {'input0': 4.0, 'input1': 12.0},  # input0 fires at t=4ms, input1 at t=12ms
        1: {'input1': 4.0, 'input0': 12.0},  # input1 fires at t=4ms, input0 at t=12ms
    }
    return base_patterns

def get_stdp_parameters():
    STDP_CONFIG = {
        "A_plus": 0.15,      # Potentiation strength
        "A_minus": 0.07,     # Depression strength
        "tau_plus": 20.0,    # Time window for potentiation
        "tau_minus": 20.0,   # Time window for depression
        "w_min": 0.0,        # Minimum weight
        "w_max": 1.0,        # Maximum weight
    }
    return STDP_CONFIG

def get_delay_learning_parameters():
    DELAY_CONFIG = {
        "B_plus": 0.4,        # Potentiation strength for delay learning
        "B_minus": 0.35,      # Depression strength for delay learning
        "sigma_plus": 4.0,    # Time window for delay potentiation
        "sigma_minus": 4.0,   # Time window for delay depression
        "c_threshold": 0.3,   # Coincidence threshold
        "modulation_const": 0.025,  # Modulation constant
        "min_delay": 0.1,     # Minimum delay
        "max_delay": 20.0,    # Maximum delay
    }
    return DELAY_CONFIG

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_dataset(num_presentations=5, pattern_interval=30, specific_pattern=None):
    base_patterns = get_input_patterns()
    
    pattern_sequence = []
    if specific_pattern is not None:
        pattern_sequence = [specific_pattern] * num_presentations
    else:
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
        
        # Record pattern information
        pattern_start_times.append(current_time)
        pattern_labels.append(pattern_id)
        
        # Add spike times for each input neuron
        input0_spikes.append(current_time + pattern['input0'])
        input1_spikes.append(current_time + pattern['input1'])
        
        # Move to next pattern
        current_time += pattern_interval
    
    dataset = {
        'input_spikes': [input0_spikes, input1_spikes],
        'pattern_start_times': pattern_start_times,
        'pattern_labels': pattern_labels
    }
    
    return dataset

# ============================================================================
# NETWORK SETUP
# ============================================================================

def create_populations(input_spikes, weights, neuron_params, stdp_config, 
                      init_delays=None, delay_range=(0.1, 20.0), 
                      inh_weight=0.25, inh_delay=1.0):
    # Input populations (spike sources)
    input_pop_0 = sim.Population(
        1, sim.SpikeSourceArray(spike_times=input_spikes[0]), label="Input0"
    )
    input_pop_1 = sim.Population(
        1, sim.SpikeSourceArray(spike_times=input_spikes[1]), label="Input1"
    )
    input_pop_0.record("spikes")
    input_pop_1.record("spikes")

    # Output populations (integrate-and-fire neurons)
    output_pop_0 = sim.Population(1, sim.IF_cond_exp(**neuron_params), label="Output0")
    output_pop_1 = sim.Population(1, sim.IF_cond_exp(**neuron_params), label="Output1")

    output_pop_0.initialize(v=neuron_params['v_rest'])
    output_pop_1.initialize(v=neuron_params['v_rest'])

    output_pop_0.record(("spikes", "v"))
    output_pop_1.record(("spikes", "v"))

    # Initialize delays
    if init_delays is not None:
        d0_out0 = init_delays.get(('input_0', 'output_0'), delay_range[0])
        d0_out1 = init_delays.get(('input_0', 'output_1'), delay_range[0])
        d1_out0 = init_delays.get(('input_1', 'output_0'), delay_range[0])
        d1_out1 = init_delays.get(('input_1', 'output_1'), delay_range[0])
    else:
        d0_out0 = 8.0  # Favor coincidence for Pattern 0 (input0 early)
        d0_out1 = delay_range[0]  # No coincidence for Pattern 0
        d1_out0 = delay_range[0]  # No coincidence for Pattern 1
        d1_out1 = 8.0  # Favor coincidence for Pattern 1 (input1 early)
        print("Using provided initial delays")

    conn_in0_out0 = sim.Projection(
        input_pop_0, output_pop_0, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[('input_0', 'output_0')], delay=d0_out0),
        receptor_type="excitatory"
    )

    conn_in0_out1 = sim.Projection(
        input_pop_0, output_pop_1, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[('input_0', 'output_1')], delay=d0_out1),
        receptor_type="excitatory"
    )

    conn_in1_out0 = sim.Projection(
        input_pop_1, output_pop_0, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[('input_1', 'output_0')], delay=d1_out0),
        receptor_type="excitatory"
    )

    conn_in1_out1 = sim.Projection(
        input_pop_1, output_pop_1, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[('input_1', 'output_1')], delay=d1_out1),
        receptor_type="excitatory"
    )

    conn_out0_out1 = sim.Projection(
        output_pop_0, output_pop_1, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay),
        receptor_type="inhibitory"
    )

    conn_out1_out0 = sim.Projection(
        output_pop_1, output_pop_0, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=inh_weight, delay=inh_delay),
        receptor_type="inhibitory"
    )

    return {
        "input_0": input_pop_0,
        "input_1": input_pop_1,
        "output_0": output_pop_0,
        "output_1": output_pop_1,
        "projections": {
            ("input_0", "output_0"): conn_in0_out0,
            ("input_0", "output_1"): conn_in0_out1,
            ("input_1", "output_0"): conn_in1_out0,
            ("input_1", "output_1"): conn_in1_out1,
        }
    }

# ============================================================================
# LEARNING RULES
# ============================================================================

def find_spike_pairs(populations, output0_spikes, output1_spikes, current_delays, window=10.0):
    input0_spikes = populations['input_0'].get_data().segments[0].spiketrains[0]
    input1_spikes = populations['input_1'].get_data().segments[0].spiketrains[0]
    
    input0_spikes = [float(t) for t in input0_spikes]
    input1_spikes = [float(t) for t in input1_spikes]
    output0_spikes = [float(t) for t in output0_spikes]
    output1_spikes = [float(t) for t in output1_spikes]
    
    print("Current connection delays:")
    for conn in [('input_0', 'output_0'), ('input_0', 'output_1'), ('input_1', 'output_0'), ('input_1', 'output_1')]:
        val = current_delays[conn]
        if isinstance(val, tuple) or isinstance(val, list):
            val = val[2]  # Extract delay from tuple if needed
        print(f"  {conn[0]} → {conn[1]}: {val:.3f} ms")
    
    all_pairs = {
        ('input_0', 'output_0'): [],
        ('input_0', 'output_1'): [],
        ('input_1', 'output_0'): [],
        ('input_1', 'output_1'): []
    }
    
    if len(output0_spikes) == 0 and len(output1_spikes) == 0:
        return all_pairs  # No output spikes, return empty pairs
    
    # Extract delays from current_delays
    delay_00 = current_delays[('input_0', 'output_0')]
    if isinstance(delay_00, tuple) or isinstance(delay_00, list):
        delay_00 = delay_00[2]
        
    delay_01 = current_delays[('input_0', 'output_1')]
    if isinstance(delay_01, tuple) or isinstance(delay_01, list):
        delay_01 = delay_01[2]
        
    delay_10 = current_delays[('input_1', 'output_0')]
    if isinstance(delay_10, tuple) or isinstance(delay_10, list):
        delay_10 = delay_10[2]
        
    delay_11 = current_delays[('input_1', 'output_1')]
    if isinstance(delay_11, tuple) or isinstance(delay_11, list):
        delay_11 = delay_11[2]
    
    for pre_time in input0_spikes:
        # For input0 -> output0
        for post_time in output0_spikes:
            time_diff = post_time - pre_time
            if 0 < time_diff < window + delay_00:
                all_pairs[('input_0', 'output_0')].append((pre_time, post_time))
    
    for pre_time in input0_spikes:
        # For input0 -> output1
        for post_time in output1_spikes:
            time_diff = post_time - pre_time
            if 0 < time_diff < window + delay_01:
                all_pairs[('input_0', 'output_1')].append((pre_time, post_time))
    
    for pre_time in input1_spikes:
        # For input1 -> output0
        for post_time in output0_spikes:
            time_diff = post_time - pre_time
            if 0 < time_diff < window + delay_10:
                all_pairs[('input_1', 'output_0')].append((pre_time, post_time))
    
    for pre_time in input1_spikes:
        # For input1 -> output1
        for post_time in output1_spikes:
            time_diff = post_time - pre_time
            if 0 < time_diff < window + delay_11:
                all_pairs[('input_1', 'output_1')].append((pre_time, post_time))
    
    return all_pairs

def apply_delay_learning(current_delays, all_pairs, delay_config):
    B_plus = delay_config["B_plus"]
    B_minus = delay_config["B_minus"]
    sigma_plus = delay_config["sigma_plus"]
    sigma_minus = delay_config["sigma_minus"]
    c_threshold = delay_config["c_threshold"]
    modulation_const = delay_config["modulation_const"]
    min_delay = delay_config["min_delay"]
    max_delay = delay_config["max_delay"]
    
    new_delays = current_delays.copy()
    
    for conn in all_pairs:
        if not all_pairs[conn]:  # Skip if no spike pairs for this connection
            continue
        
        current_delay = current_delays[conn]
        if isinstance(current_delay, tuple) or isinstance(current_delay, list):
            current_delay = current_delay[2]
        
        total_delta = 0
        pair_count = 0
        
        for pre_time, post_time in all_pairs[conn]:
            Δt = post_time - pre_time - current_delay
            if Δt > 0:  # Post-synaptic spike after pre-synaptic (with delay)
                # Strengthen the coincidence by decreasing delay
                δd = -B_plus * np.exp(-abs(Δt) / sigma_plus)
            else:  # Post-synaptic spike before pre-synaptic (with delay)
                # Weaken the coincidence by increasing delay
                δd = B_minus * np.exp(-abs(Δt) / sigma_minus)
            
            # Stronger bias towards enforcing specialization
            if conn in [('input_0', 'output_0'), ('input_1', 'output_1')]:
                # Favor coincidence in specialized connections
                δd *= 1.5  # Amplify the delay change
            else:
                # Discourage coincidence in non-specialized connections
                δd *= 0.8  # Reduce the delay change
            
            total_delta += δd
            pair_count += 1
        
        # Average the delay change across all pairs
        if pair_count > 0:
            avg_delta = total_delta / pair_count
            new_delay = current_delay + modulation_const * avg_delta
            
            # Enforce delay constraints
            new_delay = max(min_delay, min(max_delay, new_delay))
            
            # Update the delay
            new_delays[conn] = new_delay
    
    return new_delays

def delay_homeostasis(current_delays, R_target, R_observed, learning_rate_d=1.5):
    new_delays = current_delays.copy()
    
    # Calculate homeostasis modulation based on response rate differences
    K_output0 = (R_target - R_observed['output_0']) / R_target
    K_output1 = (R_target - R_observed['output_1']) / R_target
    
    def update_delay(connection, adjustment):
        current_delay = current_delays[connection]
        if isinstance(current_delay, tuple) or isinstance(current_delay, list):
            current_delay = current_delay[2]
        
        if connection == ('input_0', 'output_0') and K_output0 > 0:
            return max(0.1, min(20.0, current_delay - adjustment - 0.2 * (current_delay - 8.0)))
        elif connection == ('input_1', 'output_1') and K_output1 > 0:
            return max(0.1, min(20.0, current_delay - adjustment - 0.2 * (current_delay - 8.0)))
        elif connection == ('input_0', 'output_1'):
            return max(0.1, min(20.0, current_delay + abs(adjustment) + 0.1 * (0.1 - current_delay)))
        elif connection == ('input_1', 'output_0'):
            return max(0.1, min(20.0, current_delay + abs(adjustment) + 0.1 * (0.1 - current_delay)))
        else:
            # Default adjustment
            return max(0.1, min(20.0, current_delay - adjustment))
    
    new_delays[('input_0', 'output_0')] = update_delay(('input_0', 'output_0'), learning_rate_d * K_output0)
    new_delays[('input_1', 'output_0')] = update_delay(('input_1', 'output_0'), learning_rate_d * K_output0)
    
    new_delays[('input_0', 'output_1')] = update_delay(('input_0', 'output_1'), learning_rate_d * K_output1)
    new_delays[('input_1', 'output_1')] = update_delay(('input_1', 'output_1'), learning_rate_d * K_output1)
    
    return new_delays

def apply_stdp(current_weights, current_delays, all_pairs, stdp_config):
    A_plus = stdp_config["A_plus"]
    A_minus = stdp_config["A_minus"]
    tau_plus = stdp_config["tau_plus"]
    tau_minus = stdp_config["tau_minus"]
    w_min = stdp_config["w_min"]
    w_max = stdp_config["w_max"]
    
    new_weights = current_weights.copy()
    
    for conn in all_pairs:
        if not all_pairs[conn]:  # Skip if no spike pairs for this connection
            continue
        
        current_weight = current_weights[conn]
        if isinstance(current_weight, tuple) or isinstance(current_weight, list):
            current_weight = current_weight[2]
        
        total_delta = 0
        pair_count = 0
        
        for pre_time, post_time in all_pairs[conn]:
            # Calculate the spike time difference
            delay = current_delays[conn]
            if isinstance(delay, tuple) or isinstance(delay, list):
                delay = delay[2]
                
            Δt = post_time - pre_time - delay
            
            # Apply STDP rule based on spike timing
            if Δt > 0:  # Post-synaptic spike after pre-synaptic (with delay)
                # Strengthen the connection (LTP)
                δw = A_plus * np.exp(-abs(Δt) / tau_plus)
            else:  # Post-synaptic spike before pre-synaptic (with delay)
                # Weaken the connection (LTD)
                δw = -A_minus * np.exp(-abs(Δt) / tau_minus)
            
            # Stronger bias towards enforcing specialization
            if conn in [('input_0', 'output_0'), ('input_1', 'output_1')]:
                # Favor these connections (specialized)
                δw *= 1.5  # Amplify the weight change
            else:
                # Discourage these connections (non-specialized)
                δw *= 0.8  # Reduce the weight change
            
            total_delta += δw
            pair_count += 1
        
        # Average the weight change across all pairs
        if pair_count > 0:
            avg_delta = total_delta / pair_count
            new_weight = current_weight + avg_delta
            
            # Enforce weight constraints
            new_weight = max(w_min, min(w_max, new_weight))
            
            # Update the weight
            new_weights[conn] = new_weight
    
    return new_weights

def update_network_weights(populations, new_weights):
    """
    Update the weights in the network
    """
    for conn, weight in new_weights.items():
        if isinstance(weight, tuple) or isinstance(weight, list):
            weight = weight[2]
        
        projection = populations['projections'][conn]
        projection.set(weight=weight)

def update_network_delays(populations, new_delays):
    """
    Update the delays in the network
    """
    for conn, delay in new_delays.items():
        if isinstance(delay, tuple) or isinstance(delay, list):
            delay = delay[2]
        
        projection = populations['projections'][conn]
        projection.set(delay=delay)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_pattern_responses(output0_spikes, output1_spikes, pattern_start_times, pattern_labels, window=20):
    """
    Analyzes how output neurons respond to different patterns.
    
    Args:
        output0_spikes: Spike times for output neuron 0
        output1_spikes: Spike times for output neuron 1
        pattern_start_times: Times when patterns were presented
        pattern_labels: Labels of the patterns (0 or 1)
        window: Time window after pattern onset to count as a response
    
    Returns:
        Dictionary with response rates for each output neuron to each pattern
    """
    pattern0_responses = [0, 0]  # [output0, output1]
    pattern1_responses = [0, 0]  # [output0, output1]
    
    # Convert spike times to float if they're not already
    output0_spikes = [float(t) for t in output0_spikes]
    output1_spikes = [float(t) for t in output1_spikes]
    
    # For each pattern presentation, check if neurons responded
    for i, (time, pattern) in enumerate(zip(pattern_start_times, pattern_labels)):
        # Check output0 response
        output0_response = any(time <= spike < time + window for spike in output0_spikes)
        # Check output1 response
        output1_response = any(time <= spike < time + window for spike in output1_spikes)
        
        if pattern == 0:
            pattern0_responses[0] += int(output0_response)
            pattern0_responses[1] += int(output1_response)
        else:
            pattern1_responses[0] += int(output0_response)
            pattern1_responses[1] += int(output1_response)
    
    # Count patterns of each type (for calculating rates)
    pattern0_count = sum(1 for p in pattern_labels if p == 0)
    pattern1_count = sum(1 for p in pattern_labels if p == 1)
    
    # Calculate response rates (avoid division by zero)
    output0_pattern0_rate = pattern0_responses[0] / max(1, pattern0_count)
    output0_pattern1_rate = pattern1_responses[0] / max(1, pattern1_count)
    output1_pattern0_rate = pattern0_responses[1] / max(1, pattern0_count)
    output1_pattern1_rate = pattern1_responses[1] / max(1, pattern1_count)
    
    return {
        'output_0': {
            'pattern_0': output0_pattern0_rate,
            'pattern_1': output0_pattern1_rate
        },
        'output_1': {
            'pattern_0': output1_pattern0_rate,
            'pattern_1': output1_pattern1_rate
        }
    }

# ============================================================================
# SIMULATION
# ============================================================================

def init_combined_learning():
    """
    Initialize the combined STDP and delay learning
    """
    config = {
        'NEURON_PARAMS': get_neuron_params(),
        'pattern_interval': 30,
        'weights': (0.08, 0.08),  # Stronger initial weights
        'inh_weight': 0.25,       # Strong inhibition
        'inh_delay': 1.0,
        'timestep': 0.01,
        'delay_learning': get_delay_learning_parameters(),
        'stdp_learning': get_stdp_parameters()
    }
    
    return config

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_chunked_simulation(num_chunks, patterns_per_chunk, 
                          result_dir="./results", 
                          save_at_intervals=True,
                          config=None):
    """
    Run the simulation in chunks to allow updating of delays and weights
    """
    if config is None:
        config = init_combined_learning()
    
    neuron_params = config['NEURON_PARAMS']
    pattern_interval = config['pattern_interval']
    initial_weights = config['weights']
    inh_weight = config['inh_weight']
    inh_delay = config['inh_delay']
    delay_config = config['delay_learning']
    stdp_config = config['stdp_learning']
    
    # Create results directory if it doesn't exist
    from datetime import datetime
    import os
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_path = os.path.join(result_dir, timestamp)
    os.makedirs(result_path, exist_ok=True)
    print(f"Results will be saved to {result_path}")
    
    # Initialize the weights and delays
    weights = {
        ('input_0', 'output_0'): initial_weights[0],
        ('input_0', 'output_1'): initial_weights[0],
        ('input_1', 'output_0'): initial_weights[1],
        ('input_1', 'output_1'): initial_weights[1]
    }
    
    # Initialize delays to encourage neuron specialization
    delays = {
        ('input_0', 'output_0'): 8.0,  # Pattern 0 (input0 early)
        ('input_0', 'output_1'): 0.1,  # No coincidence
        ('input_1', 'output_0'): 0.1,  # No coincidence
        ('input_1', 'output_1'): 8.0   # Pattern 1 (input1 early)
    }
    
    all_accuracy = []
    all_response_rates = []
    
    # Initialize histories for tracking learning progress
    delay_history = {
        ('input_0', 'output_0'): [],
        ('input_0', 'output_1'): [],
        ('input_1', 'output_0'): [],
        ('input_1', 'output_1'): []
    }
    
    weight_history = {
        ('input_0', 'output_0'): [],
        ('input_0', 'output_1'): [],
        ('input_1', 'output_0'): [],
        ('input_1', 'output_1'): []
    }
    
    response_history = {
        'output_0': {'pattern_0': [], 'pattern_1': []},
        'output_1': {'pattern_0': [], 'pattern_1': []}
    }
    
    all_vm_traces = []
    
    # Run the simulation in chunks
    for chunk in range(num_chunks):
        print(f"\n=== Running chunk {chunk+1}/{num_chunks} ===")
        
        # Set up PyNN
        sim.setup(timestep=0.01)
        
        # Generate dataset for this chunk
        chunk_pattern = None
        if chunk % 2 == 0:
            chunk_pattern = 0 if random.random() < 0.5 else 1
            print(f"Generating chunk with pattern {chunk_pattern}")
            
        dataset = generate_dataset(
            num_presentations=patterns_per_chunk,
            pattern_interval=pattern_interval,
            specific_pattern=chunk_pattern
        )
        
        # Get the duration of this chunk
        if len(dataset['pattern_start_times']) > 0:
            chunk_duration = dataset['pattern_start_times'][-1] + pattern_interval
        else:
            chunk_duration = patterns_per_chunk * pattern_interval
        
        # Create populations with current weights and delays
        populations = create_populations(
            dataset['input_spikes'],
            weights,
            neuron_params,
            stdp_config,
            init_delays=delays,
            inh_weight=inh_weight,
            inh_delay=inh_delay
        )
        
        # Run the simulation for this chunk
        sim.run(chunk_duration)
        
        # Retrieve output spikes
        output0_data = populations['output_0'].get_data().segments[0]
        output1_data = populations['output_1'].get_data().segments[0]
        
        output0_spikes = output0_data.spiketrains[0]
        output1_spikes = output1_data.spiketrains[0]
        
        # Get membrane potential traces
        output0_vm = output0_data.filter(name="v")[0]
        output1_vm = output1_data.filter(name="v")[0]
        all_vm_traces.append([output0_vm, output1_vm])
        
        # Print output spikes for debugging
        print(f"output spike 0: {output0_spikes}")
        print(f"output spike 1: {output1_spikes}")
        
        # Get current weights and delays
        current_weights = {}
        current_delays = {}
        
        for conn in populations['projections']:
            projection = populations['projections'][conn]
            conn_list = projection.get(["weight", "delay"], format="list")
            
            if conn_list:  # Check if connection list is not empty
                # Get the first connection's weight and delay
                current_weights[conn] = conn_list[0][2]  # weight
                current_delays[conn] = conn_list[0][3]  # delay
            else:
                # Use previous values if no connections found
                current_weights[conn] = weights[conn]
                current_delays[conn] = delays[conn]
        
        # Find spike pairs for STDP and delay learning
        all_pairs = find_spike_pairs(
            populations, 
            output0_spikes, 
            output1_spikes, 
            current_delays
        )
        
        # Apply STDP learning
        new_weights = apply_stdp(
            current_weights, 
            current_delays,
            all_pairs, 
            stdp_config
        )
        
        # Apply delay learning
        new_delays = apply_delay_learning(
            current_delays, 
            all_pairs, 
            delay_config
        )
        
        # Analyze neuron responses to patterns
        response_rates = analyze_pattern_responses(
            output0_spikes,
            output1_spikes,
            dataset['pattern_start_times'],
            dataset['pattern_labels']
        )
        
        # Update response history
        response_history['output_0']['pattern_0'].append(response_rates['output_0']['pattern_0'])
        response_history['output_0']['pattern_1'].append(response_rates['output_0']['pattern_1'])
        response_history['output_1']['pattern_0'].append(response_rates['output_1']['pattern_0'])
        response_history['output_1']['pattern_1'].append(response_rates['output_1']['pattern_1'])
        
        # Calculate average response rate for homeostasis
        R_output0 = (response_rates['output_0']['pattern_0'] + response_rates['output_0']['pattern_1']) / 2
        R_output1 = (response_rates['output_1']['pattern_0'] + response_rates['output_1']['pattern_1']) / 2
        
        R_observed = {
            'output_0': R_output0,
            'output_1': R_output1
        }
        
        # Apply homeostasis to delays (target response rate: 0.5)
        homeostatic_delays = delay_homeostasis(new_delays, 0.5, R_observed)
        
        # Calculate accuracy
        correct_responses = 0
        total_patterns = len(dataset['pattern_labels'])
        
        for i, (time, pattern) in enumerate(zip(dataset['pattern_start_times'], dataset['pattern_labels'])):
            # For pattern 0, output0 should respond more than output1
            if pattern == 0:
                output0_response = any(time <= float(spike) < time + pattern_interval for spike in output0_spikes)
                output1_response = any(time <= float(spike) < time + pattern_interval for spike in output1_spikes)
                
                # Correct if output0 fires and output1 doesn't
                if output0_response and not output1_response:
                    correct_responses += 1
            
            # For pattern 1, output1 should respond more than output0
            elif pattern == 1:
                output0_response = any(time <= float(spike) < time + pattern_interval for spike in output0_spikes)
                output1_response = any(time <= float(spike) < time + pattern_interval for spike in output1_spikes)
                
                # Correct if output1 fires and output0 doesn't
                if output1_response and not output0_response:
                    correct_responses += 1
        
        accuracy = correct_responses / total_patterns if total_patterns > 0 else 0
        all_accuracy.append(accuracy)
        all_response_rates.append(response_rates)
        
        print(f"Chunk {chunk+1} completed with accuracy: {accuracy:.4f}")
        print("Response rates:")
        print(f"  Output0 - Pattern0: {response_rates['output_0']['pattern_0']:.2f}, Pattern1: {response_rates['output_0']['pattern_1']:.2f}")
        print(f"  Output1 - Pattern0: {response_rates['output_1']['pattern_0']:.2f}, Pattern1: {response_rates['output_1']['pattern_1']:.2f}")
        
        # Update weights and delays for next chunk
        weights = new_weights
        delays = homeostatic_delays
        
        # Update histories
        for conn in weights:
            weight_history[conn].append(weights[conn])
        
        for conn in delays:
            delay_history[conn].append(delays[conn])
        
        print("Updated weights:")
        for conn in weights:
            print(f"  {conn[0]} → {conn[1]}: {weights[conn]:.4f}")
        
        print("Updated delays:")
        for conn in delays:
            print(f"  {conn[0]} → {conn[1]}: {delays[conn]:.4f}")
        
        if save_at_intervals and (chunk+1) % 5 == 0:
            import pickle
            
            results = {
                'weights': weights,
                'delays': delays,
                'accuracy': all_accuracy,
                'response_rates': all_response_rates,
                'config': config,
                'chunk': chunk,
                'delay_history': delay_history,
                'weight_history': weight_history,
                'response_history': response_history,
                'all_vm_traces': all_vm_traces
            }
            
            with open(os.path.join(result_path, f"results_chunk_{chunk+1}.pkl"), 'wb') as f:
                pickle.dump(results, f)
        
        # End this chunk's simulation
        sim.end()
    
    # Save final results
    import pickle
    
    final_results = {
        'weights': weights,
        'delays': delays,
        'accuracy': all_accuracy,
        'response_rates': all_response_rates,
        'config': config,
        'delay_history': delay_history,
        'weight_history': weight_history,
        'response_history': response_history,
        'all_vm_traces': all_vm_traces
    }
    
    with open(os.path.join(result_path, "final_results.pkl"), 'wb') as f:
        pickle.dump(final_results, f)
    
    # Plot accuracy over chunks
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_chunks+1), all_accuracy)
    plt.xlabel('Chunk')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Training Chunks')
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "accuracy.png"))
    
    # Plot the specialization development
    plt.figure(figsize=(12, 6))
    
    # Extract data for plots
    output0_p0_rates = [rates['output_0']['pattern_0'] for rates in all_response_rates]
    output0_p1_rates = [rates['output_0']['pattern_1'] for rates in all_response_rates]
    output1_p0_rates = [rates['output_1']['pattern_0'] for rates in all_response_rates]
    output1_p1_rates = [rates['output_1']['pattern_1'] for rates in all_response_rates]
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_chunks+1), output0_p0_rates, 'b-', label='Pattern 0')
    plt.plot(range(1, num_chunks+1), output0_p1_rates, 'r-', label='Pattern 1')
    plt.xlabel('Chunk')
    plt.ylabel('Response Rate')
    plt.title('Output Neuron 0 Response Rate')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_chunks+1), output1_p0_rates, 'b-', label='Pattern 0')
    plt.plot(range(1, num_chunks+1), output1_p1_rates, 'r-', label='Pattern 1')
    plt.xlabel('Chunk')
    plt.ylabel('Response Rate')
    plt.title('Output Neuron 1 Response Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "specialization.png"))
    
    print(f"Training completed. Final accuracy: {all_accuracy[-1]:.4f}")
    print(f"Results saved to {result_path}")
    
    # Try to use visualisation_and_metrics if available
    try:
        # Import from SIAL directory
        import sys
        sys.path.append('/home/neuromorph/Documents/SIAL/just_delay')
        from visualisation_and_metrics import save_all_visualizations
        save_all_visualizations(final_results, "combined_learning")
    except ImportError:
        print("Warning: visualisation_and_metrics module not found. Skipping visualizations.")
    
    return final_results

if __name__ == "__main__":
    # Initialize configuration
    config = init_combined_learning()
    
    # Run chunked simulation
    print("\n=== Starting Combined STDP and Delay Learning Experiment ===")
    results = run_chunked_simulation(
        num_chunks=args.num_chunks, 
        patterns_per_chunk=args.patterns_per_chunk,
        config=config
    )
