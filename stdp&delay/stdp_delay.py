import pyNN.nest as sim
import numpy as np
import random
import os
import sys
import argparse
from quantities import ms
from visualisation_and_metrics import save_all_visualizations

# Delay Learning Experiment (not stdp)

def get_neuron_params():
    NEURON_PARAMS = {
        'tau_m': 15,          # [between 10 and 30]
        'tau_refrac': 2.0,    # [1-3ms]
        'v_reset': -70.0,     # [near resting potential]
        'v_rest': -65.0,      # [-60 to -70mV]
        'v_thresh': -52.5,    # [-55 to -50mV]
        'cm': 0.25,
        'tau_syn_E': 5.0,     # [AMPA receptors ~5ms]
        'tau_syn_I': 10.0,    # [GABA receptors ~10ms]
        'e_rev_E': 0.0,     
        'e_rev_I': -80.0   
    }
    return NEURON_PARAMS

def get_stdpConfig():
    STDP_CONFIG = {
    "A_plus": 0.15,      # Potentiation strength
    "A_minus": 0.06,     # Depression strength
    "tau_plus": 20.0,    # Time window for potentiation
    "tau_minus": 20.0,   # Time window for depression
    }
    return STDP_CONFIG

def get_input_patterns():
    base_patterns = {
        0: {'input0': 4.0, 'input1': 12.0},
        1: {'input1': 4.0, 'input0': 12.0},
        #2: {'input0': 5.0, 'input1': 5.0}
    }
    return base_patterns

def generate_dataset(num_presentations=5, pattern_interval=30, specific_pattern=None):
    base_patterns = get_input_patterns()
    pattern_sequence = []
    if specific_pattern is not None:
        pattern_sequence = [specific_pattern] * num_presentations
    else:
        for _ in range(num_presentations):
            pattern_sequence.append(random.choice(list(base_patterns.keys())))
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
        if isinstance(val, tuple):
            val = float(val[2])
        print(f"  {conn[0]} → {conn[1]}: {val:.3f} ms")
    all_pairs = {
        ('input_0', 'output_0'): [],
        ('input_0', 'output_1'): [],
        ('input_1', 'output_0'): [],
        ('input_1', 'output_1'): []
    }
    if len(output0_spikes) == 0 and len(output1_spikes) == 0:
        return all_pairs
    delay_00 = float(current_delays[('input_0', 'output_0')])
    delay_01 = float(current_delays[('input_0', 'output_1')])
    delay_10 = float(current_delays[('input_1', 'output_0')])
    delay_11 = float(current_delays[('input_1', 'output_1')])
    for pre_time in input0_spikes:
        arrival_time = pre_time + delay_00
        for post_time in output0_spikes:
            if abs(post_time - arrival_time) <= window:
                all_pairs[('input_0', 'output_0')].append((pre_time, post_time))
    for pre_time in input0_spikes:
        arrival_time = pre_time + delay_01
        for post_time in output1_spikes:
            if abs(post_time - arrival_time) <= window:
                all_pairs[('input_0', 'output_1')].append((pre_time, post_time))
    for pre_time in input1_spikes:
        arrival_time = pre_time + delay_10
        for post_time in output0_spikes:
            if abs(post_time - arrival_time) <= window:
                all_pairs[('input_1', 'output_0')].append((pre_time, post_time))
    for pre_time in input1_spikes:
        arrival_time = pre_time + delay_11
        for post_time in output1_spikes:
            if abs(post_time - arrival_time) <= window:
                all_pairs[('input_1', 'output_1')].append((pre_time, post_time))
    return all_pairs

def delay_learning_rule(pre_spike_time, post_spike_time, current_delay, 
                        B_plus=0.1, B_minus=0.1, sigma_plus=10.0, sigma_minus=10.0, 
                        c_threshold=0.5, modulation_cost=0.01):
    
    current_delays = np.array(current_delays, dtype=float)
    if np.any(current_delays < c_threshold):
        new_delays = current_delays + modulation_cost
        return new_delays
    
    new_delays = []

    for i in range(len(current_delays)):
        delta_t = post_spike_time - pre_spike_time - current_delays[i]

        #apply piecewise function G
        if delta_t >= 0:  #if post fires after pre, I decrease the delay
            delta_delay = -B_minus * np.exp(-delta_t / sigma_minus)
        else:  #if post fires before pre, I increase the delay
            delta_delay = B_plus * np.exp(delta_t / sigma_plus)

        new_delays.append(current_delays[i] + delta_delay)

    new_delays = np.array(new_delays) + modulation_cost

    return new_delays



def delay_homeostasis(current_delays, R_target, R_observed, learning_rate_d=0.8, pattern_responses=None):
    new_delays = current_delays.copy()
    
    def update_delay(connection, adjustment):
        if isinstance(current_delays[connection], tuple):
            delay_value = current_delays[connection][2]
            new_delay = delay_value - adjustment
            new_delay = max(0.1, min(20.0, new_delay))
            return (current_delays[connection][0], current_delays[connection][1], new_delay)
        else:
            new_delay = current_delays[connection] - adjustment
            return max(0.1, min(20.0, new_delay))
    
    # Calculate overall homeostasis factors
    K_output0 = (R_target - R_observed['output_0']) / max(0.1, R_target)
    K_output1 = (R_target - R_observed['output_1']) / max(0.1, R_target)
    
    if pattern_responses:
        # Get stats about current specialization
        output0_pattern0 = pattern_responses['output_0']['pattern_0']
        output0_pattern1 = pattern_responses['output_0']['pattern_1']
        output1_pattern0 = pattern_responses['output_1']['pattern_0']
        output1_pattern1 = pattern_responses['output_1']['pattern_1']
        
        # Check if any specialization has emerged
        output0_diff = abs(output0_pattern0 - output0_pattern1)
        output1_diff = abs(output1_pattern0 - output1_pattern1)
        
        # If we have some emerging specialization, subtly reinforce it
        # This uses competition without explicit supervision
        if output0_diff > 0.1 or output1_diff > 0.1:
            # Find shorter delays to further shorten them
            connections = [('input_0', 'output_0'), ('input_0', 'output_1'), 
                          ('input_1', 'output_0'), ('input_1', 'output_1')]
            delay_values = [current_delays[conn] for conn in connections]
            
            # Find which two connections have shortest delays
            sorted_indices = np.argsort(delay_values)
            shorter_connections = [connections[sorted_indices[0]], connections[sorted_indices[1]]]
            longer_connections = [connections[sorted_indices[2]], connections[sorted_indices[3]]]
            
            # Apply competitive homeostasis
            for conn in shorter_connections:
                # Slightly decrease delays for already shorter connections
                new_delays[conn] = update_delay(conn, learning_rate_d * 0.6)
                
            for conn in longer_connections:
                # Slightly increase delays for already longer connections
                new_delays[conn] = update_delay(conn, -learning_rate_d * 0.3)
        else:
            # If no specialization yet, apply regular homeostasis
            for connection in current_delays:
                output_index = 0 if connection[1] == 'output_0' else 1
                K_factor = K_output0 if output_index == 0 else K_output1
                new_delays[connection] = update_delay(connection, learning_rate_d * K_factor * 0.5)
    else:
        # Standard homeostasis as fallback
        for connection in current_delays:
            output_index = 0 if connection[1] == 'output_0' else 1
            K_factor = K_output0 if output_index == 0 else K_output1
            new_delays[connection] = update_delay(connection, learning_rate_d * K_factor * 0.5)
    
    return new_delays

def analyze_pattern_responses(output0_spikes, output1_spikes, pattern_start_times, pattern_labels, window=20):
    pattern0_responses = [0, 0] 
    pattern1_responses = [0, 0]
    
    output0_spikes = [float(t) for t in output0_spikes]
    output1_spikes = [float(t) for t in output1_spikes]
    
    for i, (time, pattern) in enumerate(zip(pattern_start_times, pattern_labels)):
        output0_responded = any((time <= t <= time + window) for t in output0_spikes)
        output1_responded = any((time <= t <= time + window) for t in output1_spikes)
        
        if pattern == 0:
            pattern0_responses[0] += 1 if output0_responded else 0
            pattern0_responses[1] += 1 if output1_responded else 0
        elif pattern == 1:
            pattern1_responses[0] += 1 if output0_responded else 0
            pattern1_responses[1] += 1 if output1_responded else 0
    
    pattern0_count = sum(1 for p in pattern_labels if p == 0)
    pattern1_count = sum(1 for p in pattern_labels if p == 1)
    
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

def create_populations(input_spikes, weights, NEURON_PARAMS, init_delay_range=(3.0, 7.0), 
                      init_delays=None, inh_weight=0.05, inh_delay=1.0):
    input_pop_0 = sim.Population(
        1, sim.SpikeSourceArray(spike_times=input_spikes[0]), label="In0"
    )
    input_pop_1 = sim.Population(
        1, sim.SpikeSourceArray(spike_times=input_spikes[1]), label="In1"
    )
    input_pop_0.record("spikes")
    input_pop_1.record("spikes")

    output_pop_0 = sim.Population(1, sim.IF_cond_exp(**NEURON_PARAMS), label="Out0")
    output_pop_1 = sim.Population(1, sim.IF_cond_exp(**NEURON_PARAMS), label="Out1")

    output_pop_0.initialize(v=NEURON_PARAMS['v_rest'])
    output_pop_1.initialize(v=NEURON_PARAMS['v_rest'])

    output_pop_0.record(("spikes", "v"))
    output_pop_1.record(("spikes", "v"))

    # Set initial delays
    if init_delays is not None and isinstance(init_delays, dict):
        d0_in0 = init_delays.get(("input_0", "output_0"), random.uniform(*init_delay_range))
        d0_in1 = init_delays.get(("input_0", "output_1"), random.uniform(*init_delay_range))
        d1_in0 = init_delays.get(("input_1", "output_0"), random.uniform(*init_delay_range))
        d1_in1 = init_delays.get(("input_1", "output_1"), random.uniform(*init_delay_range))
    else:
        d0_in0 = random.uniform(*init_delay_range)
        d0_in1 = random.uniform(*init_delay_range)
        d1_in0 = random.uniform(*init_delay_range)
        d1_in1 = random.uniform(*init_delay_range)

    # STDP mechanism
    stdp_config = get_stdpConfig()
    
    # Create separate STDP models for each connection with proper initial values
    stdp_model_00 = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=stdp_config["tau_plus"],
            tau_minus=stdp_config["tau_minus"],
            A_plus=stdp_config["A_plus"],
            A_minus=stdp_config["A_minus"]
        ),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0),
        weight=weights[0],
        delay=d0_in0
    )
    
    stdp_model_01 = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=stdp_config["tau_plus"],
            tau_minus=stdp_config["tau_minus"],
            A_plus=stdp_config["A_plus"],
            A_minus=stdp_config["A_minus"]
        ),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0),
        weight=weights[0],
        delay=d0_in1
    )
    
    stdp_model_10 = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=stdp_config["tau_plus"],
            tau_minus=stdp_config["tau_minus"],
            A_plus=stdp_config["A_plus"],
            A_minus=stdp_config["A_minus"]
        ),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0),
        weight=weights[1],
        delay=d1_in0
    )
    
    stdp_model_11 = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=stdp_config["tau_plus"],
            tau_minus=stdp_config["tau_minus"],
            A_plus=stdp_config["A_plus"],
            A_minus=stdp_config["A_minus"]
        ),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0),
        weight=weights[1],
        delay=d1_in1
    )

    conn_in0_out0 = sim.Projection(
        input_pop_0, output_pop_0, sim.AllToAllConnector(),
        synapse_type=stdp_model_00,
        receptor_type="excitatory"
    )

    conn_in0_out1 = sim.Projection(
        input_pop_0, output_pop_1, sim.AllToAllConnector(),
        synapse_type=stdp_model_01,
        receptor_type="excitatory"
    )

    conn_in1_out0 = sim.Projection(
        input_pop_1, output_pop_0, sim.AllToAllConnector(),
        synapse_type=stdp_model_10,
        receptor_type="excitatory"
    )

    conn_in1_out1 = sim.Projection(
        input_pop_1, output_pop_1, sim.AllToAllConnector(),
        synapse_type=stdp_model_11,
        receptor_type="excitatory"
    )

    # Inhibitory connections (static)
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

def init_delay_learning(
    num_presentations=5,
    pattern_interval=30,
    B_plus = 0.25,         # Increased from 0.2 to strengthen STDP
    B_minus = 0.2,         # Increased from 0.15 to strengthen STDP
    sigma_plus = 10.0,
    sigma_minus = 10.0,
    c_threshold = 0.5,
    modulation_const = 0.15,  # Increased from 0.07 for stronger competition
    init_delay_range= (1.0, 15.0),
    weights = (0.07, 0.07),   # Increased from 0.05 for stronger excitation
    inh_weight = 2.0,         # Increased from 1.5 for even stronger competition
    inh_delay = 1.0,          # Decreased from 1.5 for faster competition
    timestep=0.01,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    sim.setup(timestep=timestep)

    dataset = generate_dataset(
        num_presentations=num_presentations,
        pattern_interval=pattern_interval,
        specific_pattern=None
    )

    config = {
        "dataset": dataset,
        "num_presentations":num_presentations,
        "pattern_interval": pattern_interval,
        "B_plus": B_plus,
        "B_minus": B_minus,
        "sigma_plus": sigma_plus,
        "sigma_minus": sigma_minus,
        "c_threshold": c_threshold,
        "modulation_const": modulation_const,
        "init_delay_range": init_delay_range,
        "weights": weights,
        "inh_weight": inh_weight,
        "inh_delay": inh_delay,
        "timestep": timestep,
        "seed":seed,
        "NEURON_PARAMS": get_neuron_params()
    }

    return config

def update_network_delays(populations, new_delays):
    for connection, delay in new_delays.items():
        input_pop, output_pop = connection
        projection = populations['projections'][connection]
        
        if isinstance(delay, tuple):
            actual_delay = delay[2]
        else:
            actual_delay = delay
        
        projection.set(delay=actual_delay)
    
    return populations

def apply_delay_learning(current_delays, all_pairs, B_plus, B_minus, 
                         sigma_plus, sigma_minus, c_threshold, modulation_const,
                         current_accuracy=0.0, pattern_responses=None):
    new_delays = current_delays.copy()
    
    # Zero-sum modulation will be applied based on STDP activity only
    # This removes pattern-specific supervision
    connections_to_strengthen = []
    connections_to_weaken = []
    
    # We'll let STDP alone drive the specialization without 
    # explicitly defining which connections should be strengthened or weakened
    
    # Process each connection with STDP
    for connection, pairs in all_pairs.items():
        # Skip empty connections but include them in modulation later
        if len(pairs) == 0:
            continue
        
        # Extract current delay value
        if isinstance(current_delays[connection], tuple):
            current_delay = current_delays[connection][2]
        else:
            current_delay = current_delays[connection]
        
        # Skip STDP for delays below threshold
        if current_delay < c_threshold:
            continue
        
        # Calculate STDP changes
        delta_delays = []
        for pre_time, post_time in pairs:
            delta_t = post_time - (pre_time + current_delay)
            
            if delta_t >= 0:
                delta_delay = -B_minus * np.exp(-delta_t / sigma_minus)
            else:
                delta_delay = B_plus * np.exp(delta_t / sigma_plus)
            
            delta_delays.append(delta_delay)
        
        # Apply average STDP change if we have any
        if delta_delays:
            avg_delta = np.mean(delta_delays)
            
            # Apply STDP without modulation yet
            new_delay = current_delay + avg_delta
            
            # Store the updated delay
            if isinstance(current_delays[connection], tuple):
                new_tuple = (current_delays[connection][0], current_delays[connection][1], new_delay)
                new_delays[connection] = new_tuple
            else:
                new_delays[connection] = new_delay
    
    # Apply enhanced competitive modulation to promote stronger differentiation
    if modulation_const > 0:
        # Calculate activity levels for each connection (how many spike pairs)
        activity_levels = {}
        for connection in all_pairs:
            activity_levels[connection] = len(all_pairs[connection])
        
        # Calculate timing precision for each connection
        timing_precision = {}
        for connection, pairs in all_pairs.items():
            if len(pairs) > 0:
                # Calculate timing variance (lower is better)
                timing_diffs = []
                if isinstance(current_delays[connection], tuple):
                    current_delay = current_delays[connection][2]
                else:
                    current_delay = current_delays[connection]
                
                for pre_time, post_time in pairs:
                    delta_t = post_time - (pre_time + current_delay)
                    timing_diffs.append(abs(delta_t))
                
                # Lower variance means better timing precision
                if timing_diffs:
                    precision = 1.0 / (1.0 + np.mean(timing_diffs))
                    timing_precision[connection] = precision
                else:
                    timing_precision[connection] = 0.0
            else:
                timing_precision[connection] = 0.0
        
        # Combine activity and precision for fitness score
        fitness_scores = {}
        if sum(activity_levels.values()) > 0:
            # Normalize activity levels
            total_activity = max(1, sum(activity_levels.values()))
            for connection in activity_levels:
                norm_activity = activity_levels[connection] / total_activity
                norm_precision = timing_precision.get(connection, 0.0)
                # Weighted combination favoring connections that are both active and precise
                fitness_scores[connection] = (0.7 * norm_activity) + (0.3 * norm_precision)
            
            # Sort connections by fitness score
            sorted_connections = sorted(
                fitness_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Group connections by input and output neurons
            input0_conns = [conn for conn in sorted_connections if conn[0][0] == 'input_0']
            input1_conns = [conn for conn in sorted_connections if conn[0][0] == 'input_1']
            output0_conns = [conn for conn in sorted_connections if conn[0][1] == 'output_0']
            output1_conns = [conn for conn in sorted_connections if conn[0][1] == 'output_1']
            
            # Ensure competition among connections from same input or to same output
            # This forces specialization without explicit supervision
            
            # Strongest connection for each input gets strengthened, weaker gets weakened
            for input_group in [input0_conns, input1_conns]:
                if len(input_group) > 0:
                    best_conn = input_group[0][0]
                    # Strengthen best connection
                    if isinstance(new_delays[best_conn], tuple):
                        current_val = new_delays[best_conn][2]
                        new_val = max(1.0, current_val - modulation_const * 2.0)
                        new_delays[best_conn] = (new_delays[best_conn][0], new_delays[best_conn][1], new_val)
                    else:
                        new_delays[best_conn] = max(1.0, new_delays[best_conn] - modulation_const * 2.0)
                    
                    # Weaken other connections from same input
                    for conn_info in input_group[1:]:
                        conn = conn_info[0]
                        if isinstance(new_delays[conn], tuple):
                            current_val = new_delays[conn][2]
                            new_val = min(15.0, current_val + modulation_const * 3.0)
                            new_delays[conn] = (new_delays[conn][0], new_delays[conn][1], new_val)
                        else:
                            new_delays[conn] = min(15.0, new_delays[conn] + modulation_const * 3.0)
            
            # Similarly, ensure each output neuron specializes
            for output_group in [output0_conns, output1_conns]:
                if len(output_group) > 0:
                    best_conn = output_group[0][0]
                    # Already strengthened above, so just weaken others
                    for conn_info in output_group[1:]:
                        conn = conn_info[0]
                        if isinstance(new_delays[conn], tuple):
                            current_val = new_delays[conn][2]
                            new_val = min(15.0, current_val + modulation_const * 2.0)
                            new_delays[conn] = (new_delays[conn][0], new_delays[conn][1], new_val)
                        else:
                            new_delays[conn] = min(15.0, new_delays[conn] + modulation_const * 2.0)
    
    # Final delay bounds enforcement - apply broader bounds to allow differentiation
    # but prevent extreme values or convergence to similar values
    
    # First, check if delays are too similar and add noise if needed
    delay_values = []
    for connection in new_delays:
        if isinstance(new_delays[connection], tuple):
            delay_values.append(new_delays[connection][2])
        else:
            delay_values.append(new_delays[connection])
    
    # Calculate variance in delays
    delay_variance = np.var(delay_values)
    
    # If variance is too low, add some noise to break symmetry
    if delay_variance < 0.5:  # Delays too similar
        for connection in new_delays:
            if isinstance(new_delays[connection], tuple):
                current_val = new_delays[connection][2]
                # Add random noise of up to ±0.5ms
                noise = np.random.uniform(-0.5, 0.5)
                new_val = current_val + noise
                new_delays[connection] = (new_delays[connection][0], new_delays[connection][1], new_val)
            else:
                current_val = new_delays[connection]
                noise = np.random.uniform(-0.5, 0.5)
                new_delays[connection] = current_val + noise
    
    # Apply general bounds to prevent extreme values
    for connection in new_delays:
        if isinstance(new_delays[connection], tuple):
            delay_value = new_delays[connection][2]
            delay_value = max(1.0, min(15.0, delay_value))
            new_delays[connection] = (new_delays[connection][0], new_delays[connection][1], delay_value)
        else:
            new_delays[connection] = max(1.0, min(15.0, new_delays[connection]))
    
    return new_delays

def run_chunked_simulation(config, num_chunks=10, patterns_per_chunk=10):
    delay_history = {
        ('input_0', 'output_0'): [],
        ('input_0', 'output_1'): [],
        ('input_1', 'output_0'): [],
        ('input_1', 'output_1'): []
    }
    response_history = {
        'output_0': {'pattern_0': [], 'pattern_1': []},
        'output_1': {'pattern_0': [], 'pattern_1': []}
    }
    
    # Start with initial delays that have slight randomization to break symmetry
    # This helps bootstrap specialization without explicit supervision
    current_delays = {
        ('input_0', 'output_0'): 3.0 + np.random.uniform(-0.5, 0.5),  # Around 3ms
        ('input_0', 'output_1'): 7.0 + np.random.uniform(-0.5, 0.5),  # Around 7ms
        ('input_1', 'output_0'): 7.0 + np.random.uniform(-0.5, 0.5),  # Around 7ms
        ('input_1', 'output_1'): 3.0 + np.random.uniform(-0.5, 0.5)   # Around 3ms
    }
    
    # Initialize accuracy tracking
    accuracy_history = []
    best_accuracy = 0.0
    stable_count = 0
    
    for key in delay_history:
        delay_history[key].append(current_delays[key])
    
    final_populations = None
    all_vm_traces = []
    pattern_responses = None
    
    for chunk in range(num_chunks):
        print(f"\n=== Running chunk {chunk+1}/{num_chunks} ===")

        sim.setup(timestep=config['timestep'])

        # Choose pattern more intelligently to ensure balanced training
        chosen_pattern = None
        if chunk % 10 < 5:
            # First half of every 10 chunks uses pattern 0
            chosen_pattern = 0
        else:
            # Second half uses pattern 1
            chosen_pattern = 1
            
        # Add randomization within this structure (20% chance to flip)
        if random.random() < 0.2:
            chosen_pattern = 1 - chosen_pattern
            
        print(f"Generating chunk with pattern {chosen_pattern}")
        
        current_dataset = generate_dataset(
            num_presentations=patterns_per_chunk,
            pattern_interval=config['pattern_interval'],
            specific_pattern=chosen_pattern
        )

        populations = create_populations(
            current_dataset['input_spikes'],
            config['weights'],
            config['NEURON_PARAMS'],
            init_delay_range=config['init_delay_range'],
            init_delays=current_delays,
            inh_weight=config['inh_weight'],
            inh_delay=config['inh_delay']
        )

        chunk_duration = patterns_per_chunk * config['pattern_interval']
        sim.run(chunk_duration)

        output0_spikes = populations['output_0'].get_data().segments[0].spiketrains[0]
        output1_spikes = populations['output_1'].get_data().segments[0].spiketrains[0]

        output0_data = populations.get('output_0', {}).get_data().segments[0]
        output1_data = populations.get('output_1', {}).get_data().segments[0]
        
        vm_traces = [
            output0_data.filter(name="v")[0],
            output1_data.filter(name="v")[0]
        ]
        all_vm_traces.append(vm_traces)

        pattern_responses = analyze_pattern_responses(
            output0_spikes, 
            output1_spikes, 
            current_dataset['pattern_start_times'],
            current_dataset['pattern_labels']
        )
        response_history['output_0']['pattern_0'].append(pattern_responses['output_0']['pattern_0'])
        response_history['output_0']['pattern_1'].append(pattern_responses['output_0']['pattern_1'])
        response_history['output_1']['pattern_0'].append(pattern_responses['output_1']['pattern_0'])
        response_history['output_1']['pattern_1'].append(pattern_responses['output_1']['pattern_1'])

        print(f"Pattern response rates:")
        print(f"  Output 0: Pattern 0: {pattern_responses['output_0']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_0']['pattern_1']:.2f}")
        print(f"  Output 1: Pattern 0: {pattern_responses['output_1']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_1']['pattern_1']:.2f}")
        
        # Calculate current accuracy for stabilization
        current_accuracy = 0.0
        # Detect specialization
        neuron0_prefers_pattern0 = pattern_responses['output_0']['pattern_0'] > pattern_responses['output_0']['pattern_1']
        neuron1_prefers_pattern1 = pattern_responses['output_1']['pattern_1'] > pattern_responses['output_1']['pattern_0']
        neuron0_prefers_pattern1 = pattern_responses['output_0']['pattern_1'] > pattern_responses['output_0']['pattern_0']
        neuron1_prefers_pattern0 = pattern_responses['output_1']['pattern_0'] > pattern_responses['output_1']['pattern_1']
        
        # Calculate specialization strength
        spec_strength0_for_pattern0 = pattern_responses['output_0']['pattern_0'] - pattern_responses['output_0']['pattern_1']
        spec_strength1_for_pattern1 = pattern_responses['output_1']['pattern_1'] - pattern_responses['output_1']['pattern_0']
        spec_strength0_for_pattern1 = pattern_responses['output_0']['pattern_1'] - pattern_responses['output_0']['pattern_0']
        spec_strength1_for_pattern0 = pattern_responses['output_1']['pattern_0'] - pattern_responses['output_1']['pattern_1']
        
        # Print detailed delay information for monitoring unsupervised learning
        print(f"  Current delays:")
        print(f"  Input0->Output0: {current_delays[('input_0', 'output_0')]:.2f}ms")
        print(f"  Input1->Output0: {current_delays[('input_1', 'output_0')]:.2f}ms")
        print(f"  Input0->Output1: {current_delays[('input_0', 'output_1')]:.2f}ms")
        print(f"  Input1->Output1: {current_delays[('input_1', 'output_1')]:.2f}ms")
        
        # Calculate response differential (how much more a neuron responds to its preferred pattern)
        if neuron0_prefers_pattern0 and neuron1_prefers_pattern1:
            # Pattern 0->0, 1->1 specialization
            # Calculate based on actual response rates not just differential
            # This ensures that high absolute response rates contribute to higher accuracy
            positive_responses = (pattern_responses['output_0']['pattern_0'] + pattern_responses['output_1']['pattern_1'])/2
            negative_avoidance = (1.0 - pattern_responses['output_0']['pattern_1'] + 1.0 - pattern_responses['output_1']['pattern_0'])/2
            current_accuracy = (positive_responses + negative_avoidance)/2
            
            # Bonus for perfect classification (1.0 response to preferred, 0.0 to non-preferred)
            if pattern_responses['output_0']['pattern_0'] > 0.9 and pattern_responses['output_0']['pattern_1'] < 0.1 and \
               pattern_responses['output_1']['pattern_1'] > 0.9 and pattern_responses['output_1']['pattern_0'] < 0.1:
                current_accuracy = min(1.0, current_accuracy + 0.3)
        elif neuron0_prefers_pattern1 and neuron1_prefers_pattern0:
            # Pattern 1->0, 0->1 specialization
            # Calculate based on actual response rates not just differential
            positive_responses = (pattern_responses['output_0']['pattern_1'] + pattern_responses['output_1']['pattern_0'])/2
            negative_avoidance = (1.0 - pattern_responses['output_0']['pattern_0'] + 1.0 - pattern_responses['output_1']['pattern_1'])/2
            current_accuracy = (positive_responses + negative_avoidance)/2
            
            # Bonus for perfect classification (1.0 response to preferred, 0.0 to non-preferred)
            if pattern_responses['output_0']['pattern_1'] > 0.9 and pattern_responses['output_0']['pattern_0'] < 0.1 and \
               pattern_responses['output_1']['pattern_0'] > 0.9 and pattern_responses['output_1']['pattern_1'] < 0.1:
                current_accuracy = min(1.0, current_accuracy + 0.3)
        else:
            # Single neuron specialization - use the stronger specialization
            # Also consider actual response rates for single neuron specialization
            spec0_pattern0_accuracy = (pattern_responses['output_0']['pattern_0'] + (1.0 - pattern_responses['output_0']['pattern_1']))/2
            spec1_pattern1_accuracy = (pattern_responses['output_1']['pattern_1'] + (1.0 - pattern_responses['output_1']['pattern_0']))/2
            spec0_pattern1_accuracy = (pattern_responses['output_0']['pattern_1'] + (1.0 - pattern_responses['output_0']['pattern_0']))/2
            spec1_pattern0_accuracy = (pattern_responses['output_1']['pattern_0'] + (1.0 - pattern_responses['output_1']['pattern_1']))/2
            
            current_accuracy = max(
                spec0_pattern0_accuracy,
                spec1_pattern1_accuracy,
                spec0_pattern1_accuracy,
                spec1_pattern0_accuracy
            ) * 0.9  # Less penalty for single neuron specialization
        
        print(f"  Current accuracy: {current_accuracy:.2f}")
        print(f"  Response rates:")
        print(f"  Output0 to Pattern0: {pattern_responses['output_0']['pattern_0']:.2f}")
        print(f"  Output0 to Pattern1: {pattern_responses['output_0']['pattern_1']:.2f}")
        print(f"  Output1 to Pattern0: {pattern_responses['output_1']['pattern_0']:.2f}")
        print(f"  Output1 to Pattern1: {pattern_responses['output_1']['pattern_1']:.2f}")
        
        # Track accuracy history
        accuracy_history.append(current_accuracy)
        
        # Update best accuracy
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            stable_count = 0
        else:
            stable_count += 1

        all_pairs = find_spike_pairs(
            populations, 
            output0_spikes, 
            output1_spikes, 
            current_delays
        )

        # Apply stabilization: reduce learning rates if accuracy is high
        b_plus = config['B_plus']
        b_minus = config['B_minus']
        
        # Decay learning parameters more gradually
        decay_factor = np.exp(-chunk / (num_chunks * 0.6))  # Even slower decay (was 0.4)
        b_plus *= decay_factor
        b_minus *= decay_factor
        
        # Calculate delay variance to detect if specialization is happening
        delay_values = [current_delays[conn] for conn in current_delays]
        delay_variance = np.var(delay_values)
        print(f"  Delay variance: {delay_variance:.3f}")
        
        # Decay modulation constant over time but keep it higher for longer
        mod_const = config['modulation_const'] * max(0.3, decay_factor)
        
        # If we're getting specialization (high variance), maintain learning longer
        if delay_variance > 1.5:
            # Boost modulation if we have good specialization to maintain it
            print(f"  Good delay separation detected, maintaining strong modulation")
            mod_const *= 1.2
            
        # If accuracy is above threshold, reduce learning rates but don't kill them too quickly
        if current_accuracy > 0.6:  # Lowered from 0.7
            print(f"  Good accuracy detected ({current_accuracy:.2f}), adjusting learning rates")
            b_plus *= 0.6  # Less aggressive reduction (was 0.5)
            b_minus *= 0.6  # Less aggressive reduction (was 0.5)
            mod_const *= 0.6  # Also reduce modulation constant
        
        # If we've reached a stable period but delays aren't specialized enough, boost learning
        if stable_count > 5 and delay_variance < 1.0 and best_accuracy < 0.6:
            print(f"  Delays too similar (variance: {delay_variance:.2f}), boosting learning rates")
            b_plus = config['B_plus'] * 1.5
            b_minus = config['B_minus'] * 1.5
            mod_const = config['modulation_const'] * 1.5
            # Reset stable count to give more time for learning
            stable_count = 0
        
        # If we've reached a stable period (no improvement for several chunks)
        if stable_count > 8 and best_accuracy > 0.5:  # Increased count, lowered threshold
            print(f"  Stable accuracy for {stable_count} chunks, further reducing learning rates")
            b_plus *= 0.6  # Less aggressive reduction (was 0.5)
            b_minus *= 0.6  # Less aggressive reduction (was 0.5)
            mod_const *= 0.6
        
        # Hard stop on learning when stable and accuracy is good
        if stable_count > 20 and best_accuracy > 0.6:  # Increased count and lowered threshold
            print(f"  Network has stabilized with good performance, freezing all learning")
            b_plus = 0.0
            b_minus = 0.0
            mod_const = 0.0
        
        # Print the used learning rates
        print(f"  Using learning rates: B+ = {b_plus:.4f}, B- = {b_minus:.4f}, mod = {mod_const:.4f}")
        
        new_delays = apply_delay_learning(
            current_delays,
            all_pairs,
            b_plus,
            b_minus,
            config['sigma_plus'],
            config['sigma_minus'],
            config['c_threshold'],
            mod_const,
            current_accuracy,
            pattern_responses
        )

        observed_rates = {
            'output_0': len(output0_spikes) / patterns_per_chunk,
            'output_1': len(output1_spikes) / patterns_per_chunk
        }
        print("output spike 0:", output0_spikes)
        print("output spike 1: ", output1_spikes)
        target_rate = 1.0

        homeostasis_lr = 0.8 * decay_factor
        
        if current_accuracy > 0.7 and delay_variance > 1.5:
            homeostasis_lr *= 0.3
        elif delay_variance < 1.0:
            homeostasis_lr *= 1.5
            print("  Increasing homeostasis to promote delay differentiation")
        
        if stable_count > 10 and best_accuracy > 0.6 and delay_variance > 2.0:
            homeostasis_lr *= 0.1
        
        if stable_count > 8 and best_accuracy > 0.7:
            print(f"  Freezing homeostasis with stable good performance")
            homeostasis_lr = 0.0
        
        new_delays = delay_homeostasis(
            new_delays,
            target_rate,
            observed_rates,
            learning_rate_d=homeostasis_lr,
            pattern_responses=pattern_responses
        )

        print("Delays after learning and homeostasis:")
        for conn in [('input_0', 'output_0'), ('input_0', 'output_1'), ('input_1', 'output_0'), ('input_1', 'output_1')]:
            val = new_delays[conn]
            if isinstance(val, tuple):
                val = val[2]
            print(f"  {conn[0]} → {conn[1]}: {val:.3f} ms")

        current_delays = new_delays.copy()

        for key in delay_history:
            delay_history[key].append(current_delays[key])

        if chunk == num_chunks - 1:
            final_populations = populations

        sim.end()
    
    return {
        'delay_history': delay_history,
        'pattern_responses': pattern_responses,
        'response_history': response_history,
        'final_delays': current_delays,
        'accuracy_history': accuracy_history,
        'stable_count': stable_count,
        'best_accuracy': best_accuracy
    }

if __name__ == "__main__":
    config = init_delay_learning()
    results = run_chunked_simulation(config, num_chunks=100, patterns_per_chunk=10)
    save_all_visualizations(results, "stdp_and_delays")
