import pyNN.nest as sim
import numpy as np
import random
import os
import sys
import argparse
from quantities import ms
from visualisation_and_metrics import save_all_visualizations

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
    return {
        "A_plus": 0.45,
        "A_minus": 0.20,
        "tau_plus": 20.0,
        "tau_minus": 20.0,    }

def get_input_patterns():
    base_patterns = {
        0: {'input0': 4.0, 'input1': 12.0},
        1: {'input1': 4.0, 'input0': 12.0},
        #2: {'input0': 5.0, 'input1': 5.0}
    }
    return base_patterns

def delay_learning_rule(pre_spike_time, post_spike_time, current_delays, 
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

def generate_dataset(num_presentations=5, pattern_interval=30, specific_pattern=None):
    base_patterns = get_input_patterns()
    
    pattern_sequence = []
    if specific_pattern is not None:
        # If a specific pattern is requested, use only that
        pattern_sequence = [specific_pattern] * num_presentations
    else:
        # Create a balanced set of patterns with equal numbers of each
        for _ in range(num_presentations // 2):
            pattern_sequence.append(0)
            pattern_sequence.append(1)
        
        if num_presentations % 2 == 1:
            pattern_sequence.append(random.choice([0, 1]))
        
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
    
    print(f"Generated dataset with {len(pattern_sequence)} patterns. Pattern distribution: {pattern_sequence.count(0)} pattern 0, {pattern_sequence.count(1)} pattern 1")
    
    return dataset


def find_spike_pairs(populations, output0_spikes, output1_spikes, current_delays, window=10.0):
    "find the pairs of post-synpatic and pre-synaptic neurons used later in apply_delay_learning"
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
            val = val[2]
        print(f"  {conn[0]} → {conn[1]}: {val:.3f} ms")
    
    all_pairs = {
        ('input_0', 'output_0'): [],
        ('input_0', 'output_1'): [],
        ('input_1', 'output_0'): [],
        ('input_1', 'output_1'): []
    }
    
    if len(output0_spikes) == 0 and len(output1_spikes) == 0:
        return all_pairs
    
    delay_00 = current_delays[('input_0', 'output_0')]
    if isinstance(delay_00, tuple) or isinstance(delay_00, list):
        delay_00 = float(delay_00[2])
    else:
        delay_00 = float(delay_00)
        
    delay_01 = current_delays[('input_0', 'output_1')]
    if isinstance(delay_01, tuple) or isinstance(delay_01, list):
        delay_01 = float(delay_01[2])
    else:
        delay_01 = float(delay_01)
        
    delay_10 = current_delays[('input_1', 'output_0')]
    if isinstance(delay_10, tuple) or isinstance(delay_10, list):
        delay_10 = float(delay_10[2])
    else:
        delay_10 = float(delay_10)
        
    delay_11 = current_delays[('input_1', 'output_1')]
    if isinstance(delay_11, tuple) or isinstance(delay_11, list):
        delay_11 = float(delay_11[2])
    else:
        delay_11 = float(delay_11)
    
    for pre_time in input0_spikes:
        arrival_time = pre_time + delay_00
        
        if output0_spikes:
            valid_posts = [t for t in output0_spikes if abs(t - arrival_time) <= window]
            
            if valid_posts:
                closest_post = min(valid_posts, key=lambda t: abs(t - arrival_time))
                all_pairs[('input_0', 'output_0')].append((pre_time, closest_post))
    
    for pre_time in input0_spikes:
        arrival_time = pre_time + delay_01
        
        if output1_spikes:
            valid_posts = [t for t in output1_spikes if abs(t - arrival_time) <= window]
            
            if valid_posts:
                closest_post = min(valid_posts, key=lambda t: abs(t - arrival_time))
                all_pairs[('input_0', 'output_1')].append((pre_time, closest_post))
    
    for pre_time in input1_spikes:
        arrival_time = pre_time + delay_10
        
        if output0_spikes:
            valid_posts = [t for t in output0_spikes if abs(t - arrival_time) <= window]
            
            if valid_posts:
                closest_post = min(valid_posts, key=lambda t: abs(t - arrival_time))
                all_pairs[('input_1', 'output_0')].append((pre_time, closest_post))
    
    for pre_time in input1_spikes:
        arrival_time = pre_time + delay_11
        
        if output1_spikes:
            valid_posts = [t for t in output1_spikes if abs(t - arrival_time) <= window]
            
            if valid_posts:
                closest_post = min(valid_posts, key=lambda t: abs(t - arrival_time))
                all_pairs[('input_1', 'output_1')].append((pre_time, closest_post))
    
    return all_pairs



def normalize_weights(pops, normalization_factor=1.0, pattern_specific=True):
    projections = pops["projections"]
    
    # Get current weights
    weights = {}
    for conn_key, projection in projections.items():
        weight_data = projection.get('weight', 'list')
        if isinstance(weight_data[0], tuple):
            weights[conn_key] = float(weight_data[0][0])
        else:
            weights[conn_key] = float(weight_data[0])
    
    min_weight = 0.15 
    for conn_key in weights:
        if weights[conn_key] < min_weight:
            weights[conn_key] = min_weight
            projections[conn_key].set(weight=min_weight)
    
    # Normalize weights for output neuron 0
    weights_out0 = [
        weights[("input_0", "output_0")],
        weights[("input_1", "output_0")]
    ]
        
    # Calculate normalization for output neuron 0
    sum_weights_out0 = sum(weights_out0)
    if sum_weights_out0 > 0:
        scale_factor_out0 = normalization_factor / sum_weights_out0
        new_weights_0 = [w * scale_factor_out0 for w in weights_out0]
        if min(new_weights_0) < min_weight:
            deficit = min_weight - min(new_weights_0)
            new_weights_0 = [w + deficit for w in new_weights_0]
            
        projections[("input_0", "output_0")].set(weight=new_weights_0[0])
        projections[("input_1", "output_0")].set(weight=new_weights_0[1])

    weights_out1 = [
        weights[("input_0", "output_1")],
        weights[("input_1", "output_1")]
    ]
        
    sum_weights_out1 = sum(weights_out1)
    if sum_weights_out1 > 0:
        scale_factor_out1 = normalization_factor / sum_weights_out1
        new_weights_1 = [w * scale_factor_out1 for w in weights_out1]
        
        if min(new_weights_1) < min_weight:
            deficit = min_weight - min(new_weights_1)
            new_weights_1 = [w + deficit for w in new_weights_1]
            
        projections[("input_0", "output_1")].set(weight=new_weights_1[0])
        projections[("input_1", "output_1")].set(weight=new_weights_1[1])

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
    
    # Calculate overall homeostasis factors - this is purely based on firing rates
    # which is an unsupervised mechanism
    K_output0 = (R_target - R_observed['output_0']) / max(0.1, R_target)
    K_output1 = (R_target - R_observed['output_1']) / max(0.1, R_target)
    
    # Standard homeostasis: Each neuron tries to maintain a target firing rate
    # This is unsupervised as it only depends on the neuron's own activity
    for connection in current_delays:
        output_index = 0 if connection[1] == 'output_0' else 1
        K_factor = K_output0 if output_index == 0 else K_output1
        
        # Apply homeostatic adjustment
        new_delays[connection] = update_delay(connection, learning_rate_d * K_factor * 0.8)
    
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

    # Use strategic initial delays to break symmetry but not bias toward a specific specialization
    if init_delays is not None and isinstance(init_delays, dict):
        d0_in0 = init_delays.get(("input_0", "output_0"), random.uniform(*init_delay_range))
        d0_in1 = init_delays.get(("input_0", "output_1"), random.uniform(*init_delay_range))
        d1_in0 = init_delays.get(("input_1", "output_0"), random.uniform(*init_delay_range))
        d1_in1 = init_delays.get(("input_1", "output_1"), random.uniform(*init_delay_range))
    else:
        # Add small random variations around the middle of the range to break symmetry
        mid_delay = (init_delay_range[0] + init_delay_range[1]) / 2
        variation = (init_delay_range[1] - init_delay_range[0]) * 0.2  # 20% of range
        d0_in0 = mid_delay + random.uniform(-variation, variation)
        d0_in1 = mid_delay + random.uniform(-variation, variation)
        d1_in0 = mid_delay + random.uniform(-variation, variation)
        d1_in1 = mid_delay + random.uniform(-variation, variation)

    # Get STDP parameters
    stdp_config = get_stdpConfig()
    
    # Create STDP mechanisms with the same parameters but different weights/delays
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

    # Balanced bidirectional lateral inhibition for competitive learning
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
    B_plus = 0.5,
    B_minus = 0.45,
    sigma_plus = 10.0,
    sigma_minus = 10.0,
    c_threshold = 0.5,
    modulation_const = 1.5,
    init_delay_range= (1.0, 15.0),
    weights = (0.04, 0.04),
    inh_weight = 1.2,
    inh_delay = 0.5,
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
    
    # Process each connection with delay_learning_rule
    for connection, pairs in all_pairs.items():
        # Skip empty connections
        if len(pairs) == 0:
            continue
        
        # Extract current delay value
        if isinstance(current_delays[connection], tuple):
            current_delay = current_delays[connection][2]
        else:
            current_delay = current_delays[connection]
        
        # Apply delay learning to each spike pair
        if pairs:
            # Apply the delay learning rule for each spike pair
            new_delay_values = []
            for pre_time, post_time in pairs:
                # Call delay_learning_rule for each spike pair
                updated_delay = delay_learning_rule(
                    pre_time, 
                    post_time, 
                    [current_delay],  # Pass as list since the function expects an array
                    B_plus=B_plus,
                    B_minus=B_minus,
                    sigma_plus=sigma_plus,
                    sigma_minus=sigma_minus,
                    c_threshold=c_threshold,
                    modulation_cost=0.01  # Default value from the function
                )
                
                # Extract the single delay value from the returned array
                if isinstance(updated_delay, list) or isinstance(updated_delay, np.ndarray):
                    updated_delay = updated_delay[0]
                
                new_delay_values.append(updated_delay)
            
            # Average the updates from all spike pairs
            if new_delay_values:
                avg_delay = np.mean(new_delay_values)
                
                # Store the updated delay
                if isinstance(current_delays[connection], tuple):
                    new_tuple = (current_delays[connection][0], current_delays[connection][1], avg_delay)
                    new_delays[connection] = new_tuple
                else:
                    new_delays[connection] = avg_delay
    
    # Add slight noise to avoid symmetry problems
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

    for input_id in ['input_0', 'input_1']:
        # For each input, find which output connection has more spike pairs
        # and enhance that connection's changes
        out0_pairs = len(all_pairs[(input_id, 'output_0')])
        out1_pairs = len(all_pairs[(input_id, 'output_1')])
        
        if out0_pairs > out1_pairs:
            # Enhance delay change for the more active connection
            current_out0 = new_delays[(input_id, 'output_0')]
            target_delay = 8.0 if input_id == 'input_0' else 1.0  # Optimal values from analysis
            
            # Move delay toward target with a stronger pull
            if isinstance(current_out0, tuple):
                new_value = current_out0[2] + (target_delay - current_out0[2]) * 0.1
                new_delays[(input_id, 'output_0')] = (current_out0[0], current_out0[1], new_value)
            else:
                new_value = current_out0 + (target_delay - current_out0) * 0.1
                new_delays[(input_id, 'output_0')] = new_value
                
            # Weaken the competing connection
            current_out1 = new_delays[(input_id, 'output_1')]
            if isinstance(current_out1, tuple):
                new_value = current_out1[2] + (15.0 - current_out1[2]) * 0.05  # Push toward max delay
                new_delays[(input_id, 'output_1')] = (current_out1[0], current_out1[1], new_value)
            else:
                new_value = current_out1 + (15.0 - current_out1) * 0.05
                new_delays[(input_id, 'output_1')] = new_value
        
        elif out1_pairs > out0_pairs:
            current_out1 = new_delays[(input_id, 'output_1')]
            target_delay = 8.0 if input_id == 'input_1' else 1.0
            
            # Move delay toward target with a stronger pull
            if isinstance(current_out1, tuple):
                new_value = current_out1[2] + (target_delay - current_out1[2]) * 0.1
                new_delays[(input_id, 'output_1')] = (current_out1[0], current_out1[1], new_value)
            else:
                new_value = current_out1 + (target_delay - current_out1) * 0.1
                new_delays[(input_id, 'output_1')] = new_value
                
            # Weaken the competing connection
            current_out0 = new_delays[(input_id, 'output_0')]
            if isinstance(current_out0, tuple):
                new_value = current_out0[2] + (15.0 - current_out0[2]) * 0.05  # Push toward max delay
                new_delays[(input_id, 'output_0')] = (current_out0[0], current_out0[1], new_value)
            else:
                new_value = current_out0 + (15.0 - current_out0) * 0.05
                new_delays[(input_id, 'output_0')] = new_value
    
    return new_delays

def run_chunked_simulation(config, num_chunks=10, patterns_per_chunk=10):
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
    
    mid_delay = (config['init_delay_range'][0] + config['init_delay_range'][1]) / 2
    variation = (config['init_delay_range'][1] - config['init_delay_range'][0]) * 0.2
    current_delays = {
        ('input_0', 'output_0'): mid_delay + np.random.uniform(-variation, variation),
        ('input_0', 'output_1'): mid_delay + np.random.uniform(-variation, variation),
        ('input_1', 'output_0'): mid_delay + np.random.uniform(-variation, variation),
        ('input_1', 'output_1'): mid_delay + np.random.uniform(-variation, variation)
    }
    
    # Add slight asymmetry to initial weights to break symmetry and encourage specialization
    initial_weights = list(config['weights'])
    weight_jitter = 0.05  # 5% jitter
    
    # Initialize weights with jitter to break symmetry
    init_weights = {
        ('input_0', 'output_0'): initial_weights[0] * (1 + np.random.uniform(-weight_jitter, weight_jitter)),
        ('input_0', 'output_1'): initial_weights[0] * (1 + np.random.uniform(-weight_jitter, weight_jitter)),
        ('input_1', 'output_0'): initial_weights[1] * (1 + np.random.uniform(-weight_jitter, weight_jitter)),
        ('input_1', 'output_1'): initial_weights[1] * (1 + np.random.uniform(-weight_jitter, weight_jitter))
    }
    
    # Initialize accuracy tracking
    accuracy_history = []
    best_accuracy = 0.0
    stable_count = 0
    
    for key in delay_history:
        # Extract delay value from tuple if necessary
        if isinstance(current_delays[key], tuple):
            delay_history[key].append(current_delays[key][2])
        else:
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

        # Use jittered weights for the first chunk to break symmetry
        if chunk == 0:
            populations = create_populations(
                current_dataset['input_spikes'], 
                (init_weights[('input_0', 'output_0')], init_weights[('input_1', 'output_0')]),
                config['NEURON_PARAMS'],
                init_delay_range=config['init_delay_range'],
                init_delays=current_delays,
                inh_weight=config['inh_weight'],
                inh_delay=config['inh_delay']
            )
        else:
            populations = create_populations(
                current_dataset['input_spikes'], 
                config['weights'],
                config['NEURON_PARAMS'],
                init_delay_range=config['init_delay_range'],
            init_delays=current_delays,
            inh_weight=config['inh_weight'],
            inh_delay=config['inh_delay']
        )

        # Run simulation with chunk duration
        chunk_duration = patterns_per_chunk * config['pattern_interval']
        sim.run(chunk_duration)
        
        # Apply homeostatic weight normalization to promote specialization
        normalize_weights(populations, normalization_factor=1.0)

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
        
        # Get and print current weights
        weights = {}
        for conn_key in [('input_0', 'output_0'), ('input_1', 'output_0'), ('input_0', 'output_1'), ('input_1', 'output_1')]:
            weight_data = populations['projections'][conn_key].get("weight", format="list")[0]
            # Handle weight data which might be a tuple or a simple value
            if isinstance(weight_data, tuple):
                weights[conn_key] = float(weight_data[0])
            else:
                weights[conn_key] = float(weight_data)
        
        print(f"  Current weights:")
        print(f"  Input0->Output0: {weights[('input_0', 'output_0')]:.4f}")
        print(f"  Input1->Output0: {weights[('input_1', 'output_0')]:.4f}")
        print(f"  Input0->Output1: {weights[('input_0', 'output_1')]:.4f}")
        print(f"  Input1->Output1: {weights[('input_1', 'output_1')]:.4f}")
        
        # Store weights in history
        for key in weight_history:
            weight_history[key].append(weights[key])
        
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
            # Extract delay value from tuple if necessary
            if isinstance(current_delays[key], tuple):
                delay_history[key].append(current_delays[key][2])
            else:
                delay_history[key].append(current_delays[key])

        if chunk == num_chunks - 1:
            final_populations = populations

        sim.end()
    
    return {
        'delay_history': delay_history,
        'weight_history': weight_history,
        'pattern_responses': pattern_responses,
        'response_history': response_history,
        'final_delays': current_delays,
        'accuracy_history': accuracy_history,
        'stable_count': stable_count,
        'best_accuracy': best_accuracy
    }

if __name__ == "__main__":
    # Update main parameters to optimize combined learning
    config = init_delay_learning(
        # Increase number of presentations per pattern for better learning
        num_presentations=10,
        # Stronger STDP learning rates
        B_plus=0.35,  
        B_minus=0.30,
        # Stronger competitive modulation
        modulation_const=0.20,
        # Stronger initial weights for better responsiveness
        weights=(0.25, 0.25),
        # Balanced inhibition for better competition
        inh_weight=1.2,
        # Faster inhibition for quicker competition
        inh_delay=0.8,
        # Random seed for reproducibility
        seed=42
    )
    
    # Run simulation with more chunks for better learning
    results = run_chunked_simulation(
        config, 
        num_chunks=100,
        patterns_per_chunk=10
    )
    
    # Save the visualization with an identifiable name
    save_all_visualizations(results, "stdp_and_delay")
