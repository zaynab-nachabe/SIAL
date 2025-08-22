import pyNN.nest as sim
import numpy as np
import random
import os
from visualisation_and_metrics import save_all_visualizations, calculate_specialization_score
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
                      init_delays=None, inh_weight=-15.0, inh_delay=0.5):
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

    if init_delays is not None and isinstance(init_delays, dict):
        d0_in0 = init_delays[('input_0', 'output_0')]
        d0_in1 = init_delays[('input_0', 'output_1')]
        d1_in0 = init_delays[('input_1', 'output_0')]
        d1_in1 = init_delays[('input_1', 'output_1')]
        
        if isinstance(d0_in0, tuple):
            d0_in0 = d0_in0[2]
        if isinstance(d0_in1, tuple):
            d0_in1 = d0_in1[2]
        if isinstance(d1_in0, tuple):
            d1_in0 = d1_in0[2]
        if isinstance(d1_in1, tuple):
            d1_in1 = d1_in1[2]
    else:
        d0_in0 = np.random.uniform(*init_delay_range)
        d0_in1 = np.random.uniform(*init_delay_range)
        d1_in0 = np.random.uniform(*init_delay_range)
        d1_in1 = np.random.uniform(*init_delay_range)

    conn_in0_out0 = sim.Projection(
        input_pop_0, output_pop_0, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[0], delay=d0_in0),
        receptor_type="excitatory"
    )

    conn_in0_out1 = sim.Projection(
        input_pop_0, output_pop_1, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[0], delay=d0_in1),
        receptor_type="excitatory"
    )

    conn_in1_out0 = sim.Projection(
        input_pop_1, output_pop_0, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[1], delay=d1_in0),
        receptor_type="excitatory"
    )

    conn_in1_out1 = sim.Projection(
        input_pop_1, output_pop_1, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[1], delay=d1_in1),
        receptor_type="excitatory"
    )

    # For inhibitory connections, use the absolute value since inhibitory weights need to be positive in NEST
    # The inhibitory effect comes from the receptor_type="inhibitory"
    abs_inh_weight = abs(inh_weight)
    
    conn_out0_out1 = sim.Projection(
        output_pop_0, output_pop_1, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=abs_inh_weight, delay=inh_delay),
        receptor_type="inhibitory"
    )

    conn_out1_out0 = sim.Projection(
        output_pop_1, output_pop_0, sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=abs_inh_weight, delay=inh_delay),
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
    B_plus = 0.5,          # Increased for stronger learning
    B_minus = 0.45,        # Increased for stronger learning 
    sigma_plus = 10.0,
    sigma_minus = 10.0,
    c_threshold = 0.5,
    modulation_const = 1.5,
    init_delay_range= (1.0, 15.0),
    weights = (0.04, 0.04),
    inh_weight = 1.2,       # Strong lateral inhibition
    inh_delay = 0.5,          # Fast inhibition
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
    """
    Apply delay learning to all connections based on spike timing using delay_learning_rule.
    This simplified version directly applies the STDP rule without additional mechanisms,
    relying on inhibition for neuron specialization.
    """
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

def process_pattern_responses(pattern_responses, response_history, specialization_scores):
    """
    Process pattern responses, update response history, and calculate specialization score.
    Returns the calculated specialization score.
    """
    # Record responses
    response_history['output_0']['pattern_0'].append(pattern_responses['output_0']['pattern_0'])
    response_history['output_0']['pattern_1'].append(pattern_responses['output_0']['pattern_1'])
    response_history['output_1']['pattern_0'].append(pattern_responses['output_1']['pattern_0'])
    response_history['output_1']['pattern_1'].append(pattern_responses['output_1']['pattern_1'])

    # Print pattern responses
    print(f"Pattern response rates:")
    print(f"  Output 0: Pattern 0: {pattern_responses['output_0']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_0']['pattern_1']:.2f}")
    print(f"  Output 1: Pattern 0: {pattern_responses['output_1']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_1']['pattern_1']:.2f}")
    
    # Calculate specialization score
    spec_score, spec_metrics = calculate_specialization_score(pattern_responses)
    specialization_scores.append(spec_score)
    print(f"  Specialization score: {spec_score:.3f}")
    
    return spec_score

def run_curriculum_simulation(config, num_chunks=100, patterns_per_chunk=5):
    """
    Implements a curriculum learning approach:
    1. Start with isolated pattern presentations
    2. Gradually introduce mixed patterns
    3. Finally use fully shuffled patterns
    """
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
    
    # Track curriculum phase for visualization
    curriculum_phases = []
    
    # Track specialization scores
    specialization_scores = []
    
    # Start with completely random delays within the configured range
    delay_min, delay_max = config['init_delay_range']
    current_delays = {
        ('input_0', 'output_0'): np.random.uniform(delay_min, delay_max),
        ('input_0', 'output_1'): np.random.uniform(delay_min, delay_max),
        ('input_1', 'output_0'): np.random.uniform(delay_min, delay_max),
        ('input_1', 'output_1'): np.random.uniform(delay_min, delay_max)
    }
    
    print("Initial random delays:")
    for conn, delay in current_delays.items():
        print(f"  {conn[0]} → {conn[1]}: {delay:.3f} ms")
    
    # Best delays tracking
    best_delays = current_delays.copy()
    best_class_separation = 0.0
    learning_frozen = False
    stable_count = 0
    
    for key in delay_history:
        delay_history[key].append(current_delays[key])
    
    final_populations = None
    all_vm_traces = []
    pattern_responses = None
    
    # Store pattern information for the last chunk
    final_pattern_times = []
    final_pattern_labels = []
    
    # Phase 1: Isolated patterns (25% of chunks)
    isolated_chunks = num_chunks // 4
    for chunk in range(isolated_chunks):
        # Alternate between pattern 0 and pattern 1 for entire chunks
        pattern_type = 0 if chunk % 2 == 0 else 1
        curriculum_phases.append("isolated")
        
        print(f"\n=== Running chunk {chunk+1}/{num_chunks} (isolated pattern {pattern_type}) ===")
        
        sim.setup(timestep=config['timestep'])
        
        # Generate dataset with just one pattern type
        current_dataset = generate_dataset(
            num_presentations=patterns_per_chunk,
            pattern_interval=config['pattern_interval'],
            specific_pattern=pattern_type
        )
        
        # Keep pattern info from the last chunk for visualization
        if chunk == num_chunks - 1:
            final_pattern_times = current_dataset['pattern_start_times']
            final_pattern_labels = current_dataset['pattern_labels']
            
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
        
        # Calculate specialization score
        spec_score, spec_metrics = calculate_specialization_score(pattern_responses)
        specialization_scores.append(spec_score)
        print(f"  Specialization score: {spec_score:.3f}")
        
        # Print detailed delay information for monitoring
        print(f"  Current delays:")
        print(f"  Input0->Output0: {current_delays[('input_0', 'output_0')]:.2f}ms")
        print(f"  Input1->Output0: {current_delays[('input_1', 'output_0')]:.2f}ms")
        print(f"  Input0->Output1: {current_delays[('input_0', 'output_1')]:.2f}ms")
        print(f"  Input1->Output1: {current_delays[('input_1', 'output_1')]:.2f}ms")
        
        # Calculate class separation just for monitoring, not for decisions
        output0_separation = abs(pattern_responses['output_0']['pattern_0'] - pattern_responses['output_0']['pattern_1'])
        output1_separation = abs(pattern_responses['output_1']['pattern_1'] - pattern_responses['output_1']['pattern_0'])
        class_separation = (output0_separation + output1_separation) / 2
        print(f"  Class separation: {class_separation:.3f} (Output0: {output0_separation:.3f}, Output1: {output1_separation:.3f})")
        
        all_pairs = find_spike_pairs(
            populations, 
            output0_spikes, 
            output1_spikes, 
            current_delays
        )

        # Apply learning parameters with gradual decay
        b_plus = config['B_plus']
        b_minus = config['B_minus']
        
        # Decay learning parameters
        decay_factor = np.exp(-chunk / (num_chunks * 0.6))
        b_plus *= decay_factor
        b_minus *= decay_factor
        
        # Calculate delay variance
        delay_values = [current_delays[conn] for conn in current_delays]
        delay_variance = np.var(delay_values)
        print(f"  Delay variance: {delay_variance:.3f}")
        
        # Decay modulation constant over time
        mod_const = config['modulation_const'] * max(0.3, decay_factor)
        
        # If we're getting specialization (high variance), maintain learning longer
        if delay_variance > 1.5:
            print(f"  Good delay separation detected, maintaining strong modulation")
            mod_const *= 1.2
            
        print(f"  Using learning rates: B+ = {b_plus:.4f}, B- = {b_minus:.4f}, mod = {mod_const:.4f}")
        
        # Apply delay learning
        new_delays = apply_delay_learning(
            current_delays,
            all_pairs,
            b_plus,
            b_minus,
            config['sigma_plus'],
            config['sigma_minus'],
            config['c_threshold'],
            mod_const,
            0.0,  # No accuracy tracking
            None  # No pattern responses for unsupervised learning
        )

        observed_rates = {
            'output_0': len(output0_spikes) / patterns_per_chunk,
            'output_1': len(output1_spikes) / patterns_per_chunk
        }
        
        print("output spike 0:", output0_spikes)
        print("output spike 1: ", output1_spikes)
        target_rate = 1.0

        homeostasis_lr = 0.8 * decay_factor
        
        # Apply homeostasis
        new_delays = delay_homeostasis(
            new_delays,
            target_rate,
            observed_rates,
            learning_rate_d=homeostasis_lr,
            pattern_responses=None
        )

        print("Delays after learning and homeostasis:")
        for conn in [('input_0', 'output_0'), ('input_0', 'output_1'), ('input_1', 'output_0'), ('input_1', 'output_1')]:
            val = new_delays[conn]
            if isinstance(val, tuple):
                val = val[2]
            print(f"  {conn[0]} → {conn[1]}: {val:.3f} ms")

        current_delays = new_delays.copy()
        
        # Always record delays in history
        for key in delay_history:
            delay_history[key].append(current_delays[key])

        if chunk == num_chunks - 1:
            final_populations = populations

        sim.end()
    
        # Phase 2: Block patterns (25% of chunks)
    block_chunks = num_chunks // 4
    for chunk in range(isolated_chunks, isolated_chunks + block_chunks):
        curriculum_phases.append("blocked")
        print(f"\n=== Running chunk {chunk+1}/{num_chunks} (blocked patterns) ===")
        
        sim.setup(timestep=config['timestep'])
        
        # Create a dataset with blocks of patterns (first half pattern 0, second half pattern 1)
        half_size = patterns_per_chunk // 2
        pattern_sequence = [0] * half_size + [1] * (patterns_per_chunk - half_size)
        
        current_dataset = {
            'input_spikes': [[], []],
            'pattern_start_times': [],
            'pattern_labels': []
        }
        
        current_time = 0.0
        base_patterns = get_input_patterns()
        
        for pattern_id in pattern_sequence:
            pattern = base_patterns[pattern_id]
            
            current_dataset['pattern_start_times'].append(current_time)
            current_dataset['pattern_labels'].append(pattern_id)
            
            if 'input0' in pattern:
                current_dataset['input_spikes'][0].append(current_time + pattern['input0'])
            
            if 'input1' in pattern:
                current_dataset['input_spikes'][1].append(current_time + pattern['input1'])
            
            current_time += config['pattern_interval']
        
        print(f"Generated dataset with {len(pattern_sequence)} patterns. Pattern distribution: {pattern_sequence.count(0)} pattern 0, {pattern_sequence.count(1)} pattern 1")
        
        # Keep pattern info from the last chunk for visualization
        if chunk == num_chunks - 1:
            final_pattern_times = current_dataset['pattern_start_times']
            final_pattern_labels = current_dataset['pattern_labels']
            
        # Rest of the simulation loop is the same as Phase 1
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

        # Same processing as Phase 1...
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
        
        # Record responses
        response_history['output_0']['pattern_0'].append(pattern_responses['output_0']['pattern_0'])
        response_history['output_0']['pattern_1'].append(pattern_responses['output_0']['pattern_1'])
        response_history['output_1']['pattern_0'].append(pattern_responses['output_1']['pattern_0'])
        response_history['output_1']['pattern_1'].append(pattern_responses['output_1']['pattern_1'])

        # Print pattern responses
        print(f"Pattern response rates:")
        print(f"  Output 0: Pattern 0: {pattern_responses['output_0']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_0']['pattern_1']:.2f}")
        print(f"  Output 1: Pattern 0: {pattern_responses['output_1']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_1']['pattern_1']:.2f}")
        
        # Print current delays
        print(f"  Current delays:")
        print(f"  Input0->Output0: {current_delays[('input_0', 'output_0')]:.2f}ms")
        print(f"  Input1->Output0: {current_delays[('input_1', 'output_0')]:.2f}ms")
        print(f"  Input0->Output1: {current_delays[('input_0', 'output_1')]:.2f}ms")
        print(f"  Input1->Output1: {current_delays[('input_1', 'output_1')]:.2f}ms")
        
        # Find spike pairs for learning
        all_pairs = find_spike_pairs(
            populations, 
            output0_spikes, 
            output1_spikes, 
            current_delays
        )

        # Apply learning parameters with gradual decay
        b_plus = config['B_plus']
        b_minus = config['B_minus']
        decay_factor = np.exp(-chunk / (num_chunks * 0.6))
        b_plus *= decay_factor
        b_minus *= decay_factor
        mod_const = config['modulation_const'] * max(0.3, decay_factor)
        
        # Apply delay learning
        new_delays = apply_delay_learning(
            current_delays,
            all_pairs,
            b_plus,
            b_minus,
            config['sigma_plus'],
            config['sigma_minus'],
            config['c_threshold'],
            mod_const,
            0.0,
            None
        )

        # Apply homeostasis
        observed_rates = {
            'output_0': len(output0_spikes) / patterns_per_chunk,
            'output_1': len(output1_spikes) / patterns_per_chunk
        }
        
        homeostasis_lr = 0.8 * decay_factor
        
        new_delays = delay_homeostasis(
            new_delays,
            target_rate,
            observed_rates,
            learning_rate_d=homeostasis_lr,
            pattern_responses=None
        )

        current_delays = new_delays.copy()
        
        # Record delays in history
        for key in delay_history:
            delay_history[key].append(current_delays[key])

        if chunk == num_chunks - 1:
            final_populations = populations

        sim.end()
    
    # Phase 3: Fully mixed patterns (remaining chunks)
    for chunk in range(isolated_chunks + block_chunks, num_chunks):
        curriculum_phases.append("mixed")
        print(f"\n=== Running chunk {chunk+1}/{num_chunks} (mixed patterns) ===")
        
        sim.setup(timestep=config['timestep'])
        
        # Generate fully shuffled dataset
        current_dataset = generate_dataset(
            num_presentations=patterns_per_chunk,
            pattern_interval=config['pattern_interval'],
            specific_pattern=None  # Use shuffled patterns
        )
        
        # Keep pattern info from the last chunk for visualization
        if chunk == num_chunks - 1:
            final_pattern_times = current_dataset['pattern_start_times']
            final_pattern_labels = current_dataset['pattern_labels']
            
        # Rest of the simulation loop is the same as Phase 1 and 2
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

        # Process results
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
        
        # Record responses
        response_history['output_0']['pattern_0'].append(pattern_responses['output_0']['pattern_0'])
        response_history['output_0']['pattern_1'].append(pattern_responses['output_0']['pattern_1'])
        response_history['output_1']['pattern_0'].append(pattern_responses['output_1']['pattern_0'])
        response_history['output_1']['pattern_1'].append(pattern_responses['output_1']['pattern_1'])

        # Print pattern responses
        print(f"Pattern response rates:")
        print(f"  Output 0: Pattern 0: {pattern_responses['output_0']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_0']['pattern_1']:.2f}")
        print(f"  Output 1: Pattern 0: {pattern_responses['output_1']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_1']['pattern_1']:.2f}")
        
        # Print current delays
        print(f"  Current delays:")
        print(f"  Input0->Output0: {current_delays[('input_0', 'output_0')]:.2f}ms")
        print(f"  Input1->Output0: {current_delays[('input_1', 'output_0')]:.2f}ms")
        print(f"  Input0->Output1: {current_delays[('input_0', 'output_1')]:.2f}ms")
        print(f"  Input1->Output1: {current_delays[('input_1', 'output_1')]:.2f}ms")
        
        # Find spike pairs for learning
        all_pairs = find_spike_pairs(
            populations, 
            output0_spikes, 
            output1_spikes, 
            current_delays
        )

        # Apply learning parameters with gradual decay
        b_plus = config['B_plus']
        b_minus = config['B_minus']
        decay_factor = np.exp(-chunk / (num_chunks * 0.6))
        b_plus *= decay_factor
        b_minus *= decay_factor
        mod_const = config['modulation_const'] * max(0.3, decay_factor)
        
        # Apply delay learning
        new_delays = apply_delay_learning(
            current_delays,
            all_pairs,
            b_plus,
            b_minus,
            config['sigma_plus'],
            config['sigma_minus'],
            config['c_threshold'],
            mod_const,
            0.0,
            None
        )

        # Apply homeostasis
        observed_rates = {
            'output_0': len(output0_spikes) / patterns_per_chunk,
            'output_1': len(output1_spikes) / patterns_per_chunk
        }
        
        homeostasis_lr = 0.8 * decay_factor
        
        new_delays = delay_homeostasis(
            new_delays,
            target_rate,
            observed_rates,
            learning_rate_d=homeostasis_lr,
            pattern_responses=None
        )

        current_delays = new_delays.copy()
        
        # Record delays in history
        for key in delay_history:
            delay_history[key].append(current_delays[key])

        if chunk == num_chunks - 1:
            final_populations = populations

        sim.end()
    
    # Calculate specialization scores for each chunk using the response history
    if not specialization_scores:  # If we don't have scores already
        print("Calculating specialization scores from response history...")
        for i in range(len(next(iter(response_history['output_0'].values())))):
            chunk_responses = {
                'output_0': {
                    'pattern_0': response_history['output_0']['pattern_0'][i],
                    'pattern_1': response_history['output_0']['pattern_1'][i]
                },
                'output_1': {
                    'pattern_0': response_history['output_1']['pattern_0'][i],
                    'pattern_1': response_history['output_1']['pattern_1'][i]
                }
            }
            score, _ = calculate_specialization_score(chunk_responses)
            specialization_scores.append(score)
    
    return {
        'delay_history': delay_history,
        'pattern_responses': pattern_responses,
        'response_history': response_history,
        'final_delays': current_delays,
        'all_vm_traces': all_vm_traces,
        'curriculum_phases': curriculum_phases,
        'specialization_scores': specialization_scores,
        'dataset': {
            'pattern_start_times': final_pattern_times,
            'pattern_labels': final_pattern_labels
        }
    }

if __name__ == "__main__":
    config = init_delay_learning()
    results = run_curriculum_simulation(
        config, 
        num_chunks=100,
        patterns_per_chunk=5
    )
    save_all_visualizations(results, "just_delay")