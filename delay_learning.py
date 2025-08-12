import pyNN.nest as sim
import numpy as np
import random
import os
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

def get_input_patterns():
    base_patterns = {
        0: {'input0': 4.0, 'input1': 12.0},
        1: {'input1': 4.0, 'input0': 12.0},
        #2: {'input0': 5.0, 'input1': 5.0}
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



def delay_homeostasis(current_delays, R_target, R_observed, learning_rate_d=0.2):
    new_delays = current_delays.copy()
    
    K_output0 = (R_target - R_observed['output_0']) / R_target
    K_output1 = (R_target - R_observed['output_1']) / R_target
    
    def update_delay(connection, adjustment):
        if isinstance(current_delays[connection], tuple):
            delay_value = current_delays[connection][2]
            new_delay = delay_value - adjustment
            new_delay = max(0.1, min(20.0, new_delay))
            return (current_delays[connection][0], current_delays[connection][1], new_delay)
        else:
            new_delay = current_delays[connection] - adjustment
            return max(0.1, min(20.0, new_delay))
    new_delays[('input_0', 'output_0')] = update_delay(('input_0', 'output_0'), learning_rate_d * K_output0)
    new_delays[('input_1', 'output_0')] = update_delay(('input_1', 'output_0'), learning_rate_d * K_output0)
    
    new_delays[('input_0', 'output_1')] = update_delay(('input_0', 'output_1'), learning_rate_d * K_output1)
    new_delays[('input_1', 'output_1')] = update_delay(('input_1', 'output_1'), learning_rate_d * K_output1)
    
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

def create_populations(input_spikes, weights, NEURON_PARAMS, init_delay_range=(1.0, 15.0), 
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
    B_plus = 0.1,
    B_minus = 0.1,
    sigma_plus = 10.0,
    sigma_minus = 10.0,
    c_threshold = 0.5,
    modulation_const = 0.01,
    init_delay_range= (1.0, 15.0),
    weights = (0.04, 0.04),
    inh_weight = 0.5,
    inh_delay = 1.0,
    timestep=0.01,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    sim.setup(timestep=timestep)

    dataset = generate_dataset(
        num_presentations=num_presentations,
        pattern_interval=pattern_interval
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
                         sigma_plus, sigma_minus, c_threshold, modulation_const):
    new_delays = current_delays.copy()
    
    for connection, pairs in all_pairs.items():
        if len(pairs) == 0:
            if isinstance(current_delays[connection], tuple):
                delay_value = current_delays[connection][2]
                new_tuple = (current_delays[connection][0], current_delays[connection][1], 
                             delay_value + modulation_const)
                new_delays[connection] = new_tuple
            else:
                new_delays[connection] = current_delays[connection] + modulation_const
            continue
        
        if isinstance(current_delays[connection], tuple):
            current_delay = current_delays[connection][2]
        else:
            current_delay = current_delays[connection]
        
        if current_delay < c_threshold:
            if isinstance(current_delays[connection], tuple):
                new_tuple = (current_delays[connection][0], current_delays[connection][1], 
                             current_delay + modulation_const)
                new_delays[connection] = new_tuple
            else:
                new_delays[connection] = current_delay + modulation_const
            continue
        
        delta_delays = []
        for pre_time, post_time in pairs:
            delta_t = post_time - (pre_time + current_delay)
            
            if delta_t >= 0:
                delta_delay = -B_minus * np.exp(-delta_t / sigma_minus)
            else:
                delta_delay = B_plus * np.exp(delta_t / sigma_plus)
            
            delta_delays.append(delta_delay)
        
        if delta_delays:
            avg_delta = np.mean(delta_delays)
            
            new_delay = current_delay + avg_delta + modulation_const
            
            new_delay = max(0.1, min(20.0, new_delay))
            
            if isinstance(current_delays[connection], tuple):
                new_tuple = (current_delays[connection][0], current_delays[connection][1], new_delay)
                new_delays[connection] = new_tuple
            else:
                new_delays[connection] = new_delay
    
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
    
    current_delays = {
        ('input_0', 'output_0'): np.random.uniform(*config['init_delay_range']),
        ('input_0', 'output_1'): np.random.uniform(*config['init_delay_range']),
        ('input_1', 'output_0'): np.random.uniform(*config['init_delay_range']),
        ('input_1', 'output_1'): np.random.uniform(*config['init_delay_range'])
    }
    
    for key in delay_history:
        delay_history[key].append(current_delays[key])
    
    final_populations = None
    all_vm_traces = []
    pattern_responses = None
    
    for chunk in range(num_chunks):
        print(f"\n=== Running chunk {chunk+1}/{num_chunks} ===")

        sim.setup(timestep=config['timestep'])

        current_dataset = generate_dataset(
            num_presentations=patterns_per_chunk,
            pattern_interval=config['pattern_interval']
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

        {"output_0":{"pattern_0":[{}, {}, {}]}}
         


        print(f"Pattern response rates:")
        print(f"  Output 0: Pattern 0: {pattern_responses['output_0']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_0']['pattern_1']:.2f}")
        print(f"  Output 1: Pattern 0: {pattern_responses['output_1']['pattern_0']:.2f}, Pattern 1: {pattern_responses['output_1']['pattern_1']:.2f}")

        all_pairs = find_spike_pairs(
            populations, 
            output0_spikes, 
            output1_spikes, 
            current_delays
        )

        new_delays = apply_delay_learning(
            current_delays,
            all_pairs,
            config['B_plus'],
            config['B_minus'],
            config['sigma_plus'],
            config['sigma_minus'],
            config['c_threshold'],
            config['modulation_const']
        )

        observed_rates = {
            'output_0': len(output0_spikes) / patterns_per_chunk,
            'output_1': len(output1_spikes) / patterns_per_chunk
        }
        print("output spike 0:", output0_spikes)
        print("output spike 1: ", output1_spikes)
        target_rate = 1.0

        new_delays = delay_homeostasis(
            new_delays,
            target_rate,
            observed_rates,
            learning_rate_d=0.5
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
        'populations': final_populations,
        'all_vm_traces': all_vm_traces
    }

if __name__ == "__main__":
    config = init_delay_learning()
    results = run_chunked_simulation(config, num_chunks=20, patterns_per_chunk=1)
    save_all_visualizations(results, "delay_learning_experiment")
