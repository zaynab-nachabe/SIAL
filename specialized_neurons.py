import pyNN.nest as sim
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def get_neuron_params():
    NEURON_PARAMS = {
        'tau_m': 15,          # membrane time constant [between 10 and 30]
        'tau_refrac': 2.0,    # typical range 1-3ms
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
        0: {'input0': 4.0, 'input1': 12.0},  # input0 fires first, input1 follows 8ms later
        1: {'input1': 4.0, 'input0': 12.0}   # input1 fires first, input0 follows 8ms later
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

def setup_pattern0_network(input_spikes, weights, NEURON_PARAMS):
    input_pop_0 = sim.Population(
        1,
        sim.SpikeSourceArray(spike_times=input_spikes[0]),
        label="Input_0"
    )
    
    input_pop_1 = sim.Population(
        1,
        sim.SpikeSourceArray(spike_times=input_spikes[1]),
        label="Input_1"
    )
    
    input_pop_0.record("spikes")
    input_pop_1.record("spikes")
    
    output_pop_0 = sim.Population(1, sim.IF_cond_exp(**NEURON_PARAMS), label="Output_0")
    output_pop_1 = sim.Population(1, sim.IF_cond_exp(**NEURON_PARAMS), label="Output_1")
    
    # Initialize membrane potentials
    output_pop_0.initialize(v=NEURON_PARAMS['v_rest'])
    output_pop_1.initialize(v=NEURON_PARAMS['v_rest'])
    
    output_pop_0.record(["spikes", "v"])

    output_pop_1.record(["spikes", "v"])
    
    
    # Connect input 0 to output 0 coincident arrival in Pattern 0
    conn = sim.Projection(
        input_pop_0, output_pop_0,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[0], delay=8.0),  # Input0 fires at t=4ms, add 8ms delay
        receptor_type="excitatory"
    )

    # Connect input 1 to output 0 coincident arrival in Pattern 0
    conn = sim.Projection(
        input_pop_1, output_pop_0,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[0], delay=0.1),  # Input1 fires at t=12ms, minimal delay
        receptor_type="excitatory"
    )
    
    # Connect input 0 to output 1  mismatched for Pattern 0
    conn = sim.Projection(
        input_pop_0, output_pop_1,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[1], delay=1.0),
        receptor_type="excitatory"
    )
    
    # Connect input 1 to output 1 mismatched for Pattern 0
    conn = sim.Projection(
        input_pop_1, output_pop_1,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[1], delay=15.0),  #separated
        receptor_type="excitatory"
    )

    return {
        "input_0": input_pop_0,
        "input_1": input_pop_1,
        "output_0": output_pop_0,
        "output_1": output_pop_1
    }

def setup_pattern1_network(input_spikes, weights, NEURON_PARAMS):
    input_pop_0 = sim.Population(
        1,
        sim.SpikeSourceArray(spike_times=input_spikes[0]),
        label="Input_0"
    )
    
    input_pop_1 = sim.Population(
        1,
        sim.SpikeSourceArray(spike_times=input_spikes[1]),
        label="Input_1"
    )
    
    input_pop_0.record("spikes")
    input_pop_1.record("spikes")
    
    output_pop_0 = sim.Population(1, sim.IF_cond_exp(**NEURON_PARAMS), label="Output_0")
    output_pop_1 = sim.Population(1, sim.IF_cond_exp(**NEURON_PARAMS), label="Output_1")
    
    # Initialize membrane potentials
    output_pop_0.initialize(v=NEURON_PARAMS['v_rest'])
    output_pop_1.initialize(v=NEURON_PARAMS['v_rest'])
    
    # Recording
    output_pop_0.record(["spikes", "v"])
    output_pop_1.record(["spikes", "v"])
    
    
    # Connect input 0 to output 0 mismatched for Pattern 1
    conn = sim.Projection(
        input_pop_0, output_pop_0,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[0], delay=1.0),
        receptor_type="excitatory"
    )
    
    # Connect input 1 to output 0 mismatched for Pattern 1
    conn = sim.Projection(
        input_pop_1, output_pop_0,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[0], delay=15.0),  #separated
        receptor_type="excitatory"
    )
    
    # Connect input 0 to output 1 - tuned for coincident arrival in Pattern 1
    conn = sim.Projection(
        input_pop_0, output_pop_1,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[1], delay=0.1),  # Input0 fires at t=12ms, minimal delay
        receptor_type="excitatory"
    )
    
    # Connect input 1 to output 1 - tuned for coincident arrival in Pattern 1
    conn = sim.Projection(
        input_pop_1, output_pop_1,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[1], delay=8.0),  # Input1 fires at t=4ms, add 8ms delay
        receptor_type="excitatory"
    )

    return {
        "input_0": input_pop_0,
        "input_1": input_pop_1,
        "output_0": output_pop_0,
        "output_1": output_pop_1
    }

def collect_simulation_data(populations):
    input_pop_0 = populations["input_0"]
    input_pop_1 = populations["input_1"]
    output_pop_0 = populations["output_0"]
    output_pop_1 = populations["output_1"]
    
    output_data_0 = output_pop_0.get_data().segments[0]
    output_data_1 = output_pop_1.get_data().segments[0]
    
    output_spiketrains = [
        output_data_0.spiketrains[0],
        output_data_1.spiketrains[0]
    ]
    
    vm_traces = [
        output_data_0.filter(name="v")[0],
        output_data_1.filter(name="v")[0]
    ]
    
    return {
        "output_spiketrains": output_spiketrains,
        "vm_traces": vm_traces
    }

def plot_membrane_potentials(data, pattern_labels, pattern_start_times, experiment_name):
    vm_traces = data["vm_traces"]
    times = vm_traces[0].times
    neuron_params = get_neuron_params()
    colors = {0: 'blue', 1: 'red'}
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(times, vm_traces[0], 'b-', label='Membrane Potential', alpha=0.8, linewidth=1.5)
    axes[0].set_ylabel('Membrane Potential (mV)')
    axes[0].set_title('Output Neuron 0 (Pattern 0 specialist)', color='blue')
    axes[0].grid(True, alpha=0.3)
    
    axes[0].axhline(y=neuron_params['v_thresh'], color='k', linestyle='-', alpha=0.5, label='Threshold')
     
    axes[1].plot(times, vm_traces[1], 'r-', label='Membrane Potential', alpha=0.8, linewidth=1.5)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Membrane Potential (mV)')
    axes[1].set_title('Output Neuron 1 (Pattern 1 specialist)', color='red')
    axes[1].grid(True, alpha=0.3)
    
    axes[1].axhline(y=neuron_params['v_thresh'], color='k', linestyle='-', alpha=0.5, label='Threshold')
    
    for i, (pattern, start_time) in enumerate(zip(pattern_labels, pattern_start_times)):
        for ax_idx, ax in enumerate(axes):
            ax.axvline(x=start_time, color=colors[pattern], linestyle='--', alpha=0.3,
                      label=f'Pattern {pattern}' if i == 0 or i == 1 else "")
            
            y_pos = -45 if ax_idx == 0 else -45
            ax.text(start_time + 1, y_pos, f'P{pattern}', color=colors[pattern], 
                   fontsize=9, ha='left', va='bottom')
    
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    
    filename = f"{experiment_name}_membrane_potentials.png"
    plt.savefig(filename, dpi=150)    
    return filename

def plot_output_spikes(data, pattern_labels, pattern_start_times, experiment_name):
    output_spiketrains = data["output_spiketrains"]
    
    plt.figure(figsize=(12, 4))
    
    for i, spiketrain in enumerate(output_spiketrains):
        if len(spiketrain) > 0:
            color = 'blue' if i == 0 else 'red'
            plt.plot(spiketrain, [i+1] * len(spiketrain), 'o', color=color, 
                    label=f'Output {i}', markersize=8)
    
    colors = {0: 'blue', 1: 'red'}
    for i, (pattern, start_time) in enumerate(zip(pattern_labels, pattern_start_times)):
        plt.axvline(x=start_time, color=colors[pattern], linestyle='--', alpha=0.3,
                   label=f'Pattern {pattern}' if i == 0 or i == 1 else "")
        
        plt.text(start_time + 1, 0.5, f'P{pattern}', color=colors[pattern], 
                fontsize=9, ha='left', va='bottom')
    
    plt.xlabel('Time (ms)')
    plt.yticks([1, 2], ['Output 0\n(Pattern 0 specialist)', 'Output 1\n(Pattern 1 specialist)'])
    plt.title('Output Neuron Spike Times')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{experiment_name}_output_spikes.png"
    plt.savefig(filename, dpi=150)    
    return filename

def plot_results(data, NEURON_PARAMS, experiment_name):
    plot_membrane_potentials(
        data, 
        data['pattern_labels'], 
        data['pattern_start_times'], 
        experiment_name
    )
    
    plot_output_spikes(
        data, 
        data['pattern_labels'], 
        data['pattern_start_times'], 
        experiment_name
    )

def evaluate_specialization(output_spiketrains, pattern_labels, pattern_start_times):
    window_start = 8
    window_end = 25
    
    output0_correct = 0
    output1_correct = 0
    
    detections = []
    
    for i, (pattern, start_time) in enumerate(zip(pattern_labels, pattern_start_times)):
        t_start = start_time + window_start
        t_end = start_time + window_end
        
        output0_spikes = sum(1 for t in output_spiketrains[0] if t_start <= float(t) <= t_end)
        
        output1_spikes = sum(1 for t in output_spiketrains[1] if t_start <= float(t) <= t_end)
        
        output0_correct_for_pattern = (pattern == 0 and output0_spikes > 0) or (pattern == 1 and output0_spikes == 0)
        
        output1_correct_for_pattern = (pattern == 1 and output1_spikes > 0) or (pattern == 0 and output1_spikes == 0)
        
        if output0_correct_for_pattern:
            output0_correct += 1
        
        if output1_correct_for_pattern:
            output1_correct += 1
        
        detections.append({
            'pattern': pattern, 
            'start': start_time,
            'output0_spikes': output0_spikes,
            'output1_spikes': output1_spikes,
            'output0_correct': output0_correct_for_pattern,
            'output1_correct': output1_correct_for_pattern
        })
    
    total_patterns = len(pattern_labels)
    accuracy_output0 = (output0_correct / total_patterns) * 100 if total_patterns > 0 else 0
    accuracy_output1 = (output1_correct / total_patterns) * 100 if total_patterns > 0 else 0
    
    print("\n=== SUMMARY ===")
    print(f"Output Neuron 0: {accuracy_output0:.2f}%")
    print(f"Output Neuron 1: {accuracy_output1:.2f}%")
    print(f"Overall Accuracy: {((output0_correct + output1_correct) / (2 * total_patterns)) * 100:.2f}%")

def run_specialized_neurons_test(num_presentations=5):
    sim.setup(timestep=0.01)
    
    dataset = generate_dataset(num_presentations=num_presentations, pattern_interval=30)
    input_spikes = dataset['input_spikes']
    pattern_labels = dataset['pattern_labels']
    pattern_start_times = dataset['pattern_start_times']
    
    print(f"Generated dataset with {len(pattern_labels)} pattern presentations")
    print(f"Pattern sequence: {pattern_labels}")
    print(f"Pattern start times: {pattern_start_times}")
    print(f"Input 0 spikes at: {input_spikes[0]}")
    print(f"Input 1 spikes at: {input_spikes[1]}")
    
    experiment_name = f"pattern_specific_neurons_{num_presentations}x"
    
    NEURON_PARAMS = get_neuron_params()
    
    weights = [0.04, 0.04]  # [Output0 weight, Output1 weight]
    
    pattern0_input0 = []
    pattern0_input1 = []
    pattern0_times = []
    
    pattern1_input0 = []
    pattern1_input1 = []
    pattern1_times = []
    
    for i, (pattern, start_time) in enumerate(zip(pattern_labels, pattern_start_times)):
        if pattern == 0:
            pattern0_input0.append(input_spikes[0][i])
            pattern0_input1.append(input_spikes[1][i])
            pattern0_times.append(start_time)
        else:  # pattern == 1
            pattern1_input0.append(input_spikes[0][i])
            pattern1_input1.append(input_spikes[1][i])
            pattern1_times.append(start_time)
            
    pop0 = setup_pattern0_network(
        [pattern0_input0, pattern0_input1], 
        weights, 
        NEURON_PARAMS
    )
    
    pop1 = setup_pattern1_network(
        [pattern1_input0, pattern1_input1], 
        weights, 
        NEURON_PARAMS
    )
    
    end_time = pattern_start_times[-1] + 30 if pattern_start_times else 300
    sim.run(end_time)
    
    data0 = collect_simulation_data(pop0)
    data1 = collect_simulation_data(pop1)
    
    output_spiketrains = [
        data0["output_spiketrains"][0],
        data1["output_spiketrains"][1]
    ]
    
    combined_data = {
        "input_spiketrains": [
            np.array(input_spikes[0]),
            np.array(input_spikes[1])
        ],
        "output_spiketrains": output_spiketrains,
        "vm_traces": [
            data0["vm_traces"][0],
            data1["vm_traces"][1]
        ],
        "pattern_labels": pattern_labels,
        "pattern_start_times": pattern_start_times
    }
    
    plot_results(combined_data, NEURON_PARAMS, experiment_name)
    
    evaluate_specialization(output_spiketrains, pattern_labels, pattern_start_times)
    
    sim.end()
    
    return experiment_name

if __name__ == "__main__":
    run_specialized_neurons_test(num_presentations=5)
