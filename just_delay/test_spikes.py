import pyNN.nest as sim
import numpy as np
import matplotlib.pyplot as plt

def run_membrane_potential_test(experiment_type="coincident"):

    sim.setup(timestep=0.01)
    
    input_spikes = [
        [1],
        [5]
    ]
    
    # Define delays based on experiment type
    if experiment_type == "coincident":
        # Experiment 1: Delays adjusted to make spikes arrive simultaneously
        # Input 0 at 1ms + 5ms delay = 6ms arrival
        # Input 1 at 5ms + 1ms delay = 6ms arrival
        delays = np.array([
            [5, 5],  # Input 0 delay
            [1, 1]   # Input 1 delay
        ])
        experiment_name = "coincident_arrival"
        figtext = "Spikes arrive coincidentally at output neuron (6ms) - should trigger output spike"
    else:
        # Experiment 2: Delays set to create separated arrivals
        # Input 0 at 1ms + 1.5ms delay = 2.5ms arrival
        # Input 1 at 5ms + 8ms delay = 13ms arrival (10.5ms separation)
        delays = np.array([
            [1.5, 1.5],  # Input 0 delay
            [8, 8]   # Input 1 delay
        ])
        experiment_name = "separated_arrival"
        figtext = "Spikes arrive at different times at output neuron (10.5ms apart) - should NOT trigger output spike"


    # Biologically plausible neuron parameters based on cortical neurons
    NEURON_PARAMS = {
        'tau_m': 20.0,        # Membrane time constant (ms) - typical range 10-30ms [1,2]
        'tau_refrac': 2.0,    # Refractory period (ms) - typical range 1-3ms [3]
        'v_reset': -70.0,     # Reset potential (mV) - typically near resting potential [1]
        'v_rest': -65.0,      # Resting potential (mV) - typical range -60 to -70mV [1,4]
        'v_thresh': -50.0,    # Threshold potential (mV) - typical range -55 to -50mV [1,4]
        'cm': 0.25,           # Membrane capacitance (nF) - scaled for single compartment [2]
        'tau_syn_E': 5.0,     # Excitatory synaptic time constant (ms) - AMPA receptors ~5ms [2,5]
        'tau_syn_I': 10.0,    # Inhibitory synaptic time constant (ms) - GABA receptors ~10ms [2,5]
        'e_rev_E': 0.0,       # Excitatory reversal potential (mV) [1,2]
        'e_rev_I': -80.0      # Inhibitory reversal potential (mV) [1,2]
     }
     # Sources:
     # [1] Gerstner et al. (2014). Neuronal Dynamics.
     # [2] Izhikevich (2003). Simple Model of Spiking Neurons.
     # [3] Nawrot et al. (2008). Measurement of variability dynamics in cortical spike trains.
     # [4] Destexhe et al. (2003). Fluctuating synaptic conductances recreate in-vivo-like activity in neocortical neurons.
     # [5] Markram et al. (2015). Reconstruction and Simulation of Neocortical Microcircuitry.
    
    #weight set so a single input won't reach threshold but two coincident inputs will
    weights = np.array([
        [0.011, 0.011],
        [0.011, 0.011]
    ])

    
    input_pop = sim.Population(
        2,
        sim.SpikeSourceArray(spike_times=input_spikes),
        label="Input"
    )
    input_pop.record("spikes")
    
    output_pop_0 = sim.Population(1, sim.IF_cond_exp(**NEURON_PARAMS), label="Output_0")
    output_pop_1 = sim.Population(1, sim.IF_cond_exp(**NEURON_PARAMS), label="Output_1")
    
    output_pop_0.initialize(v=NEURON_PARAMS['v_rest'])
    output_pop_1.initialize(v=NEURON_PARAMS['v_rest'])
    
    output_pop_0.record(["spikes", "v"])
    output_pop_1.record(["spikes", "v"])
    
    connections = []
    # Connect input 0 to output 0
    conn = sim.Projection(
        input_pop[0:1], output_pop_0,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[0][0], delay=delays[0][0]),
        receptor_type="excitatory"
    )
    connections.append(conn)
    
    # Connect input 0 to output 1
    conn = sim.Projection(
        input_pop[0:1], output_pop_1,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[0][1], delay=delays[0][1]),
        receptor_type="excitatory"
    )
    connections.append(conn)
    
    # Connect input 1 to output 0
    conn = sim.Projection(
        input_pop[1:2], output_pop_0,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[1][0], delay=delays[1][0]),
        receptor_type="excitatory"
    )
    connections.append(conn)
    
    # Connect input 1 to output 1
    conn = sim.Projection(
        input_pop[1:2], output_pop_1,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=weights[1][1], delay=delays[1][1]),
        receptor_type="excitatory"
    )
    connections.append(conn)
    
    sim.run(100)
    
    input_data = input_pop.get_data().segments[0]
    input_spiketrains = input_data.spiketrains
    
    output_data_0 = output_pop_0.get_data().segments[0]
    output_data_1 = output_pop_1.get_data().segments[0]
    
    output_spiketrains = [
        output_data_0.spiketrains[0],
        output_data_1.spiketrains[0]
    ]
    vm_traces = []
    
    if len(output_data_0.analogsignals) > 0:
        vm_traces.append(output_data_0.analogsignals[0])
    else:
        print("WARNING: No membrane potential recorded for output neuron 0")
        vm_traces.append(None)
        
    if len(output_data_1.analogsignals) > 0:
        vm_traces.append(output_data_1.analogsignals[0])
    else:
        print("WARNING: No membrane potential recorded for output neuron 1")
        vm_traces.append(None)
    
    plt.figure(figsize=(15, 12))
    
    ax1 = plt.subplot(3, 1, 1)
    for i, spiketrain in enumerate(input_spiketrains):
        if len(spiketrain) > 0:
            plt.scatter(spiketrain, np.full(len(spiketrain), i), 
                       marker='|', s=100, label=f'Input {i}')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Neuron')
    plt.title('Input Spike Raster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 1.5)
    
    plt.figtext(0.5, 0.65, figtext, 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    ax2 = plt.subplot(3, 1, 2)
    for i, spiketrain in enumerate(output_spiketrains):
        if len(spiketrain) > 0:
            plt.scatter(spiketrain, np.full(len(spiketrain), i), 
                       marker='|', s=100, label=f'Output {i}')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Output Neuron')
    plt.title('Output Spike Raster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 1.5)
    
    print("\nInput spike times:")
    for i, spiketrain in enumerate(input_spiketrains):
        print(f"  Input neuron {i}: {[float(t) for t in spiketrain]}")
    
    print("\nOutput spike times:")
    for i, spiketrain in enumerate(output_spiketrains):
        print(f"  Output neuron {i}: {[float(t) for t in spiketrain]}")
    
    ax3 = plt.subplot(3, 1, 3)
    colors = ['r', 'b']
    for i, vm in enumerate(vm_traces):
        if vm is not None:
            plt.plot(vm.times, vm[:, 0], label=f'Output {i}', color=colors[i], linewidth=2)
        else:
            print(f"Cannot plot membrane potential for output neuron {i} - no data available")
    
    for i, spiketrain in enumerate(output_spiketrains):
        for t in spiketrain:
            plt.axvline(x=t, ymin=0.7, ymax=0.9, color=colors[i], linestyle='-', linewidth=2)
            plt.text(t, -52, f"out{i}", fontsize=10, ha='center', color=colors[i])
    
    for i, st in enumerate(input_spiketrains):
        for t in st:
            plt.axvline(x=t, ymin=0.05, ymax=0.15, color='black', linestyle='-', linewidth=1)
            plt.text(t, -72, f"in{i}", fontsize=10, ha='center')
    
    for i, input_neuron in enumerate(input_spiketrains):
        for t in input_neuron:
            t_value = float(t)
            for j in range(2):
                arrival_time = t_value + delays[i][j]
                plt.axvline(x=arrival_time, ymin=0.2, ymax=0.3, 
                           color=colors[j], linestyle='--', linewidth=1)
                plt.text(arrival_time, -60, f"arr{i}→{j}", fontsize=8, 
                        ha='center', color=colors[j])
    
    plt.axhline(y=NEURON_PARAMS['v_thresh'], color='k', linestyle='--', 
                label=f"Threshold ({NEURON_PARAMS['v_thresh']}mV)")
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Output Neuron Membrane Potentials with Spike Timing')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(-67, -45)
    
    plt.tight_layout()
    plt.savefig(f'membrane_potential_{experiment_name}.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as 'membrane_potential_{experiment_name}.png'")
    
    sim.end()

if __name__ == "__main__":
    print("\n=== EXPERIMENT 1: COINCIDENT ARRIVAL ===")
    run_membrane_potential_test("coincident")
    
    sim.end()
    
    print("\n=== EXPERIMENT 2: SEPARATED ARRIVAL ===")
    run_membrane_potential_test("separated")