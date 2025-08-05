#!/usr/bin/env python3
import brian2 as b2
from brian2 import *
import numpy as np
import random
import matplotlib.pyplot as plt

b2.prefs.codegen.target = 'numpy'

def delay_learning_rule(pre_time, post_time, current_delay, 
                       B_plus=10.0, B_minus=8.0, sigma_plus=5.0, sigma_minus=5.0):
    """Nadafian & Ganjtabesh 2020 delay learning rule with biological constraints"""
    delta_t = post_time - pre_time - current_delay
    if delta_t >= 0:
        return -B_minus * np.exp(-delta_t / sigma_minus)
    else:
        return B_plus * np.exp(delta_t / sigma_plus)

def check_stop_condition(delays, post_idx, c=8.5):
    """
    Check stop condition for delay learning to prevent hyper-myelination.
    For post-synaptic neuron j, stop delay learning on all d_i,j if any d_k,j < c
    where c > B_minus (we use c=8.5 > B_minus=8.0)
    """
    # Get all delays targeting this post-synaptic neuron
    delays_to_post = delays[:, post_idx]  # delays from all pre-synaptic neurons
    
    # Check if any delay is below the threshold
    min_delay = np.min(delays_to_post)
    stop_learning = min_delay < c
    
    return stop_learning, min_delay

def apply_delay_modulation(delays, modulation_step=0.01):
    """
    Apply constant small increase to all delays at every time step.
    This prevents convergence issues while maintaining biological plausibility.
    """
    modulated_delays = delays.copy()
    modulated_delays += modulation_step
    
    # Clip to reasonable range (don't let delays grow indefinitely)
    modulated_delays = np.clip(modulated_delays, 0.1, 8.0)
    
    return modulated_delays

def find_spike_pairs(pre_spikes, post_spikes, current_delay, window=20.0):
    pairs = []
    for pre_time in pre_spikes:
        arrival_time = pre_time + current_delay
        valid_posts = [t for t in post_spikes if abs(t - arrival_time) <= window]
        if valid_posts:
            closest_post = min(valid_posts, key=lambda t: abs(t - arrival_time))
            pairs.append((pre_time, closest_post))
    return pairs

def apply_delay_homeostasis(delays, output_spikes, simulation_time, 
                           R_target=10.0, lambda_d=0.1):
    homeostasis_delays = delays.copy()
        
    for post_idx in range(3):
        spike_count = len(output_spikes[post_idx])
        R_observed = spike_count / (simulation_time / 1000.0)
        
        K = (R_target - R_observed) / R_target
        
        for pre_idx in range(2):
            # dnew = dcurrent - λd × K
            delta_d_homeo = -lambda_d * K
            
            max_change = 0.2
            delta_d_homeo = np.clip(delta_d_homeo, -max_change, max_change)
            
            homeostasis_delays[pre_idx, post_idx] += delta_d_homeo
            homeostasis_delays[pre_idx, post_idx] = np.clip(
                homeostasis_delays[pre_idx, post_idx], 0.1, 8.0)
    
    return homeostasis_delays

def apply_weight_homeostasis(synapses, output_spikes, simulation_time, 
                            R_target=15.0, lambda_w=0.01):
    
    for post_idx in range(3):
        spike_count = len(output_spikes[post_idx])
        R_observed = spike_count / (simulation_time / 1000.0)
        
        K = (R_target - R_observed) / R_target
        
        for pre_idx in range(2):
            synapse_idx = pre_idx * 3 + post_idx
            current_weight = synapses[synapse_idx].w[0]
            
            delta_w_homeo = lambda_w * K
            new_weight = current_weight + delta_w_homeo
            
            new_weight = np.clip(new_weight, 1.0, 10.0)
            synapses[synapse_idx].w = new_weight
    
    return synapses

def plot_delay_evolution(all_delay_states, save_path=None):
    n_cycles = len(all_delay_states)
    cycles = range(1, n_cycles + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Delay Evolution Across Simulation Cycles', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for pre_idx in range(2):
        for post_idx in range(3):
            row = pre_idx
            col = post_idx
            ax = axes[row, col]
            
            delay_values = [delay_state[pre_idx, post_idx] for delay_state in all_delay_states]
            
            ax.plot(cycles, delay_values, 'o-', 
                   color=colors[pre_idx * 3 + post_idx], 
                   linewidth=2, markersize=6, 
                   label=f'Input{pre_idx+1}→Output{post_idx+1}')
            
            ax.set_title(f'Input{pre_idx+1} → Output{post_idx+1}', fontweight='bold')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Delay (ms)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, n_cycles + 0.5)
            
            final_delay = delay_values[-1]
            ax.annotate(f'{final_delay:.2f}ms', 
                       xy=(n_cycles, final_delay), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Delay evolution plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig

def plot_delay_evolution_summary(all_delay_states, save_path=None):
    n_cycles = len(all_delay_states)
    cycles = range(1, n_cycles + 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Delay Evolution Summary - All Connections', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-', '--', '-', '--']
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    for pre_idx in range(2):
        for post_idx in range(3):
            connection_idx = pre_idx * 3 + post_idx
            
            delay_values = [delay_state[pre_idx, post_idx] for delay_state in all_delay_states]
            
            ax.plot(cycles, delay_values, 
                   color=colors[connection_idx], 
                   linestyle=linestyles[connection_idx],
                   marker=markers[connection_idx],
                   linewidth=2, markersize=8, 
                   label=f'Input{pre_idx+1}→Output{post_idx+1}')
    
    ax.set_xlabel('Cycle', fontsize=12)
    ax.set_ylabel('Delay (ms)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, n_cycles + 0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.text(0.02, 0.98, f'Cycles: {n_cycles}', transform=ax.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Delay evolution summary plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig

def plot_accuracy_analysis(confusion_matrix, overall_accuracy, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Network Performance Analysis (Overall Accuracy: {overall_accuracy:.1%})', 
                fontsize=16, fontweight='bold')
    
    cm_transposed = confusion_matrix.T
    
    im1 = ax1.imshow(cm_transposed, cmap='Blues', aspect='auto')
    
    pattern_names = ['P1', 'P2', 'P3']
    pred_names = ['P1', 'P2', 'P3', 'None']
    
    for i in range(3):
        for j in range(4): 
            text = ax1.text(j, i, int(cm_transposed[i, j]), 
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(pred_names)
    ax1.set_yticklabels(pattern_names)
    ax1.set_xlabel('Predicted Pattern')
    ax1.set_ylabel('True Pattern')
    ax1.set_title('Confusion Matrix')
    
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    total_per_pattern = np.sum(cm_transposed, axis=1)
    correct_per_pattern = np.diag(confusion_matrix[:3, :3])
    incorrect_per_pattern = total_per_pattern - correct_per_pattern
    no_response_per_pattern = confusion_matrix[3, :]
    
    x_pos = np.arange(3)
    width = 0.6
    
    p1 = ax2.bar(x_pos, correct_per_pattern, width, label='Correct', color='green', alpha=0.8)
    p2 = ax2.bar(x_pos, incorrect_per_pattern, width, bottom=correct_per_pattern, 
                label='Misclassified', color='orange', alpha=0.8)
    p3 = ax2.bar(x_pos, no_response_per_pattern, width, 
                bottom=correct_per_pattern + incorrect_per_pattern,
                label='No Response', color='red', alpha=0.8)
    
    ax2.set_xlabel('True Pattern')
    ax2.set_ylabel('Number of Presentations')
    ax2.set_title('Response Distribution by Pattern')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pattern_names)
    ax2.legend()
    
    for i in range(3):
        total = total_per_pattern[i]
        if total > 0:
            correct_pct = (correct_per_pattern[i] / total) * 100
            ax2.text(i, total + 0.5, f'{correct_pct:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy analysis plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig

def main():
    print("=" * 60)
    print("BRIAN2 DELAY LEARNING EXPERIMENT")
    print("=" * 60)
    
    start_scope()
    
    patterns = [
        {"name": "pattern_1", "delay": 1.0, "first_neuron": 0},
        {"name": "pattern_2", "delay": 3.0, "first_neuron": 1}, 
        {"name": "pattern_3", "delay": 0.0, "simultaneous": True}
    ]
    
    input_spikes = [[], []]
    pattern_sequence = []
    current_time = 10
    
    all_patterns = []
    for _ in range(20):
        for pattern in patterns:
            all_patterns.append(pattern.copy())
    
    random.seed(42)
    random.shuffle(all_patterns)
    
    for pattern in all_patterns:
        pattern_copy = pattern.copy()
        pattern_copy["start_time"] = current_time
        pattern_sequence.append(pattern_copy)
        
        if pattern.get("simultaneous", False):
            input_spikes[0].append(current_time)
            input_spikes[1].append(current_time)
        else:
            first_neuron = pattern["first_neuron"]
            second_neuron = 1 - first_neuron
            delay = pattern["delay"]
            
            input_spikes[first_neuron].append(current_time)
            input_spikes[second_neuron].append(current_time + delay)
        
        current_time += 15
    
    input_spikes[0] = sorted(input_spikes[0])
    input_spikes[1] = sorted(input_spikes[1])
    
    pattern_counts = {}
    for pattern in pattern_sequence:
        name = pattern["name"]
        pattern_counts[name] = pattern_counts.get(name, 0) + 1
    
    input1_times = np.array(input_spikes[0]) * ms
    input1_indices = [0] * len(input_spikes[0])
    
    input2_times = np.array(input_spikes[1]) * ms
    input2_indices = [0] * len(input_spikes[1])
    
    input1_group = SpikeGeneratorGroup(1, input1_indices, input1_times)
    input2_group = SpikeGeneratorGroup(1, input2_indices, input2_times)
    
    eqs = '''
    dv/dt = (v_rest - v + I_syn + I_noise) / tau_m : volt (unless refractory)
    dI_noise/dt = -I_noise/tau_noise + sigma_noise * sqrt(2/tau_noise) * xi : volt
    I_syn : volt
    v_rest : volt (constant)
    v_thresh : volt (constant)
    v_reset : volt (constant)
    tau_m : second (constant)
    tau_noise : second (constant)
    sigma_noise : volt (constant)
    '''
    
    output_group = NeuronGroup(3, eqs,
                              threshold='v > v_thresh',
                              reset='v = v_reset',
                              refractory=2*ms,
                              method='euler')  # Changed to euler for noise integration
    
    output_group.v = -65*mV
    output_group.v_rest = -65*mV
    output_group.v_thresh = -55*mV  # Lowered threshold from -52mV to -55mV to make firing easier
    output_group.v_reset = -70*mV
    output_group.tau_m = 20*ms
    output_group.tau_noise = 10*ms
    output_group.sigma_noise = 1.0*mV  # Increased noise from 0.5mV to 1.0mV
    output_group.I_noise = 0*mV
    
    np.random.seed(42)
    synapses = []
    initial_delays = np.zeros((2, 3))
    initial_weights = np.zeros((2, 3))
    
    print("Creating synapses...")
    
    # Debug: print initial weights and delays for analysis
    
    # Input1 -> All outputs
    for post_idx in range(3):
        # More balanced weight initialization - different ranges for each output
        if post_idx == 0:
            weight = np.random.uniform(4.0, 6.0)  # Higher weights for output1
        elif post_idx == 1:
            weight = np.random.uniform(4.0, 6.0)  # Higher weights for output2
        else:
            weight = np.random.uniform(2.0, 4.0)  # Lower weights for output3
            
        delay = np.random.uniform(0.5, 3.0)  # Narrower delay range
        
        syn = Synapses(input1_group, output_group,
                      'w : 1', on_pre='I_syn_post += w * mV')
        syn.connect(i=0, j=post_idx)
        syn.w = weight
        syn.delay = delay * ms
        
        synapses.append(syn)
        initial_delays[0, post_idx] = delay
        initial_weights[0, post_idx] = weight
        print(f"  Input1→Output{post_idx+1}: w={weight:.2f}, d={delay:.2f}ms")
        print(f"  Input2->Output{post_idx+1}: w={weight:.2f}, d={delay:.2f}ms")
    # Input2 -> All outputs  
    for post_idx in range(3):
        if post_idx == 0:
            weight = np.random.uniform(4.0, 6.0)
        elif post_idx == 1:
            weight = np.random.uniform(4.0, 6.0) 
        else:
            weight = np.random.uniform(2.0, 4.0)
            
        delay = np.random.uniform(0.5, 3.0)

        syn = Synapses(input2_group, output_group,
                      'w : 1', on_pre='I_syn_post += w * mV')
        syn.connect(i=0, j=post_idx)
        syn.w = weight
        syn.delay = delay * ms
        
        synapses.append(syn)
        initial_delays[1, post_idx] = delay
        initial_weights[1, post_idx] = weight
    
    inhibition = Synapses(output_group, output_group,
                         '''w_inh : 1
                            adaptive_inh : 1''', 
                         on_pre='I_syn_post -= (w_inh + adaptive_inh) * mV')
    inhibition.connect(condition='i != j')
    inhibition.w_inh = 0.05
    inhibition.adaptive_inh = 0.0
    
    # Create monitors
    input1_monitor = SpikeMonitor(input1_group)
    input2_monitor = SpikeMonitor(input2_group)
    output_monitor = SpikeMonitor(output_group)
    
    # Run simulation with iterative homeostasis
    simulation_time = 1000  # Each cycle duration
    n_cycles = 10  # Increased from 5 to 10 for more gradual adaptation
    current_delays = initial_delays.copy()  # Track current delays through cycles
    current_weights = initial_weights.copy()  # Track current weights through cycles
    
    print(f"\nRunning {n_cycles} cycles of {simulation_time}ms each for homeostatic adaptation...")
    
    all_output_spikes = []
    all_delay_states = []  # Track delay evolution
    all_input1_spikes = []  # Track input spikes from all cycles
    all_input2_spikes = []  # Track input spikes from all cycles
    
    for cycle in range(n_cycles):
        print(f"Cycle {cycle+1}/{n_cycles}...", end=" ")
        
        # Start fresh scope for each cycle to reset simulation time
        start_scope()
        
        # Recreate input groups with fresh timing for this cycle
        input1_times_cycle = np.array(input_spikes[0]) * ms
        input1_indices_cycle = [0] * len(input_spikes[0])
        
        input2_times_cycle = np.array(input_spikes[1]) * ms
        input2_indices_cycle = [0] * len(input_spikes[1])
        
        input1_group_cycle = SpikeGeneratorGroup(1, input1_indices_cycle, input1_times_cycle)
        input2_group_cycle = SpikeGeneratorGroup(1, input2_indices_cycle, input2_times_cycle)
        
        # Recreate output group with same parameters
        output_group_cycle = NeuronGroup(3, eqs,
                                        threshold='v > v_thresh',
                                        reset='v = v_reset',
                                        refractory=2*ms,
                                        method='euler')  # Changed to euler for noise
        
        output_group_cycle.v = -65*mV
        output_group_cycle.v_rest = -65*mV
        output_group_cycle.v_thresh = -55*mV  # Lowered threshold
        output_group_cycle.v_reset = -70*mV
        output_group_cycle.tau_m = 20*ms
        output_group_cycle.tau_noise = 10*ms
        output_group_cycle.sigma_noise = 1.0*mV  # Increased noise
        output_group_cycle.I_noise = 0*mV
        
        # Recreate synapses with current learned parameters
        synapses_cycle = []
        
        # Input1 -> All outputs
        for post_idx in range(3):
            weight = current_weights[0, post_idx]
            delay = current_delays[0, post_idx]
            
            syn = Synapses(input1_group_cycle, output_group_cycle,
                          'w : 1', on_pre='I_syn_post += w * mV')
            syn.connect(i=0, j=post_idx)
            syn.w = weight
            syn.delay = delay * ms
            
            synapses_cycle.append(syn)
        
        # Input2 -> All outputs  
        for post_idx in range(3):
            weight = current_weights[1, post_idx]
            delay = current_delays[1, post_idx]

            syn = Synapses(input2_group_cycle, output_group_cycle,
                          'w : 1', on_pre='I_syn_post += w * mV')
            syn.connect(i=0, j=post_idx)
            syn.w = weight
            syn.delay = delay * ms
            
            synapses_cycle.append(syn)
        
        # Recreate adaptive inhibition
        inhibition_cycle = Synapses(output_group_cycle, output_group_cycle,
                                   '''w_inh : 1
                                      adaptive_inh : 1''', 
                                   on_pre='I_syn_post -= (w_inh + adaptive_inh) * mV')
        inhibition_cycle.connect(condition='i != j')
        inhibition_cycle.w_inh = 0.05  # Much reduced base inhibition
        
        # Set adaptive inhibition based on previous cycle performance
        if cycle > 0:
            prev_max_rate = max([len(all_output_spikes[cycle-1][neuron_idx])/(simulation_time/1000) for neuron_idx in range(3)])
            if prev_max_rate > 80:  # Higher threshold for adaptive inhibition
                adaptive_inh_value = min(0.2, prev_max_rate / 300.0)  # Much gentler adaptive inhibition
                inhibition_cycle.adaptive_inh = adaptive_inh_value
            else:
                inhibition_cycle.adaptive_inh = 0.0
        else:
            inhibition_cycle.adaptive_inh = 0.0
        
        # Create monitors for this cycle
        input1_monitor = SpikeMonitor(input1_group_cycle)
        input2_monitor = SpikeMonitor(input2_group_cycle)
        output_monitor = SpikeMonitor(output_group_cycle)
        
        # Run this cycle
        run(simulation_time*ms)
        
        # Analyze results - show brief summary
        for neuron_idx in range(3):
            neuron_spikes = [t for i, t in zip(output_monitor.i, output_monitor.t/ms) if i == neuron_idx]
            rate = len(neuron_spikes)/(simulation_time/1000)
            if neuron_idx == 0:
                print(f"O1:{rate:.0f}Hz", end=" ")
            elif neuron_idx == 1:
                print(f"O2:{rate:.0f}Hz", end=" ")
            else:
                print(f"O3:{rate:.0f}Hz")
        
        # Store spikes for learning analysis (convert to absolute time for consistency)
        output_spikes = [[] for _ in range(3)]
        for i, t in zip(output_monitor.i, output_monitor.t):
            output_spikes[i].append(float(t/ms))
        all_output_spikes.append(output_spikes)
        
        # Store input spikes for final analysis
        input1_spikes_cycle = [float(t/ms) for t in input1_monitor.t]
        input2_spikes_cycle = [float(t/ms) for t in input2_monitor.t]
        all_input1_spikes.extend(input1_spikes_cycle)
        all_input2_spikes.extend(input2_spikes_cycle)
        
        # Store current delay state
        all_delay_states.append(current_delays.copy())
        
        # Apply homeostasis after each cycle (suppress detailed output)
        if cycle < n_cycles - 1:
            # Apply delay homeostasis with much gentler adaptation
            current_delays = apply_delay_homeostasis(
                current_delays, output_spikes, simulation_time,
                R_target=15.0, lambda_d=0.05  # Reduced from 0.3 to 0.05 - much gentler
            )
            
            # Apply weight homeostasis manually (since we recreate synapses each cycle)
            R_target = 15.0  # Define the target firing rate
            for post_idx in range(3):
                spike_count = len(output_spikes[post_idx])
                R_observed = spike_count / (simulation_time / 1000.0)
                
                K = (R_target - R_observed) / R_target
                
                for pre_idx in range(2):
                    current_weight = current_weights[pre_idx, post_idx]
                    
                    delta_w_homeo = 0.005 * K  # Reduced from 0.02 to 0.005 - much gentler
                    new_weight = current_weight + delta_w_homeo
                    
                    new_weight = np.clip(new_weight, 1.0, 10.0)
                    current_weights[pre_idx, post_idx] = new_weight
    
    # Plot delay evolution across cycles
    print("\nGenerating delay evolution plots...")
    plot_delay_evolution(all_delay_states, save_path="delay_evolution_detailed.png")
    plot_delay_evolution_summary(all_delay_states, save_path="delay_evolution_summary.png")
    
    # Use the final delay state for analysis
    final_delays = current_delays.copy()
    
    # Use the final cycle's data for detailed analysis
    final_output_spikes = all_output_spikes[-1]
    
    # Final results summary - suppressed for minimal output
    
    if len(output_monitor.t) > 0:
        # Output spike details - suppressed for minimal output
        
        # Show spikes per neuron - suppressed for minimal output
        
        # Network working confirmation - suppressed for minimal output
        
        # Extract spike data for learning - use final cycle spikes since input generators are recreated each cycle
        input1_spikes_final = [float(t/ms) for t in input1_monitor.t]
        input2_spikes_final = [float(t/ms) for t in input2_monitor.t]
        
        # Comprehensive delay learning - minimal output
        
        # Apply learning to all connections - use final cycle input data
        input_data = [input1_spikes_final, input2_spikes_final]
        
        # Initial delays - suppressed for minimal output
        
        # Final delays - suppressed for minimal output
        
        # Delay evolution over cycles - suppressed for minimal output
        
        # Multiple delay learning iterations with stopping condition (STDP-like delay learning) 
        # Use final cycle spikes for learning
        # STDP delay learning with biological stopping - minimal output
        learning_rate = 0.5
        c_threshold = 8.5  # Stop condition threshold (c > B_minus = 8.0)
        modulation_step = 0.01  # Constant delay increase per iteration
        
        for iteration in range(10):
            total_updates = 0
            stopped_neurons = []
            
            # First, apply constant delay modulation to all delays
            final_delays = apply_delay_modulation(final_delays, modulation_step)
            
            # Check stop condition for each post-synaptic neuron
            neuron_stop_status = {}
            for post_idx in range(3):
                stop_learning, min_delay = check_stop_condition(final_delays, post_idx, c_threshold)
                neuron_stop_status[post_idx] = {
                    'stop': stop_learning,
                    'min_delay': min_delay
                }
                if stop_learning:
                    stopped_neurons.append(post_idx)
            
            # Stop condition output - suppressed for minimal output
            
            # Apply delay learning only to neurons that haven't hit stop condition
            for pre_idx in range(2):
                for post_idx in range(3):
                    if neuron_stop_status[post_idx]['stop']:
                        continue
                    
                    if not input_data[pre_idx] or not final_output_spikes[post_idx]:
                        continue
                    
                    current_delay = final_delays[pre_idx, post_idx]
                    
                    pairs = find_spike_pairs(input_data[pre_idx], final_output_spikes[post_idx], 
                                           current_delay, window=20.0)
                    
                    if pairs:
                        updates = []
                        for pre_time, post_time in pairs:
                            delta_delay = delay_learning_rule(pre_time, post_time, current_delay)
                            updates.append(delta_delay)
                        
                        if updates:
                            avg_update = np.mean(updates) * learning_rate
                            final_delays[pre_idx, post_idx] += avg_update
                            final_delays[pre_idx, post_idx] = np.clip(
                                final_delays[pre_idx, post_idx], 0.1, 8.0)
                            total_updates += 1
            
            active_neurons = [neuron_idx for neuron_idx in range(3) if not neuron_stop_status[neuron_idx]['stop']]
            if total_updates > 0:
                pass
            else:
                if len(active_neurons) == 0:
                    print(f"  Iteration {iteration+1}: All neurons stopped - delay learning complete!")
                    break
                else:
                    # No updates output - suppressed for minimal output
                    pass
            
            # Current delay ranges - suppressed for minimal output
        
        # Delays after learning - suppressed for minimal output
        
        # Fixed weight distribution - suppressed for minimal output
        
        all_weights = initial_weights.flatten()
        # Weight statistics - suppressed for minimal output
        
        # Pattern classification accuracy (use final cycle spikes)
        # Pattern classification analysis - minimal output
        pattern_to_label = {"pattern_1": 0, "pattern_2": 1, "pattern_3": 2}
        label_to_pattern = {0: "pattern_1", 1: "pattern_2", 2: "pattern_3"}
        predictions = []
        true_labels = []
        detailed_results = []
        
        # Initialize confusion matrix
        confusion_matrix = np.zeros((4, 3))  # 4 rows: 3 patterns + no response, 3 cols: 3 patterns
        pattern_responses = {"pattern_1": [], "pattern_2": [], "pattern_3": []}
        
        # Pattern analysis - minimal output only
        for pattern_idx, pattern in enumerate(pattern_sequence):
            pattern_start = pattern["start_time"]
            pattern_end = pattern_start + 15  # Use full pattern interval (15ms)
            pattern_name = pattern["name"]
            
            # Count spikes per output neuron in this pattern window (final cycle only)
            neuron_spikes = [0, 0, 0]
            spike_details = [[], [], []]  # Store actual spike times for each neuron
            
            for neuron_idx, spike_time in zip(output_monitor.i, output_monitor.t/ms):
                if pattern_start <= spike_time < pattern_end:
                    neuron_spikes[neuron_idx] += 1
                    spike_details[neuron_idx].append(spike_time)
            
            # Predict based on max spikes
            if max(neuron_spikes) > 0:
                predicted = neuron_spikes.index(max(neuron_spikes))
                predicted_pattern = label_to_pattern[predicted]
            else:
                predicted = -1  # No response
                predicted_pattern = "none"
            
            true_label = pattern_to_label[pattern_name]
            predictions.append(predicted)
            true_labels.append(true_label)
            
            # Store detailed results
            result = {
                "pattern_idx": pattern_idx+1,
                "true_pattern": pattern_name,
                "predicted_pattern": predicted_pattern,
                "start_time": pattern_start,
                "neuron_spikes": neuron_spikes,
                "spike_details": spike_details,
                "correct": predicted == true_label
            }
            detailed_results.append(result)
            
            # Update confusion matrix
            if predicted == -1:
                confusion_matrix[3, true_label] += 1
            else:
                confusion_matrix[predicted, true_label] += 1
            
            pattern_responses[pattern_name].append({
                "spikes": neuron_spikes,
                "predicted": predicted_pattern,
                "correct": predicted == true_label
            })
            
            # Only show first 10 patterns and errors - suppress rest for minimal output
            if pattern_idx < 5 or not result["correct"]:
                pass  # Suppress individual pattern outputs
        
        import builtins
        correct = builtins.sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        total_patterns = len(true_labels)
        overall_accuracy = correct / total_patterns
        
        print(f"\nOVERALL ACCURACY: {overall_accuracy:.1%} ({correct}/{total_patterns})")
        
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"True      P1   P2   P3  None")
        pattern_names = ["P1", "P2", "P3"]
        for true_idx in range(3):
            row_str = f"{pattern_names[true_idx]:>4s}  "
            for pred_idx in range(3):
                row_str += f"{int(confusion_matrix[pred_idx, true_idx]):4d} "
            row_str += f"{int(confusion_matrix[3, true_idx]):4d}"  # No response
            print(row_str)
        
        plot_accuracy_analysis(confusion_matrix, overall_accuracy, save_path="accuracy_analysis.png")
        
        return overall_accuracy, confusion_matrix, detailed_results        
    else:
        print("FAILED: No output spikes recorded!")

if __name__ == "__main__":
    main()
