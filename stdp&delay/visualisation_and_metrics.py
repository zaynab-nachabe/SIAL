import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def plot_all_membrane_potentials(all_vm_traces, v_thresh=-52.5, save_path=None):
    if not all_vm_traces:
        print("Warning: No membrane potential data available for plotting")
        return None, None
    
    num_chunks = len(all_vm_traces)
    neuron_colors = ['#1f77b4', '#ff7f0e'] 
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    time_offset = 0
    chunk_boundaries = [0] 
    
    for chunk_idx, vm_traces in enumerate(all_vm_traces):
        if not vm_traces or len(vm_traces) < 2:
            continue
            
        times = np.array(vm_traces[0].times)
        chunk_duration = float(times[-1] - times[0])
        
        shifted_times = times + time_offset
        
        ax.plot(shifted_times, vm_traces[0], color=neuron_colors[0], linewidth=1.2, 
                label=f'Output 0' if chunk_idx == 0 else "_nolegend_")
        ax.plot(shifted_times, vm_traces[1], color=neuron_colors[1], linewidth=1.2, 
                label=f'Output 1' if chunk_idx == 0 else "_nolegend_")
        
        time_offset += chunk_duration
        chunk_boundaries.append(time_offset)
    
    ax.axhline(y=v_thresh, color='k', linestyle='--', alpha=0.7, label='Threshold')
    
    for boundary in chunk_boundaries[1:-1]:
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.6)
    
    for i in range(len(chunk_boundaries)-1):
        chunk_middle = (chunk_boundaries[i] + chunk_boundaries[i+1]) / 2
        ax.text(chunk_middle, ax.get_ylim()[1] - 2, f'Chunk {i+1}', 
                horizontalalignment='center', fontsize=10)
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax.legend(unique_handles, unique_labels, loc='upper right')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title('Membrane Potentials of Output Neurons Across All Training Chunks')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved all membrane potentials plot to {save_path}")
    
    return fig, ax

def plot_spike_raster(input_spiketrains, output_spiketrains, pattern_times=None, pattern_labels=None, save_path=None):
    colors = {0: 'blue', 1: 'red'}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    neuron_labels = []
    y_positions = []
    
    current_pos = 0
    
    for i, spiketrain in enumerate(input_spiketrains):
        if len(spiketrain) > 0:
            ax.scatter(spiketrain, np.full(len(spiketrain), current_pos), 
                      marker='|', s=100, color='green', label=f'Input {i}' if i == 0 else "_")
            neuron_labels.append(f'Input {i}')
            y_positions.append(current_pos)
            current_pos += 0.5
    
    for i, spiketrain in enumerate(output_spiketrains):
        if len(spiketrain) > 0:
            ax.scatter(spiketrain, np.full(len(spiketrain), current_pos), 
                      marker='|', s=100, color='blue' if i == 0 else 'red', 
                      label=f'Output {i}' if i == 0 else "_")
            neuron_labels.append(f'Output {i}')
            y_positions.append(current_pos)
            current_pos += 0.5
    
    if pattern_times is not None and pattern_labels is not None:
        for time, pattern in zip(pattern_times, pattern_labels):
            color = colors[pattern]
            ax.axvline(x=time, color=color, linestyle='-', alpha=0.2)
            ax.text(time + 1, 0.25, f'P{pattern}', color=color, fontsize=8, alpha=0.7)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(neuron_labels)
    ax.set_xlabel('Time (ms)')
    ax.set_title('Spike Raster Plot')
    ax.grid(True, alpha=0.3)
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax.legend(unique_handles, unique_labels, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spike raster plot to {save_path}")
    
    return fig, ax

def plot_pattern_responses(response_history, save_path=None):
    if not response_history or not response_history.get('output_0') or not response_history.get('output_1'):
        print("Warning: No pattern response data available")
        return None, None
    
    chunks = list(range(len(response_history['output_0']['pattern_0'])))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot neuron 0 responses
    axes[0].plot(chunks, response_history['output_0']['pattern_0'], 'b-o', label='Pattern 0')
    axes[0].plot(chunks, response_history['output_0']['pattern_1'], 'r-o', label='Pattern 1')
    
    # Plot neuron 1 responses
    axes[1].plot(chunks, response_history['output_1']['pattern_0'], 'b-o', label='Pattern 0')
    axes[1].plot(chunks, response_history['output_1']['pattern_1'], 'r-o', label='Pattern 1')
    
    # Highlight final window (last 10 chunks)
    if len(chunks) > 10:
        final_window_start = chunks[-10]
        for ax in axes:
            ax.axvspan(final_window_start, chunks[-1], color='lightgreen', alpha=0.2, label='Final window')
    
    axes[0].set_xlabel('Training Chunk')
    axes[0].set_ylabel('Response Rate')
    axes[0].set_title('Output Neuron 0 Response Rates')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    
    axes[1].set_xlabel('Training Chunk')
    axes[1].set_ylabel('Response Rate')
    axes[1].set_title('Output Neuron 1 Response Rates')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved pattern response plot to {save_path}")
    
    return fig, axes

def plot_delay_evolution(delay_history, save_path=None):
    if not delay_history:
        print("Warning: No delay history data available")
        return None, None
    
    chunks = list(range(len(next(iter(delay_history.values())))))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.axhspan(6.0, 9.0, color='lightgreen', alpha=0.2, label='Target delayed zone')
    ax.axhspan(0.1, 2.0, color='lightyellow', alpha=0.2, label='Target immediate zone')
    
    # Extract just the delay values if they are stored as tuples
    plot_data = {}
    for conn, delays in delay_history.items():
        plot_data[conn] = []
        for d in delays:
            if isinstance(d, tuple) and len(d) >= 3:
                plot_data[conn].append(d[2])  # Extract delay value from tuple
            else:
                plot_data[conn].append(d)
    
    ax.plot(chunks, plot_data[('input_0', 'output_0')], 'b-o', label='Input 0 → Output 0')
    ax.plot(chunks, plot_data[('input_0', 'output_1')], 'b--o', label='Input 0 → Output 1')
    ax.plot(chunks, plot_data[('input_1', 'output_0')], 'r-o', label='Input 1 → Output 0')
    ax.plot(chunks, plot_data[('input_1', 'output_1')], 'r--o', label='Input 1 → Output 1')
    
    if len(chunks) > 0:
        final_delays = {k: plot_data[k][-1] for k in plot_data}
        
        for conn, delay in final_delays.items():
            ax.plot(chunks[-1], delay, 'ko', markersize=8)
            ax.text(chunks[-1] + 1, delay, f"{delay:.3f}ms", 
                   fontsize=9, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Training Chunk')
    ax.set_ylabel('Delay (ms)')
    ax.set_title('Synaptic Delay Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    return fig, ax



def plot_weight_evolution(weight_history, save_path=None):
    """
    Plots the evolution of synaptic weights over training chunks.
    
    Args:
        weight_history (dict): Dictionary with connection tuples as keys and lists of weight values as values
        save_path (str, optional): Path to save the figure. Defaults to None.
    
    Returns:
        tuple: Figure and axes objects
    """
    if not weight_history:
        print("Warning: No weight history data available")
        return None, None
    
    chunks = list(range(len(next(iter(weight_history.values())))))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract just the weight values if they are stored as tuples
    plot_data = {}
    for conn, weights in weight_history.items():
        plot_data[conn] = []
        for w in weights:
            if isinstance(w, tuple) and len(w) >= 1:
                plot_data[conn].append(w[0])  # Extract weight value from tuple
            else:
                plot_data[conn].append(w)
    
    ax.plot(chunks, plot_data[('input_0', 'output_0')], 'b-o', label='Input 0 → Output 0')
    ax.plot(chunks, plot_data[('input_0', 'output_1')], 'b--o', label='Input 0 → Output 1')
    ax.plot(chunks, plot_data[('input_1', 'output_0')], 'r-o', label='Input 1 → Output 0')
    ax.plot(chunks, plot_data[('input_1', 'output_1')], 'r--o', label='Input 1 → Output 1')
    
    if len(chunks) > 0:
        final_weights = {k: plot_data[k][-1] for k in plot_data}
        
        for conn, weight in final_weights.items():
            ax.plot(chunks[-1], weight, 'ko', markersize=8)
            ax.text(chunks[-1] + 1, weight, f"{weight:.3f}", 
                   fontsize=9, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Training Chunk')
    ax.set_ylabel('Weight Strength')
    ax.set_title('Synaptic Weight Evolution (STDP)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved weight evolution plot to {save_path}")
    
    return fig, ax



def plot_average_response_bar(response_history, save_path=None):
    if not response_history or not response_history.get('output_0') or not response_history.get('output_1'):
        print("Warning: No response history data available")
        return None, None
    
    output0_pattern0_avg = np.mean(response_history['output_0']['pattern_0'])
    output0_pattern1_avg = np.mean(response_history['output_0']['pattern_1'])
    output1_pattern0_avg = np.mean(response_history['output_1']['pattern_0'])
    output1_pattern1_avg = np.mean(response_history['output_1']['pattern_1'])
    
    labels = ['Pattern 0', 'Pattern 1']
    output0_rates = [output0_pattern0_avg, output0_pattern1_avg]
    output1_rates = [output1_pattern0_avg, output1_pattern1_avg]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, output0_rates, width, label='Output 0', color='blue')
    ax.bar(x + width/2, output1_rates, width, label='Output 1', color='red')
    
    ax.set_xlabel('Pattern Type')
    ax.set_ylabel('Response Rate')
    ax.set_title('Pattern Recognition Performance (All Chunks)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, v in enumerate(output0_rates):
        ax.text(i - width/2, v + 0.05, f'{v:.2f}', ha='center')
    for i, v in enumerate(output1_rates):
        ax.text(i + width/2, v + 0.05, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved average response bar chart to {save_path}")
    
    return fig, ax

def create_combined_visualization(result_data, save_path=None, show_plot=True):
    delay_history = result_data.get('delay_history', {})
    response_history = result_data.get('response_history', {})
    all_vm_traces = result_data.get('all_vm_traces', [])
    
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    chunks = list(range(len(next(iter(delay_history.values())))))
    
    ax1.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
    ax1.axhline(y=8.0, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
    
    ax1.plot(chunks, delay_history[('input_0', 'output_0')], 'b-o', label='Input 0 → Output 0')
    ax1.plot(chunks, delay_history[('input_0', 'output_1')], 'b--o', label='Input 0 → Output 1')
    ax1.plot(chunks, delay_history[('input_1', 'output_0')], 'r-o', label='Input 1 → Output 0')
    ax1.plot(chunks, delay_history[('input_1', 'output_1')], 'r--o', label='Input 1 → Output 1')
    
    ax1.set_xlabel('Training Chunk')
    ax1.set_ylabel('Delay (ms)')
    ax1.set_title('Delay Learning Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    if response_history and response_history.get('output_0') and response_history.get('output_1'):
        output0_pattern0_avg = np.mean(response_history['output_0']['pattern_0'])
        output0_pattern1_avg = np.mean(response_history['output_0']['pattern_1'])
        output1_pattern0_avg = np.mean(response_history['output_1']['pattern_0'])
        output1_pattern1_avg = np.mean(response_history['output_1']['pattern_1'])
        
        labels = ['Pattern 0', 'Pattern 1']
        output0_rates = [output0_pattern0_avg, output0_pattern1_avg]
        output1_rates = [output1_pattern0_avg, output1_pattern1_avg]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax2.bar(x - width/2, output0_rates, width, label='Output 0', color='blue')
        ax2.bar(x + width/2, output1_rates, width, label='Output 1', color='red')
        
        ax2.set_xlabel('Pattern Type')
        ax2.set_ylabel('Average Response Rate')
        ax2.set_title('Average Pattern Recognition Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        for i, v in enumerate(output0_rates):
            ax2.text(i - width/2, v + 0.05, f'{v:.2f}', ha='center')
        for i, v in enumerate(output1_rates):
            ax2.text(i + width/2, v + 0.05, f'{v:.2f}', ha='center')
    
    ax3 = fig.add_subplot(gs[1, 0])
    
    if response_history and response_history.get('output_0') and response_history.get('output_1'):
        chunks = range(len(response_history['output_0']['pattern_0']))
        
        ax3.plot(chunks, response_history['output_0']['pattern_0'], 'b-o', label='Output 0 - Pattern 0')
        ax3.plot(chunks, response_history['output_0']['pattern_1'], 'b--o', label='Output 0 - Pattern 1')
        ax3.plot(chunks, response_history['output_1']['pattern_0'], 'r-o', label='Output 1 - Pattern 0')
        ax3.plot(chunks, response_history['output_1']['pattern_1'], 'r--o', label='Output 1 - Pattern 1')
        
        ax3.set_xlabel('Training Chunk')
        ax3.set_ylabel('Response Rate')
        ax3.set_title('Pattern Response Evolution')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    ax4 = fig.add_subplot(gs[1, 1])
    
    if all_vm_traces:
        time_offset = 0
        chunk_boundaries = [0]
        
        for chunk_idx, vm_traces in enumerate(all_vm_traces):
            if not vm_traces or len(vm_traces) < 2:
                continue
                
            times = np.array(vm_traces[0].times)
            chunk_duration = float(times[-1] - times[0])
            
            shifted_times = times + time_offset
            
            label0 = 'Output 0' if chunk_idx == 0 else "_nolegend_"
            label1 = 'Output 1' if chunk_idx == 0 else "_nolegend_"
            ax4.plot(shifted_times, vm_traces[0], color='blue', linewidth=0.8, alpha=0.7, label=label0)
            ax4.plot(shifted_times, vm_traces[1], color='red', linewidth=0.8, alpha=0.7, label=label1)
            
            time_offset += chunk_duration
            chunk_boundaries.append(time_offset)
        
        ax4.axhline(y=-52.5, color='k', linestyle='--', alpha=0.7, label='Threshold')
        
        for boundary in chunk_boundaries[1:-1]:
            ax4.axvline(x=boundary, color='gray', linestyle='--', alpha=0.4)
        
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Membrane Potential (mV)')
        ax4.set_title('Membrane Potentials Across All Chunks')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined visualization to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_results_directory(experiment_name):
    # Use absolute path for results
    results_dir = os.path.join('/home/neuromorph/Documents/SIAL/results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Creating results directory: {results_dir}")
    return results_dir

def calculate_model_accuracy(response_history, window_size=5, final_window_size=10):
    if not response_history or 'output_0' not in response_history or 'output_1' not in response_history:
        print("Warning: No response history available for accuracy calculation")
        return {
            'overall_accuracy': 0.0,
            'recent_accuracy': 0.0,
            'specialization_map': {},
            'confusion_matrix': np.zeros((2, 2))
        }
    
    output0_pattern0 = np.array(response_history['output_0']['pattern_0'])
    output0_pattern1 = np.array(response_history['output_0']['pattern_1'])
    output1_pattern0 = np.array(response_history['output_1']['pattern_0'])
    output1_pattern1 = np.array(response_history['output_1']['pattern_1'])
    
    # Calculate overall rates (across all training)
    overall_rates = {
        ('output_0', 'pattern_0'): np.mean(output0_pattern0),
        ('output_0', 'pattern_1'): np.mean(output0_pattern1),
        ('output_1', 'pattern_0'): np.mean(output1_pattern0),
        ('output_1', 'pattern_1'): np.mean(output1_pattern1)
    }
    
    # Calculate final rates (last N chunks) - this is our primary metric for final performance
    final_size = min(final_window_size, len(output0_pattern0))
    final_rates = {
        ('output_0', 'pattern_0'): np.mean(output0_pattern0[-final_size:]),
        ('output_0', 'pattern_1'): np.mean(output0_pattern1[-final_size:]),
        ('output_1', 'pattern_0'): np.mean(output1_pattern0[-final_size:]),
        ('output_1', 'pattern_1'): np.mean(output1_pattern1[-final_size:])
    }
    
    # Determine specialization based on final performance (last N chunks)
    scenario1_final_accuracy = (final_rates[('output_0', 'pattern_0')] + final_rates[('output_1', 'pattern_1')]) / 2
    scenario2_final_accuracy = (final_rates[('output_0', 'pattern_1')] + final_rates[('output_1', 'pattern_0')]) / 2
    
    if scenario1_final_accuracy >= scenario2_final_accuracy:
        specialization = {
            'pattern_0': 'output_0',
            'pattern_1': 'output_1'
        }
    else:
        specialization = {
            'pattern_0': 'output_1',
            'pattern_1': 'output_0'
        }
    
    # Create overall confusion matrix (for historical comparison)
    confusion_matrix = np.zeros((2, 2))
    p0_specialist = specialization['pattern_0']
    p1_specialist = specialization['pattern_1']
   
    confusion_matrix[0, 0] = overall_rates[(p0_specialist, 'pattern_0')]
    confusion_matrix[0, 1] = overall_rates[(p1_specialist, 'pattern_0')]
    confusion_matrix[1, 0] = overall_rates[(p0_specialist, 'pattern_1')]
    confusion_matrix[1, 1] = overall_rates[(p1_specialist, 'pattern_1')]
    
    # Create final confusion matrix (most important for measuring performance)
    final_confusion_matrix = np.zeros((2, 2))
    final_confusion_matrix[0, 0] = final_rates[(p0_specialist, 'pattern_0')]
    final_confusion_matrix[0, 1] = final_rates[(p1_specialist, 'pattern_0')]
    final_confusion_matrix[1, 0] = final_rates[(p0_specialist, 'pattern_1')]
    final_confusion_matrix[1, 1] = final_rates[(p1_specialist, 'pattern_1')]
    
    # Calculate accuracies
    # Enhanced accuracy calculation that considers both positive responses and negative avoidance
    pos_response_acc = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / 2
    neg_avoid_acc = (1 - confusion_matrix[1, 0] + 1 - confusion_matrix[0, 1]) / 2
    overall_accuracy = (pos_response_acc + neg_avoid_acc) / 2
    
    # Calculate final accuracy with enhanced method
    final_pos_acc = (final_confusion_matrix[0, 0] + final_confusion_matrix[1, 1]) / 2
    final_neg_acc = (1 - final_confusion_matrix[1, 0] + 1 - final_confusion_matrix[0, 1]) / 2
    final_accuracy = (final_pos_acc + final_neg_acc) / 2
    
    # Apply bonus for perfect specialization
    if final_confusion_matrix[0, 0] > 0.9 and final_confusion_matrix[1, 1] > 0.9 and \
       final_confusion_matrix[0, 1] < 0.1 and final_confusion_matrix[1, 0] < 0.1:
        final_accuracy = min(1.0, final_accuracy + 0.2)
    
    # Calculate recent accuracy (for backward compatibility)
    if len(output0_pattern0) >= window_size:
        recent_output0_pattern0 = np.mean(output0_pattern0[-window_size:])
        recent_output0_pattern1 = np.mean(output0_pattern1[-window_size:])
        recent_output1_pattern0 = np.mean(output1_pattern0[-window_size:])
        recent_output1_pattern1 = np.mean(output1_pattern1[-window_size:])
        
        recent_rates = {
            (p0_specialist, 'pattern_0'): recent_output0_pattern0 if p0_specialist == 'output_0' else recent_output1_pattern0,
            (p1_specialist, 'pattern_1'): recent_output1_pattern1 if p1_specialist == 'output_1' else recent_output0_pattern1,
            (p0_specialist, 'pattern_1'): recent_output0_pattern1 if p0_specialist == 'output_0' else recent_output1_pattern1,
            (p1_specialist, 'pattern_0'): recent_output1_pattern0 if p1_specialist == 'output_1' else recent_output0_pattern1
        }
        
        # Enhanced accuracy calculation
        recent_pos_acc = (recent_rates[(p0_specialist, 'pattern_0')] + recent_rates[(p1_specialist, 'pattern_1')]) / 2
        recent_neg_acc = (1 - recent_rates[(p0_specialist, 'pattern_1')] + 1 - recent_rates[(p1_specialist, 'pattern_0')]) / 2
        recent_accuracy = (recent_pos_acc + recent_neg_acc) / 2
        
        # Apply bonus for perfect classification
        if recent_rates[(p0_specialist, 'pattern_0')] > 0.9 and recent_rates[(p1_specialist, 'pattern_1')] > 0.9 and \
           recent_rates[(p0_specialist, 'pattern_1')] < 0.1 and recent_rates[(p1_specialist, 'pattern_0')] < 0.1:
            recent_accuracy = min(1.0, recent_accuracy + 0.2)
    else:
        recent_accuracy = overall_accuracy
    
    # Calculate selectivity indices
    selectivity = {}
    for neuron in ['output_0', 'output_1']:
        if neuron == 'output_0':
            selectivity[neuron] = overall_rates[('output_0', 'pattern_0')] - overall_rates[('output_0', 'pattern_1')]
        else:
            selectivity[neuron] = overall_rates[('output_1', 'pattern_1')] - overall_rates[('output_1', 'pattern_0')]
    
    # Calculate final selectivity
    final_selectivity = {}
    for neuron in ['output_0', 'output_1']:
        if neuron == 'output_0':
            final_selectivity[neuron] = final_rates[('output_0', 'pattern_0')] - final_rates[('output_0', 'pattern_1')]
        else:
            final_selectivity[neuron] = final_rates[('output_1', 'pattern_1')] - final_rates[('output_1', 'pattern_0')]
    
    # Calculate accuracy over time for plotting
    accuracy_evolution = []
    for i in range(len(output0_pattern0)):
        chunk_rates = {
            (p0_specialist, 'pattern_0'): output0_pattern0[i] if p0_specialist == 'output_0' else output1_pattern0[i],
            (p1_specialist, 'pattern_1'): output1_pattern1[i] if p1_specialist == 'output_1' else output0_pattern1[i],
            (p0_specialist, 'pattern_1'): output0_pattern1[i] if p0_specialist == 'output_0' else output1_pattern1[i],
            (p1_specialist, 'pattern_0'): output1_pattern0[i] if p1_specialist == 'output_1' else output0_pattern0[i]
        }
        
        # Enhanced accuracy calculation
        chunk_pos_acc = (chunk_rates[(p0_specialist, 'pattern_0')] + chunk_rates[(p1_specialist, 'pattern_1')]) / 2
        chunk_neg_acc = (1 - chunk_rates[(p0_specialist, 'pattern_1')] + 1 - chunk_rates[(p1_specialist, 'pattern_0')]) / 2
        chunk_accuracy = (chunk_pos_acc + chunk_neg_acc) / 2
        
        # Apply bonus for perfect classification
        if chunk_rates[(p0_specialist, 'pattern_0')] > 0.9 and chunk_rates[(p1_specialist, 'pattern_1')] > 0.9 and \
           chunk_rates[(p0_specialist, 'pattern_1')] < 0.1 and chunk_rates[(p1_specialist, 'pattern_0')] < 0.1:
            chunk_accuracy = min(1.0, chunk_accuracy + 0.2)
            
        accuracy_evolution.append(chunk_accuracy)
    
    return {
        'overall_accuracy': overall_accuracy,
        'recent_accuracy': recent_accuracy,
        'final_accuracy': final_accuracy,
        'specialization_map': specialization,
        'confusion_matrix': confusion_matrix,
        'final_confusion_matrix': final_confusion_matrix,
        'selectivity': selectivity,
        'final_selectivity': final_selectivity,
        'response_rates': overall_rates,
        'final_rates': final_rates,
        'accuracy_evolution': accuracy_evolution
    }

def plot_accuracy_metrics(accuracy_metrics, save_path=None):
    if not accuracy_metrics:
        print("Warning: No accuracy metrics available for plotting")
        return None, None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    final_confusion_matrix = accuracy_metrics.get('final_confusion_matrix', 
                                                 accuracy_metrics.get('confusion_matrix', np.zeros((2, 2))))
    
    im = ax1.imshow(final_confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    
    specialization = accuracy_metrics['specialization_map']
    p0_specialist = specialization['pattern_0']
    p1_specialist = specialization['pattern_1']
    
    ax1.set_xticklabels([f"{p0_specialist}", f"{p1_specialist}"])
    ax1.set_yticklabels(['Pattern 0', 'Pattern 1'])
    
    ax1.set_xlabel('Output Neuron')
    ax1.set_ylabel('True Pattern')
    ax1.set_title('Final Confusion Matrix (Last 10 Chunks)')
    
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{final_confusion_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if final_confusion_matrix[i, j] < 0.7 else "white")
    
    fig.colorbar(im, ax=ax1, label='Response Rate')
    
    # Plot accuracy evolution over time
    if 'accuracy_evolution' in accuracy_metrics:
        chunks = list(range(len(accuracy_metrics['accuracy_evolution'])))
        ax2.plot(chunks, accuracy_metrics['accuracy_evolution'], 'g-o', label='Accuracy')
        
        # Highlight final 10 chunks
        if len(chunks) > 10:
            final_window_start = chunks[-10]
            ax2.axvspan(final_window_start, chunks[-1], color='lightgreen', alpha=0.2, label='Final window')
        
        final_acc = accuracy_metrics.get('final_accuracy', 0)
        overall_acc = accuracy_metrics.get('overall_accuracy', 0)
        
        ax2.axhline(y=final_acc, color='r', linestyle='--', alpha=0.7, label=f'Final Acc: {final_acc:.2f}')
        ax2.axhline(y=overall_acc, color='b', linestyle='--', alpha=0.5, label=f'Overall Acc: {overall_acc:.2f}')
        
        ax2.set_xlabel('Training Chunk')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.05)
    else:
        # Use final selectivity if accuracy evolution not available
        selectivity = accuracy_metrics.get('final_selectivity', accuracy_metrics.get('selectivity', {}))
        
        neuron_labels = list(selectivity.keys())
        selectivity_values = [selectivity[neuron] for neuron in neuron_labels]
        
        colors = ['blue' if neuron == 'output_0' else 'red' for neuron in neuron_labels]
        ax2.bar(neuron_labels, selectivity_values, color=colors)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        final_acc = accuracy_metrics.get('final_accuracy', accuracy_metrics.get('overall_accuracy', 0))
        overall_acc = accuracy_metrics.get('overall_accuracy', 0)
        
        ax2.text(0.5, 0.95, f'Final Accuracy (Last 10): {final_acc:.2f}', transform=ax2.transAxes, 
                ha='center', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.8))
        ax2.text(0.5, 0.85, f'Overall Accuracy: {overall_acc:.2f}', transform=ax2.transAxes, 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        ax2.text(0.5, 0.75, f'Pattern 0 → {specialization["pattern_0"]}', transform=ax2.transAxes,
                ha='center', fontsize=11, bbox=dict(facecolor='lightblue', alpha=0.6))
        ax2.text(0.5, 0.65, f'Pattern 1 → {specialization["pattern_1"]}', transform=ax2.transAxes,
                ha='center', fontsize=11, bbox=dict(facecolor='lightblue', alpha=0.6))
        
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_ylabel('Selectivity Index')
        ax2.set_title('Final Neuronal Selectivity')
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy metrics plot to {save_path}")
    
    return fig, (ax1, ax2)

def save_all_visualizations(result_data, experiment_name, final_window_size=10):
    results_dir = create_results_directory(experiment_name)
    
    # Convert from pattern_responses format if needed
    if 'response_history' in result_data:
        response_history = result_data['response_history']
    elif 'pattern_responses' in result_data:
        # Backward compatibility
        response_history = {
            'output_0': {
                'pattern_0': [],
                'pattern_1': []
            },
            'output_1': {
                'pattern_0': [],
                'pattern_1': []
            }
        }
        
        for chunk_resp in result_data.get('pattern_responses', []):
            for output, patterns in chunk_resp.items():
                for pattern, rate in patterns.items():
                    response_history[output][f"pattern_{pattern}"].append(rate)
        
        result_data['response_history'] = response_history
    else:
        response_history = {}
    
    # Calculate accuracy metrics with focus on final window
    accuracy_metrics = calculate_model_accuracy(response_history, final_window_size=final_window_size)
    
    # Save all visualizations
    delay_path = os.path.join(results_dir, f"{experiment_name}_delays.png")
    plot_delay_evolution(result_data.get('delay_history', {}), save_path=delay_path)
    
    # Add weight evolution plot
    if 'weight_history' in result_data and result_data['weight_history']:
        weight_path = os.path.join(results_dir, f"{experiment_name}_weights.png")
        plot_weight_evolution(result_data.get('weight_history', {}), save_path=weight_path)
    
    resp_evol_path = os.path.join(results_dir, f"{experiment_name}_response_evolution.png")
    plot_pattern_responses(response_history, save_path=resp_evol_path)
    
    accuracy_path = os.path.join(results_dir, f"{experiment_name}_accuracy.png")
    plot_accuracy_metrics(accuracy_metrics, save_path=accuracy_path)
    
    # Save metrics to text file
    metrics_text_path = os.path.join(results_dir, f"{experiment_name}_accuracy_metrics.txt")
    with open(metrics_text_path, 'w') as f:
        f.write(f"Overall Accuracy: {accuracy_metrics.get('overall_accuracy', 0):.4f}\n")
        f.write(f"Final Accuracy (Last {final_window_size}): {accuracy_metrics.get('final_accuracy', 0):.4f}\n")
        f.write(f"Recent Accuracy (Last 5): {accuracy_metrics.get('recent_accuracy', 0):.4f}\n\n")
        
        f.write("Specialization:\n")
        for pattern, neuron in accuracy_metrics.get('specialization_map', {}).items():
            f.write(f"  {pattern} → {neuron}\n")
        
        f.write("\nFinal Response Rates (Last 10 chunks):\n")
        for (neuron, pattern), rate in accuracy_metrics.get('final_rates', {}).items():
            f.write(f"  {neuron} → {pattern}: {rate:.4f}\n")
    
    print(f"Saved accuracy metrics to {metrics_text_path}")
    
    return accuracy_metrics