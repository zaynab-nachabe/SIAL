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
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(chunks, response_history['output_0']['pattern_0'], 'b-o', label='Pattern 0')
    axes[0].plot(chunks, response_history['output_0']['pattern_1'], 'r-o', label='Pattern 1')
    axes[0].set_xlabel('Training Chunk')
    axes[0].set_ylabel('Response Rate')
    axes[0].set_title('Output Neuron 0 Response Rates')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    
    axes[1].plot(chunks, response_history['output_1']['pattern_0'], 'b-o', label='Pattern 0')
    axes[1].plot(chunks, response_history['output_1']['pattern_1'], 'r-o', label='Pattern 1')
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
    
    ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
    ax.axhline(y=8.0, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
    
    ax.plot(chunks, delay_history[('input_0', 'output_0')], 'b-o', label='Input 0 → Output 0')
    ax.plot(chunks, delay_history[('input_0', 'output_1')], 'b--o', label='Input 0 → Output 1')
    ax.plot(chunks, delay_history[('input_1', 'output_0')], 'r-o', label='Input 1 → Output 0')
    ax.plot(chunks, delay_history[('input_1', 'output_1')], 'r--o', label='Input 1 → Output 1')
    
    ax.set_xlabel('Training Chunk')
    ax.set_ylabel('Delay (ms)')
    ax.set_title('Synaptic Delay Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved delay evolution plot to {save_path}")
    
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
    
    # Create a 2x2 grid layout
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Delay Evolution (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    chunks = list(range(len(next(iter(delay_history.values())))))
    
    # Reference lines (without labels)
    ax1.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
    ax1.axhline(y=8.0, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
    
    # Delay evolution lines
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
    results_dir = os.path.join('results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_all_visualizations(result_data, experiment_name):
    
    results_dir = create_results_directory(experiment_name)
    
    combined_path = os.path.join(results_dir, f"{experiment_name}_combined.png")
    create_combined_visualization(result_data, save_path=combined_path, show_plot=False)
    
    delay_path = os.path.join(results_dir, f"{experiment_name}_delays.png")
    plot_delay_evolution(result_data.get('delay_history', {}), save_path=delay_path)
    
    resp_evol_path = os.path.join(results_dir, f"{experiment_name}_response_evolution.png")
    plot_pattern_responses(result_data.get('response_history', {}), save_path=resp_evol_path)
    
    avg_response_path = os.path.join(results_dir, f"{experiment_name}_avg_responses.png")
    plot_average_response_bar(result_data.get('response_history', {}), save_path=avg_response_path)
    
    # Plot membrane potentials from all chunks if available
    all_vm_traces = result_data.get('all_vm_traces', [])
    if all_vm_traces:
        membrane_path = os.path.join(results_dir, f"{experiment_name}_membrane.png")
        plot_all_membrane_potentials(all_vm_traces, save_path=membrane_path)