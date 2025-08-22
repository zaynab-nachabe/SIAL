import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def plot_combined_membrane_potentials(vm_traces, v_thresh=-52.5, save_path=None):
    if not vm_traces:
        print("Warning: No membrane potential data available for plotting")
        return None, None
    
    neuron_colors = ['#1f77b4', '#ff7f0e']  # Blue for output 0, orange for output 1
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    time_offset = 0
    chunk_boundaries = [0]
    
    for chunk_idx, chunk_traces in enumerate(vm_traces):
        if not chunk_traces or len(chunk_traces) < 2:
            continue
            
        # Get time base from first neuron
        times = np.array(chunk_traces[0].times)
        chunk_duration = float(times[-1] - times[0])
        shifted_times = times + time_offset
        
        # Plot output neuron 0 (with transparency)
        ax.plot(shifted_times, chunk_traces[0].magnitude, color=neuron_colors[0], 
                linewidth=0.8, alpha=0.7, label='Output 0' if chunk_idx == 0 else "")
        
        # Plot output neuron 1 (with transparency)
        ax.plot(shifted_times, chunk_traces[1].magnitude, color=neuron_colors[1], 
                linewidth=0.8, alpha=0.7, label='Output 1' if chunk_idx == 0 else "")
        
        time_offset += chunk_duration
        chunk_boundaries.append(time_offset)
    
    # Add threshold line
    ax.axhline(y=v_thresh, color='r', linestyle='--', alpha=0.7, label='Threshold')
    
    # Add chunk boundaries as vertical lines
    num_chunks = len(vm_traces)
    if num_chunks > 20:
        interval = max(1, num_chunks // 10)
        for i in range(1, len(chunk_boundaries)-1, interval):
            ax.axvline(x=chunk_boundaries[i], color='gray', linestyle='--', alpha=0.3)
    else:
        for boundary in chunk_boundaries[1:-1]:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
    
    # Set title and labels
    ax.set_title('Combined Membrane Potentials of Output Neurons', fontsize=14)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Membrane Potential (mV)', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add legend with both neuron colors
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels while preserving order
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
        print(f"Saved combined membrane potentials plot to {save_path}")
    
    return fig, ax

def plot_delay_evolution(delay_history, save_path=None, chunk_modes=None):
    if not delay_history:
        print("Warning: No delay history data available")
        return None, None
    
    chunks = list(range(len(next(iter(delay_history.values())))))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Add background to show training vs testing chunks if available
    if chunk_modes is not None:
        for i, mode in enumerate(chunk_modes):
            if mode == "TRAINING":
                ax.axvspan(i-0.4, i+0.4, color='lightblue', alpha=0.2)
            else:
                ax.axvspan(i-0.4, i+0.4, color='lightgray', alpha=0.2)
    
    # Plot delay evolution with transparency
    ax.plot(chunks, delay_history[('input_0', 'output_0')], 'b-o', label='Input 0 → Output 0', alpha=0.8)
    ax.plot(chunks, delay_history[('input_0', 'output_1')], 'b--o', label='Input 0 → Output 1', alpha=0.8)
    ax.plot(chunks, delay_history[('input_1', 'output_0')], 'r-o', label='Input 1 → Output 0', alpha=0.8)
    ax.plot(chunks, delay_history[('input_1', 'output_1')], 'r--o', label='Input 1 → Output 1', alpha=0.8)
    
    if len(chunks) > 0:
        final_delays = {k: v[-1] for k, v in delay_history.items()}
        
        for conn, delay in final_delays.items():
            ax.plot(chunks[-1], delay, 'ko', markersize=8)
            ax.text(chunks[-1] + 1, delay, f"{delay:.1f}ms", 
                   fontsize=9, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Training Chunk')
    ax.set_ylabel('Delay (ms)')
    ax.set_title('Synaptic Delay Evolution')
    ax.grid(True, alpha=0.3)
    
    # Add custom legend entries for train/test if applicable
    if chunk_modes is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.2, label='Training Chunk'),
            Patch(facecolor='lightgray', alpha=0.2, label='Testing Chunk')
        ]
        
        # Combine with regular legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + legend_elements, loc='upper right')
    else:
        ax.legend(loc='upper right')
    
    if len(chunks) > 0:
        ax.text(chunks[-1] * 0.05, 8.0, 'Target coincident zone', 
               fontsize=8, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7))
        ax.text(chunks[-1] * 0.05, 0.5, 'Target non-coincident zone', 
               fontsize=8, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved delay evolution plot to {save_path}")
    
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

def plot_weight_evolution(weight_history, save_path=None):
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

def create_results_directory(experiment_name):
    results_dir = os.path.join('/home/neuromorph/Documents/SIAL/results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Creating results directory: {results_dir}")
    return results_dir  

def save_all_visualizations(result_data, experiment_name, final_window_size=10):
    results_dir = create_results_directory(experiment_name)
    
    if 'response_history' in result_data:
        response_history = result_data['response_history']
    elif 'pattern_responses' in result_data:
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
        
    # Save all visualizations
    delay_path = os.path.join(results_dir, f"{experiment_name}_delays.png")
    plot_delay_evolution(result_data.get('delay_history', {}), save_path=delay_path)
    
    # Add weight evolution plot
    if 'weight_history' in result_data and result_data['weight_history']:
        weight_path = os.path.join(results_dir, f"{experiment_name}_weights.png")
        plot_weight_evolution(result_data.get('weight_history', {}), save_path=weight_path)
    
    resp_evol_path = os.path.join(results_dir, f"{experiment_name}_response_evolution.png")
    plot_pattern_responses(response_history, save_path=resp_evol_path)