import numpy as np
import matplotlib.pyplot as plt
import os

def plot_combined_membrane_potentials(vm_traces, v_thresh=-52.5, save_path=None):
    if not vm_traces:
        print("Warning: No membrane potential data available for plotting")
        return None, None
    
    neuron_colors = ['#1f77b4', '#ff7f0e']
    
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
            else:  # TESTING
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

def calculate_specialization_score(pattern_responses):
    """
    Calculate a specialization score between 0 and 1, where:
    - 0 means no specialization (both neurons respond equally to both patterns)
    - 1 means perfect specialization (each neuron responds exclusively to one pattern)
    """
    output0_p0 = pattern_responses['output_0']['pattern_0']
    output0_p1 = pattern_responses['output_0']['pattern_1']
    output1_p0 = pattern_responses['output_1']['pattern_0']
    output1_p1 = pattern_responses['output_1']['pattern_1']
    
    if output0_p0 + output0_p1 > 0:
        output0_preference = abs(output0_p0 - output0_p1) / (output0_p0 + output0_p1)
    else:
        output0_preference = 0
        
    if output1_p0 + output1_p1 > 0:
        output1_preference = abs(output1_p0 - output1_p1) / (output1_p0 + output1_p1)
    else:
        output1_preference = 0
    
    # Determine which pattern each neuron prefers
    output0_prefers_p0 = output0_p0 > output0_p1
    output1_prefers_p0 = output1_p0 > output1_p1
    
    # Complementary specialization bonus (1 if neurons prefer different patterns, 0 if same)
    complementary = 1.0 if output0_prefers_p0 != output1_prefers_p0 else 0.0

    specialization_score = (output0_preference + output1_preference) / 2 * complementary
    
    metrics = {
        'output0_preference': output0_preference,
        'output1_preference': output1_preference,
        'complementary': complementary,
        'output0_prefers_p0': output0_prefers_p0,
        'output1_prefers_p0': output1_prefers_p0,
        'specialization_score': specialization_score
    }
    
    return specialization_score, metrics

def create_results_directory(experiment_name):
    results_dir = os.path.join('results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def plot_specialization_score(specialization_scores, curriculum_phases=None, save_path=None):
    plt.figure(figsize=(12, 6))
    
    chunks = list(range(len(specialization_scores)))
    
    if curriculum_phases:
        phase_colors = {
            'isolated': '#e6f7ff',
            'blocked': '#fff2e6',  
            'mixed': '#f2ffe6'
        }
        
        for i, phase in enumerate(curriculum_phases):
            plt.axvspan(i, i+1, facecolor=phase_colors.get(phase, 'white'), alpha=0.3)
    
    plt.plot(chunks, specialization_scores, 'b-', linewidth=2)
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    plt.text(chunks[-1] * 0.02, 0.52, 'Moderate Specialization', fontsize=9, color='gray')
    plt.text(chunks[-1] * 0.02, 0.82, 'Good Specialization', fontsize=9, color='green')
    
    final_score = specialization_scores[-1]
    plt.text(chunks[-1] * 0.8, final_score, f'Final: {final_score:.3f}', 
             fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlabel('Chunk')
    plt.ylabel('Specialization Score')
    plt.title('Neuron Specialization Evolution')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved specialization score plot to {save_path}")
    else:
        plt.show()

def save_all_visualizations(result_data, experiment_name):
    results_dir = create_results_directory(experiment_name)
    
    all_vm_traces = result_data.get('all_vm_traces', [])
    
    if all_vm_traces:
        combined_path = os.path.join(results_dir, f"{experiment_name}_combined_membrane.png")
        plot_combined_membrane_potentials(all_vm_traces, save_path=combined_path)
    else:
        print("No membrane potential data available for visualization")
    
    if 'delay_history' in result_data:
        delay_path = os.path.join(results_dir, f"{experiment_name}_delays.png")
        chunk_modes = result_data.get('curriculum_phases', None)
        plot_delay_evolution(result_data['delay_history'], save_path=delay_path, chunk_modes=chunk_modes)
    else:
        print("No delay history data available for visualization")

    if 'specialization_scores' in result_data:
        spec_path = os.path.join(results_dir, f"{experiment_name}_specialization.png")
        plot_specialization_score(
            result_data['specialization_scores'],
            curriculum_phases=result_data.get('curriculum_phases', None),
            save_path=spec_path
        )
    
    return results_dir
