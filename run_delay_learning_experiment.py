#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Delay Learning Experiment Runner
--------------------------------
References:
- König et al. (1996): Integrator or coincidence detector? The role of the cortical neuron revisited
- Rossant et al. (2011): Sensitivity of noisy neurons to coincident inputs
- Nadafian & Ganjtabesh (2020): Bio-plausible Unsupervised Delay Learning for Extracting 
  Temporal Features in Spiking Neural Networks
"""

import pyNN.nest as sim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from SIAL import create_pattern_spike_trains, create_populations, create_connections_and_inhibition
from SIAL import calculate_accuracy, calculate_detailed_accuracy, analyze_pattern_mistakes
from SIAL import plot_spike_trains, plot_membrane_potentials, plot_final_weights, plot_delays
from SIAL import PATTERNS, PATTERN_OCCURRENCES, PATTERN_INTERVAL, NEURON_PARAMS, STDP_CONFIG
from delay_learning import apply_delay_learning

def run_delay_learning_experiment(
        simulation_time=1000,
        fixed_weights=True,
        weight_value=5.0,
        learning_epochs=3,
        B_plus=0.1,
        B_minus=0.1, 
        sigma_plus=10.0, 
        sigma_minus=10.0,
        learning_rate=0.05,
        noise_enabled=False,
        jitter=1.0,
        missing_prob=0.1,
        extra_prob=0.1):
    """
    Run a complete delay learning experiment with multiple learning epochs.
    
    Parameters:
    - simulation_time: Total simulation time per epoch (ms)
    - fixed_weights: Whether to use fixed weights or STDP learning
    - weight_value: Value for fixed weights (if fixed_weights=True)
    - learning_epochs: Number of learning epochs to run
    - B_plus, B_minus: Parameters controlling the magnitude of delay changes
    - sigma_plus, sigma_minus: Time constants for the delay learning rule
    - learning_rate: Learning rate for delay updates
    - noise_enabled: Whether to add noise to spike trains
    - jitter, missing_prob, extra_prob: Noise parameters
    
    Returns:
    - final_accuracy: Accuracy after the final epoch
    - final_weights: Weights after the final epoch
    - final_delays: Delays after the final epoch
    - output_data: Output data from the final epoch
    - pattern_sequence: Sequence of patterns presented
    """
    # Set up the noise configuration
    noise_config = {
        "enabled": noise_enabled,
        "jitter_std": jitter,
        "missing_prob": missing_prob,
        "extra_prob": extra_prob
    }
    
    print("=" * 80)
    print("DELAY LEARNING EXPERIMENT")
    print("=" * 80)
    print("Configuration:")
    if fixed_weights:
        print(f"- Fixed weights ({weight_value})")
    else:
        print("- STDP weight learning")
    print(f"- Delay learning parameters: B_plus={B_plus}, B_minus={B_minus}")
    print(f"- Learning epochs: {learning_epochs}")
    if noise_enabled:
        print(f"- Noise: jitter={jitter}ms, missing_prob={missing_prob}, extra_prob={extra_prob}")
    print("=" * 80)
    
    # Store all results across epochs
    epoch_results = []
    
    # Setup initial weights and delays
    initial_weights = None
    initial_delays = None
    current_delays = None
    
    # 1. Generate spike trains once
    input_spikes, pattern_sequence = create_pattern_spike_trains(
        PATTERNS, PATTERN_OCCURRENCES, PATTERN_INTERVAL,
        noise_config=noise_config
    )
    
    # Create command line args object for connection setup
    class Args:
        def __init__(self):
            self.fixed_weights = fixed_weights
            self.weight_value = weight_value
            self.learn_delays = True
            self.B_plus = B_plus
            self.B_minus = B_minus
            self.sigma_plus = sigma_plus
            self.sigma_minus = sigma_minus
    
    # 4. Run learning epochs
    for epoch in range(learning_epochs):
        print("\n" + "=" * 60)
        print(f"EPOCH {epoch+1}/{learning_epochs}")
        print("=" * 60)
        
        # For each epoch, set up the simulator from scratch to avoid NEST reset issues
        sim.setup(timestep=0.01)
        
        # 2. Create populations
        input_pop, output_pop = create_populations(input_spikes, NEURON_PARAMS)
        
        args = Args()
        
        # 3. Create connections
        connections, inhibitory_connections = create_connections_and_inhibition(
            input_pop, output_pop, STDP_CONFIG, args
        )
        
        # Get the initial weights and delays
        weights = connections.get("weight", format="array")
        delays = connections.get("delay", format="array")
        
        # On first epoch, store the initial values
        if epoch == 0:
            initial_weights = weights.copy()
            initial_delays = delays.copy()
            current_delays = delays.copy()
            print(f"Initial weights: {initial_weights}")
            print(f"Initial delays: {initial_delays}")
        else:
            # Use the updated delays from previous epoch
            if current_delays is not None:
                try:
                    # Apply the delays from the previous epoch
                    connection_list = []
                    for i in range(len(input_spikes)):  # For each input neuron
                        for j in range(3):  # For each output neuron
                            connection_list.append((i, j, weights[i][j], current_delays[i][j]))
                    
                    # Recreate the connections with the updated delays
                    connections = sim.Projection(
                        input_pop, output_pop,
                        sim.FromListConnector(connection_list),
                        synapse_type=sim.StaticSynapse() if fixed_weights else sim.STDPMechanism(
                            timing_dependence=sim.SpikePairRule(
                                tau_plus=STDP_CONFIG["tau_plus"],
                                tau_minus=STDP_CONFIG["tau_minus"],
                                A_plus=STDP_CONFIG["A_plus"],
                                A_minus=STDP_CONFIG["A_minus"]
                            ),
                            weight_dependence=sim.AdditiveWeightDependence(
                                w_min=0.0,
                                w_max=5.0
                            )
                        ),
                        receptor_type="excitatory",
                        label="connections"
                    )
                    print(f"Applied delays from previous epoch: {current_delays}")
                except Exception as e:
                    print(f"Error applying delays from previous epoch: {e}")
        
        # Set up recording
        input_pop.record("spikes")
        output_pop.record(("spikes", "v"))
        
        # Run the simulation for this epoch
        print(f"Running simulation for {simulation_time} ms...")
        sim.run(simulation_time)
        
        # Get the spike data
        input_data = input_pop.get_data().segments[-1]
        output_data = output_pop.get_data().segments[-1]
        
        # Extract spike times as simple lists
        input_spike_times = []
        for st in input_data.spiketrains:
            input_spike_times.append([float(t) for t in st])
            
        output_spike_times = []
        for st in output_data.spiketrains:
            output_spike_times.append([float(t) for t in st])
        
        # Print spike counts
        print(f"Input spikes: {[len(spikes) for spikes in input_spike_times]}")
        print(f"Output spikes: {[len(spikes) for spikes in output_spike_times]}")
        
        # Calculate accuracy
        accuracy, predictions, true_labels = calculate_accuracy(
            output_data, pattern_sequence, PATTERN_INTERVAL, start_time=10
        )
        print(f"Accuracy: {accuracy:.2%}")
        
        # Calculate detailed accuracy
        detailed_results = calculate_detailed_accuracy(
            output_data, pattern_sequence, PATTERN_INTERVAL, start_time=10
        )
        
        print("\nDetailed Results:")
        print(f"Overall Accuracy: {detailed_results['overall_accuracy']:.2%}")
        print("Per-class Accuracies:")
        for pattern, acc in detailed_results['class_accuracies'].items():
            print(f"  {pattern}: {acc:.2%}")
        
        # Apply the delay learning rule
        print("\nApplying delay learning rule...")
        try:
            # Only apply if we have both input and output spikes
            if sum(len(spikes) for spikes in input_spike_times) > 0 and \
               sum(len(spikes) for spikes in output_spike_times) > 0:
                
                new_delays = apply_delay_learning(
                    connections, input_spike_times, output_spike_times,
                    learning_rate=learning_rate, min_delay=0.1, max_delay=10.0,
                    B_plus=B_plus, B_minus=B_minus, 
                    sigma_plus=sigma_plus, sigma_minus=sigma_minus,
                    window_size=20.0
                )
                
                # Update the delays for the next epoch
                current_delays = new_delays
                
                # Print delay changes
                print("\nDelay changes:")
                for i in range(initial_delays.shape[0]):
                    for j in range(initial_delays.shape[1]):
                        change = new_delays[i][j] - initial_delays[i][j]
                        print(f"  Input{i+1}->Output{j+1}: {initial_delays[i][j]:.4f} -> {new_delays[i][j]:.4f} (Δ{change:+.4f})")
            else:
                print("No spikes detected for delay learning.")
                
        except Exception as e:
            print(f"Error applying delay learning: {e}")
        
        # End this epoch's simulation
        sim.end()
        
        # Store the results for this epoch
        epoch_results.append({
            'epoch': epoch + 1,
            'accuracy': accuracy,
            'detailed_results': detailed_results,
            'weights': weights.copy(),
            'delays': current_delays.copy() if current_delays is not None else delays.copy(),
            'input_spikes': input_spike_times,
            'output_spikes': output_spike_times
        })
    
    # 5. Visualizations and final analysis
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    # Get the final results
    final_results = epoch_results[-1]
    final_weights = final_results['weights']
    final_delays = final_results['delays']
    final_output_data = output_data  # From the last epoch
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot([r['epoch'] for r in epoch_results], 
             [r['accuracy'] for r in epoch_results], 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning Progress')
    plt.grid(True)
    plt.savefig("delay_learning_progress.png")
    print("Learning curve saved to delay_learning_progress.png")
    
    # Plot final spike trains
    plot_spike_trains(input_data, final_output_data)
    plt.savefig("final_spike_trains.png")
    print("Final spike trains saved to final_spike_trains.png")
    
    # Plot membrane potentials
    plot_membrane_potentials(final_output_data, NEURON_PARAMS['v_thresh'])
    plt.savefig("final_membrane_potentials.png")
    print("Final membrane potentials saved to final_membrane_potentials.png")
    
    # Plot final weights
    plot_final_weights(final_weights, title="Final Synaptic Weights")
    plt.savefig("final_weights.png")
    print("Final weights saved to final_weights.png")
    
    # Plot final delays
    plot_delays(final_delays)
    plt.savefig("final_delays.png")
    print("Final delays saved to final_delays.png")
    
    # Plot delay changes
    plt.figure(figsize=(10, 6))
    delay_changes = final_delays - initial_delays
    plt.imshow(delay_changes, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Delay Change (ms)')
    plt.xlabel('Output Neuron')
    plt.ylabel('Input Neuron')
    plt.title('Synaptic Delay Changes')
    plt.xticks(range(final_delays.shape[1]), ['Out 1', 'Out 2', 'Out 3'])
    plt.yticks(range(final_delays.shape[0]), ['In 1', 'In 2'])
    for i in range(final_delays.shape[0]):
        for j in range(final_delays.shape[1]):
            plt.text(j, i, f"{delay_changes[i, j]:.3f}", 
                     ha='center', va='center', 
                     color='white' if abs(delay_changes[i, j]) > 0.5 else 'black')
    plt.savefig("delay_changes.png")
    print("Delay changes saved to delay_changes.png")
    
    # Final mistake analysis
    analyze_pattern_mistakes(final_output_data, pattern_sequence, PATTERN_INTERVAL)
    
    return final_results['accuracy'], final_weights, final_delays, final_output_data, pattern_sequence

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run delay learning experiments')
    
    # Learning parameters
    parser.add_argument('--fixed-weights', action='store_true', help='Use fixed weights (no STDP)')
    parser.add_argument('--weight-value', type=float, default=5.0, help='Value for fixed weights')
    parser.add_argument('--epochs', type=int, default=3, help='Number of learning epochs')
    parser.add_argument('--B-plus', type=float, default=0.1, help='Delay increase parameter')
    parser.add_argument('--B-minus', type=float, default=0.1, help='Delay decrease parameter')
    parser.add_argument('--sigma-plus', type=float, default=10.0, help='Causal time constant')
    parser.add_argument('--sigma-minus', type=float, default=10.0, help='Anti-causal time constant')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate for delay updates')
    
    # Noise parameters
    parser.add_argument('--noise', action='store_true', help='Enable noise in spike trains')
    parser.add_argument('--jitter', type=float, default=1.0, help='Temporal jitter (ms)')
    parser.add_argument('--missing', type=float, default=0.1, help='Missing spike probability')
    parser.add_argument('--extra', type=float, default=0.1, help='Extra spike probability')
    
    # Simulation parameters
    parser.add_argument('--sim-time', type=int, default=1000, help='Simulation time (ms)')
    
    args = parser.parse_args()
    
    # Run the experiment
    start_time = time.time()
    
    accuracy, weights, delays, output_data, pattern_sequence = run_delay_learning_experiment(
        simulation_time=args.sim_time,
        fixed_weights=args.fixed_weights,
        weight_value=args.weight_value,
        learning_epochs=args.epochs,
        B_plus=args.B_plus,
        B_minus=args.B_minus,
        sigma_plus=args.sigma_plus,
        sigma_minus=args.sigma_minus,
        learning_rate=args.learning_rate,
        noise_enabled=args.noise,
        jitter=args.jitter,
        missing_prob=args.missing,
        extra_prob=args.extra
    )
    
    end_time = time.time()
    print(f"\nExperiment completed in {end_time - start_time:.2f} seconds")
    print(f"Final accuracy: {accuracy:.2%}")
    
if __name__ == "__main__":
    main()
