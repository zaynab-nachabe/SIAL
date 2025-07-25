import numpy as np
import pyNN.nest as sim

def delay_learning_rule(pre_spike_time, post_spike_time, current_delay, 
                        B_plus=0.1, B_minus=0.1, sigma_plus=10.0, sigma_minus=10.0):
    """
    Implements the delay learning rule as described in the literature.
    Based on the works of:
    - König et al. (1996): Integrator or coincidence detector? The role of the cortical neuron revisited
    - Rossant et al. (2011): Sensitivity of noisy neurons to coincident inputs
    - Nadafian & Ganjtabesh (2020): Bio-plausible Unsupervised Delay Learning for Extracting 
      Temporal Features in Spiking Neural Networks
    
    Arguments:
    pre_spike_time -- time when the presynaptic neuron fired
    post_spike_time -- time when the postsynaptic neuron fired
    current_delay -- current delay value for this synapse
    B_plus -- magnitude of delay increase for causal spike pairs
    B_minus -- magnitude of delay decrease for anti-causal spike pairs
    sigma_plus -- time constant for causal spike pairs
    sigma_minus -- time constant for anti-causal spike pairs
    
    Returns:
    delta_delay -- the change in delay to be applied
    """
    # Calculate the time difference adjusted for current delay
    delta_t = post_spike_time - pre_spike_time - current_delay
    
    #apply piecewise function G
    if delta_t >= 0:  #if post fires after pre, I decrease the delay
        delta_delay = -B_minus * np.exp(-delta_t / sigma_minus)
    else:  #if post fires befre pre, I increase delay
        delta_delay = B_plus * np.exp(delta_t / sigma_plus)
    
    return delta_delay

def apply_delay_learning(connections, input_spikes, output_spikes,
                         learning_rate=0.01, min_delay=0.1, max_delay=10.0,
                         B_plus=0.1, B_minus=0.1, sigma_plus=10.0, sigma_minus=10.0,
                         window_size=20.0):
    """
    Applies the delay learning rule to update connection delays based on spike timing.
    
    Arguments:
    connections -- PyNN projection object containing the connections to update
    input_spikes -- list of spike times for each input neuron
    output_spikes -- list of spike times for each output neuron
    learning_rate -- scaling factor for delay updates
    min_delay -- minimum allowed delay value
    max_delay -- maximum allowed delay value
    B_plus, B_minus, sigma_plus, sigma_minus -- parameters for delay learning rule
    window_size -- time window to consider for spike pairing (ms)
    
    Returns:
    updated_delays -- the new delay matrix after learning
    """
    print(f"Input spikes: {[len(spk) for spk in input_spikes]}")
    print(f"Output spikes: {[len(spk) for spk in output_spikes]}")
    
    connection_list = connections.get(["weight", "delay"], format="list")
    
    delay_updates = {}
    
    for conn in connection_list:
        pre_idx = int(conn[0]) 
        post_idx = int(conn[1])
        current_delay = float(conn[3])
        
        print(f"Processing connection from pre_idx={pre_idx} to post_idx={post_idx}, current delay={current_delay}")
        
        if pre_idx >= len(input_spikes) or post_idx >= len(output_spikes):
            print(f"Warning: Index out of range - pre_idx={pre_idx}, post_idx={post_idx}")
            continue
            
        if not input_spikes[pre_idx] or not output_spikes[post_idx]:
            print(f"No spikes for this connection: pre_idx={pre_idx}, post_idx={post_idx}")
            continue
        
        spike_pairs = find_nearest_spikes(
            input_spikes[pre_idx], 
            output_spikes[post_idx],
            current_delay, 
            window_size
        )
        
        conn_updates = []
        for pre_time, post_time in spike_pairs:
            delta_delay = delay_learning_rule(
                pre_time, post_time, current_delay,
                B_plus, B_minus, sigma_plus, sigma_minus
            )
            conn_updates.append(delta_delay)
            
        if conn_updates:
            avg_update = learning_rate * np.mean(conn_updates)
            key = (pre_idx, post_idx)
            delay_updates[key] = avg_update
    
    current_delays = np.array(connections.get("delay", format="array"))
    new_delays = np.copy(current_delays)
    
    print("Connection matrix shape:", current_delays.shape)
    
    if delay_updates:
        print("Delay updates to be applied:")
        for (pre, post), update in delay_updates.items():
            print(f"  Connection {pre}->{post}: {update:+.4f} ms")
    else:
        print("No delay updates to apply")
    
    for pre_idx in range(current_delays.shape[0]):
        for post_idx in range(current_delays.shape[1]):
            key = (pre_idx, post_idx)
            if key in delay_updates:
                new_delay = current_delays[pre_idx][post_idx] + delay_updates[key]
                new_delays[pre_idx][post_idx] = np.clip(new_delay, min_delay, max_delay)
                print(f"Updated delay {pre_idx}->{post_idx}: {current_delays[pre_idx][post_idx]:.4f} -> {new_delays[pre_idx][post_idx]:.4f}")
    
    connections.set(delay=new_delays)
    
    return connections.get("delay", format="array")

def find_nearest_spikes(pre_spike_times, post_spike_times, current_delay=0.0, 
                        time_window=20.0):
    """
    Finds the nearest pre-post spike pairs within a given time window,
    accounting for the current delay value.
    
    Returns a list of tuples (pre_time, post_time) for each valid pair.
    """
    spike_pairs = []
    
    pre_spike_times = [float(t) for t in pre_spike_times]
    post_spike_times = [float(t) for t in post_spike_times]
    
    for pre_time in pre_spike_times:
        arrival_time = pre_time + current_delay
        
        valid_post_times = [t for t in post_spike_times 
                          if abs(t - arrival_time) <= time_window]
        
        if valid_post_times:
            closest_post = min(valid_post_times, 
                              key=lambda t: abs(t - arrival_time))
            spike_pairs.append((pre_time, closest_post))
    
    return spike_pairs

def ensure_list_of_floats(spike_times):
    if hasattr(spike_times, '__len__'):
        try:
            return [float(t) for t in spike_times]
        except (TypeError, ValueError):
            print(f"Warning: Could not convert spike times to list. Type: {type(spike_times)}")
            return []
    else:
        print(f"Warning: Spike times not in expected format. Type: {type(spike_times)}")
        return []
