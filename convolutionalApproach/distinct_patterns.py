import numpy as np

def create_synthetic_patterns(x_size, y_size, pattern_duration):
    """
    Create synthetic spike patterns that are spatially and temporally distinct.
    
    Parameters:
    -----------
    x_size, y_size : dimensions of the input grid
    pattern_duration : duration of each pattern in simulation time
    
    Returns:
    --------
    Dict of patterns with spike times
    """
    patterns = {}
    
    # Pattern 0: Diagonal pattern (bottom-left to top-right)
    pattern0_spikes = []
    for i in range(min(x_size, y_size)):
        neuron_idx = i * x_size + i  # Diagonal index
        
        # Create 5 spikes per position with increasing times
        for j in range(5):
            spike_time = 5.0 + j*5.0 + i*3.0  # Time increases with position
            if spike_time < pattern_duration:
                pattern0_spikes.append((neuron_idx, spike_time))
    
    # Add some spikes in the bottom-left quadrant
    for x in range(x_size//2):
        for y in range(y_size//2):
            neuron_idx = y * x_size + x
            # Sparse spike pattern
            if (x + y) % 3 == 0:
                spike_time = 10.0 + (x+y)*2.0
                if spike_time < pattern_duration:
                    pattern0_spikes.append((neuron_idx, spike_time))
    
    # Pattern 1: Anti-diagonal pattern (top-left to bottom-right)
    pattern1_spikes = []
    for i in range(min(x_size, y_size)):
        neuron_idx = i * x_size + (x_size - 1 - i)  # Anti-diagonal index
        
        # Create 5 spikes per position with different timing
        for j in range(5):
            spike_time = 15.0 + j*5.0 + i*2.0  # Different timing pattern
            if spike_time < pattern_duration:
                pattern1_spikes.append((neuron_idx, spike_time))
    
    # Add some spikes in the top-right quadrant
    for x in range(x_size//2, x_size):
        for y in range(y_size//2):
            neuron_idx = y * x_size + x
            # Different sparse pattern
            if (x * y) % 4 == 0:
                spike_time = 20.0 + x + y*2.0
                if spike_time < pattern_duration:
                    pattern1_spikes.append((neuron_idx, spike_time))
    
    # Pattern 2: Center-outward pattern
    pattern2_spikes = []
    center_x, center_y = x_size // 2, y_size // 2
    center_idx = center_y * x_size + center_x
    
    # Central spikes
    for j in range(8):
        spike_time = 5.0 + j*3.0
        if spike_time < pattern_duration:
            pattern2_spikes.append((center_idx, spike_time))
    
    # Radial spikes
    for radius in range(1, min(x_size, y_size)//2):
        for angle in np.linspace(0, 2*np.pi, 8*radius, endpoint=False):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # Ensure within bounds
            if 0 <= x < x_size and 0 <= y < y_size:
                neuron_idx = y * x_size + x
                spike_time = 25.0 + radius*4.0
                if spike_time < pattern_duration:
                    pattern2_spikes.append((neuron_idx, spike_time))
    
    # Sort by time
    pattern0_spikes.sort(key=lambda x: x[1])
    pattern1_spikes.sort(key=lambda x: x[1])
    pattern2_spikes.sort(key=lambda x: x[1])
    
    patterns[0] = pattern0_spikes
    patterns[1] = pattern1_spikes
    patterns[2] = pattern2_spikes
    
    return patterns
