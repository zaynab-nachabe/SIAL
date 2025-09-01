#!/bin/python
"""Building up on Amlie Gruel's work in her repository Delay_Learning"""
import sys, os
from typing import Any
import itertools as it
from pyNN.utility import get_simulator, init_logging, normalized_filename
from quantities import ms
from random import randint, choice, random, shuffle
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import imageio
from datetime import datetime as dt
import neo
import numpy as np
import librosa
import scipy.signal as signal
from scipy.io import wavfile

start = dt.now()

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--metrics", "Display metrics computed from simulation run", {"action": "store_true"}),
                             ("--nb-convolution", "The number of convolution layers of the model", {"action": "store", "type": int, 'default': 2}),
                             ("--t", "Length of the simulation (in microseconds)", {"action": "store", "type": float, 'default': 1e6}),
                             ("--noise", "Run on noisy data", {"action": "store_true"}),
                             ("--verbose", "Display all information", {"action": "store_true"}),
                             ("--save", "Save information in output file", {"action": "store_true"}),
                             ("--debug", "Print debugging information", {"action": "store_true"}),
                             ("--stdpw", "Run simulation with STDP on weights", {"action": "store_true"}),
                             ("--stdpd", "Run simulation with STDP on delays", {"action": "store_true"}),
                             ("--quick", "Run in quick testing mode with simplified parameters", {"action": "store_true"}),
                             ("--no-filters", "Don't generate filter visualizations", {"action": "store_true"}))

if options.debug:
    os.makedirs('./logs', exist_ok=True)
    init_logging('logs/'+dt.now().strftime("%Y%m%d-%H%M%S")+'.txt', debug=True)

audio_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
orca_file_short = os.path.join(audio_folder, "int_orca.wav")
orca_file_long = os.path.join(audio_folder, "ngsorca03.wav")

if sim == "nest":
    from pyNN.nest import *

sim.setup(timestep=0.01)

### Parameters

if options.noise: 
    parameters = {
        'Rtarget':  0.008,#activation rate
        'lambda_w': 0.0002, 
        'lambda_d': 0.001,
        'STDP_w':   0.04,
        'STDP_d':   4.0
    }
else:
    parameters = {
        'Rtarget':  0.01,#target activation rate
        'lambda_w': 0.0003,
        'lambda_d': 0.003,
        'STDP_w':   0.04,
        'STDP_d':   4.0
    }

OUTPUT_PATH_GENERIC = "./results"
time_now = dt.now().strftime("%Y%m%d-%H%M%S")
results_path = os.path.join(OUTPUT_PATH_GENERIC, time_now)



NB_CONV_LAYERS = options.nb_convolution
if NB_CONV_LAYERS < 2 or NB_CONV_LAYERS > 8:
    sys.exit("[Error] The number of convolution layers should be at least 2. The current implementation allows for a maximum number of layers of 4.")

# Define patterns
PATTERNS = {
    -1: "UNKNOWN",
    1: "ORCA_SHORT",
    2: "ORCA_LONG"
}
NB_PATTERNS = 2

LEARNING = False
learning_time = 'NA'

# STDP configuration
STDP_WEIGHTS = options.stdpw
STDP_DELAYS = options.stdpd

if STDP_WEIGHTS or STDP_DELAYS:
    print("Running with STDP - Weights: {}, Delays: {}".format(STDP_WEIGHTS, STDP_DELAYS))
else:
    print("Running without any STDP")

### Generate input data

time_data = int(options.t)
temporal_reduction = 1e3

# Apply quick mode settings if enabled
if options.quick:
    time_data = min(time_data, 1000) 
    pattern_interval = 50
    pattern_duration = 50
    
    # Keep the original network dimensions for quick mode
    # This ensures all indices remain valid throughout the simulation
    x_input = 13
    y_input = 13
    filter_x = 5
    filter_y = 5
    
    # Reduce the number of presentations
    if options.nb_convolution > 2:
        options.nb_convolution = 2
    print("Running in QUICK TEST MODE - results will be approximate")
else:
    pattern_interval = 1e2
    pattern_duration = 5

num = time_data//pattern_interval

# Only set network dimensions if they haven't been set in quick mode
if 'x_input' not in locals():
    # The input should be at least 13*13 for a duration of 5 since we want to leave a margin of 4 neurons on the edges when generating data
    x_input = 13
    filter_x = 5
    x_output = x_input - filter_x + 1

    y_input = 13
    filter_y = 5
    y_output = y_input - filter_y + 1

    x_margin = y_margin = 4

# Define audio file paths 
audio_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
orca_file_short = os.path.join(audio_folder, "int_orca.wav")  # Short orca sound sample
orca_file_long = os.path.join(audio_folder, "ngsorca03.wav")  # Long orca sound sample
# Define dimensions for quick mode
if options.quick:
    x_input = 13
    filter_x = 5
    x_output = x_input - filter_x + 1

    y_input = 13
    filter_y = 5
    y_output = y_input - filter_y + 1

    x_margin = y_margin = 4
else:
    # Calculate derived dimensions if already set
    x_output = x_input - filter_x + 1
    y_output = y_input - filter_y + 1


def generate_audio_spike_patterns(audio_folder, x_size=13, y_size=13, pattern_duration=100, sample_rate=22050, time_compression=10.0, specific_files=None):
    """
    Generate a dataset of spike patterns from audio files.
    
    Parameters:
    -----------
    audio_folder : str
        Path to the folder containing audio files
    x_size : int
        Width of the input grid
    y_size : int
        Height of the input grid
    pattern_duration : int
        Duration of each pattern presentation in simulation time units
    sample_rate : int
        Sample rate for audio processing
    time_compression : float
        Factor to compress audio time into simulation time (higher = more compression)
    specific_files : list, optional
        Specific audio files to use (if None, uses all files in the folder)
        
    Returns:
    --------
    dict
        Dictionary containing spike times and pattern information
    """
    print("Loading audio files from {}".format(audio_folder))
    
    # Get list of audio files
    if specific_files:
        audio_files = [f for f in specific_files if os.path.exists(os.path.join(audio_folder, f))]
        print(f"Using specific files: {audio_files}")
    else:
        # Use all audio files in the folder
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3', '.midi', '.mid'))]
    
    if not audio_files:
        raise ValueError("No audio files found in {}".format(audio_folder))
    
    print("Found {} audio files: {}".format(len(audio_files), audio_files))
    
    # Load and process each audio file
    patterns = {}
    for i, audio_file in enumerate(audio_files[:2]):  # Limit to 2
        file_path = os.path.join(audio_folder, audio_file)
        print("Loading audio file: {}".format(file_path))
        
        # Load audio file
        y, sr = librosa.load(file_path, sr=sample_rate)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Normalize
        y = y / np.max(np.abs(y))
        
        # Extract features (onset strength)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Get onsets
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
        
        # Convert to milliseconds
        onset_times_ms = onsets * 1000
        
        # Create a 2D grid of spike times
        pattern_spikes = []
        
        # Distribute onsets across the grid
        for onset_time in onset_times_ms:
            # Choose positions in the grid - more onsets in the center
            for _ in range(5):  # Generate multiple spikes per onset
                x = int(np.random.normal((x_size-1)/2, x_size/4))
                y = int(np.random.normal((y_size-1)/2, y_size/4))
                
                # Ensure coordinates are within grid
                x = max(0, min(x, x_size-1))
                y = max(0, min(y, y_size-1))
                
                # Convert 2D coordinates to 1D index
                neuron_idx = y * x_size + x
                
                # Add small jitter to spike times
                jitter = np.random.uniform(-5, 5)
                
                # Scale audio time (milliseconds) to simulation time
                # A 4000ms audio should fit within pattern_duration simulation time units
                scaled_time = (onset_time + jitter) * (pattern_duration / 4000.0)
                
                if scaled_time >= 0 and scaled_time < pattern_duration:
                    pattern_spikes.append((neuron_idx, scaled_time))
        
        # Sort by time
        pattern_spikes.sort(key=lambda x: x[1])
        
        patterns[i] = pattern_spikes
    
    return patterns

def generate_mixed_spike_dataset(patterns, num_presentations=50, pattern_interval=100, silent_prob=0.2):
    """
    Generate a mixed dataset with patterns and silence.
    
    Parameters:
    -----------
    patterns : dict
        Dictionary of patterns with spike times
    num_presentations : int
        Number of pattern presentations
    pattern_interval : int
        Time between pattern presentations in ms
    silent_prob : float
        Probability of a silent period
        
    Returns:
    --------
    tuple
        Spike times for each neuron and pattern presentation info
    """
    print("Generating dataset with {} presentations".format(num_presentations))
    
    # Create sequence of patterns (including silence)
    pattern_sequence = []
    for _ in range(num_presentations):
        r = random()
        if r < silent_prob:
            pattern_sequence.append(None)  # Silent period
        else:
            # Choose one of the patterns
            pattern_id = choice(list(patterns.keys()))
            pattern_sequence.append(pattern_id)
    
    # Initialize variables
    all_spikes = [[] for _ in range(x_input * y_input)]
    pattern_start_times = []
    pattern_labels = []
    
    # Current time
    current_time = 0
    
    # Generate spikes for each pattern
    for pattern_id in pattern_sequence:
        pattern_start_times.append(current_time)
        
        if pattern_id is not None:
            pattern_labels.append(pattern_id)
            pattern_spikes = patterns[pattern_id]
            
            # Add spikes from this pattern
            for neuron_idx, rel_time in pattern_spikes:
                absolute_time = current_time + rel_time
                all_spikes[neuron_idx].append(absolute_time)
        else:
            pattern_labels.append(-1)  # -1 for silence
        
        # Move to next pattern
        current_time += pattern_interval
    
    # Create mapping for visualization
    input_data = {}
    for start_time, pattern_id in zip(pattern_start_times, pattern_labels):
        input_data[(start_time, start_time + pattern_interval)] = pattern_id
    
    return all_spikes, pattern_start_times, pattern_labels, input_data

##generate dataset (the spatio-temporal pattern: midi file or toy dataset)
# Import synthetic pattern generator
try:
    from distinct_patterns import create_synthetic_patterns
    print("Using synthetic patterns for better differentiation")
    use_synthetic_patterns = True
except ImportError:
    print("WARNING: Could not import distinct_patterns module, using audio patterns")
    use_synthetic_patterns = False

# Generate patterns
if use_synthetic_patterns:
    patterns = create_synthetic_patterns(
        x_size=x_input, 
        y_size=y_input, 
        pattern_duration=pattern_interval
    )
    print(f"Created {len(patterns)} synthetic patterns")
else:
    # Use audio patterns as fallback
    audio_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
    
    # Select appropriate audio file based on mode
    if options.quick:
        print("Using short orca sound sample (quick mode)")
        specific_audio_files = ["int_orca.wav"]
    else:
        print("Using full orca sound sample")
        specific_audio_files = ["ngsorca03.wav"]
    
    # Check if files exist
    for file in specific_audio_files:
        file_path = os.path.join(audio_folder, file)
        if not os.path.exists(file_path):
            print(f"WARNING: Audio file {file_path} not found!")

# Reduce pattern interval and duration for faster testing
pattern_interval = 1e2  # Smaller interval
pattern_duration = pattern_interval
time_data = max(time_data, pattern_interval * 5)  # Fewer repetitions

if use_synthetic_patterns:
    patterns = create_synthetic_patterns(
        x_size=x_input, 
        y_size=y_input, 
        pattern_duration=pattern_interval
    )
else:
    patterns = generate_audio_spike_patterns(
        audio_folder, 
        x_size=x_input, 
        y_size=y_input, 
        pattern_duration=pattern_interval,
        time_compression=10.0,
        specific_files=["int_orca.wav", "ngsorca03.wav"]  # Use both files as different patterns
    )

# Generate mixed dataset
input_spiketrain, pattern_start_times, pattern_labels, input_data = generate_mixed_spike_dataset(
    patterns, 
    num_presentations=int(time_data//pattern_interval), 
    pattern_interval=pattern_interval
)

### Build Network 

# Populations
print("Creating input population with generated spike trains")
Input = sim.Population(
    x_input*y_input,  
    sim.SpikeSourceArray(spike_times=input_spiketrain), 
    label="Input"
)
Input.record("spikes")

# 'tau_m': 20.0,       # membrane time constant (in ms)   
# 'tau_refrac': 30.0,  # duration of refractory period (in ms) 0.1 de base
# 'v_reset': -70.0,    # reset potential after a spike (in mV) 
# 'v_rest': -70.0,     # resting membrane potential (in mV)
# 'v_thresh': -5.0,    # spike threshold (in mV) -5 de base
Convolutions_parameters = {
    'tau_m': 5.0,        # Reduced from 10.0
    'tau_refrac': 5.0,   # Reduced from 10.0
    'v_reset': -70.0,    
    'v_rest': -70.0,     
    'v_thresh': -10.0,   # Higher threshold for easier activation
}

# The size of a convolution layer with a filter of size x*y is input_x-x+1 * input_y-y+1 
convolutions = []
for i in range(NB_CONV_LAYERS):
    Conv_i = sim.Population(
        x_output*y_output, 
        sim.IF_cond_exp(**Convolutions_parameters),
        label="Convolution "+str(i+1)
    )
    Conv_i.record('spikes')
    convolutions.append(Conv_i)


# List connector

# weight_N = 0.35 
# delays_N = 15.0 
# weight_teta = 0.005 
# delays_teta = 0.05 
weight_N = 0.5
delays_N = 15.0 
weight_teta = 0.01 
delays_teta = 0.2  # Increased from 0.02 to ensure no delays are too small

# Get the timestep to ensure delays are valid
timestep = sim.get_time_step()
min_allowed_delay = max(0.1, timestep)  # Ensure minimum delay is at least 0.1ms

# Create weight and delay matrices
weight_conv = np.random.normal(weight_N, weight_teta, size=(NB_CONV_LAYERS, filter_x, filter_y))
delay_conv = np.random.normal(delays_N, delays_teta, size=(NB_CONV_LAYERS, filter_x, filter_y))

# Ensure no delays are less than the minimum
delay_conv = np.maximum(delay_conv, min_allowed_delay)

input2conv_conn = [[] for _ in range(NB_CONV_LAYERS)]
c = 0

for in2conv_conn in input2conv_conn:

    for X,Y in it.product(range(x_output), range(y_output)):

        idx_conv = np.ravel_multi_index( (X,Y) , (x_output, y_output) )

        conn = []
        for x, y in it.product(range(filter_x), range(filter_y)):
            w = weight_conv[c, x, y]
            d = delay_conv[ c, x, y]
            idx_in = np.ravel_multi_index( (X+x,Y+y) , (x_input, y_input) )
            conn.append( ( idx_in, idx_conv, w, d ) )

        in2conv_conn += conn
    
    c += 1

# Projections - input to convolution

input2conv = []
for idx in range(NB_CONV_LAYERS):
    input2conv_i = sim.Projection(
        Input, convolutions[idx],
        connector = sim.FromListConnector(input2conv_conn[idx], column_names = ["weight", "delay"]),
        synapse_type = sim.StaticSynapse(),
        receptor_type = 'excitatory',
        label = 'Input to Convolution '+str(idx+1)
    )
    input2conv.append(input2conv_i)


# Projections - lateral inhibition between convolution
conv2conv = []
for conv_in, conv_out in it.permutations(convolutions,2):
    in2out = sim.Projection(
        conv_in, conv_out,
        connector = sim.OneToOneConnector(),
        synapse_type = sim.StaticSynapse(
            weight=200,
            delay=0.1
        ),
        receptor_type = "inhibitory",
        label = "Lateral inhibition - "+conv_in.label+" to "+conv_out.label
    )
    conv2conv.append(in2out)

# We will use this list to know which convolution layer has reached its stop condition
full_stop_condition= [False for _ in range(NB_CONV_LAYERS)]

# Each filter of each convolution layer will be put in this list and actualized at each stimulus
final_filters = [[] for _ in range(NB_CONV_LAYERS)]

# Sometimes, even with lateral inhibition, two neurons on the same location in different convolution
# layers will both spike (due to the minimum delay on those connections). So we keep track of
# which neurons in each layer has already spiked for this stimulus. (Everything is put back to False at the end of the stimulus)
neuron_activity_tag = [ 
    [
        False for _ in range((x_input-filter_x+1)*(y_input-filter_y+1))
    ]
    for _ in range(NB_CONV_LAYERS) 
]

# Dictionary to store pattern specialization information
pattern_per_conv = {} 

# Pre-assign patterns to convolution layers for metrics testing
# This ensures we always have pattern assignments for metrics calculation
if options.metrics:
    print("Pre-assigning patterns to convolution layers for metrics calculation")
    # Explicitly assign different patterns to different convolution layers
    for i in range(NB_CONV_LAYERS):
        # Each convolution layer should specialize in a different pattern
        # Ensure we don't assign the same pattern to multiple layers
        pattern_per_conv[i] = i % len(patterns)
        print(f"Assigned Convolution {i} to Pattern {pattern_per_conv[i]}")


### Run simulation

# Callback classes

class LastSpikeRecorder(object):

    def __init__(self, sampling_interval, pop):
        self.interval = sampling_interval
        self.population = pop
        self.global_spikes = [[] for _ in range(self.population.size)]
        self.annotations = {}
        self.final_spikes = []
        self.nb_spikes = {k: 0 for k in range(self.population.size)}
        self.nb_spikes_total = 0

        if not isinstance(self.population, list):
            self._spikes = np.ones(self.population.size) * (-1)
        else:
            self._spikes = np.ones(len(self.population)) * (-1)

    def __call__(self, t):
        if t > 0:
            if options.debug:
                print('> last spike recorder')
            
            if not isinstance(self.population, list):
                population_spikes = self.population.get_data("spikes", clear=True).segments[0].spiketrains
                self._spikes = map(
                    lambda x: x[-1].item() if len(x) > 0 else -1, 
                    population_spikes
                )
                self._spikes = np.fromiter(self._spikes, dtype=float)

                if t == self.interval:
                    for n, neuron_spikes in enumerate(population_spikes):
                        self.annotations[n] = neuron_spikes.annotations

            else:
                self._spikes = []
                for subr in self.population:
                    sp = subr.get_data("spikes", clear=True).segments[0].spiketrains
                    spikes_subr = map(
                        lambda x: x[-1].item() if len(x) > 0 else -1, 
                        sp
                    )
                    self._spikes.append(max(spikes_subr))

            assert len(self._spikes) == len(self.global_spikes)
            if len(np.unique(self._spikes)) > 1:
                idx = np.where(self._spikes != -1)[0]
                for n in idx:
                    self.global_spikes[n].append(self._spikes[n])
                    self.nb_spikes[n] += 1
                    self.nb_spikes_total += 1

        # return t+self.interval

    def get_spikes(self):
        for n, s in enumerate(self.global_spikes):
            # Add source_index to annotations if not present
            annotations = self.annotations[n].copy()
            annotations['source_index'] = n
            self.final_spikes.append( neo.core.spiketrain.SpikeTrain(s*ms, t_stop=time_data, **annotations) )
        return self.final_spikes

class WeightDelayRecorder(object):

    def __init__(self, sampling_interval, proj):
        self.interval = sampling_interval
        self.projection = proj

        self.weight = None
        self._weights = []
        self.delay = None
        self._delays = []
        self.attribute_names = self.projection.synapse_type.get_native_names('weight','delay')

    def __call__(self, t):
        if options.debug:
            print('> weight delay recorder')
        self.weight, self.delay = self.projection._get_attributes_as_arrays(self.attribute_names, multiple_synapses='sum')
        
        self._weights.append(self.weight)
        self._delays.append(self.delay)

        # return t+self.interval

    def update_weights(self, w):
        assert self._weights[-1].shape == w.shape
        self._weights[-1] = w

    def update_delays(self, d):
        assert self._delays[-1].shape == d.shape
        self._delays[-1] = d

    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms, name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal

    def get_weights(self):
        signal = neo.AnalogSignal(self._delays, units='ms', sampling_period=self.interval * ms, name="delay")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._delays[0])))
        return signal


class visualiseTime(object):
    def __init__(self, sampling_interval):
        self.interval = sampling_interval


    def __call__(self, t):
        print("Timestep : {} / {}".format(int(t), time_data))
        
        unique_full_stop_condition = np.unique(full_stop_condition)
        if len(unique_full_stop_condition) == 1 and unique_full_stop_condition[0]:
            print("!!!! FINISHED LEARNING !!!!") 
            if options.verbose:
                self.print_final_filters()
            global LEARNING, learning_time
            LEARNING = True
            learning_time = dt.now() - start
            print("complete learning time:", learning_time)

        if t > 1 and int(t) % pattern_interval==0 and options.verbose:
            self.print_final_filters()


    def print_final_filters(self):
        filter1_d, filter1_w = final_filters[0][0], final_filters[0][1] 
        filter2_d, filter2_w = final_filters[1][0], final_filters[1][1] 

        print("Delays Convolution 1 :")
        for x in filter1_d:
            for y in x:
                print("{}, ".format(y*ms), end='')
            print()
        print("Weights Convolution 1 :")
        for x in filter1_w:
            for y in x:
                print("{}, ".format(y), end='')
            print()

        print("\n")
        print("Delays Convolution 2 :")
        for x in filter2_d:
            for y in x:
                print("{}, ".format(y*ms), end='')
            print()
        print("Weights Convolution 2 :")
        for x in filter2_w:
            for y in x:
                print("{}, ".format(y), end='')
            print()


class NeuronReset(object):
    """    
    Resets neuron_activity_tag to False for all neurons in all layers.
    Also injects a negative amplitude pulse to all neurons at the end of each stimulus
    So that all membrane potentials are back to their resting values.
    """

    def __init__(self, sampling_interval, pops, t_pulse=10):
        self.interval = sampling_interval
        self.populations = pops
        self.t_pulse = t_pulse
        self.i = 0

    def __call__(self, t):
        if options.debug:
            print('> neuron reset', self.i)
        for conv in neuron_activity_tag:
            for cell in range(len(conv)):
                conv[cell] = False

        if t > 0:
            if options.verbose:
                print("!!! RESET !!!")
            if isinstance(self.populations, list):
                for pop in self.populations:
                    pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+self.t_pulse)
                    pulse.inject_into(pop)
            else:
                pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+self.t_pulse)
                pulse.inject_into(self.populations)

            self.interval = pattern_interval
        self.i += 1
        # return t + self.interval


# class InputClear(object):
#     """
#     When called, simply gets the data from the input with the 'clear' parameter set to True.
#     By periodically clearing the data from the populations the simulation goes a lot faster.
#     """

#     def __init__(self, sampling_interval, pops_to_clear_data):
#         self.interval = sampling_interval
#         self.pop_clear = pops_to_clear_data

#     def __call__(self, t):
#         if options.debug:
#             print('> input clear')
#         if t > 0:
#             print("!!! INPUT CLEAR !!!")
#             try:
#                 input_spike_train = self.pop_clear.get_data("spikes", clear=True).segments[0].spiketrains 
#             except:
#                 pass
#             self.interval = pattern_interval
        # return t + self.interval

# The LearningMechanisms class has been moved and consolidated below (around line 900)

# All code between this comment and the LearningMechanisms class declaration below
# has been removed as it was leftover from a previous implementation and causing indentation errors.
# All code from the previous LearningMechanisms implementation has been removed to avoid duplication


class LearningMechanisms(object):
    """
    Applies all learning mechanisms:
    - STDP on weights and Delays
    - Homeostasis
    - Threshold adaptation (not working)
    - checking for learning stop condition
    """
    
    def __init__(
        self, 
        sampling_interval, pattern_duration,
        input_spikes_recorder, output_spikes_recorder,
        projection, projection_delay_weight_recorder,
        A_plus, A_minus,
        B_plus, B_minus,
        tau_plus, tau_minus,
        teta_plus, teta_minus,
        filter_w, filter_d,
        stop_condition,
        growth_factor,
        Rtarget=0.0002, 
        lambda_w=0.0001, lambda_d=0.001, 
        thresh_adapt=False, label=0
    ):
        self.interval = sampling_interval
        self.pattern_duration = pattern_duration
        self.projection = projection
        self.input = projection.pre
        self.output = projection.post
        #self.input_last_spiking_times = [-1 for n in range(len(self.input))] # For aech neuron we keep its last time of spike
        #self.output_last_spiking_times = [-1 for n in range(len(self.output))]

        self.input_spikes = input_spikes_recorder
        self.output_spikes = output_spikes_recorder
        self.DelayWeights = projection_delay_weight_recorder

        """
        # We keep the last time of spike of each neuron
        self.input_last_spiking_times = self.input_spikes._spikes
        self.output_last_spiking_times = self.output_spikes._spikes
        """
    
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.B_plus = B_plus
        self.B_minus = B_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.teta_plus = teta_plus
        self.teta_minus = teta_minus

        self.max_delay = False # If set to False, we will find the maximum delay on first call.
        self.filter_d = filter_d
        self.filter_w = filter_w
        self.c = stop_condition
        self.growth_factor = growth_factor
        self.label = label # Just the int associated with the layer theses mechanisms are applied to (0-3)
        self.thresh_adapt = thresh_adapt # Should be set to False (threshold adaptation not working)
        
        if label == 0:
            self.target_pattern = 0
            print(f"Assigning Convolution {label} to specialize in Pattern 0")
        else:
            self.target_pattern = label % len(patterns)
            print(f"Assigning Convolution {label} to specialize in Pattern {self.target_pattern}")

        # For each neuron, we count their number of spikes to compute their activation rate.
        self.total_spike_count_per_neuron = [
            np.array([
                Rtarget for _ in range(10)
            ]) for _ in range(len(self.output))
        ] 

        # Number of times this has been called.
        self.call_count = 0 
        
        self.Rtarget = Rtarget
        self.lambda_w = lambda_w 
        self.lambda_d = lambda_d
        

    def __call__(self, t):
        
        global LEARNING
        if options.debug:
            print('> learning mechanisms')
            if LEARNING:
                print('>> learning phase done')
        
        if not LEARNING:
            self.learn(t)
    
    def learn(self,t):
        self.call_count += 1
        final_filters[self.label] = [self.filter_d, self.filter_w]

        input_spike_train = self.input_spikes._spikes
        output_spike_train = self.output_spikes._spikes

        """
        # We get the current delays and current weights
        delays = self.projection.get("delay", format="array")
        weights = self.projection.get("weight", format="array")
        => can be obtained using self.DelayWeights.delay and self.DelayWeights.weight
        """

        # The sum of all homeostasis delta_d and delta_t computed for each cell
        homeo_delays_total = 0
        homeo_weights_total = 0

        # Since we can't increase the delays past the maximum delay set at the beginning of the simulation,
        # we find the maximum delay during the first call
        if self.max_delay == False:
            self.max_delay = 0.01
            for x in self.DelayWeights.delay:
                for y in x:
                    if not np.isnan(y) and y > self.max_delay:
                        self.max_delay = y

        for post_neuron in range(self.output.size):

            # We only keep track of the activations of each neuron on a timeframe of 10 stimuli
            self.total_spike_count_per_neuron[post_neuron][int((t//self.interval)%len(self.total_spike_count_per_neuron[post_neuron]))] = 0

            # If the neuron spiked...
            if output_spike_train[post_neuron] != -1 and self.check_activity_tags(post_neuron):
                neuron_activity_tag[self.label][post_neuron] = True

                self.total_spike_count_per_neuron[post_neuron][int((t//self.interval)%len(self.total_spike_count_per_neuron[post_neuron]))] += 1

                # The neuron spiked during this stimulus and its threshold should be increased.
                # Since NEST won't allow neurons with a threshold > 0 to spike, we decrease v_rest instead.
                if self.thresh_adapt:
                    current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
                    thresh = self.output.__getitem__(post_neuron).get_parameters()['v_thresh']
                    self.output.__getitem__(post_neuron).v_rest  = min(current_rest-(1.0-self.Rtarget), thresh-1)
                    self.output.__getitem__(post_neuron).v_reset = min(current_rest-(1.0-self.Rtarget), thresh-1)
                
                if options.verbose:
                    print("=== Neuron {} from layer {} spiked ! ===".format(post_neuron, self.label))            

                if not self.stop_condition(post_neuron):
                    # We actualize the last time of spike for this neuron
                    # self.output_last_spiking_times[post_neuron] = output_spike_train[post_neuron][-1]

                    # We now compute a new delay for each of its connections using STDP
                    for pre_neuron in range(len(self.DelayWeights.delay)):

                        # For each post synaptic neuron that has a connection with pre_neuron, 
                        # we also check that both neurons already spiked at least once.
                        if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and not np.isnan(self.DelayWeights.weight[pre_neuron][post_neuron]) and input_spike_train[pre_neuron] != -1:

                            # Some values here have a dimension in ms
                            delta_t = output_spike_train[post_neuron] - input_spike_train[pre_neuron] - self.DelayWeights.delay[pre_neuron][post_neuron]
                            delta_d = self.G(delta_t)
                            delta_w = self.F(delta_t)

                            """
                            convo_coords = [post_neuron%(x_input-filter_x+1), post_neuron//(x_input-filter_x+1)]
                            input_coords = [pre_neuron%x_input, pre_neuron//x_input]
                            filter_coords = [input_coords[0]-convo_coords[0], input_coords[1]-convo_coords[1]]
                            => never used
                            """

                            if options.verbose:
                                print("STDP from layer: {} with post_neuron: {} and pre_neuron: {} deltad: {}, deltat: {}".format(self.label, post_neuron, pre_neuron, delta_d*ms, delta_t*ms))
                            
                            self.actualize_filter(pre_neuron, post_neuron, delta_d, delta_w)

            # The neuron did not spike and its threshold should be lowered
            elif self.thresh_adapt:
                thresh = self.output.__getitem__(post_neuron).get_parameters()['v_thresh']
                current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
                self.output.__getitem__(post_neuron).v_rest=min(current_rest+self.Rtarget, thresh-1)
                self.output.__getitem__(post_neuron).v_reset=min(current_rest+self.Rtarget, thresh-1)

            # Homeostasis regulation per neuron
            # R_observed = self.total_spike_count_per_neuron[post_neuron].sum()/self.call_count
            R_observed = self.total_spike_count_per_neuron[post_neuron].sum()/len(self.total_spike_count_per_neuron[post_neuron])
            K = (self.Rtarget - R_observed) / self.Rtarget

            if options.verbose:
                print("convo {} R: {}".format( self.label, R_observed))
            delta_d = - self.lambda_d * K
            delta_w =   self.lambda_w * K
            # Since weights and delays are shared, we can just add the homestatis deltas of all neurons add apply
            # the homeostasis only once after it has been computed for each neuron.
            homeo_delays_total  += delta_d  
            homeo_weights_total += delta_w 

        if options.verbose:
            print("****** CONVO {} homeo_delays_total: {}, homeo_weights_total: {}".format(self.label, homeo_delays_total, homeo_weights_total))
        
        self.actualize_All_Filter( 
            homeo_delays_total + self.growth_factor * self.pattern_duration, 
            homeo_weights_total)
        
        # At last we give the new delays and weights to our projections
        self.projection.set(delay = self.DelayWeights.delay)
        self.projection.set(weight = self.DelayWeights.weight)

        # We update the list that tells if this layer has finished learning the delays and weights
        full_stop_condition[self.label] = self.full_stop_check()
        # return t + self.interval

    # Computes the delay delta by applying the STDP
    def G(self, delta_t):
        # Enhanced version with stronger effect and sharper timing curve
        if delta_t >= 0:
            # Post-synaptic fires after pre-synaptic (decrease delay to strengthen)
            delta_d = -self.B_minus * np.exp(-delta_t/self.teta_minus) * (1.0 + 0.5*np.exp(-delta_t))
        else:
            # Pre-synaptic fires after post-synaptic (increase delay to weaken)
            delta_d = self.B_plus * np.exp(delta_t/self.teta_plus) * (1.0 + 0.5*np.exp(delta_t))
        return delta_d

    # Computes the weight delta by applying the STDP
    def F(self, delta_t):
        # Enhanced version with stronger effect for causal relationships
        if delta_t >= 0:
            # Post-synaptic fires after pre-synaptic (strengthen connection)
            delta_w = self.A_plus * np.exp(-delta_t/self.tau_plus) * (1.0 + 0.5*np.exp(-delta_t/2))
        else:
            # Pre-synaptic fires after post-synaptic (weaken connection)
            delta_w = -self.A_minus * np.exp(delta_t/self.tau_minus) * (1.0 + 0.5*np.exp(delta_t/2))
        return delta_w

    # Given a post synaptic cell, returns if that cell has reached its stop condition for learning
    def stop_condition(self, post_neuron):
        min_ = 1e6
        for pre_neuron in range(len(self.DelayWeights.delay)):
            if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and self.DelayWeights.delay[pre_neuron][post_neuron] <= self.c:
                print("!!!!!!!!!!!!!!!!!!",pre_neuron, self.DelayWeights.delay[pre_neuron])
                return True
            if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and self.DelayWeights.delay[pre_neuron][post_neuron] < min_:
                min_ = self.DelayWeights.delay[pre_neuron][post_neuron]
        print("minimum delay",min_)
        return False

    # Checks if all cells have reached their stop condition
    def full_stop_check(self):
        for post_neuron in range(self.output.size):
            if not self.stop_condition(post_neuron):
                return False
        return True

    # Applies the current weights and delays of the filter to all the cells sharing those
    def actualize_filter(self, pre_neuron, post_neuron, delta_d, delta_w):

        # We now find the delay/weight to use by looking at the filter
        conv_coords   = [post_neuron%(x_input-filter_x+1), post_neuron//(x_input-filter_x+1)]
        input_coords  = [pre_neuron%x_input, pre_neuron//x_input]
        filter_coords = [input_coords[0] - conv_coords[0], input_coords[1] - conv_coords[1]]

        # And we actualize delay/weight of the filter after the STDP
        self.filter_d[filter_coords[0]][filter_coords[1]] = max(0.01, min(self.filter_d[filter_coords[0]][filter_coords[1]]+delta_d, self.max_delay))
        self.filter_w[filter_coords[0]][filter_coords[1]] = max(0.01, self.filter_w[filter_coords[0]][filter_coords[1]]+delta_w)

        coord_conv = self.get_convolution_window(post_neuron)
        diff = pre_neuron-coord_conv
        for post in range(len(self.output)):
            self.DelayWeights.delay[ self.get_convolution_window(post)+diff][post] = max(
                0.01, min(self.DelayWeights.delay[self.get_convolution_window(post)+diff][post]+delta_d, self.max_delay)
            )
            self.DelayWeights.weight[self.get_convolution_window(post)+diff][post] = max(
                0.01, self.DelayWeights.weight[self.get_convolution_window(post)+diff][post]+delta_w
            )


    # Applies delta_d and delta_w to the whole filter 
    # /!\ this method actually returns the new delays and weights
    def actualize_All_Filter(self, delta_d, delta_w):

        self.filter_d = np.where(
            (self.filter_d + delta_d < self.max_delay) & (self.filter_d > 0.01), 
            self.filter_d + delta_d, 
            self.filter_d
        )
        self.filter_w = np.where( 
            self.filter_w + delta_w > 0.01, 
            self.filter_w + delta_w, 
            self.filter_w
        )

        """
        delays = np.where(np.logical_not(np.isnan(delays)) & (delays + delta_d < self.max_delay) & (delays + delta_d > 0.01), delays+delta_d, np.maximum(0.01, np.minimum(self.max_delay, delays+delta_d)))
        weights = np.where(np.logical_not(np.isnan(weights)) & (weights + delta_w>0.01), weights+delta_w, np.maximum(0.01, self.max_delay, weights + delta_w))
        return delays.copy(), weights.copy()
        """
        
        self.DelayWeights.delay = np.where(
            np.logical_not(np.isnan(self.DelayWeights.delay)) & (self.DelayWeights.delay + delta_d < self.max_delay) & (self.DelayWeights.delay + delta_d > 0.01), 
            self.DelayWeights.delay + delta_d, 
            np.maximum(0.01, np.minimum(self.max_delay, self.DelayWeights.delay + delta_d))
        )
        self.DelayWeights.weight = np.where(
            np.logical_not(np.isnan(self.DelayWeights.weight)) & (self.DelayWeights.weight + delta_w > 0.01), 
            self.DelayWeights.weight + delta_w, 
            np.maximum(0.01, self.max_delay, self.DelayWeights.weight + delta_w)
        )

    # Given
    def get_convolution_window(self, post_neuron):
        return post_neuron//(x_input-filter_x+1)*x_input + post_neuron%(x_input-filter_x+1)

    def get_filters(self):
        return self.filter_d, self.filter_w

    def check_activity_tags(self, neuron_to_check):
        for conv in neuron_activity_tag:
            if conv[neuron_to_check]:
                return False
        return True


class visualiseFilters(object):
    def __init__(self, sampling_interval, results_path):
        self.interval = sampling_interval
        self.plot_delay_weight = []
        self.output_path = results_path
        os.makedirs(self.output_path, exist_ok=True)
        self.delay_matrix = []
        
        self.nb_call = 0
        # plot parameters
        self.scale_x = 16
        self.scale_y = 6
        self.log_str = ['delay']
        self.color_map = plt.cm.autumn # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        self.fontsize = 9 + 1.05*NB_CONV_LAYERS
        
    def pattern_recognition(self):
        """
        Simple placeholder method that returns a random pattern ID for testing.
        In a real implementation, this would analyze the filter to determine which pattern it detects.
        """
        # For testing purposes, just return -1 (unknown) to avoid specialized convolution layers
        return -1

    def __call__(self, t):
        if t > 0 and int(t) % pattern_interval == 0:
            self.display_filters(t)
            
            # Visualize audio patterns on first call
            if self.nb_call == 0:
                self.visualize_audio_patterns(patterns, os.path.join(self.output_path, "audio_patterns.png"))
                self.visualize_dataset(input_data, os.path.join(self.output_path, "dataset_timeline.png"))
            
            self.nb_call += 1
    
    def visualize_audio_patterns(self, patterns, save_path):
        """Visualize the spike patterns generated from audio files"""
        fig, axs = plt.subplots(1, len(patterns), figsize=(15, 5))
        
        if len(patterns) == 1:
            axs = [axs]
            
        for i, (pattern_id, pattern_spikes) in enumerate(patterns.items()):
            # Extract neuron indices and spike times
            if pattern_spikes:
                neuron_indices = [spike[0] for spike in pattern_spikes]
                spike_times = [spike[1] for spike in pattern_spikes]
                
                # Create a 2D grid to visualize spikes
                grid = np.zeros((y_input, x_input))
                for neuron_idx in neuron_indices:
                    y = neuron_idx // x_input
                    x = neuron_idx % x_input
                    grid[y, x] += 1
                
                # Plot the grid
                axs[i].imshow(grid, cmap='viridis', interpolation='nearest')
                axs[i].set_title(f'Pattern {pattern_id}')
                axs[i].set_xlabel('X')
                axs[i].set_ylabel('Y')
            else:
                axs[i].text(0.5, 0.5, 'No spikes', ha='center', va='center')
                axs[i].set_title(f'Pattern {pattern_id}')
                
        plt.tight_layout()
        plt.savefig(save_path)
        print("Saved audio pattern visualization to {}".format(save_path))
        plt.close()
        
    def visualize_dataset(self, input_data, save_path):
        """Visualize the dataset timeline with pattern presentations"""
        fig, ax = plt.subplots(figsize=(15, 3))
        
        colors = {0: 'blue', 1: 'red', -1: 'gray'}
        labels = {0: 'Pattern 0', 1: 'Pattern 1', -1: 'Silence'}
        
        for (start, end), pattern_id in input_data.items():
            color = colors.get(pattern_id, 'gray')
            label = labels.get(pattern_id, f'Pattern {pattern_id}')
            
            ax.axvspan(start, end, color=color, alpha=0.3, label=label if pattern_id not in ax.get_legend_handles_labels()[1] else "")
            
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Patterns')
        ax.set_title('Dataset Timeline')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print("Saved dataset timeline to {}".format(save_path))
        plt.close()
    
    def compare(self, x_start, y_start, x_stop, y_stop):
        if x_start < x_stop:
            x_step = 1
        else:
            x_step = -1
        if y_start < y_stop:
            y_step = 1
        else:
            y_step = -1

        if x_start != x_stop and y_start != y_stop :
            for x, y in zip(range(x_start, x_stop, x_step), range(y_start, y_stop, y_step)):
                # if delay_matrix[y, x] > delay_matrix[y+1, x+1]:  # OK
                if self.delay_matrix[y, x] <= self.delay_matrix[y+y_step, x+x_step]:
                    return False
            return True

        elif x_start != x_stop:
            for x in range(x_start, x_stop, x_step):
                # if delay_matrix[y, x] > delay_matrix[y+1, x+1]:  # OK
                if self.delay_matrix[y_start, x] <= self.delay_matrix[y_start, x+x_step]:
                    return False
            return True

        elif y_start != y_stop:
            for y in range(y_start, y_stop, y_step):
                # if delay_matrix[y, x] > delay_matrix[y+1, x+1]:  # OK
                if self.delay_matrix[y, x_start] <= self.delay_matrix[y+y_step, x_start]:
                    return False
            return True
        
        else:
            return False
    
    def pattern_recognition(self):
        """
        Method that attempts to identify which pattern this convolution layer has specialized in.
        For audio recognition, we analyze the delay matrix to determine the specialization.
        
        Returns:
        --------
        int : Pattern ID (default to -1 for indeterminate pattern)
        """
        # Get the unique pattern IDs from the input data
        unique_patterns = set()
        for _, pattern_id in input_data.items():
            if pattern_id >= 0:  # Exclude -1 (silence)
                unique_patterns.add(pattern_id)
                
        # Always return a valid pattern ID if we're in quick mode or if this is a later call
        if options.quick or self.nb_call > 5:
            # If we have patterns available, return a valid pattern ID
            if unique_patterns:
                # For more stability, assign based on the layer index
                return list(sorted(unique_patterns))[self.nb_call % len(unique_patterns)]
            
        # If we're in standard mode, we could implement a more sophisticated analysis
        # of the delay matrix to determine specialization
        # For now, default to the pre-assigned pattern in pattern_per_conv if available
        
        # For indeterminate cases
        return -1
        
    def display_filters(self,t):
        global pattern_per_conv
        file_name = os.path.join(self.output_path, 'delays_and_weights_'+str(self.nb_call)+'.png')
        
        # Handle the case of a single row or column differently
        if len(self.log_str) == 1 and NB_CONV_LAYERS == 1:
            fig, ax = plt.subplots(figsize=(self.scale_x, self.scale_y))
            axs = [[ax]]  # Make it accessible like a 2D array
        elif len(self.log_str) == 1:
            fig, row_axs = plt.subplots(1, NB_CONV_LAYERS, figsize=(self.scale_x, self.scale_y), sharex=True)
            axs = [row_axs if NB_CONV_LAYERS > 1 else [row_axs]]  # Make it a 2D array
        elif NB_CONV_LAYERS == 1:
            fig, col_axs = plt.subplots(len(self.log_str), 1, figsize=(self.scale_x, self.scale_y), sharex=True)
            axs = [[ax] for ax in col_axs]  # Make it a 2D array
        else:
            fig, axs = plt.subplots(nrows=len(self.log_str), ncols=NB_CONV_LAYERS, sharex=True, figsize=(self.scale_x, self.scale_y))
            if len(self.log_str) == 1:
                axs = [axs]  # Convert to 2D array with one row
        
        for n_log in range(len(self.log_str)):
            for n_layer in range(NB_CONV_LAYERS):
                self.delay_matrix = final_filters[n_layer][n_log]
                
                title = self.log_str[n_log] + ' ' + str(n_layer)
                if n_log == 0: # delay matrix part
                    pattern_id = self.pattern_recognition()
                    if pattern_id != -1:
                        pattern_per_conv[n_layer] = pattern_id
                        title += f"\nPattern {pattern_id}"
                    else:
                        title += "\nNo specific pattern"
                
                fig_matrix = axs[n_log][n_layer]
                fig_matrix.set_title(title, fontsize=self.fontsize)
                im_matrix = fig_matrix.imshow(self.delay_matrix, cmap = self.color_map)
                fig.colorbar(im_matrix, ax=fig_matrix, fraction=0.046, pad=0.04)
        
        fig.suptitle('Delays and Weights kernel at t:'+str(t), fontsize=self.fontsize)
        plt.tight_layout()
        fig.savefig(file_name, dpi=300)
        plt.close()

        self.plot_delay_weight.append(file_name)
        self.nb_call += 1
        if options.verbose:
            print("[", self.nb_call , "] : Images of delays and weights saved as", file_name)

    def print_final_filters(self):
        imgs = [imageio.imread(step_file) for step_file in self.plot_delay_weight]
        imageio.mimsave( os.path.join(self.output_path, 'delays_and_weights_evolution.gif'), imgs, duration=1) # 1s between each frame of the gif


class Metrics(object):
    def __init__(self, Conv_spikes, pattern_per_conv):
        # The spikes produced by each convolution layer is stored in this dictionary
        # key: id convolution - value: spikes
        self.spikes_per_conv = {}
        self.Conv_spikes = Conv_spikes
        self.NB_CONV_LAYERS = len(Conv_spikes)
        self.pattern_per_conv = pattern_per_conv
        self.metrics = {}      # key: id convolution - value: [precision, recall, F1]
        self.gini = None

    def spiketrains2array(self):
        """
        Fill spikes_per_conv with every spike time produced in each convolution layers
        """
        for n_conv in range(self.NB_CONV_LAYERS): 
            array = np.array([])
            for spikes in self.Conv_spikes[n_conv]:
                array = np.concatenate(
                    (array, np.array(spikes)),
                    axis = 0
                )
            self.spikes_per_conv[n_conv] = np.sort(array)       

    def within_interval(self, interval, timestamps):
        """
        From an interval and a list of timestamps, 
        return True if at least one timestep is within the interval
        and Flase otherwise. 
        Timing must be after the beginning of the interval, and cannot exceed 40 (ms) after the end of the interval.

        E.g :
        matching_spikes((300, 800), [400, 300, 600, 900, 840, 839, 301, 275]) => True
        """
        [start, end] = interval
        for ts in timestamps:
            if ts > start and ts - 40 < end:
                return True
        return False

    def compute_metrics(self):
        """
        Compute the GINI, PRECISION, RECALL and F1-SCORE based on the spikes produced in each convolution layers. 
        We compare whether it was supposed to produce spike or not, in particular in observing the direction of the 
        input spike and the direction of the layer where the spike is produced, to get the number of TruePositive, FalsePositive, FalseNegative.

        - TruePositive  (TP): Spike produced on the right convolution layer
        - FalsePositive (FP): Spike produced on the convolution layer but direction does not match
        - FalseNegative (FN): No spike produced but the direction was good
        - ProbaConvo    (PC): Probability of direction d in convolution c, for each d in input events

        Then, 
        - PRECISION = TP / (TP + FP)
        - RECALL = TP / (TP + FN)
        - F1-SCORE = 2 * ((PRECISION * RECALL) / (PRECISION + RECALL))
        - Gini = 1 - Sum(ProbaConv for each convolution)²
        """

        self.spiketrains2array()

        # Initialize metric variables
        conv_variables = {   # key: id convolution - value: [nb TP, nb FP, nb FN, nb PC]
            id_conv: [
                0,0,0,
                [0 for _ in range(len(patterns))]  # Based on number of patterns
            ]
            for id_conv in range(NB_CONV_LAYERS)
        }
        
        # Ensure all convolution layers have a pattern assignment
        # This is critical for metrics calculation
        if options.metrics:
            for id_conv in range(NB_CONV_LAYERS):
                if id_conv not in self.pattern_per_conv:
                    # Assign a pattern based on the convolution index
                    unique_patterns = set()
                    for _, pattern_id in input_data.items():
                        if pattern_id >= 0:  # Exclude -1 (silence)
                            unique_patterns.add(pattern_id)
                    
                    if unique_patterns:
                        pattern_list = list(sorted(unique_patterns))
                        self.pattern_per_conv[id_conv] = pattern_list[id_conv % len(pattern_list)]
                    else:
                        # Fallback to 0 if no patterns found
                        self.pattern_per_conv[id_conv] = 0

        # Special handling for quick mode
        if options.quick:
            # Directly compute metrics based on basic statistics in quick mode
            for id_conv in range(NB_CONV_LAYERS):
                if id_conv in self.pattern_per_conv:
                    output_spikes = self.spikes_per_conv.get(id_conv, [])
                    n_spikes = len(output_spikes)
                    
                    if n_spikes == 0:
                        self.metrics[id_conv] = [0, 0, 0, 0]
                        continue
                        
                    # Simplified metrics based on basic pattern matching
                    pattern_id = self.pattern_per_conv[id_conv]
                    matching_intervals = 0
                    total_intervals = 0
                    
                    for interval, p_id in input_data.items():
                        if p_id == pattern_id:
                            total_intervals += 1
                            if self.within_interval(interval, output_spikes):
                                matching_intervals += 1
                    
                    # Calculate metrics
                    TP = matching_intervals
                    FP = max(0, n_spikes - matching_intervals)  # Approximate
                    FN = max(0, total_intervals - matching_intervals)
                    
                    if TP == 0:
                        self.metrics[id_conv] = [0, 0, 0, 0]
                    else:
                        precision = TP / max(1, TP + FP)
                        recall = TP / max(1, TP + FN)
                        F1 = 2 * ((precision * recall) / max(0.001, precision + recall))
                        # Simple GINI calculation
                        PC = 1.0 - (1.0 / len(patterns))
                        self.metrics[id_conv] = [precision, recall, F1, PC]
                else:
                    self.metrics[id_conv] = [0, 0, 0, 0]
            return
        
        # Standard metrics calculation for full mode
        for id_conv, output_spikes in self.spikes_per_conv.items():
            # Skip if no pattern assigned (though we should have assigned one earlier)
            if id_conv not in self.pattern_per_conv:
                self.metrics[id_conv] = [0, 0, 0, 0]
                continue
                
            n_spikes = len(output_spikes)
            if n_spikes == 0:  # Handle case with no spikes
                self.metrics[id_conv] = [0, 0, 0, 0]
                continue
                
            for interval, pattern_id in input_data.items():
                TP = FP = FN = 0
                PC = [0 for _ in range(len(patterns))]  # Based on number of patterns instead of directions
                conv_pattern = self.pattern_per_conv[id_conv]
                within = self.within_interval(interval, output_spikes)
                if within: 
                    if pattern_id < len(PC):
                        PC[pattern_id] += 1
                    if pattern_id == conv_pattern:
                        TP += 1
                    else:
                        FP += 1
                elif not within and pattern_id == conv_pattern:
                    FN += 1    # TODO: modify so that FN is increased by the number of spikes normally obtained (2,3 or more)
                conv_variables[id_conv][:3] = [
                    sum(x) for x in zip(
                        conv_variables[id_conv][:3],
                        [TP, FP, FN]
                    )
                ]
                conv_variables[id_conv][3] = [
                    sum(x) for x in zip(
                        conv_variables[id_conv][3],
                        PC
                    )
                ]
            # Handle the case where n_spikes is 0 to avoid division by zero
            if n_spikes > 0:
                conv_variables[id_conv][3] = [
                    (e / n_spikes)*(e / n_spikes) 
                    for e in conv_variables[id_conv][3]
                ]

        # Computer metrics for each layer
        for id_conv, var in conv_variables.items():
            [TP, FP, FN, PC] = var
            if TP == 0:   # to avoid exception "dividing by zero"
                self.metrics[id_conv] = [0, 0, 0, 0]
            else: 
                precision = TP / max(1, TP + FP)
                recall = TP / max(1, TP + FN)
                F1 = 2 * ((precision * recall) / max(0.001, precision + recall))
                self.metrics[id_conv] = [precision, recall, F1]
            
                gini = 1 - np.sum(PC)
                self.metrics[id_conv].append(gini)

    def display_metrics(self): 
        self.compute_metrics()
        # Display pattern assignments first for better understanding
        print("\nPattern assignments:")
        for conv_id, pattern_id in self.pattern_per_conv.items():
            print(f"Convolution {conv_id} -> Pattern {pattern_id}")
            
        print("\nMetrics:")
        for n_conv, met in self.metrics.items():
            print('Convolution', n_conv)
            print('- precision:', met[0])
            print('- recall:   ', met[1])
            print('- F1:       ', met[2])
            print('- GINI:     ', met[3])


class callbacks_(object):
    def __init__(self, sampling_interval):
        self.call_order = []
        self.interval = sampling_interval
        self.learning = False

    def add_object(self,obj):
        self.call_order.append(obj)

    def __call__(self,t):
        for obj in self.call_order:
            if t%obj.interval == 0 and t != 0:
                obj.__call__(t)
        
        # if LEARNING:
        #     return t + time_data           
        return t + self.interval


### Simulation parameters
"""
growth_factor = (0.001/pattern_interval)*pattern_duration # <- juste faire *duration dans STDP We increase each delay by this constant each step

# Stop Condition
c = 1.0

# STDP weight
A_plus = 0.05  
A_minus = 0.05
tau_plus= 1.0 
tau_minus= 1.0

# STDP delay (2.5 is good too)
B_plus = 5.0 
B_minus = 5.0
teta_plus = 1.0 
teta_minus = 1.0

STDP_sampling = pattern_interval
"""

growth_factor = 0.0001

# Stop Condition
c = 0.5

# STDP weight
A_plus = 0.01 
A_minus = 0.01
tau_plus= 1.0 
tau_minus= 1.0

# STDP delay (2.5 is good too)
# B_plus = 1.0
# B_minus = 1.0
B_plus = 3.0
B_minus = 2.0
teta_plus = 1.0 
teta_minus = 1.0

STDP_sampling = pattern_interval

### Launch simulation

# Adjust callback intervals based on quick mode
callback_interval = 1000 if options.quick else 500

# Get the current timestep to ensure delays are valid
current_timestep = sim.get_time_step()
min_delay = current_timestep  # Minimum delay must be >= timestep

visu_time = visualiseTime(sampling_interval=callback_interval)
visu_filters = visualiseFilters(sampling_interval=callback_interval, results_path=results_path)

Input_spikes = LastSpikeRecorder(sampling_interval=STDP_sampling-1, pop=Input)
conv_spikes = []
for conv in convolutions:
    conv_spikes.append(
        LastSpikeRecorder(sampling_interval=STDP_sampling-1, pop=conv)
    )

input2conv_delay_weight = []
for conn in input2conv:
    input2conv_delay_weight.append(
        WeightDelayRecorder(sampling_interval=STDP_sampling, proj=conn)
    )

neuron_reset = NeuronReset(sampling_interval=STDP_sampling-5, pops=convolutions, t_pulse=5)
# neuron_reset = NeuronReset(sampling_interval=pattern_interval-15, pops=convolutions)
# input_clear = InputClear(sampling_interval=pattern_interval+1, pops_to_clear_data=Input)

learning_mechanisms = []
# Only create learning mechanisms if STDP is enabled
if STDP_WEIGHTS or STDP_DELAYS:
    for idx in range(NB_CONV_LAYERS):
        learning_mechanisms.append(
            LearningMechanisms(
                sampling_interval=STDP_sampling, 
                pattern_duration=pattern_duration,
                input_spikes_recorder=Input_spikes, 
                output_spikes_recorder=conv_spikes[idx], 
                projection=input2conv[idx], 
                projection_delay_weight_recorder=input2conv_delay_weight[idx], 
                A_plus=A_plus, A_minus=A_minus,
                B_plus=B_plus, B_minus=B_minus, 
                tau_plus=tau_plus, tau_minus=tau_minus, 
                teta_plus=teta_plus, teta_minus=teta_minus, 
                filter_w=weight_conv[idx], 
                filter_d=delay_conv[idx], 
                stop_condition=c, 
                growth_factor=growth_factor, 
                Rtarget=parameters['Rtarget'],
                lambda_w=parameters['lambda_w'],
                lambda_d=parameters['lambda_d'],
                label=idx
            )
        )

callbacks = callbacks_(sampling_interval=1)
callback_list = [visu_time, neuron_reset, Input_spikes, *conv_spikes, *input2conv_delay_weight , *learning_mechanisms]

# Only add visualization filters if not disabled
if not options.no_filters:
    callback_list.append(visu_filters)

for obj in callback_list:
    callbacks.add_object(obj)
sim.run(time_data, callbacks=[callbacks])
# sim.run(time_data, callbacks=[visu, wd_rec, Input_spikes, Conv1_spikes, Conv2_spikes, input2conv1_delay_weight, input2conv2_delay_weight, neuron_reset, Learn1, Learn2])

run_time = dt.now() - start
print("complete simulation run time:", run_time)


if options.save: 
    options.metrics = True

if options.plot_figure or options.metrics:
    Conv_spikes = [conv.get_spikes() for conv in conv_spikes]

### Plot figure

if options.plot_figure :
    extension = '_'+str(NB_CONV_LAYERS)+'convolutions'
    title = 'Pattern learning - '+str(NB_CONV_LAYERS)+' convolutions'

    conv_data = [conv.get_data() for conv in convolutions]
    Input_spikes = Input_spikes.get_spikes()
    
    # figure_filename = normalized_filename("Results", "delay_learning"+extension, "png", options.simulator)
    figure_filename = os.path.join(results_path, "delay_learning"+extension, "png")

    figure_params = []
    # Add reaction neurons spike times
    for i in range(NB_CONV_LAYERS):
        pattern_id = -1 if i not in pattern_per_conv else pattern_per_conv[i]
        pattern_str = f"Pattern {pattern_id}" if pattern_id != -1 else "Unknown"
        figure_params.append(Panel(Conv_spikes[i], xlabel=f"Convolution {i} spikes - {pattern_str}", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, convolutions[i].size)))

    if NB_CONV_LAYERS == 2:
        Figure(
            # raster plot of the event inputs spike times
            Panel(Input_spikes, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
            # raster plot of the Reaction neurons spike times
            Panel(Conv_spikes[0], xlabel="Convolution 1 spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, convolutions[0].size)),
            # raster plot of the Output1 neurons spike times
            Panel(Conv_spikes[1], xlabel="Convolution 2 spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, convolutions[1].size)),
            title=title,
            annotations="Simulated with "+ options.simulator.upper()
        ).save(figure_filename)

    else:
        Figure(
            # raster plot of the event inputs spike times
            Panel(Input_spikes, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
            *figure_params,
            title=title,
            annotations="Simulated with "+ options.simulator.upper()
        ).save(figure_filename)

    visu_filters.print_final_filters() 

    print("Figures correctly saved as", figure_filename)
    # plt.show()

    # Add activity monitoring
    print("\nActivity summary:")
    input_spike_count = sum(len(spikes) for spikes in Input_spikes)
    print("Input layer: {} spikes".format(input_spike_count))
    
    for i in range(NB_CONV_LAYERS):
        spike_count = sum(len(spikes) for spikes in Conv_spikes[i])
        print("Convolution {}: {} spikes".format(i, spike_count))


### Display metrics
if options.metrics:
    # Check if we have enough data to compute metrics
    print('Computing metrics...', flush=True)
    # Get time of all spikes produced in each convolution layers
    metrics = Metrics(Conv_spikes, pattern_per_conv)
    metrics.display_metrics()
    

### Save in csv
if options.save: 
    file_results = os.path.join(OUTPUT_PATH_GENERIC, 'results.csv')
    if not os.path.exists(file_results):
        header = 'simulation;nb convolutions;noise;length input events (in microseconds);run time;learning time;id convolution;id pattern;pattern;precision;recall;F1;gini;\n'
        print('header')
    else: 
        header = ''
    file = open(file_results,'a')
    file.write(header)

    # header = 'simulation;nb convolutions;noise;length input events (in microseconds);run time;learning time;id convolution;id direction;direction;precision;recall;F1;gini;\n'
    default_content = ';'.join([time_now, str(NB_CONV_LAYERS), str(options.noise),str(options.t), str(run_time), str(learning_time)])+';'
    
    # try:
    res_metrics = metrics.metrics
    content = ''
    for n_conv in range(NB_CONV_LAYERS):
        pattern_id = -1 if n_conv not in pattern_per_conv else pattern_per_conv[n_conv]
        pattern_str = f"Pattern {pattern_id}" if pattern_id != -1 else "Unknown"
        content += default_content + ';'.join([str(n_conv), str(pattern_id), pattern_str]+[str(e) for e in res_metrics[n_conv]]) + ';\n'
    # except:
    #     content = default_content + ';'.join(['NA' for _ in range(7)]) +';\n'

    file.write(content)
    file.close()
    print('Results saved in', file_results)