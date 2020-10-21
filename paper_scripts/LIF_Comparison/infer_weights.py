import numpy as np
import matplotlib.pyplot as plt
import re
import sys

# This import statement assumes you execute this python script from within this folder
sys.path.insert(0, '../..')
from weight_inference import fitter

# Network Settings
num_input_neurons = 100
num_output_neurons = 10
timestep = 0.25
simulation_time = 1000 * 1e3  # X * 1000ms

seeds = [1]  # np.arange(1,11)
ratio_active = 0.2

print("Ratio Active: " + str(ratio_active))
for seed in seeds:
    print("Seed: " + str(seed))

    # First we must load the data pertaining to the network activity
    path = "./" + str(num_input_neurons) + "Inputs_" + str(num_output_neurons) + "Outputs_" + str(
        ratio_active) + "Perc_" + str(seed) + "Seed/"

    original_weight_matrix = np.fromfile(path + "IO_weight_matrix.npy").reshape(num_output_neurons, num_input_neurons)
    input_neuron_acc = np.fromfile(path + "input_neuron_acc.npy").reshape(num_input_neurons, -1)
    input_neuron_mem = np.fromfile(path + "input_neuron_mem.npy").reshape(num_input_neurons, -1)
    output_neuron_xpsps = np.fromfile(path + "output_neuron_xpsps.npy").reshape(num_output_neurons, -1)
    print("---- Weights and Membrane Voltages Loaded----")

    input_neuron_spiketimes = []
    for i_indx in range(num_input_neurons):
        train = timestep*np.fromfile(path + str(i_indx) + "_input_neuron_spiketimes.npy")
        input_neuron_spiketimes.append(train)

    output_neuron_spiketimes = []
    for o_indx in range(num_output_neurons):
        train = timestep*np.fromfile(path + str(o_indx) + "_output_neuron_spiketimes.npy")
        output_neuron_spiketimes.append(train)
    print("---- Spiking Data Loaded ----")

    # Initialising a random "guess" matrix which all solvers will use as a prior
    r = np.random.RandomState(seed=42)
    initial_guess_matrix = 0.001 * (r.uniform(size=(num_output_neurons, num_input_neurons)) - 0.5)

    # Setting up parameters for learning
    learning_rate = 5e-4
    check_interval = 10

    stimulus_length = 100.0
    nb_timesteps_per_stimulus = int(stimulus_length / timestep)
    num_stimuli = int(simulation_time / stimulus_length)

    # Fitting weights with the Akrout method
    batch_sizes = [] #[1000]  # [10,100,1000]
    for batch_size in batch_sizes:
        akrout_guess_dumps = fitter.akrout(
            initial_guess_matrix,
            input_neuron_spiketimes,
            output_neuron_spiketimes,
            simulation_time,
            stimulus_length,
            batch_size,
            learning_rate,
            check_interval)
        akrout_guess_dumps = np.asarray(akrout_guess_dumps)
        akrout_guess_dumps.tofile(path + "akrout_dump_" + str(batch_size) + "batch.npy")
        print("---- Akrout Method Complete, Batch Size: " + str(batch_size) + " ----")

    # Fitting weights with the STDWI method
    a_fast = 1.0
    taus_fast = [40.0]  # [5.0,10.0,20.0,40.0,60.0,80.0,100.0,120.0,140.0,160.0,180.0,200.0,500.0]
    taus_slow = [200.0]  # [5.0,10.0,20.0,40.0,60.0,80.0,100.0,120.0,140.0,160.0,180.0,200.0,500.0]
    for f_indx, t_fast in enumerate(taus_fast):
        for t_slow in taus_slow:
            a_slow = a_fast * (t_fast / t_slow)
            stdwi_guess_dumps = fitter.stdwi(
                initial_guess_matrix,
                input_neuron_spiketimes,
                output_neuron_spiketimes,
                simulation_time,
                stimulus_length,
                timestep,
                a_slow, t_slow,
                a_fast, t_fast,
                learning_rate, check_interval, decay_factor=0.1)
            stdwi_guess_dumps = np.asarray(stdwi_guess_dumps)
            stdwi_guess_dumps.tofile(path + "stdwi_dump_" + str(t_fast) + "fast_" + str(t_slow) + "slow.npy")
            print("---- STDWI Method Complete, Fast Tau: " + str(t_fast) + ", Slow Tau: " + str(t_slow) + " ----")

    # Fitting weights with the RDD method
    alphas = [] #[0.025]  # [0.005,0.01,0.025,0.05,0.1,0.15,0.20,0.25,0.30,0.35] # The boundary about threshold
    windows = [] #[35]  # [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] #ms
    for alpha in alphas:
        for window in windows:
            window_size = np.round(window / timestep).astype(int)  # The window about events (30ms)
            threshold = 1.0
            rdd_guess_dumps = fitter.rdd(
                initial_guess_matrix,
                input_neuron_mem,
                input_neuron_acc,
                output_neuron_xpsps,
                alpha, window_size,
                threshold,
                timestep,
                learning_rate,
                check_interval)
            rdd_guess_dumps = np.asarray(rdd_guess_dumps)
            rdd_guess_dumps.tofile(path + "rdd_dump_" + str(alpha) + "bound_" + str(window) + "window.npy")
            print("---- RDD Method Complete, alpha: " + str(alpha) + ", window: " + str(window) + " ----")
