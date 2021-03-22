import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import json

# This import statement assumes you execute this python script from within this folder
sys.path.insert(0, '../..')
from weight_inference import fitter

# Network Settings
num_input_neurons = 100
num_output_neurons = 10
timestep = 0.25
simulation_time = 2500 * 1e3  # X * 1000ms

seeds = np.arange(2,11)
correlations = [0.0] #[0.9,1.0] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ratio_active = 0.2

print("Ratio Active: " + str(ratio_active))
for seed in seeds:
    print("Seed: " + str(seed))
    for correlation in correlations:
        print("Correlation: " + str(correlation))
        parampath = "./plot_scripts/_plots/" + str(ratio_active) + "Perc_" + str(correlation) + "Corr/"
        with open(parampath + 'optimal_parameters.json', 'r') as fp:
            parameters = json.load(fp)

        # First we must load the data pertaining to the network activity
        path = "./" + str(num_input_neurons) + "Inputs_" + str(num_output_neurons) + "Outputs_" + str(
            ratio_active) + "Perc_" + str(correlation) + "Corr_" + str(seed) + "Seed/"

        original_weight_matrix = np.fromfile(path + "IO_weight_matrix.npy").reshape(num_output_neurons, num_input_neurons)
        input_neuron_acc = np.fromfile(path + "input_neuron_acc.npy").reshape(num_input_neurons, -1)
        input_neuron_mem = np.fromfile(path + "input_neuron_mem.npy").reshape(num_input_neurons, -1)
        output_neuron_xpsps = np.fromfile(path + "output_neuron_xpsps.npy").reshape(num_output_neurons, -1)
        print("---- Weights and Membrane Voltages Loaded----")

        input_neuron_spiketimes = []
        for i_indx in range(num_input_neurons):
            train = np.fromfile(path + str(i_indx) + "_input_neuron_spiketimes.npy")
            input_neuron_spiketimes.append(train)

        output_neuron_spiketimes = []
        for o_indx in range(num_output_neurons):
            train = np.fromfile(path + str(o_indx) + "_output_neuron_spiketimes.npy")
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
        akrout_guess_dumps = fitter.akrout(
            initial_guess_matrix,
            input_neuron_spiketimes,
            output_neuron_spiketimes,
            simulation_time,
            stimulus_length,
            parameters['Akrout']['batch_size'],
            learning_rate,
            check_interval,
            decay_factor=parameters['Akrout']['batch_size'])
        akrout_guess_dumps = np.asarray(akrout_guess_dumps)
        akrout_guess_dumps.tofile(path + "akrout_dump_" + str(parameterrs['Akrout']['batch_size']) + "batch_" + str(parameterrs['Akrout']['decay_value']) + "decay.npy")
        print("---- Akrout Method Complete, Batch Size: " + str(parameterrs['Akrout']['batch_size']) + ", Decay Factor:" + str(parameterrs['Akrout']['decay_value']) + " ----")

        # Fitting weights with the STDWI method
        a_fast = 1.0
        a_slow = a_fast * (parameters['STDWI']['tau_fast'] / parameters['STDWI']['tau_slow'])
        stdwi_guess_dumps = fitter.stdwi(
            initial_guess_matrix,
            input_neuron_spiketimes,
            output_neuron_spiketimes,
            simulation_time,
            stimulus_length,
            timestep,
            a_slow, parameters['STDWI']['tau_slow'],
            a_fast, parameters['STDWI']['tau_fast'],
            learning_rate, check_interval, decay_factor=0.1)
        stdwi_guess_dumps = np.asarray(stdwi_guess_dumps)
        stdwi_guess_dumps.tofile(path + "stdwi_dump_" + str(parameters['STDWI']['tau_fast']) + "fast_" + str(parameters['STDWI']['tau_slow']) + "slow.npy")
        print("---- STDWI Method Complete, Fast Tau: " + str(parameters['STDWI']['tau_fast']) + ", Slow Tau: " + str(parameters['STDWI']['tau_slow']) + " ----")

        # Fitting weights with the RDD method
        window_size = np.round(parameters['RDD']['window'] / timestep).astype(int)  # The window about events (30ms)
        threshold = 1.0
        rdd_guess_dumps = fitter.rdd(
            initial_guess_matrix,
            input_neuron_mem,
            input_neuron_acc,
            output_neuron_xpsps,
            parameters['RDD']['bound'], window_size,
            threshold,
            timestep,
            learning_rate,
            check_interval)
        rdd_guess_dumps = np.asarray(rdd_guess_dumps)
        rdd_guess_dumps.tofile(path + "rdd_dump_" + str(parameters['RDD']['bound']) + "bound_" + str(parameters['RDD']['window']) + "window.npy")
        print("---- RDD Method Complete, alpha: " + str(parameters['RDD']['bound']) + ", window: " + str(parameters['RDD']['window']) + " ----")
