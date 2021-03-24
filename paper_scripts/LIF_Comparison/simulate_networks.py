import numpy as np
import os
from shutil import copyfile
import time

import sys
sys.path.append("../..")
from weight_inference import simulator

# Network Settings
num_input_neurons = 100
num_output_neurons = 10

# Simulation Settings
timestep = 0.25     # ms  
sim_time = 4000*1e3  # X * 1000ms
nb_timesteps = int(sim_time / timestep)

# Stimulation Protocol
stimulation_FR = 200 / 1000 # spikes / ms
stimulus_length = 100 #ms
num_stimuli = int(sim_time / stimulus_length)
ratio_input_neurons_stimulated = 0.2
seeds = np.arange(1,11)
correlations = [0.0] #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print("---- Ratio Stimulated: " + str(ratio_input_neurons_stimulated) + " ----")
for seed in seeds:
    for correlation in correlations:
        print("---- Beginning Simulation with Seed " + str(seed) + ", and Stimulation Correlation " + str(correlation) + " ----")
        path = "./" + str(num_input_neurons) + "Inputs_" + str(num_output_neurons) + "Outputs_" + str(ratio_input_neurons_stimulated) + "Perc_" + str(correlation) + "Corr_" + str(seed) + "Seed/"
        os.makedirs(path, exist_ok=True)

        start_time = time.time()
        # Producing Stimulation for the whole length of time
        stimulation_spike_trains = simulator.correlated_poisson_spike_train(
            num_input_neurons,
            stimulation_FR,
            correlation,
            sim_time,
            timestep,
            seed=seed)
        print("---- Stimulation Spike Trains Simulated ---- Time: " + str(time.time() - start_time))
        start_time = time.time()

        # In reality, we want only some perc of neurons firing at a time
        simulator.random_sample_spike_train(stimulation_spike_trains,
                                            sim_time, timestep,
                                            stimulus_length, ratio_input_neurons_stimulated)
        print("---- Stimuli Separated in Spike Times ---- Time: ", time.time() - start_time)
        start_time = time.time()


        # Converting stimulation spike trains to Post-Synaptic Potentials
        stimulation_xpsps = simulator.spike_trains_to_xpsps(
            stimulation_spike_trains,
            sim_time,
            timestep)
        print("---- Stimulation XPSPs Produced ---- Time: " + str(time.time() - start_time))
        start_time = time.time()

        # Creating weight matrices from stimulation to input and input to output
        scale = 12.0
        stim_weights = scale*np.eye(num_input_neurons, num_input_neurons)
        # stim_weights *= scale * 0.2 / ratio_input_neurons_stimulated

        scale *= 7.5 # Compensating for hidden layer mean weight, firing rates and number)
        r = np.random.RandomState(seed=seed)
        io_weights = scale*((0.5/np.sqrt(num_input_neurons*ratio_input_neurons_stimulated))*r.normal(size=(num_output_neurons, num_input_neurons)) + (1 / (num_input_neurons*ratio_input_neurons_stimulated)))

        # Using input stimulation XPSPs and weight matrices to simulate network dynamics
        input_neuron_acc, input_neuron_mem, input_neuron_spiketimes = simulator.lif_dynamics(
            stimulation_xpsps, stim_weights, timestep)
        input_neuron_acc.tofile(path + "input_neuron_acc.npy")
        input_neuron_mem.tofile(path + "input_neuron_mem.npy")
        input_neuron_xpsps = simulator.spike_trains_to_xpsps(input_neuron_spiketimes, sim_time, timestep)
        for i_indx in range(num_input_neurons):
            input_neuron_spiketimes[i_indx].tofile(path + str(i_indx) + "_input_neuron_spiketimes.npy")
        print("---- Input Network Dynamics Simulated ---- Time: " + str(time.time() - start_time))
        start_time = time.time()

        output_neuron_acc, output_neuron_mem, output_neuron_spiketimes = simulator.lif_dynamics(
            input_neuron_xpsps, io_weights, timestep)
        output_neuron_xpsps = simulator.spike_trains_to_xpsps(
            output_neuron_spiketimes, sim_time, timestep)
        print("---- Network Dynamics Simulated ---- Time: " + str(time.time() - start_time))
        start_time = time.time()

        input_firing_rates = simulator.firing_rates(input_neuron_spiketimes, sim_time)
        output_firing_rates = simulator.firing_rates(output_neuron_spiketimes, sim_time)
        print("     Input Neurons Firing Rate: " + str(np.mean(input_firing_rates)))
        print("     Output Neurons Firing Rate: " + str(np.mean(output_firing_rates)))

        """
        plt.hist(input_firing_rates)
        plt.hist(output_firing_rates)
        plt.show()
        exit()
        """

        # Dumping data to file for loading
        io_weights.tofile(path + "IO_weight_matrix.npy")


        output_neuron_acc.tofile(path + "output_neuron_acc.npy")
        output_neuron_mem.tofile(path + "output_neuron_mem.npy")
        output_neuron_xpsps.tofile(path + "output_neuron_xpsps.npy")
        for o_indx in range(num_output_neurons):
            output_neuron_spiketimes[o_indx].tofile(path + str(o_indx) + "_output_neuron_spiketimes.npy")

        # Copying this file to directory also
        thisfile = os.path.realpath(__file__)
        copyfile(thisfile, path + os.path.basename(__file__))
