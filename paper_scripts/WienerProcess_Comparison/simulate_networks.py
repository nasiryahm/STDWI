import numpy as np
import matplotlib.pyplot as plt
import os, sys
from shutil import copyfile

sys.path.insert(0,'../..')
from weight_inference import simulator

# Simulation Settings
timestep = 0.25     # ms
drift = 10.0
diffusion = 0.5
threshold = 1.0
reset = 0.0
simulation_length = 250*1e3
nb_timesteps = int(simulation_length / timestep)

for seed in np.arange(1,11):
    print("---- Beginning with Seed: " + str(seed) + " ----")
    # Trials are a synonym for the different weights of pre-synaptic 
    nb_trials = 501
    weights = 0.1*((np.arange(nb_trials) - ((nb_trials-1)/2)) / ((nb_trials-1)/2))
    pre_firing_rate = 10 / 1000 # 10Hz

    # Construct the weiner process data for the trials and timesteps
    membrane_voltages = simulator.wiener_process(nb_trials, nb_timesteps, timestep, drift*1e-3, diffusion*1e-3, seed=seed)

    # Constructing a poisson spike train for inputs to all of these trials
    input_spike_trains = simulator.poisson_spike_train(nb_trials, pre_firing_rate, simulation_length, timestep, seed=seed)
    cumulative_input_spike_train = np.cumsum(np.concatenate([np.zeros((nb_trials,1)), simulator.binary_spike_matrix(input_spike_trains, simulation_length, timestep)[:,:-1]], axis=1), axis=1)

    # Adding the value of the weight to the weiner processes at each presynaptic spike
    membrane_voltages += weights[:,np.newaxis]*cumulative_input_spike_train
    print("---- Input Spikes, Wiener Processes, and Weight Inputs Computed ----")

    # Computing the spike times of the output neurons
    output_spike_trains = [[] for t in range(nb_trials)]
    for trial in range(nb_trials):
        while True:
            loc = np.where(membrane_voltages[trial,:] >= threshold)
            if (loc[0].shape[0] == 0):
                break
            output_spike_trains[trial].append(loc[0][0]*timestep)
            membrane_voltages[trial,loc[0][0]:] -= (threshold - reset)
    for trial in range(nb_trials):
        output_spike_trains[trial] = np.asarray(output_spike_trains[trial])
    print("---- Output Spikes Computed ----")

    # Storing input and output spike times, membrane voltage, and weights
    path = "./" + str(nb_trials) + "Trials_" + str(drift) + "Drift_" + str(diffusion) + "Diffusion_" + str(seed) + "Seed/"  
    os.makedirs(path, exist_ok=True)
    weights.tofile(path + "IO_weight_matrix.npy")

    for trial in range(nb_trials):
        input_spike_trains[trial].tofile(path + str(trial) + "_input_spike_trains.npy")

    membrane_voltages.tofile(path + "output_neuron_mem.npy")
    for trial in range(nb_trials):
        output_spike_trains[trial].tofile(path + str(trial) + "_output_spike_trains.npy")
    print("---- Weights and Spike Times Dumped----")

    ## Create a plot of some of the trajectories and save (just for future)
    #plt.plot(membrane_voltages[:10,:int(500/timestep)].transpose())
    #plt.xlabel("Time (0.1 ms)")
    #plt.ylabel("Normalized Voltage")
    #plt.savefig(path + 'ExampleTrajectories.png', bbox_inches='tight')
    #plt.clf()
    #print("---- Figures and Files Dumped ----")
