import numpy as np
import matplotlib.pyplot as plt
import re
import sys, os

sys.path.insert(0,'../..')
from weight_inference import methods
from weight_inference import fitter

# First we must load the data pertaining to the network activity
nb_trials = 501
weights = 0.1*((np.arange(nb_trials) - ((nb_trials-1)/2)) / ((nb_trials-1)/2))
drift = 10.0
diffusion = 0.5

seeds = np.arange(1,11)
for seed in seeds:
    path = "./" + str(nb_trials) + "Trials_" + str(drift) + "Drift_" + str(diffusion) + "Diffusion_" + str(seed) + "Seed/" 
    outpath = "./_plots/"
    os.makedirs(outpath, exist_ok=True)
    check_interval = 1

    # Network Settings
    timestep = 0.25
    threshold = 1.0

    original_weight_matrix = np.fromfile(path + "IO_weight_matrix.npy").reshape(-1,1)
    original_weight_matrix = original_weight_matrix*np.eye(len(original_weight_matrix))
    output_neuron_mem = np.fromfile(path + "output_neuron_mem.npy").reshape(nb_trials, -1)

    input_spike_trains = []
    for i_indx in range(nb_trials):
        train = np.fromfile(path + str(i_indx) + "_input_spike_trains.npy")
        input_spike_trains.append(train)

    output_spike_trains = []
    for o_indx in range(nb_trials):
        train = np.fromfile(path + str(o_indx) + "_output_spike_trains.npy")
        output_spike_trains.append(train)
    print("---- Spiking Network Data Loaded, Seed: " + (str(seed)) + " ----")

    # Initialising a random "guess" matrix which all solvers will use as a prior
    r = np.random.RandomState(seed=42)
    initial_guess_matrix = 0.001*(r.uniform(size=original_weight_matrix.shape) - 0.5)

    # Setting up parameters for learning
    learning_rate = 0.001
    simulation_time = output_neuron_mem.shape[1]*timestep
    check_interval = 1

    '''
    # Fitting weights with the STDWI method
    a_fast = 1.0
    t_fast = 20.0
    t_slow = 200.0
    ratio = t_fast/t_slow
    a_slow = a_fast*ratio
    stdwi_guess_dumps = fitter.stdwi(
        initial_guess_matrix,
        input_spike_trains,
        output_spike_trains,
        simulation_time,
        100.0, # stimulation length
        timestep,
        a_slow, t_slow,
        a_fast, t_fast,
        learning_rate, check_interval, decay_factor=0.1, offsetanalysis=10, alltoall=False)
    stdwi_acc = []
    stdwi_r = []
    for stdwi_guess in stdwi_guess_dumps:
        stdwi_acc.append(methods.sign_alignment(np.diagonal(stdwi_guess), np.diagonal(original_weight_matrix)))
        stdwi_r.append(np.corrcoef(np.diagonal(original_weight_matrix).flatten(), np.diagonal(stdwi_guess).flatten())[0,1])
    print(stdwi_acc[-1], stdwi_r[-1])
    stdwi_guess_dumps = np.asarray(stdwi_guess_dumps)
    stdwi_guess_dumps.tofile(path + "stdwi_dump_" + str(t_fast) + "fast_" + str(t_slow) + "slow.npy")
    print("---- STDWI Method Complete, Fast Tau: " + str(t_fast) + ", Slow Tau: " + str(t_slow) + " ----")
    '''

    # Fitting weights with a bayesian update rule based upon hitting-time
    bayes_guess_dumps, bayes_var_dumps  = fitter.bayes(
        initial_guess_matrix,
        input_spike_trains,
        output_spike_trains,
        output_neuron_mem,
        simulation_time,
        100.0, # Stimulation time
        timestep,
        check_interval,
        drift*1e-3, diffusion*1e-3, variance_bound=0.00, offsetanalysis=10)
    bayes_acc = []
    bayes_r = []
    for bayes_guess in bayes_guess_dumps:
        bayes_acc.append(methods.sign_alignment(np.diagonal(bayes_guess), np.diagonal(original_weight_matrix)))
        bayes_r.append(np.corrcoef(np.diagonal(original_weight_matrix).flatten(), np.diagonal(bayes_guess).flatten())[0,1])
    print(bayes_acc[-1], bayes_r[-1])
    bayes_guess_dumps = np.asarray(bayes_guess_dumps)
    bayes_guess_dumps.tofile(path + "bayes_dump_" + str(drift) + "drift_" + str(diffusion) + "diff.npy")
    print("---- Bayes Method Complete, Drift: " + str(drift) + ", Diffusion D: " + str(diffusion) + " ----") #+ ", Variance Bound: " + str(bound) + " ----")
    # plt.scatter(np.diagonal(original_weight_matrix).flatten(),np.diagonal(bayes_guess_dumps[-1]).flatten())
    # plt.show()

plt.scatter(np.diagonal(original_weight_matrix).flatten(), np.diagonal(stdwi_guess_dumps[-1]).flatten(), color='red', label='STDWI',s=10,alpha=0.5)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.scatter(np.diagonal(original_weight_matrix).flatten(), np.diagonal(bayes_guess_dumps[-1]).flatten(), color='blue', label='Bayes',s=10,alpha=0.5)
plt.xlabel("True Weight")
plt.ylabel("Estimated Weight")
plt.legend
plt.savefig(outpath + 'WeightScatter.png', bbox_inches='tight')
plt.clf()

plt.plot(bayes_acc, label='Bayes', color='blue')
plt.plot(stdwi_acc, label='STDWI', color='red')
plt.xlabel("Trials")
plt.ylabel("Sign Accuracy")
plt.legend()
plt.savefig(outpath + 'SignAccuracy.png', bbox_inches='tight')
plt.clf()

plt.plot(bayes_r, label='Bayes', color='blue')
plt.plot(stdwi_r, label='STDWI', color='red')
plt.xlabel("Trials")
plt.ylabel("Pearson Correlation -- r")
plt.legend()
plt.savefig(outpath + 'PearsonCorr.png', bbox_inches='tight')
plt.clf()

exit(0)
