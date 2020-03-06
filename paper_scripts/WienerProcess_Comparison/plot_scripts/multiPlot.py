import numpy as np
import matplotlib.pyplot as plt
import re
import sys, os
import seaborn as sns

sys.path.insert(0,'..')
import inferenceMethods
import modelFitter

# Simulation Details 
nb_trials = 501
weights = 0.1*((np.arange(nb_trials) - ((nb_trials-1)/2)) / ((nb_trials-1)/2))
drift = 10.0
diffusion = 0.5
seeds = np.arange(1,11)

# Network Settings
timestep = 0.5
threshold = 1.0
check_interval = 1

# STDWI Data Loading
t_fast = 20.0
t_slow = 200.0

bayes_acc_set, bayes_r_set = [], []
stdwi_acc_set, stdwi_r_set = [], []
    
outpath = "./combinedPlots/"
os.makedirs(outpath, exist_ok=True)

for seed in seeds:
    # Path and folder creation
    path = "./" + str(nb_trials) + "Trials_" + str(drift) + "Drift_" + str(diffusion) + "Diffusion_" + str(seed) + "Seed/" 


    original_weight_matrix = np.fromfile(path + "IO_weight_matrix.npy").reshape(-1,1)
    original_weight_matrix = original_weight_matrix*np.eye(len(original_weight_matrix))
    output_neuron_mem = np.fromfile(path + "output_neuron_mem.npy").reshape(nb_trials, -1)


    stdwi_filepath = path + "stdwi_dump_" + str(t_fast) + "fast_" + str(t_slow) + "slow.npy"
    stdwi_weight_estimates = np.fromfile(stdwi_filepath).reshape(-1, nb_trials, nb_trials)
    num_epochs = stdwi_weight_estimates.shape[0]

    stdwi_acc = []
    stdwi_r = []
    for e in range(num_epochs):
        stdwi_acc.append(inferenceMethods.sign_alignment(np.diagonal(original_weight_matrix), np.diagonal(stdwi_weight_estimates[e])))
        stdwi_r.append(np.corrcoef(np.diagonal(original_weight_matrix).flatten(), np.diagonal(stdwi_weight_estimates[e]).flatten())[0,1])
    stdwi_acc_set.append(stdwi_acc)
    stdwi_r_set.append(stdwi_r)


    # Bayes Data Loading
    bayes_filepath = path + "bayes_dump_" + str(drift) + "drift_" + str(diffusion) + "diff.npy"
    bayes_weight_estimates = np.fromfile(bayes_filepath).reshape(-1, nb_trials, nb_trials)
    num_epochs = stdwi_weight_estimates.shape[0]

    bayes_acc = []
    bayes_r = []
    for e in range(num_epochs):
        bayes_acc.append(inferenceMethods.sign_alignment(np.diagonal(original_weight_matrix), np.diagonal(bayes_weight_estimates[e])))
        bayes_r.append(np.corrcoef(np.diagonal(original_weight_matrix).flatten(), np.diagonal(bayes_weight_estimates[e]).flatten())[0,1])
    bayes_acc_set.append(bayes_acc)
    bayes_r_set.append(bayes_r)


stdwi_acc_set = np.asarray(stdwi_acc_set)
stdwi_r_set = np.asarray(stdwi_r_set)
bayes_acc_set = np.asarray(bayes_acc_set)
bayes_r_set = np.asarray(bayes_r_set)

mean_stdwi_acc = np.mean(stdwi_acc_set, axis=0)
std_stdwi_acc = np.std(stdwi_acc_set, axis=0)
mean_stdwi_r = np.mean(stdwi_r_set, axis=0)
std_stdwi_r = np.std(stdwi_r_set, axis=0)

mean_bayes_acc = np.mean(bayes_acc_set, axis=0)
std_bayes_acc = np.std(bayes_acc_set, axis=0)
mean_bayes_r = np.mean(bayes_r_set, axis=0)
std_bayes_r = np.std(bayes_r_set, axis=0)

plt.figure(figsize=(4,3), dpi=200)
ax = plt.gca()
# Bayes with std
plt.plot(mean_bayes_acc, color='blue', label="Bayes")
ax.fill_between(range(len(mean_bayes_acc)), mean_bayes_acc - std_bayes_acc, mean_bayes_acc + std_bayes_acc, color='blue', alpha=0.25)
# STDWI and std
plt.plot(mean_stdwi_acc, color='black', label="STDWI")
ax.fill_between(range(len(mean_stdwi_acc)), mean_stdwi_acc - std_stdwi_acc, mean_stdwi_acc + std_stdwi_acc, color='black', alpha=0.25)
plt.legend(frameon=False)
plt.ylim([0.4,1.0])
plt.xticks(np.arange(num_epochs,step=100), np.arange(num_epochs, step=100)*0.1)
plt.ylabel("Sign Accuracy")
plt.xlabel("Time (s)")
sns.despine()
plt.savefig(outpath + 'SignAccuracy.png', bbox_inches='tight')
plt.clf()
# plt.show()


plt.figure(figsize=(4,3), dpi=200)
ax = plt.gca()
plt.plot(mean_bayes_r, color='blue', label="Bayes")
ax.fill_between(range(len(mean_bayes_r)), mean_bayes_r - std_bayes_r, mean_bayes_r + std_bayes_r, color='blue', alpha=0.25)
plt.plot(mean_stdwi_r, color='black', label="STDWI")
ax.fill_between(range(len(mean_stdwi_r)), mean_stdwi_r - std_stdwi_r, mean_stdwi_r + std_stdwi_r, color='black', alpha=0.25)
plt.legend(frameon=False)
plt.ylim([0.0,1.0])
plt.xticks(np.arange(num_epochs,step=100), np.arange(num_epochs, step=100)*0.1)
plt.ylabel("Pearson Correlation -- r")
plt.xlabel("Time (s)")
sns.despine()
plt.savefig(outpath + 'PearsonCorrelation.png', bbox_inches='tight')
plt.clf()
# plt.show()

plt.figure(figsize=(4,3), dpi=200)
#plt.subplot(1,2,1)
plt.scatter(np.diagonal(original_weight_matrix), np.diagonal(bayes_weight_estimates[-1]), color='blue', label="Bayes", alpha=0.25)
plt.ylabel("Inferred Jump Width")
plt.xlabel("True Jump Width")
rangevals = 1.05*np.max(np.abs(bayes_weight_estimates[-1]))
plt.ylim([-rangevals, rangevals])
sns.despine()
plt.savefig(outpath + 'BayesScatterPlot.png', bbox_inches='tight')
plt.clf()

plt.figure(figsize=(4,3), dpi=200)
#plt.subplot(1,2,2)
plt.scatter(np.diagonal(original_weight_matrix), np.diagonal(stdwi_weight_estimates[-1]), color='black', label="STDWI", alpha=0.25)
plt.ylabel("Inferred Jump Width")
plt.xlabel("True Jump Width")
rangevals = 1.05*np.max(np.abs(stdwi_weight_estimates[-1]))
plt.ylim([-rangevals, rangevals])
sns.despine()
# plt.show()
plt.savefig(outpath + 'STDWIScatterPlot.png', bbox_inches='tight')
plt.clf()

