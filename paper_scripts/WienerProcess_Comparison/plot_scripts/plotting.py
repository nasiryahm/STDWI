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

# Path and folder creation
path = "./" + str(nb_trials) + "Trials_" + str(drift) + "Drift_" + str(diffusion) + "Diffusion/" 
outpath = path + "_plots/"
os.makedirs(outpath, exist_ok=True)
check_interval = 1

# Network Settings
timestep = 0.25
threshold = 1.0

original_weight_matrix = np.fromfile(path + "IO_weight_matrix.npy").reshape(-1,1)
original_weight_matrix = original_weight_matrix*np.eye(len(original_weight_matrix))
output_neuron_mem = np.fromfile(path + "output_neuron_mem.npy").reshape(nb_trials, -1)

# STDWI Data Loading
t_fast = 5.0
t_slow = 500.0

stdwi_filepath = path + "stdwi_dump_" + str(t_fast) + "fast_" + str(t_slow) + "slow.npy"
stdwi_weight_estimates = np.fromfile(stdwi_filepath).reshape(-1, nb_trials, nb_trials)
num_epochs = stdwi_weight_estimates.shape[0]

stdwi_acc = []
stdwi_r = []
for e in range(num_epochs):
    stdwi_acc.append(inferenceMethods.sign_alignment(np.diagonal(original_weight_matrix), np.diagonal(stdwi_weight_estimates[e])))
    stdwi_r.append(np.corrcoef(np.diagonal(original_weight_matrix).flatten(), np.diagonal(stdwi_weight_estimates[e]).flatten())[0,1])


# Bayes Data Loading
bayes_filepath = path + "bayes_dump_" + str(drift) + "drift_" + str(diffusion) + "diff.npy"
bayes_weight_estimates = np.fromfile(bayes_filepath).reshape(-1, nb_trials, nb_trials)
num_epochs = stdwi_weight_estimates.shape[0]

bayes_acc = []
bayes_r = []
for e in range(num_epochs):
    bayes_acc.append(inferenceMethods.sign_alignment(np.diagonal(original_weight_matrix), np.diagonal(bayes_weight_estimates[e])))
    bayes_r.append(np.corrcoef(np.diagonal(original_weight_matrix).flatten(), np.diagonal(bayes_weight_estimates[e]).flatten())[0,1])

plt.figure(figsize=(4,3), dpi=100)
plt.plot(bayes_acc, color='blue', label="Bayes")
plt.plot(stdwi_acc, color='black', label="STDWI")
plt.legend(frameon=False)
plt.ylim([0.4,1.0])
plt.xticks(np.arange(num_epochs,step=100), np.arange(num_epochs, step=100)*0.1)
plt.ylabel("Sign Accuracy")
plt.xlabel("Time (s)")
sns.despine()
plt.savefig(outpath + 'SignAccuracy.png', bbox_inches='tight')
plt.clf()
# plt.show()


plt.figure(figsize=(4,3), dpi=100)
plt.plot(bayes_r, color='blue', label="Bayes")
plt.plot(stdwi_r, color='black', label="STDWI")
plt.legend(frameon=False)
plt.ylim([0.0,1.0])
plt.xticks(np.arange(num_epochs,step=100), np.arange(num_epochs, step=100)*0.1)
plt.ylabel("Pearson Correlation -- r")
plt.xlabel("Time (s)")
sns.despine()
plt.savefig(outpath + 'PearsonCorrelation.png', bbox_inches='tight')
plt.clf()
# plt.show()

plt.figure(figsize=(8,3), dpi=100)
plt.subplot(1,2,1)
plt.scatter(np.diagonal(original_weight_matrix), np.diagonal(bayes_weight_estimates[-1]), color='blue', label="Bayes", alpha=0.25)
plt.ylabel("Inferred Jump Width")
plt.xlabel("True Jump Width")
rangevals = 1.05*np.max(np.abs(bayes_weight_estimates[-1]))
plt.ylim([-rangevals, rangevals])
plt.subplot(1,2,2)
plt.scatter(np.diagonal(original_weight_matrix), np.diagonal(stdwi_weight_estimates[-1]), color='black', label="STDWI", alpha=0.25)
# plt.ylabel("Inferred Jump Width")
plt.xlabel("True Jump Width")
rangevals = 1.05*np.max(np.abs(stdwi_weight_estimates[-1]))
plt.ylim([-rangevals, rangevals])
sns.despine()
# plt.show()
plt.savefig(outpath + 'ScatterPlot.png', bbox_inches='tight')
plt.clf()


exit(0)
# Pearson Correlation Plots
plt.imshow(stdwi_r_map.T, cmap="gray", vmin=0.5, vmax=1.0)
plt.xticks(np.arange(len(stdwi_taus)), stdwi_taus, rotation='vertical')
plt.yticks(np.arange(len(stdwi_taus)), stdwi_taus)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Pearson Correlation, r', rotation=270)
plt.xlabel("Fast Trace Tau (ms)")
plt.ylabel("Slow Trace Tau (ms)")
plt.savefig(outpath + 'rSTDWIParameterMap.png', bbox_inches='tight')
plt.clf()

plt.imshow(bayes_r_map.T, cmap="gray", vmin=0.5, vmax=1.0)
plt.xticks(np.arange(len(bayes_drifts)), bayes_drifts, rotation='vertical')
plt.yticks(np.arange(len(bayes_Ds)), bayes_Ds)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Pearson Correlation, r', rotation=270)
plt.xlabel("Drift")
plt.ylabel("Diffusion")
plt.savefig(outpath + 'rBayesParameterMap.png', bbox_inches='tight')
plt.clf()



# Sign Alignment Plots
plt.imshow(stdwi_map.T, cmap="gray", vmin=0.5, vmax=1.0)
plt.xticks(np.arange(len(stdwi_taus)), stdwi_taus, rotation='vertical')
plt.yticks(np.arange(len(stdwi_taus)), stdwi_taus)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Sign Alignment', rotation=270)
plt.xlabel("Fast Trace Tau (ms)")
plt.ylabel("Slow Trace Tau (ms)")
plt.savefig(outpath + 'STDWIParameterMap.png', bbox_inches='tight')
plt.clf()

plt.imshow(bayes_map.T, cmap="gray", vmin=0.5, vmax=1.0)
plt.xticks(np.arange(len(bayes_drifts)), bayes_drifts, rotation='vertical')
plt.yticks(np.arange(len(bayes_Ds)), bayes_Ds)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Sign Alignment', rotation=270)
plt.xlabel("Drift")
plt.ylabel("Diffusion")
plt.savefig(outpath + 'BayesParameterMap.png', bbox_inches='tight')
plt.clf()


print("STDWI, Bayes")
print(stdwi_maxval, bayes_maxval)

print("STDWI Best Params: Fast=" + str(stdwi_taus[stdwi_argmax[0]]) + " Slow=" + str(stdwi_taus[stdwi_argmax[1]]))
print("Bayes Best Params: Drift=" + str(bayes_drifts[bayes_argmax[0]]) + " Diffusion=" + str(bayes_Ds[bayes_argmax[1]]))

# Put best parameters in a file


