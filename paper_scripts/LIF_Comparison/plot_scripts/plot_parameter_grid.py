import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
import seaborn as sns

sys.path.insert(0,'../../..')
from weight_inference import methods

# Network Settings
num_input_neurons = 100
num_output_neurons = 10
timestep = 0.25
simulation_time = 1000*1e3  # X * 1000ms
ratio_active = 0.2
seed = 1
correlation = 0.0

# First we must load the data pertaining to the network activity
path = "../" + str(num_input_neurons) + "Inputs_" + str(num_output_neurons) + "Outputs_" + str(ratio_active) + "Perc_" + str(correlation) + "Corr_" + str(seed) + "Seed/" 

outpath = "./_plots/" + str(ratio_active) + "Perc_" str(correlation) + "Corr/"
os.makedirs(outpath, exist_ok=True)

# Loading weight matrices (from original plus dumped estimation methods)
original_weight_matrix = np.fromfile(path + "IO_weight_matrix.npy").reshape(num_output_neurons, num_input_neurons)

# Now outlining the set of parameters which we wish to investigate
akrout_batch_sizes = [10,100,1000]
akrout_decay_values = [0.01,0.05,0.1,0.2,0.3,0.4,0.5, 0.6, 0.7]
stdwi_taus = [5.0,10.0,20.0,40.0,60.0,80.0,100.0,120.0,140.0,160.0,180.0,200.0,500.0]
rdd_alphas = [0.005,0.01,0.025,0.05,0.1,0.15,0.20,0.25,0.30,0.35] # The boundary about threshold
rdd_windows = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] #ms

# Constructing maps for pearson correlation and accuracies
akrout_r_map = np.zeros((len(akrout_batch_sizes), len(akrout_decay_values)))
stdwi_r_map = np.zeros((len(stdwi_taus), len(stdwi_taus)))
stdwi_r_map[:,:] = None
rdd_r_map = np.zeros((len(rdd_alphas), len(rdd_windows)))

akrout_map = np.zeros((len(akrout_batch_sizes), len(akrout_decay_values))
stdwi_map = np.zeros((len(stdwi_taus), len(stdwi_taus)))
stdwi_map[:,:] = None
rdd_map = np.zeros((len(rdd_alphas), len(rdd_windows)))

# Populating the maps
# Akrout
akrout_maxval = 0.0
akrout_r_maxval = 0.0
akrout_argmax = (0,0)
for b_indx, batch_size in enumerate(akrout_batch_sizes):
    for d_indx, decay_value in enumerate(akrout_decay_values):
        filepath = path + "akrout_dump_" + str(batch_size) + "batch_" + str(decay_value) + "decay.npy"
        weight_estimates = np.fromfile(filepath).reshape(-1, num_output_neurons, num_input_neurons)
        akrout_map[b_indx, d_indx] = methods.sign_alignment(original_weight_matrix, weight_estimates[-1])
        akrout_r_map[b_indx, d_indx] = np.corrcoef(original_weight_matrix.flatten(), weight_estimates[-1].flatten())[0,1]
        if akrout_map[b_indx, d_indx] > akrout_maxval:
            akrout_maxval = akrout_map[b_indx, d_indx]
            akrout_r_maxval = akrout_r_map[b_indx, d_indx]
            akrout_argmax = (b_indx, d_indx)

# STDWI
stdwi_maxval = 0.0
stdwi_r_maxval = 0.0
stdwi_argmax = (0,0)
for f_indx, t_fast in enumerate(stdwi_taus):
    for s_indx, t_slow in enumerate(stdwi_taus[f_indx+1:]):
        filepath = path + "stdwi_dump_" + str(t_fast) + "fast_" + str(t_slow) + "slow.npy"
        weight_estimates = np.fromfile(filepath).reshape(-1, num_output_neurons, num_input_neurons)
        stdwi_map[f_indx, f_indx + s_indx] = methods.sign_alignment(original_weight_matrix, weight_estimates[-1])
        stdwi_r_map[f_indx, f_indx + s_indx] = np.corrcoef(original_weight_matrix.flatten(), weight_estimates[-1].flatten())[0,1]
        if (stdwi_map[f_indx, f_indx + s_indx] > stdwi_maxval):
            stdwi_maxval = stdwi_map[f_indx, f_indx + s_indx]
            stdwi_r_maxval = stdwi_r_map[f_indx, f_indx + s_indx]
            stdwi_argmax = (f_indx, f_indx + s_indx)


# RDD
rdd_maxval = 0.0
rdd_r_maxval = 0.0
rdd_argmax = (0,0)
for a_indx, alpha in enumerate(rdd_alphas):
    for w_indx, window_size in enumerate(rdd_windows):
        filepath = path + "rdd_dump_" + str(alpha) + "bound_" + str(window_size) + "window.npy"
        weight_estimates = np.fromfile(filepath).reshape(-1, num_output_neurons, num_input_neurons)
        rdd_map[a_indx, w_indx] = methods.sign_alignment(original_weight_matrix, weight_estimates[-1])
        rdd_r_map[a_indx, w_indx] = np.corrcoef(original_weight_matrix.flatten(), weight_estimates[-1].flatten())[0,1]
        if (rdd_map[a_indx, w_indx] > rdd_maxval):
            rdd_maxval = rdd_map[a_indx, w_indx]
            rdd_r_maxval = rdd_r_map[a_indx, w_indx]
            rdd_argmax = (a_indx, w_indx)

# Pearson Correlation Plots
plt.imshow(akrout_r_map.T, cmap="magma", vmin=0.5, vmax=1.0)
plt.xlabel("Baseline Window Sizes")
plt.ylabel("Decay Factor")
plt.xticks(np.arange(len(akrout_batch_sizes)), akrout_batch_sizes, rotation='vertical')
plt.yticks(np.arange(len(akrout_decay_values)), akrout_decay_values)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Pearson Correlation, r', rotation=270)
plt.savefig(outpath + 'rAkroutParameterMap.png', bbox_inches='tight')
plt.clf()

plt.imshow(stdwi_r_map.T, cmap="magma", vmin=0.5, vmax=1.0)
plt.xticks(np.arange(len(stdwi_taus)), stdwi_taus, rotation='vertical')
plt.yticks(np.arange(len(stdwi_taus[1:])), stdwi_taus[1:])
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Pearson Correlation, r', rotation=270)
plt.xlabel("Fast Trace Tau (ms)")
plt.ylabel("Slow Trace Tau (ms)")
plt.savefig(outpath + 'rSTDWIParameterMap.png', bbox_inches='tight')
plt.clf()

plt.imshow(rdd_r_map.T, cmap="magma", vmin=0.5, vmax=1.0)
plt.xticks(np.arange(len(rdd_alphas)), rdd_alphas, rotation='vertical')
plt.yticks(np.arange(len(rdd_windows)), rdd_windows)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Pearson Correlation, r', rotation=270)
plt.xlabel("Alpha -- Boundary to Spike")
plt.ylabel("Window Size (ms)")
plt.savefig(outpath + 'rRDDParameterMap.png', bbox_inches='tight')
plt.clf()

# Sign Alignment Plots
plt.imshow(akrout_map.T, cmap="magma", vmin=0.5, vmax=1.0)
plt.xlabel("Baseline Window Sizes")
plt.ylabel("Decay Factor")
plt.xticks(np.arange(len(akrout_batch_sizes)), akrout_batch_sizes, rotation='vertical')
plt.yticks(np.arange(len(akrout_decay_values)), akrout_decay_values)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Sign Alignment', rotation=270)
plt.savefig(outpath + 'AkroutParameterMap.png', bbox_inches='tight')
plt.clf()

plt.imshow(stdwi_map.T, cmap="magma", vmin=0.5, vmax=1.0)
plt.xticks(np.arange(len(stdwi_taus)), stdwi_taus, rotation='vertical')
plt.yticks(np.arange(len(stdwi_taus[1:])), stdwi_taus[1:])
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Sign Alignment', rotation=270)
plt.xlabel("Fast Trace Tau (ms)")
plt.ylabel("Slow Trace Tau (ms)")
plt.savefig(outpath + 'STDWIParameterMap.png', bbox_inches='tight')
plt.clf()

plt.imshow(rdd_map.T, cmap="magma", vmin=0.5, vmax=1.0)
plt.xticks(np.arange(len(rdd_alphas)), rdd_alphas, rotation='vertical')
plt.yticks(np.arange(len(rdd_windows)), rdd_windows)
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('SignAlignment', rotation=270)
plt.xlabel("Alpha -- Boundary to Spike")
plt.ylabel("Window Size (ms)")
plt.savefig(outpath + 'RDDParameterMap.png', bbox_inches='tight')
plt.clf()

print("Best Sign Accuracy Measurements:")
print("Akrout, STDWI, RDD")
print(akrout_maxval, stdwi_maxval, rdd_maxval)
print("Best Pearson Correlation Measurements:")
print("Akrout, STDWI, RDD")
print(akrout_r_maxval, stdwi_r_maxval, rdd_r_maxval)

print("Akrout Params: Window=" + str(akrout_batch_sizes[akrout_argmax[0]]) + " Decay=" + str(akrout_decay_values[akrout_argmax[1]]))
print("STDWI Params: Fast=" + str(stdwi_taus[stdwi_argmax[0]]) + " Slow=" + str(stdwi_taus[stdwi_argmax[1]]))
print("RDD Params: Alpha=" + str(rdd_alphas[rdd_argmax[0]]) + " Window=" + str(rdd_windows[rdd_argmax[1]]))

# Creating Scatter Plots with best params:

# RDD Scatter
filepath = path + "rdd_dump_" + str(rdd_alphas[rdd_argmax[0]]) + "bound_" + str(rdd_windows[rdd_argmax[1]]) + "window.npy"
rdd_weight_estimates = np.fromfile(filepath).reshape(-1, num_output_neurons, num_input_neurons)
print(len(rdd_weight_estimates))
plt.figure(figsize=(4,3), dpi=200)
plt.scatter(original_weight_matrix, rdd_weight_estimates[-1], color='blue', label="RDD", alpha=0.25)
plt.ylabel("Inferred Weight")
plt.xlabel("True Weight")
rangevals = 1.05*np.max(np.abs(rdd_weight_estimates[-1]))
plt.ylim([-rangevals, rangevals])
sns.despine()
plt.savefig(outpath + 'RDDScatterPlot.png', bbox_inches='tight')
plt.clf()

# Akrout Scatter
filepath = path + "akrout_dump_" + str(akrout_batch_sizes[akrout_argmax[0]]) + "batch_" + str(akrout_decay_values[akrout_argmax[1]]) + "decay.npy"
akrout_weight_estimates = np.fromfile(filepath).reshape(-1, num_output_neurons, num_input_neurons)
plt.figure(figsize=(4,3), dpi=200)
plt.scatter(original_weight_matrix, akrout_weight_estimates[-1], color='red', label="Akrout", alpha=0.25)
plt.ylabel("Inferred Weight")
plt.xlabel("True Weight")
rangevals = 1.05*np.max(np.abs(akrout_weight_estimates[-1]))
plt.ylim([-rangevals, rangevals])
sns.despine()
plt.savefig(outpath + 'AkroutScatterPlot.png', bbox_inches='tight')
plt.clf()

# STDWI Scatter
filepath = path + "stdwi_dump_" + str(stdwi_taus[stdwi_argmax[0]]) + "fast_" + str(stdwi_taus[stdwi_argmax[1]]) + "slow.npy"
stdwi_weight_estimates = np.fromfile(filepath).reshape(-1, num_output_neurons, num_input_neurons)
plt.figure(figsize=(4,3), dpi=200)
plt.scatter(original_weight_matrix, stdwi_weight_estimates[-1], color='black', label="STDWI", alpha=0.25)
plt.ylabel("Inferred Weight")
plt.xlabel("True Weight")
rangevals = 1.05*np.max(np.abs(stdwi_weight_estimates[-1]))
plt.ylim([-rangevals, rangevals])
sns.despine()
plt.savefig(outpath + 'STDWIScatterPlot.png', bbox_inches='tight')
