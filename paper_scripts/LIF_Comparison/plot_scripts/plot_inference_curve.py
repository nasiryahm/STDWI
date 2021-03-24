import numpy as np
import matplotlib.pyplot as plt
import re
import sys, os
import seaborn as sns
import json

sys.path.insert(0,'../../..')
from weight_inference import methods

# Network Settings
num_input_neurons = 100
num_output_neurons = 10
timestep = 0.25
threshold = 1.0
check_interval = 10

# Result sets
stdwi_acc_set, stdwi_r_set = [], []
akrout_acc_set, akrout_r_set = [], []
rdd_acc_set, rdd_r_set = [], []

seed = 1
ratio_active = 0.2
correlation = 0.0
weight = np.random.randint(num_input_neurons*num_output_neurons)

outpath = "./_plots/" + str(ratio_active) + "Perc_" + str(correlation) + "Corr/"

akrout_batch = 100
akrout_decay = 0.2

stdwi_fast = 20.0
stdwi_slow = 200.0

rdd_alpha = 0.025
rdd_window = 35

# Path and folder creation
path = "../" + str(num_input_neurons) + "Inputs_" + str(num_output_neurons) + "Outputs_" + str(ratio_active) + "Perc_" + str(correlation) + "Corr_" + str(seed) + "Seed/" 

original_weight_matrix = np.fromfile(path + "IO_weight_matrix.npy").reshape(num_output_neurons, num_input_neurons)

# STDWI results loading
stdwi_filepath = path + "stdwi_dump_" + str(stdwi_fast) + "fast_" + str(stdwi_slow) + "slow.npy"
stdwi_weight_estimates = np.fromfile(stdwi_filepath).reshape(-1, num_output_neurons*num_input_neurons)
stdwi_weight_estimates -= stdwi_weight_estimates[0,:]
stdwi_weight_estimates /= np.max(np.abs(stdwi_weight_estimates), axis=0)
num_epochs = stdwi_weight_estimates.shape[0]

# Akrout results loading
akrout_filepath = path + "akrout_dump_" + str(akrout_batch) + "batch_" + str(akrout_decay) + "decay.npy"
akrout_weight_estimates = np.fromfile(akrout_filepath).reshape(-1, num_output_neurons*num_input_neurons)
akrout_weight_estimates -= akrout_weight_estimates[0,:]
akrout_weight_estimates /= np.max(np.abs(akrout_weight_estimates), axis=0)

# RDD results loading
rdd_filepath = path + "rdd_dump_" + str(rdd_alpha) + "bound_" + str(rdd_window) + "window.npy"
rdd_weight_estimates = np.fromfile(rdd_filepath).reshape(-1, num_output_neurons*num_input_neurons)
rdd_weight_estimates -= rdd_weight_estimates[0,:]
rdd_weight_estimates /= np.max(np.abs(rdd_weight_estimates), axis=0)

plt.figure(figsize=(4,3), dpi=200)
ax = plt.gca()
plt.plot(rdd_weight_estimates[:,weight], color='blue', label="RDD")
plt.plot(akrout_weight_estimates[:,weight], color='red', label="Akrout")
plt.plot(stdwi_weight_estimates[:,weight], color='black', label="STDWI")
plt.xticks([0, num_epochs], [0, num_epochs])
plt.yticks([])
plt.ylabel("Normalized Inferred Weight")
plt.xlabel("Time (s)")
sns.despine()
plt.savefig(outpath + 'inference_curve.png', bbox_inches='tight')
plt.clf()
# plt.show()
