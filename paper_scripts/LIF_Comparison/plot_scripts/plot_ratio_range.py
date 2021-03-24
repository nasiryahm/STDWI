import numpy as np
import matplotlib.pyplot as plt
import re
import sys, os
import seaborn as sns

sys.path.insert(0,'../../..')
from weight_inference import methods

# Network Settings
num_input_neurons = 100
num_output_neurons = 10
timestep = 0.25
threshold = 1.0
check_interval = 10

# These are the best parameters -- should be determined by a grid search
# STDWI Params
t_fast = 40.0
t_slow = 200.0
# Akrout params
akrout_batch = 1000
akrout_decay = 0.1
# RDD params
alpha = 0.025
window = 35

# Result sets
stdwi_acc_set, stdwi_r_set = [], []
akrout_acc_set, akrout_r_set = [], []
rdd_acc_set, rdd_r_set = [], []

outpath = "./_plots/"
os.makedirs(outpath, exist_ok=True)

seed = 1 
ratios_active = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0][::-1]

for ratio_active in ratios_active:
    # Path and folder creation
    path = "../" + str(num_input_neurons) + "Inputs_" + str(num_output_neurons) + "Outputs_" + str(ratio_active) + "Perc_" + str(seed) + "Seed/" 

    original_weight_matrix = np.fromfile(path + "IO_weight_matrix.npy").reshape(num_output_neurons, num_input_neurons)

    # STDWI results loading
    stdwi_filepath = path + "stdwi_dump_" + str(t_fast) + "fast_" + str(t_slow) + "slow.npy"
    stdwi_weight_estimates = np.fromfile(stdwi_filepath).reshape(-1, num_output_neurons, num_input_neurons)
    num_epochs = stdwi_weight_estimates.shape[0]

    stdwi_acc = []
    stdwi_r = []
    for e in range(num_epochs):
        stdwi_acc.append(methods.sign_alignment(original_weight_matrix, stdwi_weight_estimates[e]))
        stdwi_r.append(np.corrcoef(original_weight_matrix.flatten(), stdwi_weight_estimates[e].flatten())[0,1])
    stdwi_acc_set.append(stdwi_acc)
    stdwi_r_set.append(stdwi_r)


    # Akrout results loading
    akrout_filepath = path + "akrout_dump_" + str(akrout_batch) + "batch_" + str(akrout_decay) + "decay.npy"
    akrout_weight_estimates = np.fromfile(akrout_filepath).reshape(-1, num_output_neurons, num_input_neurons)

    akrout_acc = []
    akrout_r = []
    for e in range(num_epochs):
        akrout_acc.append(methods.sign_alignment(original_weight_matrix, akrout_weight_estimates[e]))
        akrout_r.append(np.corrcoef(original_weight_matrix.flatten(), akrout_weight_estimates[e].flatten())[0,1])
    akrout_acc_set.append(akrout_acc)
    akrout_r_set.append(akrout_r)
    
    
    # RDD results loading
    rdd_filepath = path + "rdd_dump_" + str(alpha) + "bound_" + str(window) + "window.npy"
    rdd_weight_estimates = np.fromfile(rdd_filepath).reshape(-1, num_output_neurons, num_input_neurons)

    rdd_acc = []
    rdd_r = []
    for e in range(num_epochs):
        rdd_acc.append(methods.sign_alignment(original_weight_matrix, rdd_weight_estimates[e]))
        rdd_r.append(np.corrcoef(original_weight_matrix.flatten(), rdd_weight_estimates[e].flatten())[0,1])
    rdd_acc_set.append(rdd_acc)
    rdd_r_set.append(rdd_r)


stdwi_acc_set, stdwi_r_set = np.asarray(stdwi_acc_set), np.asarray(stdwi_r_set)
akrout_acc_set, akrout_r_set = np.asarray(akrout_acc_set), np.asarray(akrout_r_set)
rdd_acc_set, rdd_r_set = np.asarray(rdd_acc_set), np.asarray(rdd_r_set)


plt.figure(figsize=(4,3*len(ratios_active)), dpi=200)
for indx in range(len(ratios_active)):
    plt.subplot(len(ratios_active), 1, indx + 1)
    plt.bar(range(3), [stdwi_acc_set[indx][-1], akrout_acc_set[indx][-1], rdd_acc_set[indx][-1]],
            color=["black", "blue", "red"])
    plt.xticks([]) #range(3), ["STDWI", "Akrout", "RDD"])
    plt.yticks([])
    plt.ylim([0.7,1.0])
    plt.box(False)
plt.savefig(outpath + "RatioSearch.png", bbox_inches='tight')
plt.clf()
# plt.show()
