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

# STDWI Params
t_fast = 40.0
t_slow = 200.0

# Akrout params
akrout_batch = 100
akrout_decay = 0.1

# RDD params
alpha = 0.025
window = 35

seed = 1
ratio_active = 1.0
seeds = np.arange(1,11)
correlations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

outpath = "./_plots/" + str(ratio_active) + "Perc/"
os.makedirs(outpath, exist_ok=True)

# Result sets
stdwi_acc_set, stdwi_r_set = [[] for c in correlations], [[] for c in correlations]
akrout_acc_set, akrout_r_set = [[] for c in correlations], [[] for c in correlations]
rdd_acc_set, rdd_r_set = [[] for c in correlations], [[] for c in correlations]


for c_indx, correlation in enumerate(correlations):
    for seed in seeds:
    
        # Path and folder creation
        path = "../" + str(num_input_neurons) + "Inputs_" + str(num_output_neurons) + "Outputs_" + str(ratio_active) + "Perc_" + str(correlation) + "Corr_" + str(seed) + "Seed/" 
    
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
        stdwi_acc_set[c_indx].append(stdwi_acc)
        stdwi_r_set[c_indx].append(stdwi_r)
    
    
        # Akrout results loading
        akrout_filepath = path + "akrout_dump_" + str(akrout_batch) + "batch_" + str(akrout_decay) + "decay.npy"
        akrout_weight_estimates = np.fromfile(akrout_filepath).reshape(-1, num_output_neurons, num_input_neurons)
    
        akrout_acc = []
        akrout_r = []
        for e in range(num_epochs):
            akrout_acc.append(methods.sign_alignment(original_weight_matrix, akrout_weight_estimates[e]))
            akrout_r.append(np.corrcoef(original_weight_matrix.flatten(), akrout_weight_estimates[e].flatten())[0,1])
        akrout_acc_set[c_indx].append(akrout_acc)
        akrout_r_set[c_indx].append(akrout_r)
        
        
        # RDD results loading
        rdd_filepath = path + "rdd_dump_" + str(alpha) + "bound_" + str(window) + "window.npy"
        rdd_weight_estimates = np.fromfile(rdd_filepath).reshape(-1, num_output_neurons, num_input_neurons)
    
        rdd_acc = []
        rdd_r = []
        for e in range(num_epochs):
            rdd_acc.append(methods.sign_alignment(original_weight_matrix, rdd_weight_estimates[e]))
            rdd_r.append(np.corrcoef(original_weight_matrix.flatten(), rdd_weight_estimates[e].flatten())[0,1])
        rdd_acc_set[c_indx].append(rdd_acc)
        rdd_r_set[c_indx].append(rdd_r)


stdwi_acc_set, stdwi_r_set = np.asarray(stdwi_acc_set), np.asarray(stdwi_r_set)
akrout_acc_set, akrout_r_set = np.asarray(akrout_acc_set), np.asarray(akrout_r_set)
rdd_acc_set, rdd_r_set = np.asarray(rdd_acc_set), np.asarray(rdd_r_set)

mean_stdwi_acc, std_stdwi_acc = np.mean(stdwi_acc_set, axis=1)[:,-1], np.std(stdwi_acc_set, axis=1)[:,-1]
mean_stdwi_r, std_stdwi_r = np.mean(stdwi_r_set, axis=1)[:,-1], np.std(stdwi_r_set, axis=1)[:,-1]

mean_akrout_acc, std_akrout_acc = np.mean(akrout_acc_set, axis=1)[:,-1], np.std(akrout_acc_set, axis=1)[:,-1]
mean_akrout_r, std_akrout_r = np.mean(akrout_r_set, axis=1)[:,-1], np.std(akrout_r_set, axis=1)[:,-1]

mean_rdd_acc, std_rdd_acc = np.mean(rdd_acc_set, axis=1)[:,-1], np.std(rdd_acc_set, axis=1)[:,-1]
mean_rdd_r, std_rdd_r = np.mean(rdd_r_set, axis=1)[:,-1], np.std(rdd_r_set, axis=1)[:,-1]

#plt.figure(figsize=(4,3), dpi=200)
#ax = plt.gca()
## RDD with std
#plt.plot(mean_rdd_acc[:,-1], color='blue', label="RDD")
## Akrout with std
#plt.plot(akrout_acc_set[:,-1], color='red', label="Akrout")
## STDWI and std
#plt.plot(stdwi_acc_set[:,-1], color='black', label="STDWI")
#plt.legend(frameon=False)
#plt.ylim([0.0,1.0])
#plt.xticks(correlations)
#plt.ylabel("Sign Accuracy")
#plt.xlabel("Stimulation Correlation Coefficient")
#sns.despine()
#plt.savefig(outpath + 'SignAccuracy_Correlation.png', bbox_inches='tight')
#plt.clf()
## plt.show()
#
#
#plt.figure(figsize=(4,3), dpi=200)
#ax = plt.gca()
## RDD with std
#plt.plot(rdd_r_set[:,-1], color='blue', label="RDD")
## Akrout with std
#plt.plot(akrout_r_set[:,-1], color='red', label="Akrout")
## STDWI and std
#plt.plot(stdwi_r_set[:,-1], color='black', label="STDWI")
#plt.legend(frameon=False)
#plt.ylim([0.0,1.0])
#plt.xticks(correlations)
#plt.ylabel("Pearson Correlation -- r")
#plt.xlabel("Stimulation Correlation Coefficient")
#sns.despine()
#plt.savefig(outpath + 'PearsonCorrelation_Correlation.png', bbox_inches='tight')
#plt.clf()
## plt.show()

plt.figure(figsize=(4,3), dpi=200)
ax = plt.gca()
# RDD with std
plt.errorbar(correlations, mean_rdd_acc, yerr=std_rdd_acc, fmt='o', color='blue', label='RDD')
#plt.plot(mean_rdd_acc, color='blue', label="RDD")
#ax.fill_between(range(len(mean_rdd_acc)), mean_rdd_acc - std_rdd_acc, mean_rdd_acc + std_rdd_acc, color='blue', alpha=0.25)
# Akrout with std
plt.errorbar(correlations, mean_akrout_acc, yerr=std_akrout_acc, fmt='o', color='red', label='Akrout')
#plt.plot(mean_akrout_acc, color='red', label="Akrout")
#ax.fill_between(range(len(mean_akrout_acc)), mean_akrout_acc - std_akrout_acc, mean_akrout_acc + std_akrout_acc, color='red', alpha=0.25)
# STDWI and std
#plt.plot(mean_stdwi_acc, color='black', label="STDWI")
plt.errorbar(correlations, mean_stdwi_acc, yerr=std_stdwi_acc, fmt='o', color='black', label='STDWI')
#ax.fill_between(range(len(mean_stdwi_acc)), mean_stdwi_acc - std_stdwi_acc, mean_stdwi_acc + std_stdwi_acc, color='black', alpha=0.25)
plt.legend(frameon=False)
#plt.xticks(np.arange(11), correlations)
plt.ylabel("Sign Accuracy")
plt.xlabel("Correlation Count Coefficient (Stimulation)")
sns.despine()
plt.savefig(outpath + 'SignAccuracy_CorrelationEffect.png', bbox_inches='tight')
plt.clf()
# plt.show()


plt.figure(figsize=(4,3), dpi=200)
ax = plt.gca()
# RDD with std
plt.errorbar(correlations, mean_rdd_r, yerr=std_rdd_r, fmt='o', color='blue', label="RDD")
#plt.plot(mean_rdd_r, color='blue', label="RDD")
#ax.fill_between(range(len(mean_rdd_r)), mean_rdd_r - std_rdd_r, mean_rdd_r + std_rdd_r, color='blue', alpha=0.25)
# Akrout with std
plt.errorbar(correlations, mean_akrout_r, yerr=std_akrout_r, fmt='o', color='red', label='Akrout')
#plt.plot(mean_akrout_r, color='red', label="Akrout")
#ax.fill_between(range(len(mean_akrout_r)), mean_akrout_r - std_akrout_r, mean_akrout_r + std_akrout_r, color='red', alpha=0.25)
# STDWI and std
plt.errorbar(correlations, mean_stdwi_r, yerr=std_stdwi_r, fmt='o', color='black', label='STDWI')
#plt.plot(mean_stdwi_r, color='black', label="STDWI")
#ax.fill_between(range(len(mean_stdwi_r)), mean_stdwi_r - std_stdwi_r, mean_stdwi_r + std_stdwi_r, color='black', alpha=0.25)
plt.legend(frameon=False)
#plt.xticks(np.arange(11), correlations)
plt.ylabel("Pearson Correlation -- r")
plt.xlabel("Correlation Count Coefficient (Stimulation)")
sns.despine()
plt.savefig(outpath + 'PearsonCorrelation_CorrelationEffect.png', bbox_inches='tight')
plt.clf()
# plt.show()
