# -*- coding: utf-8 -*-
# @Time    : 9/23/21 11:33 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : collect_summary.py

# collect summery of repeated experiment.

import argparse
import os
import numpy as np
import pandas as pd
import os, shutil
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-dir", type=str, default="../exp/gopt-1e-3-4-1-1-24-gopt-kmeans_distance-br", help="directory to dump experiments")
args = parser.parse_args()

result = []
# for each repeat experiment
for i in range(0, 10):
    cur_exp_dir = args.exp_dir + '-' + str(i)

    if os.path.isfile(cur_exp_dir + '/result.csv'):
        try:
            
            print(cur_exp_dir)
            cur_res = np.loadtxt(cur_exp_dir + '/result.csv', delimiter=',',skiprows=1)
            result.append(cur_res)
        except:
            pass
        
result = np.array(result)
# get mean / std of the repeat experiments.
result_mean = np.mean(result, axis=0)
result_std = np.std(result, axis=0)
if os.path.exists(args.exp_dir) == False:
    os.mkdir(args.exp_dir)
np.savetxt(args.exp_dir + '/result_summary.csv', [result_mean, result_std], delimiter=',')

df = pd.read_csv(args.exp_dir + '/result_summary.csv', names=['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning_rate',"utt_train_mse_accuracy","utt_train_mse_completeness","utt_train_mse_fluency","utt_train_mse_prosodic","utt_train_mse_total","utt_train_pcc_accuracy","utt_train_pcc_completeness","utt_train_pcc_fluency","utt_train_pcc_prosodic","utt_train_pcc_total","utt_test_mse_accuracy","utt_test_mse_completeness","utt_test_mse_fluency","utt_test_mse_prosodic","utt_test_mse_total","utt_test_pcc_accuracy","utt_test_pcc_completeness","utt_test_pcc_fluency","utt_test_pcc_prosodic","utt_test_pcc_total","word_train_pcc_accuracy","word_train_pcc_stress","word_train_pcc_total","word_test_pcc_accuracy","word_test_pcc_stress","word_test_pcc_total","mean_distance","sum_distance","f1_score","precision","recall"])
df.drop(["learning_rate","utt_train_mse_accuracy","utt_train_mse_completeness","utt_train_mse_fluency","utt_train_mse_prosodic","utt_train_mse_total","utt_test_mse_fluency","utt_test_mse_prosodic","utt_test_mse_total","utt_test_mse_accuracy","utt_test_mse_completeness"],axis=1,inplace=True)

print()
print("Results:")
accum_mean = 0
accum_std = 0
i = 0
print("--------------------------------------------------")
for name in df.columns:
    print(f"{name}: {df[name].values[0]:.3f}")
    print(f"std: {df[name].values[1]:.3f}")
    
    if "mse" not in name:
        accum_mean+=df[name].values[0]
        accum_std+=df[name].values[1]
        i+=1
        
    print("--------------------------------------------------")

print(f"Mean Correlation: {accum_mean/i}")
print(f"Mean STD: {accum_std/i}")

