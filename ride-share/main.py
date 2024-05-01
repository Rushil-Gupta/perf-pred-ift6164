import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse
import pickle 
import sys
from utils.utilsrm import *
import scipy.linalg  as sla
import random

import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange

loc_cap=11
lam1=70
lam2=70
loc_lst_index=list(range(0,loc_cap))
price_lst_index=list(range(0,3))

loss_fname = 'loss_shift_perf_p1.png'

MAXITER=40
loc_lst_index=list(range(0,loc_cap))
price_lst_index=list(range(0,3))
seeds_to_run_for = 40
all_seeds_loss_p1 = []
all_seeds_loss_p1_t_avg = []
all_seeds_loss_p1_no_perf = []

for _ in range(seeds_to_run_for):
    x0=np.random.rand(2,loc_cap)
    ddgame=ddrideshare(loc_lst_index,price_lst_index, seed=None,lam=[lam1,lam2], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=30, make_non_diag=True)
    ddgame.setup_distribution()
    
    #Last iterate method
    dic_rrm_no_perf=ddgame.run_RRM_demand_samples_no_perf(x0,eta=0.001,BATCH=10,MAXITER=MAXITER, perform_rrm=[True,True])

    #Duo averaging
    dic_rrm=ddgame.run_RRM_demand_samples(x0,eta=0.001,BATCH=10,MAXITER=MAXITER, perform_rrm=[True,True])

    #T Averaging
    dic_rrm_t_avg=ddgame.run_RRM_demand_samples(x0,eta=0.001,BATCH=10,MAXITER=MAXITER, perform_rrm=[True,True], is_t_avg=True)

    loss_p1=np.asarray(dic_rrm['loss_p1'])
    loss_p1_t_avg=np.asarray(dic_rrm_t_avg['loss_p1'])
    loss_p1_no_perf=np.asarray(dic_rrm_no_perf['loss_p1'])

    all_seeds_loss_p1.append(loss_p1)
    all_seeds_loss_p1_t_avg.append(loss_p1_t_avg)
    all_seeds_loss_p1_no_perf.append(loss_p1_no_perf)

all_seeds_loss_p1 = np.stack(all_seeds_loss_p1, axis=0)
all_seeds_loss_p1_t_avg = np.stack(all_seeds_loss_p1_t_avg, axis=0)
all_seeds_loss_p1_no_perf = np.stack(all_seeds_loss_p1_no_perf, axis=0)

#PROCESSING AND PLOTTING THE GRAPH

all_seeds_abs_gap = np.abs(all_seeds_loss_p1[:,1:] - all_seeds_loss_p1[:,:-1])
all_seeds_abs_gap_t_avg = np.abs(all_seeds_loss_p1_t_avg[:,1:] - all_seeds_loss_p1_t_avg[:,:-1])
all_seeds_abs_gap_no_perf = np.abs(all_seeds_loss_p1_no_perf[:,1:] - all_seeds_loss_p1_no_perf[:,:-1])
norm_val = np.max(np.stack([all_seeds_abs_gap, all_seeds_abs_gap_no_perf, all_seeds_abs_gap_t_avg]))
all_seeds_abs_gap = all_seeds_abs_gap/norm_val
all_seeds_abs_gap_t_avg = all_seeds_abs_gap_t_avg/norm_val
all_seeds_abs_gap_no_perf = all_seeds_abs_gap_no_perf/norm_val

all_seeds_gap = np.mean(all_seeds_abs_gap, axis=0)
all_seeds_gap_t_avg = np.mean(all_seeds_abs_gap_t_avg, axis=0)
all_seeds_gap_no_perf = np.mean(all_seeds_abs_gap_no_perf, axis=0)

idx_to_take = np.arange(1,MAXITER, step=2)

x_vals = np.arange(0,MAXITER+1)

x_labels = np.arange(0,len(idx_to_take))
plt.errorbar(x_labels, all_seeds_gap[idx_to_take], yerr = np.std(all_seeds_abs_gap, axis=0)[idx_to_take],  color = 'red', marker='*', linestyle='-', label="duo_averaging")
plt.errorbar(x_labels, all_seeds_gap_t_avg[idx_to_take], yerr = np.std(all_seeds_abs_gap_t_avg, axis=0)[idx_to_take], color = 'green', marker='*', linestyle='-', label="t_averaging")
plt.errorbar(x_labels, all_seeds_gap_no_perf[idx_to_take], yerr = np.std(all_seeds_abs_gap_no_perf, axis=0)[idx_to_take], color = 'blue', marker = '*', linestyle='-', label ="last_iterate")
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss shift due to performativity')
plt.rcParams['ytick.labelsize']=14
plt.rcParams['xtick.labelsize']=14
plt.savefig(loss_fname)
plt.close()