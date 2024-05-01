import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import scipy.linalg  as sla
import pickle 
from numpy import linalg as la
from tqdm import tqdm, trange
import copy

### Utility Functions
    
class ddrideshare:
    def __init__(self, loc_lst,price_lst,seed=None,lam=[0.0,0.0], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10,tot_rev=0, make_non_diag = False):
        self.loc_lst = loc_lst
        self.price_lst=price_lst
        self.n=2
        self.d=len(loc_lst)
        self.m=1
        self.lam1=lam[0]; 
        self.lam2=lam[1]
        self.l=[-maxx for i in range(self.d)]
        self.u=[maxx for i in range(self.d)]
        self.tot_rev=0
        if base:
            self.params=getparams(loc_lst, make_non_diag)
            self.A1=self.params['A1']
            self.A2=self.params['A2']
            self.Ac1=self.params['Ac1']
            self.Ac2=self.params['Ac2']
        else:
            self.params=params
            self.A1=self.params['A1']
            self.A2=self.params['A2']
            self.Ac1=self.params['Ac1']
            self.Ac2=self.params['Ac2']
        self.I=np.eye(self.d)
        self.seed=seed
        np.random.seed(self.seed)
        self.q_dic=get_data_dic()
        self.locations_ = self.q_dic['locations']
        self.dates_ = self.q_dic['dates']
        self.prices_=self.q_dic['prices']
        self.qlbar=self.q_dic['lyft_mean']
        self.qubar=self.q_dic['uber_mean']
        self.verbose=False

        
    def setup_distribution(self,centered=False):
        if centered:
            self.qu=self.q_dic['uber centered']
            self.ql=self.q_dic['lyft centered']
        else:
            self.qu=self.q_dic['uber']
            self.ql=self.q_dic['lyft']

        self.qlbar_=np.zeros((len(self.loc_lst),len(self.price_lst)))
        self.qubar_=np.zeros((len(self.loc_lst),len(self.price_lst)))
        self.ql_=np.zeros((len(self.dates_),len(self.loc_lst),len(self.price_lst)))
        self.qu_=np.zeros((len(self.dates_),len(self.loc_lst),len(self.price_lst)))
        for loc in self.loc_lst:
            for p in self.price_lst:
                self.qlbar_[loc,p]=self.qlbar[loc,p]
                self.qubar_[loc,p]=self.qubar[loc,p]
                self.ql_[:,loc,p]=self.ql[:,loc,p]
                self.qu_[:,loc,p]=self.qu[:,loc,p]
                
        self.locations = {'Haymarket Square':(42.3628, -71.0583), 'Back Bay':(42.3503, -71.0810),
                     'North End':(42.3647, -71.0542), 'North Station':(42.3661, -71.0631),
                     'Beacon Hill':(42.3588, -71.0707), 'Boston University':(42.3505, -71.1054),
                     'Fenway':(42.3467, -71.0972), 'South Station':(42.3519, -71.0552),
                     'Theatre District':(42.3519, -71.0643),# 'West End':(42.3644, -71.0661),
                     'Financial District':(42.3559, -71.0550), 'Northeastern University':(42.3398, -71.0892)}

        self.sources = []
        self.lats = []
        self.lons = []
        self.locs_ids={}
        for source in self.locations_:
            self.locs_ids[source]=self.locations[source]
        for source, coord in self.locs_ids.items():
            self.sources.append(source)
            self.lats.append(coord[0])
            self.lons.append(coord[1])
                
    def proj(self,x):
        y=np.zeros(np.shape(x))
        for i in range(self.n):
            for j in range(self.d):
                if x[i][j]<=self.l[j]:
                    y[i][j]=self.l[j]
                elif self.l[j]<x[i][j] and x[i][j]<self.u[j]:
                    y[i][j]=x[i][j]
                else:
                    y[i][j]=self.u[j]
        return y
    
    def get_gradient_Z_fixed(self, x, demand, player, noise=None):
        if not player:
            grad = -np.mean(demand, axis=0) + self.lam1 * x[player]
            
        else:
            grad = -np.mean(demand, axis=0) + self.lam2 * x[player]
            
        return grad

    def optimize_player(self, pr_t, demand, eta, player, noise=None, max_iter=30):
        print(f"Inside optimize player: Optimizing player {player}")
        pr_start = copy.deepcopy(pr_t)
        n_iter = 0
        pr_updated = copy.deepcopy(pr_start)
        while n_iter < max_iter:
            grad = self.get_gradient_Z_fixed(pr_start, demand, player, noise=noise)
            pr_updated[player] = pr_start[player] - eta*grad
            pr_updated = self.proj(pr_updated) #Clipping!
            pr_change_norm = la.norm(pr_updated[player] - pr_start[player])
            
            if pr_change_norm <= self.change_tol:
                print(f"Inside optimize player: Finished optimizing player {player} in {n_iter + 1} iterations")
                break
            else:
                n_iter += 1
                pr_start = copy.deepcopy(pr_updated)
        
        return pr_updated

    def sample_demand(self, x, player, base_z, x_prev=None, if_for_optim=False, num_samples=1, t_avg=False):
        noise = np.squeeze(np.random.normal(loc=base_z[player][0], scale=np.ones(11), size=(num_samples,11)))
        if player:
            demand_shift = self.A2@x[1]+self.Ac2@x[0]
        else:
            # print(f"Before performativity: {x[1]}")
            if x_prev is not None and if_for_optim:
                print("Doing performativity!")
                if not t_avg:
                    x = (x + x_prev)*0.5
                else:
                    idx = [-1-i for i in range(0, len(x_prev), 2)]
                    x = np.mean(np.array(x_prev)[idx], axis = 0)
            demand_shift = self.A1@x[0]+self.Ac1@x[1]
        
        demand = noise + demand_shift
        return demand, noise

    def revenue_RRM(self, x, demand, player):
        return np.mean(demand@(x[player]+self.tot_rev*self.prices_[self.price_index_rrm]))

    def revenue_loc_RRM(self, x, demand, player):
        return np.mean(np.multiply(demand,x[player]+self.tot_rev*self.prices_[self.price_index_rrm]), axis=0)

    def loss_RRM(self, x, demand, player): #demand is (1000,11)
        if player==0:
            return -np.mean(demand@x[player])+(self.lam1 * 0.5)*(la.norm(x[player])**2)
        else:
            return -np.mean(demand@x[player])+(self.lam2 * 0.5)*(la.norm(x[player])**2)

    def get_gaussian_base_demand_params(self, q_):
        mean_param = np.mean(q_, axis=1) #11 x 1000
        std_param = np.std(q_, axis=1)
        return [mean_param, std_param]

    def run_RRM_demand_samples(self, x0, price_index=0,eta=0.001,BATCH=10,MAXITER=10, verbose=False, perform_rrm=[True,True], RETURN=True, MYOPIC=False, tot_rev=1, is_t_avg = False):
        '''
        Runs for one price bin and all locations
        '''
        self.stepsize_rrm=eta
        self.batch_rrm=BATCH
        self.maxiter_rrm=MAXITER
        self.perform_comp_rrm=perform_rrm
        self.tot_rev=tot_rev
        self.price_index_rrm=price_index
        self.change_tol = 1e-4
        self.num_demand_samples = 25
        print("Price we are running at : ", self.prices_[self.price_index_rrm])
        q_lyft_=self.ql_[:,:,self.price_index_rrm].T
        q_uber_=self.qu_[:,:,self.price_index_rrm].T
        base_z = []
        base_z.append(self.get_gaussian_base_demand_params(q_lyft_))
        base_z.append(self.get_gaussian_base_demand_params(q_uber_))
        
        
        self.xo_rrm=x0
        self.x_rrm=[x0]; 

        demand_p1,_ = self.sample_demand(self.x_rrm[-1], 0, base_z, num_samples=100)
        demand_p2,_ = self.sample_demand(self.x_rrm[-1], 1, base_z, num_samples=100)
        self.rev_rrm_p1=[self.revenue_RRM(self.x_rrm[-1], demand_p1, 0)]
        self.rev_rrm_p2=[self.revenue_RRM(self.x_rrm[-1], demand_p2, 1)]
        self.rev_rrm_p1_loc=[self.revenue_loc_RRM(self.x_rrm[-1], demand_p1, 0)]
        self.rev_rrm_p2_loc=[self.revenue_loc_RRM(self.x_rrm[-1], demand_p2, 1)]
        self.demand_rrm_p1=[np.mean(demand_p1, axis=0)]
        self.demand_rrm_p2=[np.mean(demand_p2, axis=0)]
        self.loss_rrm_p1=[self.loss_RRM(self.x_rrm[-1], demand_p1, 0)]
        self.loss_rrm_p2=[self.loss_RRM(self.x_rrm[-1], demand_p2, 1)]

        '''
        Alternately optimize the prices of individual firms completely keeping the other's price constant.
        Lyft is player 1 and uber is player 2. 
        Starting prices are sampled randomly.
        '''
        count = 0
        for i in range(self.maxiter_rrm):
            if len(self.x_rrm) > 1 and not count: #count is the player!
                if is_t_avg:
                    z,noise=self.sample_demand(self.x_rrm[-1], count, base_z, x_prev=self.x_rrm, if_for_optim = True, num_samples=self.num_demand_samples, t_avg=True) #Shape is (100, 2, 11)
                else:
                    z,noise=self.sample_demand(self.x_rrm[-1], count, base_z, x_prev=self.x_rrm[-3], if_for_optim = True, num_samples=self.num_demand_samples) #Shape is (100, 2, 11)
            else:
                z,noise=self.sample_demand(self.x_rrm[-1], count, base_z, if_for_optim = True, num_samples=self.num_demand_samples) #Shape is (100, 2, 11)
            
            self.x_rrm.append(self.optimize_player(self.x_rrm[-1], z, eta, player=count, noise=noise)) #x_rrm element is of shape (2, 11)
            
            if count == 1:
                demand_p1,_ = self.sample_demand(self.x_rrm[-1], 0, base_z, num_samples=100)
                demand_p2,_ = self.sample_demand(self.x_rrm[-1], 1, base_z, num_samples=100)
            
            self.rev_rrm_p1.append(self.revenue_RRM(self.x_rrm[-1], demand_p1, 0))
            self.rev_rrm_p2.append(self.revenue_RRM(self.x_rrm[-1], demand_p2, 1))
            self.rev_rrm_p1_loc.append(self.revenue_loc_RRM(self.x_rrm[-1], demand_p1, 0))
            self.rev_rrm_p2_loc.append(self.revenue_loc_RRM(self.x_rrm[-1], demand_p2, 1))
            self.demand_rrm_p1.append(np.mean(demand_p1, axis=0))
            self.demand_rrm_p2.append(np.mean(demand_p2, axis=0))
            
            self.loss_rrm_p1.append(self.loss_RRM(self.x_rrm[-1], demand_p1, 0))
            self.loss_rrm_p2.append(self.loss_RRM(self.x_rrm[-1], demand_p2, 1))
            count = (count+1)%self.n
        if RETURN:
            dic={}
            dic['x']=self.x_rrm
            dic['loss_p1']=np.asarray(self.loss_rrm_p1)
            dic['loss_p2']=np.asarray(self.loss_rrm_p2)
            dic['revenue_total_p1']=np.asarray(self.rev_rrm_p1)
            dic['revenue_total_p2']=np.asarray(self.rev_rrm_p2)
            dic['revenue_by_loc_p1']=np.asarray(self.rev_rrm_p1_loc)
            dic['revenue_by_loc_p2']=np.asarray(self.rev_rrm_p2_loc)
            dic['demand_p1']=np.asarray(self.demand_rrm_p1)
            dic['demand_p2']=np.asarray(self.demand_rrm_p2)
            return dic
    
    def run_RRM_demand_samples_no_perf(self, x0, price_index=0,eta=0.001,BATCH=10,MAXITER=10, verbose=False, perform_rrm=[True,True], RETURN=True, MYOPIC=False, tot_rev=1):
        '''
        Runs for one price bin and all locations
        '''
        self.stepsize_rrm=eta
        self.batch_rrm=BATCH
        self.maxiter_rrm=MAXITER
        self.perform_comp_rrm=perform_rrm
        self.tot_rev=tot_rev
        self.price_index_rrm=price_index
        self.change_tol = 1e-4
        self.num_demand_samples = 1000
        print("Price we are running at : ", self.prices_[self.price_index_rrm])
        q_lyft_=self.ql_[:,:,self.price_index_rrm].T
        q_uber_=self.qu_[:,:,self.price_index_rrm].T
        base_z = []
        base_z.append(self.get_gaussian_base_demand_params(q_lyft_))
        base_z.append(self.get_gaussian_base_demand_params(q_uber_))
        #print(np.shape(q_lyft_))
        
        self.xo_rrm=x0
        self.x_rrm=[x0]; 

        demand_p1,_ = self.sample_demand(self.x_rrm[-1], 0, base_z, num_samples=100)
        demand_p2,_ = self.sample_demand(self.x_rrm[-1], 1, base_z, num_samples=100)
        self.rev_rrm_p1=[self.revenue_RRM(self.x_rrm[-1], demand_p1, 0)]
        self.rev_rrm_p2=[self.revenue_RRM(self.x_rrm[-1], demand_p2, 1)]
        self.rev_rrm_p1_loc=[self.revenue_loc_RRM(self.x_rrm[-1], demand_p1, 0)]
        self.rev_rrm_p2_loc=[self.revenue_loc_RRM(self.x_rrm[-1], demand_p2, 1)]
        self.demand_rrm_p1=[np.mean(demand_p1, axis=0)]
        self.demand_rrm_p2=[np.mean(demand_p2, axis=0)]
        self.loss_rrm_p1=[self.loss_RRM(self.x_rrm[-1], demand_p1, 0)]
        self.loss_rrm_p2=[self.loss_RRM(self.x_rrm[-1], demand_p2, 1)]

        '''
        Alternately optimize the prices of individual firms completely keeping the other's price constant.
        Lyft is player 1 and uber is player 2. 
        Starting prices are sampled randomly.
        '''
        count = 0
        for i in range(self.maxiter_rrm):
            if len(self.x_rrm) > 1: #count is the player!
                z,noise=self.sample_demand(self.x_rrm[-1], count, base_z, x_prev=self.x_rrm[-2], if_for_optim = False, num_samples=self.num_demand_samples) #Shape is (100, 2, 11)
            else:
                z,noise=self.sample_demand(self.x_rrm[-1], count, base_z, if_for_optim = False, num_samples=self.num_demand_samples) #Shape is (100, 2, 11)
            
            self.x_rrm.append(self.optimize_player(self.x_rrm[-1], z, eta, player=count, noise=noise)) #x_rrm element is of shape (2, 11)
            
            if count == 1:
                demand_p1,_ = self.sample_demand(self.x_rrm[-1], 0, base_z, num_samples=100)
                demand_p2,_ = self.sample_demand(self.x_rrm[-1], 1, base_z, num_samples=100)
            
            self.rev_rrm_p1.append(self.revenue_RRM(self.x_rrm[-1], demand_p1, 0))
            self.rev_rrm_p2.append(self.revenue_RRM(self.x_rrm[-1], demand_p2, 1))
            self.rev_rrm_p1_loc.append(self.revenue_loc_RRM(self.x_rrm[-1], demand_p1, 0))
            self.rev_rrm_p2_loc.append(self.revenue_loc_RRM(self.x_rrm[-1], demand_p2, 1))
            self.demand_rrm_p1.append(np.mean(demand_p1, axis=0))
            self.demand_rrm_p2.append(np.mean(demand_p2, axis=0))
            
            self.loss_rrm_p1.append(self.loss_RRM(self.x_rrm[-1], demand_p1, 0))
            self.loss_rrm_p2.append(self.loss_RRM(self.x_rrm[-1], demand_p2, 1))
            count = (count+1)%self.n
        if RETURN:
            dic={}
            dic['x']=self.x_rrm
            dic['loss_p1']=np.asarray(self.loss_rrm_p1)
            dic['loss_p2']=np.asarray(self.loss_rrm_p2)
            dic['revenue_total_p1']=np.asarray(self.rev_rrm_p1)
            dic['revenue_total_p2']=np.asarray(self.rev_rrm_p2)
            dic['revenue_by_loc_p1']=np.asarray(self.rev_rrm_p1_loc)
            dic['revenue_by_loc_p2']=np.asarray(self.rev_rrm_p2_loc)
            dic['demand_p1']=np.asarray(self.demand_rrm_p1)
            dic['demand_p2']=np.asarray(self.demand_rrm_p2)
            return dic
   
def get_data_dic(filename='./data/datadic.p'):
    return pickle.load(open(filename,'rb'))
        
def make_matrix_non_diag(A1, A2, Ac1, Ac2):
    off_diag_options = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20,25, 30]
    mat_list = [A1, A2, Ac1, Ac2]

    for i, mat in enumerate(mat_list):
        chosen_opt = np.random.normal(size=(10))
        # print(f"Values chosen for matrix is: {chosen_opt}")
        update_val = np.diagflat(chosen_opt, k=1)
        # import pdb; pdb.set_trace()
        mat_list[i] = mat + update_val
    return mat_list[0], mat_list[1], mat_list[2], mat_list[3]

def getparams(loc_lst, make_non_diag=False):
    mu=np.load('./data/mu_est.npy')
    gamma=np.load('./data/gamma_est.npy')
    who=['Lyft values','Uber values']
    A={}
    B={}
    for i in range(2):
        A[who[i]]=[]
        B[who[i]]=[]
        for j in loc_lst:
            B[who[i]].append(gamma[j][i][0,0])
            A[who[i]].append(mu[j][i][0,0])

        B[who[i]]=np.asarray(B[who[i]])
        A[who[i]]=np.asarray(A[who[i]])
    A1=np.diag(A[who[0]])
    A2=np.diag(A[who[1]])
    Ac1=np.diag(B[who[0]])
    Ac2=np.diag(B[who[1]])
    if make_non_diag:
        A1, A2, Ac1, Ac2 = make_matrix_non_diag(A1, A2, Ac1, Ac2)
    # import pdb; pdb.set_trace()
    dic={}
    dic['A1']=A1
    dic['A2']=A2
    dic['Ac1']=Ac1
    dic['Ac2']=Ac2
    return dic

