##### Variational Implicit Processes
##### This code implememts VIP-BNN on an example UCI dataset
##### To use this code: simply run Main.py
##### minor differences from the paper might occur due to randomness.


from import_libs import*
rs = 42 # random seed
np.random.seed(rs)
tf.set_random_seed(rs)

#### VIP parameters
n_optimize = 1
lr = 0.01
n_bt = 20 # number of random functions for VIP
n_mc = 1
structure = [10,10,1] # BNN structure
alpha = 0.5 # for alpha divergence energy

##### load data, yacht uci data
original_file = sys.argv[ 1 ]

Data = np.loadtxt(original_file).astype(np.float32)
size_feature = Data.shape[1]-1
n_split = 20 # number of splits
ratio_train = 0.9
llh_VIP_agg = np.zeros(n_split) # array for storing llh results
rmse_VIP_agg = np.zeros(n_split) # array for storing rmse results

#### validation phase
# grid on hyper parameter validation
grid1 = np.array([0.5,5,1000]) # for $\tau$, i.e., precision of the observation noise
grid2 = np.array([0.08,0.1,0.2,0.5,1,2,5,10]) # for inverse Wishart distribution parameter $\psi$ in Appendix E.3
exec(open("validation.py").read())
#### test phase
exec(open("test.py").read())
