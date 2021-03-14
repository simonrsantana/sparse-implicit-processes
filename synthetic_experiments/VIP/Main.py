##### Variational Implicit Processes
##### This code implememts VIP-BNN on an example UCI dataset
##### To use this code: simply run Main.py
##### minor differences from the paper might occur due to randomness.

import os
import pandas as pd
from import_libs import*
rs = 1 # random seed
np.random.seed(rs)
tf.set_random_seed(rs)

#### VIP parameters
n_optimize =  2000
lr = 1e-3 # 0.01
n_bt = 100 # 40 # number of random functions for VIP
n_mc = 1
structure = [50, 50, 1] # BNN structure
alpha = 0.5 # for alpha divergence energy

# SRS
n_batch = 10 # Introduce batch training

##### load data, yacht uci data
original_file = sys.argv[ 1 ]


Data = np.loadtxt(original_file).astype(np.float32)
size_feature = Data.shape[1]-1
n_split = 1 # 20  # int(sys.argv[ 2 ])  # 20 # number of splits
ratio_train = 0.9
llh_VIP_agg = np.zeros(n_split) # array for storing llh results
rmse_VIP_agg = np.zeros(n_split) # array for storing rmse results
crps_VIP_agg = np.zeros(n_split) # array for storing crps results

# Create a folder to store the screen prints
if not os.path.isdir("prints"):
    os.makedirs("prints")


# SRS Since bacth training is much slower, we will paralellize the different splits 

# Loop over the number of splits
# for split in range(n_split):
#     # Create a file to store the results of the run (or empty the previously existing one)
#     if os.path.isfile("prints/print_vip_" + str(split) + "_" +  original_file):
#         with open("prints/print_vip_" + str(split) + "_" +  original_file, "w") as res_file:
#            res_file.close()

# Introduce the split as a new input to the system
split = int( sys.argv[ 2 ] )

if os.path.isfile("prints/print_vip_" + str(split) + "_" +  original_file):
    with open("prints/print_vip_" + str(split) + "_" +  original_file, "w") as res_file:
       res_file.close()



# Create the folder to save all the results
if not os.path.isdir("res"):
    os.makedirs("res")

if not os.path.isdir("res/"):
    os.makedirs("res/")




#### validation phase
# grid on hyper parameter validation
grid1 = np.array([0.5,5,1000]) # for $\tau$, i.e., precision of the observation noise
grid2 = np.array([0.08,0.1,0.2,0.5,1,2,5,10]) # for inverse Wishart distribution parameter $\psi$ in Appendix E.3

# SRS - OJO, cuando se le pasa yita en el código es haciendo yita = 1/grid2[i], por lo tanto los valores son el inverso de psi


# import pdb; pdb.set_trace()

exec(open("validation.py").read())
#### test phase
exec(open("test.py").read())
