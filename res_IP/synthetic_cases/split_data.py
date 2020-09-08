### Code for splitting datasets randomly and creating copies

import numpy as np
import os
import sys

os.chdir("/home/simon/Desktop/implicit-variational-inference/implicit-processes/synthetic_cases/")

##############

data_name = sys.argv[ 1 ]

np.random.seed(1)
splits = 20

# Import the dataset in question
data = np.loadtxt(data_name)

# Create the splits folder if there isn't any
#if not os.path.isdir("splits"):
#    os.makedirs("splits")

# Create the (splits)-permutations and store them
permutation = list()
for i in range(splits):
    permutation.append(np.random.choice(range(data.shape[0]), data.shape[ 0 ], replace = False))

    #Export the dataset once permutations are done
    # np.savetxt("splits/split_" + str(i) + "_" + data_name, data[permutation[i]])

np.savetxt("permutations_" + data_name, np.array(permutation), fmt = "%s", delimiter = ",")
