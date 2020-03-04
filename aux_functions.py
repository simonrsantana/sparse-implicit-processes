
################################################################################
# Define auxiliary functions that allow us to calculate the relevant moments   #
# and associated quantities from the sampled function values in the main code  #
################################################################################

import tensorflow as tf
import numpy as np

# Means
def mean_f(f_samples):
    # Calculate the mean of the samples of functions
    means_f = tf.reduce_mean(f_samples, axis = [ 1 ])
    return means_f


# Deviations
def deltas_f(f_samples, mean_f):
    # Calculate the deviation from the mean of sampled functions' values
    delta_s_f = f_samples - tf.expand_dims(mean_f, -1)  # Shape is (batch_size x n_samples)
    return delta_s_f


# Calculate the covariance matrix given two delta matrices
def calculate_covariances(delta_1, delta_2):

    # Check whether the number of samples for each delta is the same
    #if tf.cast(tf.shape(delta_1)[ 1 ], tf.float32) != tf.cast(tf.shape(delta_2)[ 1 ], tf.float32):
    #    print("Number of samples on the inputs: " + str(delta_1.shape) + " and " + str(delta_2.shape) + "do not coincide")

    # Matrix multiplications averaging out in sample size
    K_ff = (1 / tf.cast(tf.shape(delta_1)[ 1 ], tf.float32) ) * tf.matmul( delta_1,  delta_2, transpose_b = True)

    return K_ff # dims are (delta_1 batchsize x delta_2 batchsize)
