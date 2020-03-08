
################################################################################
# Here we include the code for the prior distribution sampler, which consists on
# a BNN and auxiliar functions
################################################################################


from test_code import w_variable_mean, w_variable_variance

import tensorflow as tf
import numpy as np

##### Function: create_bnn
# Create the variables that are going to be used in the prior sampling
# Inputs:
#       prior_bnn_structure     :   list with the shape of the BNN that samples the prior
#       dim_data                :   dimensions of the input data (x)
# Outputs:
#       {}                      :   collection of variables that represent the moments of weights for the prior BNN
#
def create_bnn(dim_data, prior_bnn_structure, n_layers_bnn):

    # Create the means and variances variables for the weights for the prior-sampling BNN

    W1_mean_prior = w_variable_mean([ prior_bnn_structure[ 0 ], dim_data ])
    W1_log_sigma2_prior = w_variable_variance([ prior_bnn_structure[ 0 ], dim_data ])

    bias1_prior  = w_variable_mean([ prior_bnn_structure[ 0 ] ])

    W2_mean_prior = w_variable_mean([ prior_bnn_structure [ 0 ], prior_bnn_structure[ 1 ] ])
    W2_log_sigma2_prior = w_variable_variance([ prior_bnn_structure [ 0 ], prior_bnn_structure[ 1 ] ])

    bias2_prior  = w_variable_mean([ prior_bnn_structure[ 1 ] ])

    W3_mean_prior = w_variable_mean([ prior_bnn_structure [ 1 ], prior_bnn_structure[ 2 ] ])
    W3_log_sigma2_prior = w_variable_variance([ prior_bnn_structure [ 1 ], prior_bnn_structure[ 2 ] ])

    bias3_prior  = w_variable_mean([ prior_bnn_structure[ 2 ] ])

    return { 'W1_mean_prior': W1_mean_prior, 'W1_log_sigma2_prior': W1_log_sigma2_prior, 'bias1_prior': bias1_prior, \
        'W2_mean_prior': W2_mean_prior, 'W2_log_sigma2_prior': W2_log_sigma2_prior, 'bias2_prior': bias2_prior, \
        'W3_mean_prior': W3_mean_prior, 'W3_log_sigma2_prior': W3_log_sigma2_prior, 'bias3_prior': bias3_prior, \
        'n_layers_bnn': n_layers_bnn }

##### Function: get_variables_prior
# Extract the variables employed in the prior sampler
# Inputs:
#       prior_network           :   prior network constructed earlier
# Outputs:
#       []                      :   list of variables in the prior BNN
#
def get_variables_bnn(prior_network):
    return [ prior_network['W1_mean_prior'], prior_network['W1_log_sigma2_prior'], prior_network['bias1_prior'], \
        prior_network['W2_mean_prior'], prior_network['W2_log_sigma2_prior'], prior_network['bias2_prior'], \
        prior_network['W3_mean_prior'], prior_network['W3_log_sigma2_prior'], prior_network['bias3_prior'] ]



###### Function: compute_samples_bnn
# Obtain the samples from p(·) using the previously created weight moments
#
# Inputs:
#   prior_network               :   previously created list with all the parameters for the computations (weights and biases)
#   n_layers                    :   number of layers employed (input onscreen)
#   x_input                     :   input data points to sample the functions in
#   n_samples                   :   number of samples required
#   dim_data                    :   dimension of the data input
#   prior_structure             :   shape of the BNN in a list
#
# Outputs:
#   fx_samples                  :   sampled values of functions f_s(x), dimension = (batchsize x n_samples)
#
def compute_samples_bnn(prior_network, n_layers, x_input, n_samples, dim_data, prior_structure):

    # Number of units in each layer
    n_units1, n_units2 = prior_structure[ 0 ], prior_structure[ 1 ]

    batch_size = tf.shape(x_input)[ 0 ]


    # Extract the weights' moments and the (not random) biases
    W1_mean_prior = tf.reshape(prior_network['W1_mean_prior'], shape = [1, 1, n_units1, dim_data])
    W1_sigma2_prior = tf.reshape( tf.exp( prior_network['W1_log_sigma2_prior'] ), shape = [1, 1, n_units1, dim_data])
    bias1_prior =  prior_network['bias1_prior']

    W2_mean_prior = tf.reshape( prior_network['W2_mean_prior'], shape = [ 1, 1, n_units1, n_units2 ])
    W2_sigma2_prior = tf.reshape( tf.exp( prior_network['W2_log_sigma2_prior'] ), shape = [ 1, 1, n_units1, n_units2 ])
    bias2_prior =  prior_network['bias2_prior']

    W3_mean_prior = tf.reshape( prior_network['W3_mean_prior'], shape = [ 1, 1, n_units2, 1 ])
    W3_sigma2_prior = tf.reshape( tf.exp( prior_network['W3_log_sigma2_prior'] ), shape = [ 1, 1, n_units2, 1 ])
    bias3_prior =  prior_network['bias3_prior']


    # Sample random noise for the sampling of the activations
    noise_1 = tf.random_normal(shape = [ batch_size, n_samples, n_units1 ]) #, seed = seed)
    noise_2 = tf.random_normal(shape = [ batch_size, n_samples, n_units2 ]) #, seed = seed)
    noise_3 = tf.random_normal(shape = [ batch_size, n_samples, 1 ]) #, seed = seed)

    # import pdb; pdb.set_trace()

    # Compute the output of the network employing the local reparametrization trick
    x_expanded = tf.reshape(x_input, shape = [ batch_size, 1, 1, dim_data]) # Extend dimensions
    gamma_1 = tf.reduce_sum( x_expanded * W1_mean_prior, axis = 3)
    diff_1 = tf.reduce_sum( tf.math.square( x_expanded ) * W1_sigma2_prior, axis = 3)

    A1 = (gamma_1 + bias1_prior) + tf.math.multiply(tf.math.sqrt(diff_1), noise_1) # Local reparam. trick first layer
    h1 = tf.nn.leaky_relu(A1) # h1 is batch_size x n_samples x n_units

    if prior_network['n_layers_bnn'] == 2:
        gamma_2 = tf.reduce_sum( tf.expand_dims(h1, -1) * W2_mean_prior, axis = 2)
        diff_2 = tf.reduce_sum( tf.math.square( tf.expand_dims(h1, -1) ) * W2_sigma2_prior, axis = 2)
        A2 = (gamma_2 + bias2_prior) + tf.math.multiply(tf.math.sqrt(diff_2), noise_2)
        h2 = tf.nn.leaky_relu(A2) # dims(h2) are (batch_size x n_samples x n_units)

        gamma_3 = tf.reduce_sum( tf.expand_dims(h2, -1) * W3_mean_prior, axis = 2)
        diff_3 = tf.reduce_sum( tf.math.square( tf.expand_dims(h2, -1) ) * W3_sigma2_prior, axis = 2)
        A3 = (gamma_3 + bias3_prior) + tf.math.multiply(tf.math.sqrt(diff_3), noise_3)
    else:
        gamma_3 = tf.reduce_sum( tf.expand_dims(h1, -1) * W3_mean_prior, axis = 2)
        diff_3 = tf.reduce_sum( tf.math.square( tf.expand_dims(h1, -1) ) * W3_sigma2_prior, axis = 2)
        A3 = (gamma_3 + bias3_prior) + tf.math.multiply(tf.math.sqrt(diff_3), noise_3) # A3 dims are (batchsize x n_samples x 1)


    # MLE solution estimate for the sampled functions
    fx_samples = A3[:,:,0]

    return fx_samples


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Functions to calculate the moments of the functions samples

# Means
#def mean_fx(fx_samples):
    # Calculate the mean of the samples of functions
#    means_fx = tf.reduce_mean(fx_samples, axis = [ 1 ])
#    return means_fx

# Deviations
#def deltas_fx(fx_samples, mean_fx):
    # Calculate the deviation from the mean of the sampled functions' values
#    delta_s_fx = fx_samples - mean_fx # Shape is (batch_size x n_samples)
#    return delta_s_fx
