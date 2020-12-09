################################################################################
# Here we include the code for the prior distribution sampler, which consists on
# a BNN and auxiliar functions
################################################################################


from AIP_bnn import w_variable_mean
from AIP_bnn import w_variable_variance

# from test_code import w_variable_mean_prior as w_variable_mean
# from test_code import w_variable_variance_prior as w_variable_variance

import tensorflow as tf
import numpy as np

##### Function: create_bnn
# Create the variables that are going to be used in the posterior sampling
# Inputs:
#       posterior_bnn_structure     :   list with the shape of the BNN that samples the posterior
#       dim_data                :   dimensions of the input data (x)
# Outputs:
#       {}                      :   collection of variables that represent the moments of weights for the posterior BNN
#
def create_posterior(dim_data, posterior_bnn_structure, n_layers_posterior):

    # Create the means and variances variables for the weights for the posterior-sampling BNN

    W1_mean_posterior = w_variable_mean([ posterior_bnn_structure[ 0 ], dim_data ])
    W1_log_sigma2_posterior = w_variable_variance([ posterior_bnn_structure[ 0 ], dim_data ])

    bias1_posterior  = w_variable_mean([ posterior_bnn_structure[ 0 ] ])

    W2_mean_posterior = w_variable_mean([ posterior_bnn_structure [ 0 ], posterior_bnn_structure[ 1 ] ])
    W2_log_sigma2_posterior = w_variable_variance([ posterior_bnn_structure [ 0 ], posterior_bnn_structure[ 1 ] ])

    bias2_posterior  = w_variable_mean([ posterior_bnn_structure[ 1 ] ])

    W3_mean_posterior = w_variable_mean([ posterior_bnn_structure [ 1 ], posterior_bnn_structure[ 2 ] ])
    W3_log_sigma2_posterior = w_variable_variance([ posterior_bnn_structure [ 1 ], posterior_bnn_structure[ 2 ] ])

    bias3_posterior  = w_variable_mean([ posterior_bnn_structure[ 2 ] ])

    return { 'W1_mean_posterior': W1_mean_posterior, 'W1_log_sigma2_posterior': W1_log_sigma2_posterior, 'bias1_posterior': bias1_posterior, \
        'W2_mean_posterior': W2_mean_posterior, 'W2_log_sigma2_posterior': W2_log_sigma2_posterior, 'bias2_posterior': bias2_posterior, \
        'W3_mean_posterior': W3_mean_posterior, 'W3_log_sigma2_posterior': W3_log_sigma2_posterior, 'bias3_posterior': bias3_posterior, \
        'n_layers_posterior': n_layers_posterior }

##### Function: get_variables_posterior
# Extract the variables employed in the posterior sampler
# Inputs:
#       posterior_network           :   posterior network constructed earlier
# Outputs:
#       []                      :   list of variables in the posterior BNN
#
def get_variables_posterior(posterior_network):
    return [ posterior_network['W1_mean_posterior'], posterior_network['W1_log_sigma2_posterior'], posterior_network['bias1_posterior'], \
        posterior_network['W2_mean_posterior'], posterior_network['W2_log_sigma2_posterior'], posterior_network['bias2_posterior'], \
        posterior_network['W3_mean_posterior'], posterior_network['W3_log_sigma2_posterior'], posterior_network['bias3_posterior'] ]



###### Function: compute_samples_bnn
# Obtain the samples from p() using the previously created weight moments
#
# Inputs:
#   posterior_network           :   previously created list with all the parameters for the computations (weights and biases)
#   n_layers                    :   number of layers employed (input onscreen)
#   x_input                     :   input data points to sample the functions in - list [x,z]
#   n_samples                   :   number of samples required
#   dim_data                    :   dimension of the data input
#   posterior_structure         :   shape of the BNN in a list
#
# Outputs:
#   fx_samples                  :   sampled values of functions f_s(x), dimension = (batchsize x n_samples)
#
def compute_samples_posterior(posterior_network, n_layers, x_input, n_samples, dim_data, posterior_structure):

    # Number of units in each layer
    n_units1, n_units2 = posterior_structure[ 0 ], posterior_structure[ 1 ]

    batch_size = tf.shape(x_input)[ 0 ]

    output_size = posterior_structure[-1]

    # Extract the weights' moments and the (not random) biases
    W1_mean_posterior = tf.reshape(posterior_network['W1_mean_posterior'], shape = [1, dim_data, n_units1 ])
    W1_sigma2_posterior = tf.reshape( tf.exp( posterior_network['W1_log_sigma2_posterior'] ), shape = [1, dim_data, n_units1])
    bias1_posterior =  posterior_network['bias1_posterior']

    W2_mean_posterior = tf.reshape( posterior_network['W2_mean_posterior'], shape = [1, n_units1, n_units2 ])
    W2_sigma2_posterior = tf.reshape( tf.exp( posterior_network['W2_log_sigma2_posterior'] ), shape = [1, n_units1, n_units2 ])
    bias2_posterior =  posterior_network['bias2_posterior']

    W3_mean_posterior = tf.reshape( posterior_network['W3_mean_posterior'], shape = [1, n_units2, output_size ])
    W3_sigma2_posterior = tf.reshape( tf.exp( posterior_network['W3_log_sigma2_posterior'] ), shape = [1, n_units2, output_size ])
    bias3_posterior =  posterior_network['bias3_posterior']


    # Sample random noise for the sampling of the activations
    noise_1 = tf.random_normal(shape = [ n_samples, dim_data, n_units1 ], dtype = tf.float64) #, seed = seed)
    noise_2 = tf.random_normal(shape = [ n_samples, n_units1, n_units2 ], dtype = tf.float64) #, seed = seed)
    noise_3 = tf.random_normal(shape = [ n_samples, n_units2, output_size ], dtype = tf.float64) #, seed = seed)

    # Construct the weights
    W1_posterior = W1_mean_posterior + tf.math.multiply(W1_sigma2_posterior, noise_1)
    W2_posterior = W2_mean_posterior + tf.math.multiply(W2_sigma2_posterior, noise_2)
    W3_posterior = W3_mean_posterior + tf.math.multiply(W3_sigma2_posterior, noise_3)

    A1 =  tf.tensordot(x_input, W1_posterior, axes = [1, 1]) + bias1_posterior
    h1 = tf.nn.leaky_relu(A1) # h1 is batch_size x n_samples x n_units

    if posterior_network['n_layers_posterior'] == 2:


        A2 = tf.reduce_sum( tf.multiply(tf.expand_dims(h1, -1), W2_posterior), axis = 2) + bias2_posterior
        h2 = tf.nn.leaky_relu(A2) # dims(h2) are (batch_size x n_samples x n_units)

        A3 = tf.reduce_sum( tf.multiply(tf.expand_dims(h2, -1), W3_posterior), axis = 2) + bias3_posterior

    else:

        A3 = tf.reduce_sum( tf.multiply(tf.expand_dims(h1, -1), W3_posterior), axis = 2) + bias3_posterior


    # MLE solution estimate for the sampled functions
    fx_samples = A3[:,:,0]

    return fx_samples


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def compute_merged_samples_bnn(prior_network, n_layers, data, n_samples, dim_data, prior_structure, number_ips):

    # Number of units in each layer
    n_units1, n_units2 = prior_structure[ 0 ], prior_structure[ 1 ]

    x_input = tf.concat([data[ 0 ], data[ 1 ]], axis = 0)
    batch_size = tf.shape(x_input)[ 0 ]

    # Extract the weights' moments and the (not random) biases
    W1_mean_prior = tf.reshape(prior_network['W1_mean_prior'], shape = [1, dim_data, n_units1 ])
    W1_sigma2_prior = tf.reshape( tf.exp( prior_network['W1_log_sigma2_prior'] ), shape = [1, dim_data, n_units1])
    bias1_prior =  prior_network['bias1_prior']

    W2_mean_prior = tf.reshape( prior_network['W2_mean_prior'], shape = [1, n_units1, n_units2 ])
    W2_sigma2_prior = tf.reshape( tf.exp( prior_network['W2_log_sigma2_prior'] ), shape = [1, n_units1, n_units2 ])
    bias2_prior =  prior_network['bias2_prior']

    W3_mean_prior = tf.reshape( prior_network['W3_mean_prior'], shape = [1, n_units2, 1 ])
    W3_sigma2_prior = tf.reshape( tf.exp( prior_network['W3_log_sigma2_prior'] ), shape = [1, n_units2, 1 ])
    bias3_prior =  prior_network['bias3_prior']


    # Sample random noise for the sampling of the activations
    noise_1 = tf.random_normal(shape = [ n_samples, dim_data, n_units1 ]) #, seed = seed)
    noise_2 = tf.random_normal(shape = [ n_samples, n_units1, n_units2 ]) #, seed = seed)
    noise_3 = tf.random_normal(shape = [ n_samples, n_units2, 1 ]) #, seed = seed)

    # Construct the weights
    W1_prior = W1_mean_prior + tf.math.multiply(W1_sigma2_prior, noise_1)
    W2_prior = W2_mean_prior + tf.math.multiply(W2_sigma2_prior, noise_2)
    W3_prior = W3_mean_prior + tf.math.multiply(W3_sigma2_prior, noise_3)

    A1 =  tf.tensordot(x_input, W1_prior, axes = [1, 1]) + bias1_prior
    h1 = tf.nn.leaky_relu(A1) # h1 is batch_size x n_samples x n_units

    if prior_network['n_layers_bnn'] == 2:


        A2 = tf.reduce_sum( tf.multiply(tf.expand_dims(h1, -1), W2_prior), axis = 2) + bias2_prior
        h2 = tf.nn.leaky_relu(A2) # dims(h2) are (batch_size x n_samples x n_units)

        A3 = tf.reduce_sum( tf.multiply(tf.expand_dims(h2, -1), W3_prior), axis = 2) + bias3_prior

    else:

        A3 = tf.reduce_sum( tf.multiply(tf.expand_dims(h1, -1), W3_prior), axis = 2) + bias3_prior


    # MLE solution estimate for the sampled functions
    fx_samples = A3[: (batch_size - number_ips),:,0]
    fz_samples = A3[(batch_size - number_ips) :,:,0]

    return fx_samples, fz_samples


