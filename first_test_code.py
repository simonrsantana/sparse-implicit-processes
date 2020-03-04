###############################################################################
##### ADVERSARIAL VARIATIONAL BAYES for regression with alpha divergences #####
###############################################################################
#
# This code is the first (and wrong) iteration for the IP approach 
#
###############################################################################
###############################################################################


# Import the relevant packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

import os
os.chdir(".")

seed = 123

# =============================================================================
# Complete system parameters (VAE, encoder, decoder, data specs, etc.)
# =============================================================================

# File to be analyzed

original_file = sys.argv[ 4 ]

# This is the total number of training samples

n_samples_train = 10
n_samples_test = 100

n_batch = 10
n_epochs = 100

# Structural parameters of the main NN
main_structure = [50, 50]
n_layers_nn = len(main_structure)  # Number of layers in the NN

# Structure of the generator
generator_structure = [50, 50]              # Array with the structure of the generator
n_layers_gen = len(generator_structure)     # Number of layers in the generator
noise_comps_gen = 100                       # Number of gaussian variables inputed in the generator

# Structure of the discriminator
discriminator_structure = [50, 50]              # Structure of the discriminator
n_layers_disc = len(discriminator_structure)    # Number of layers in the discriminator

# Learning rates
primal_rate = 1e-4 # Actual Bayesian NN
dual_rate = 1e-3   # Discriminator

# This is the total number of training samples

kl_factor_limit = int(n_epochs / 10)

ratio_train = 0.9 # Percentage of the data devoted to train

# We define the following two functions to simplify the rest of the code

def w_variable_mean(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = seed) # mean 0 stddev 1
  return tf.Variable(initial)

def w_variable_variance(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = seed) - 5.0 # mean 0 stddev 1
  return tf.Variable(initial)


###############################################################################
##########################   Network structure  ###############################
##############################################################################a

def create_generator(gen_structure, noise_comps_gen, n_layers_gen):

    # Sample noise to input in the generator
    mean_noise = w_variable_mean([ 1, 1, noise_comps_gen ])
    log_var_noise = w_variable_variance([ 1, 1, noise_comps_gen ])

    W1_gen = w_variable_mean([ noise_comps_gen, gen_structure[ 0 ] ])
    bias1_gen  = w_variable_mean([ gen_structure[ 0 ] ])

    W2_gen = w_variable_mean([ gen_structure[ 0 ], gen_structure[ 1 ] ])
    bias2_gen  = w_variable_mean([ gen_structure[ 1 ] ])

    W3_gen = w_variable_mean([ gen_structure[ 1 ], gen_structure[ 2 ] ])
    bias3_gen  = w_variable_mean([ gen_structure[ 2 ] ])
    # gen_structure[ 2 ] is just the total number of weights needed

    return {'mean_noise': mean_noise, 'log_var_noise': log_var_noise, 'W1_gen': W1_gen, 'bias1_gen': bias1_gen, \
        'W2_gen': W2_gen, 'bias2_gen': bias2_gen, 'W3_gen': W3_gen, 'bias3_gen': bias3_gen, 'n_layers_gen': n_layers_gen}

def get_variables_generator(generator):
    return [ generator['mean_noise'], generator['log_var_noise'], generator['W1_gen'], \
        generator['W2_gen'], generator['W3_gen'], generator['bias1_gen'], generator['bias2_gen'], generator['bias3_gen'] ]

def compute_output_generator(generator, batchsize, n_samples, noise_comps_gen):

    mean_noise = generator['mean_noise']
    log_var_noise = generator['log_var_noise']
    W1_gen = generator['W1_gen']
    W2_gen = generator['W2_gen']
    W3_gen = generator['W3_gen']

    bias1_gen = generator['bias1_gen']
    bias2_gen = generator['bias2_gen']
    bias3_gen = generator['bias3_gen']

    pre_init_noise = tf.random_normal(shape = [ batchsize, n_samples, noise_comps_gen ], seed = seed)

    init_noise =  mean_noise + tf.sqrt(tf.exp( log_var_noise )) * pre_init_noise

    # Process the noises through the network

    A1_gen = tf.tensordot(init_noise, W1_gen, axes = [[2], [0]]) + bias1_gen
    h1_gen = tf.nn.leaky_relu(A1_gen)

    if generator['n_layers_gen'] == 1:
        A3_gen = tf.tensordot(h1_gen, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights
    else:
        A2_gen = tf.tensordot(h1_gen, W2_gen, axes = [[2], [0]]) + bias2_gen
        h2_gen = tf.nn.leaky_relu(A2_gen)
        A3_gen = tf.tensordot(h2_gen, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights

    return A3_gen


##################################################################################################
########################## CHANGES TO THE BASIC GENERATOR ########################################
##################################################################################################


def create_new_generator(gen_structure, noise_comps_gen, n_layers_gen):

    # Sample noise to input in the generator
    mean_noise = w_variable_mean([ 1, 1, noise_comps_gen ])
    log_var_noise = w_variable_variance([ 1, 1, noise_comps_gen ])

    W1_gen = w_variable_mean([ noise_comps_gen, gen_structure[ 0 ] ])
    bias1_gen  = w_variable_mean([ gen_structure[ 0 ] ])

    W2_gen = w_variable_mean([ gen_structure[ 0 ], gen_structure[ 1 ] ])
    bias2_gen  = w_variable_mean([ gen_structure[ 1 ] ])

    # We double the output dimension so we have a pair (mean, variance) for each
    # one of the weights of the main NN
    W3_gen = w_variable_mean([ gen_structure[ 1 ], (2 * gen_structure[ 2 ]) ])
    bias3_gen  = w_variable_mean([ (2 * gen_structure[ 2 ]) ])
    # gen_structure[ 2 ] is just the total number of weights needed

    return {'mean_noise': mean_noise, 'log_var_noise': log_var_noise, 'W1_gen': W1_gen, 'bias1_gen': bias1_gen, \
        'W2_gen': W2_gen, 'bias2_gen': bias2_gen, 'W3_gen': W3_gen, 'bias3_gen': bias3_gen, 'n_layers_gen': n_layers_gen}


def compute_new_output_generator(generator, noise_comps_gen, n_samples):

    # This function computes the mean and variances for the weights in the main NN
    # in order to use the local reparametrization trick there

    mean_noise = generator['mean_noise']
    log_var_noise = generator['log_var_noise']

    W1_gen = generator['W1_gen']
    W2_gen = generator['W2_gen']
    W3_gen = generator['W3_gen']

    bias1_gen = generator['bias1_gen']
    bias2_gen = generator['bias2_gen']
    bias3_gen = generator['bias3_gen']

    # The only difference is here, where we do not require the extra dimensions of
    # n_batch and n_samples now
    pre_init_noise = tf.random_normal(shape = [ n_samples, noise_comps_gen ], seed = seed)

    init_noise =  mean_noise + tf.sqrt(tf.exp( log_var_noise )) * pre_init_noise

    # Process the noises through the network

    A1_gen = tf.tensordot(init_noise, W1_gen, axes = [[2], [0]]) + bias1_gen
    h1_gen = tf.nn.leaky_relu(A1_gen)

    if generator['n_layers_gen'] == 1:
        A3_gen = tf.tensordot(h1_gen, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights
    else:
        A2_gen = tf.tensordot(h1_gen, W2_gen, axes = [[2], [0]]) + bias2_gen
        h2_gen = tf.nn.leaky_relu(A2_gen)
        A3_gen = tf.tensordot(h2_gen, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights

    # import pdb; pdb.set_trace()

    # The dimension of the weights-containing tensor is [ 1, n_samples, total_number_weights * 2 ]

    return A3_gen

##################################################################################################
##################################################################################################
##################################################################################################



def create_discriminator(disc_structure, n_layers_disc, total_weights):

    W1_disc = w_variable_mean([ total_weights, disc_structure [ 0 ]])
    bias1_disc = w_variable_mean([ disc_structure [ 0 ] ])

    W2_disc = w_variable_mean([ disc_structure [ 0 ], disc_structure [ 1 ] ])
    bias2_disc = w_variable_mean([ disc_structure [ 1 ] ])

    W3_disc = w_variable_mean([ disc_structure [ 1 ] ,  1 ])
    bias3_disc = w_variable_mean([ 1 ])

    return {'W1_disc': W1_disc, 'bias1_disc': bias1_disc, 'W2_disc': W2_disc, \
        'bias2_disc': bias2_disc, 'W3_disc': W3_disc, 'bias3_disc': bias3_disc, 'n_layers_disc': n_layers_disc}

def get_variables_discriminator(discriminator):
    return [ discriminator['W1_disc'], discriminator['W2_disc'], discriminator['W3_disc'], discriminator['bias1_disc'], \
        discriminator['bias2_disc'], discriminator['bias3_disc'] ]

def compute_output_discriminator(discriminator, weights, n_layers):

    W1_disc = discriminator['W1_disc']
    W2_disc = discriminator['W2_disc']
    W3_disc = discriminator['W3_disc']

    bias1_disc = discriminator['bias1_disc']
    bias2_disc = discriminator['bias2_disc']
    bias3_disc = discriminator['bias3_disc']

    A1_disc = tf.tensordot(weights, W1_disc, axes = [[2], [0]]) + bias1_disc
    h1_disc = tf.nn.leaky_relu(A1_disc)

    if discriminator['n_layers_disc'] == 2:
        A2_disc = tf.tensordot(h1_disc, W2_disc, axes = [[2], [0]]) + bias2_disc
        h2_disc = tf.nn.leaky_relu(A2_disc)

        A3_disc = tf.tensordot(h2_disc, W3_disc, axes = [[2], [0]]) + bias3_disc
    else:
        A3_disc = tf.tensordot(h1_disc, W3_disc, axes = [[2], [0]]) + bias3_disc

    return A3_disc[ :, :, 0 ]


##################################################################################################
##################################### MAIN NEURAL NETWORK ########################################
##################################################################################################


def create_main_NN(main_structure, n_layers):

    # Only the variables of that Network (prior variance and noise variance)

    log_vars_prior = tf.Variable(tf.cast(np.log(1.0), dtype = tf.float32))
    log_sigma2_noise = tf.Variable(tf.cast((1.0 / 10.0), dtype = tf.float32))

    # The biases are not random

    bias_A1 = w_variable_mean([ main_structure[ 0 ] ])
    bias_A2 = w_variable_mean([ main_structure[ 1 ] ])
    bias_A3 = w_variable_mean([ 1 ])

    return {'log_vars_prior': log_vars_prior, 'log_sigma2_noise': log_sigma2_noise, \
        'bias_A1': bias_A1, 'bias_A2': bias_A2, 'bias_A3': bias_A3, 'n_layers': n_layers}

def get_variables_main_NN(network):
    return [ network['log_sigma2_noise'], network['log_vars_prior'], network['bias_A1'], network['bias_A2'], network['bias_A3'] ]

def compute_outputs_main_NN(network, x_input, y_target, weights, alpha, n_samples, dim_data, mean_targets, std_targets):

    log_sigma2_noise = network['log_sigma2_noise']
    n_units1, n_units2 = main_structure[ 0 ], main_structure[ 1 ]
    bias_A1 = network['bias_A1']
    bias_A2 = network['bias_A2']
    bias_A3 = network['bias_A3']

    batch_size = tf.shape(x_input)[ 0 ]
    W1 = tf.reshape(weights[:,:, : (dim_data * n_units1) ], shape = [ batch_size, n_samples, n_units1, dim_data ])

    if network['n_layers'] == 2:
        W2 = tf.reshape(weights[:,:, (dim_data * n_units1) : (dim_data * n_units1 + n_units1 * n_units2) ], \
        shape = [ batch_size, n_samples, n_units1, n_units2 ])
    else:
        W2 = tf.reshape(weights[:,:, (dim_data * n_units1) : ], shape = [ batch_size, n_samples, n_units1, 1 ])

    W3 = tf.reshape(weights[:,:, (dim_data * n_units1 + n_units1 * n_units2) : ], shape = [ batch_size, n_samples, n_units2, 1 ])

    A1 = tf.reduce_sum(tf.reshape(x_input, shape = [ batch_size, 1, 1, dim_data]) * W1, axis = 3) + bias_A1
    h1 = tf.nn.leaky_relu(A1) # h1 is batch_size x n_samples x n_units

    if network['n_layers'] == 2:
        A2 = tf.reduce_sum(tf.reshape(h1, shape = [ batch_size, n_samples, n_units1, 1 ]) * W2, axis = 2) + bias_A2
        h2 = tf.nn.leaky_relu(A2) # h2 is batch_size x n_samples x n_units
        A3 = tf.reduce_sum(tf.reshape(h2, shape = [ batch_size, n_samples, n_units2, 1 ]) * W3, axis = 2) + bias_A3
    else:
        A3 = tf.reduce_sum(tf.reshape(h1, shape = [ batch_size, n_samples, n_units1, 1 ]) * W2, axis = 2) + bias_A3

    # Regression results alpha-energy estimate
    res_train = ( 1.0/alpha) * (- tf.log(tf.cast(n_samples, tf.float32)) + tf.reduce_logsumexp( alpha * \
        (-0.5 * (tf.log(2.0 * np.pi) + log_sigma2_noise + (A3 - tf.reshape(y_target, shape = [ batch_size, 1, 1 ]))**2 / tf.exp(log_sigma2_noise))) , axis = [ 1 ]))


    # This is to compute the test log loglikelihood
    log_prob_data = tf.reduce_sum( (tf.reduce_logsumexp(-0.5 * tf.log(2.0 * np.pi * (tf.exp( log_sigma2_noise ) * std_targets**2)) \
        - 0.5 * (A3 * std_targets + mean_targets - tf.reshape(y_target, shape = [ batch_size, 1, 1 ]))**2 / (tf.exp(log_sigma2_noise) * std_targets**2), axis = [ 1 ])) \
        -tf.log(tf.cast(n_samples, tf.float32)) )

    # Estimate the squared error
    squared_error = tf.reduce_sum((tf.reduce_mean(A3, axis = [ 1 ]) * std_targets + mean_targets - y_target )**2)

    return res_train, squared_error, log_prob_data

###############################################################################
############## COMPUTE THE RESULTS WITH THE NEW NN VERSION ####################
###############################################################################

# This function returns the optimal estimated mean function values and $\Delta$
# for a given input x_input and n_samples of the moments of the weights prior
# distribution (given by the generator)
def compute_res_new_NN(network, x_input, weights, alpha, n_samples, dim_data, total_weights): #, mean_targets, std_targets):

    log_sigma2_noise = network['log_sigma2_noise']
    n_units1, n_units2 = main_structure[ 0 ], main_structure[ 1 ]
    bias_A1 = network['bias_A1']    # The biases are not random
    bias_A2 = network['bias_A2']
    bias_A3 = network['bias_A3']

    batch_size = tf.shape(x_input)[ 0 ]

    # Create the new weight-moments matrices, sampled from the generator
    mean_weights = weights[:,:, : (total_weights)]
    log_sigma2_weights = weights[:,:, (total_weights) :]

    # import pdb; pdb.set_trace()

    W1_mean = tf.reshape(mean_weights[0,:, : (dim_data * n_units1)], shape = [ 1, n_samples, n_units1, dim_data ])
    W1_var = tf.reshape( tf.exp(log_sigma2_weights[0,:, : (dim_data * n_units1)] ), shape = [ 1, n_samples, n_units1, dim_data ])
    #W1 = tf.reshape(weights[:,:, : (dim_data * n_units1) ], shape = [ batch_size, n_samples, n_units1, dim_data ])

    # if network['n_layers'] == 2:
    #     W2 = tf.reshape(weights[:,:, (dim_data * n_units1) : (dim_data * n_units1 + n_units1 * n_units2) ], \
    #     shape = [ batch_size, n_samples, n_units1, n_units2 ])
    # else:
    #     W2 = tf.reshape(weights[:,:, (dim_data * n_units1) : ], shape = [ batch_size, n_samples, n_units1, 1 ])

    if network['n_layers'] == 2:
        W2_mean = tf.reshape(mean_weights[0,:, (dim_data * n_units1) : (dim_data * n_units1 + n_units1 * n_units2) ], \
            shape = [ 1, n_samples, n_units1, n_units2 ])
        W2_var = tf.reshape( tf.exp(log_sigma2_weights[0,:, (dim_data * n_units1) : (dim_data * n_units1 + n_units1 * n_units2) ] ), \
            shape = [ 1, n_samples, n_units1, n_units2 ])

    else:
        W2_mean = tf.reshape(mean_weights[0,:, (dim_data * n_units1) : ], shape = [ 1, n_samples, n_units1, 1 ])
        W2_var = tf.reshape( tf.exp(log_sigma2_weights[0,:, (dim_data * n_units1) :] ), shape = [ 1, n_samples, n_units1, 1 ])


    W3_mean = tf.reshape(mean_weights[0,:, (dim_data * n_units1 + n_units1 * n_units2) : ], shape = [ 1, n_samples, n_units2, 1 ])
    W3_var = tf.reshape( tf.exp(log_sigma2_weights[0,:, (dim_data * n_units1 + n_units1 * n_units2) : ] ), shape = [ 1, n_samples, n_units2, 1 ])


    # Sample random noise for the sampling of the activations
    noise_1 = tf.random_normal(shape = [ batch_size, n_samples, n_units1 ]) #, seed = seed)
    noise_2 = tf.random_normal(shape = [ batch_size, n_samples, n_units2 ]) #, seed = seed)
    noise_3 = tf.random_normal(shape = [ batch_size, n_samples, 1 ]) #, seed = seed)


    # Compute the output of the network employing the local reparametrization trick
    x_expanded = tf.reshape(x_input, shape = [ batch_size, 1, 1, dim_data]) # Extend dimensions
    gamma_1 = tf.reduce_sum( x_expanded * W1_mean, axis = 3)
    diff_1 = tf.reduce_sum( tf.math.square( x_expanded ) * W1_var, axis = 3)

    A1 = (gamma_1 + bias_A1) + tf.math.multiply(tf.math.sqrt(diff_1), noise_1) # Local reparam. trick first layer
    h1 = tf.nn.leaky_relu(A1) # h1 is batch_size x n_samples x n_units

    if network['n_layers'] == 2:
        gamma_2 = tf.reduce_sum( tf.expand_dims(h1, -1) * W2_mean, axis = 2)
        diff_2 = tf.reduce_sum( tf.math.square( tf.expand_dims(h1, -1) ) * W2_var, axis = 2)
        A2 = (gamma_2 + bias_A2) + tf.math.multiply(tf.math.sqrt(diff_2), noise_2)
        h2 = tf.nn.leaky_relu(A2) # dims(h2) are (batch_size x n_samples x n_units)

        gamma_3 = tf.reduce_sum( tf.expand_dims(h2, -1) * W3_mean, axis = 2)
        diff_3 = tf.reduce_sum( tf.math.square( tf.expand_dims(h2, -1) ) * W3_var, axis = 2)
        A3 = (gamma_3 + bias_A3) + tf.math.multiply(tf.math.sqrt(diff_3), noise_3)
    else:
        gamma_3 = tf.reduce_sum( tf.expand_dims(h1, -1) * W3_mean, axis = 2)
        diff_3 = tf.reduce_sum( tf.math.square( tf.expand_dims(h1, -1) ) * W3_var, axis = 2)
        A3 = (gamma_3 + bias_A3) + tf.math.multiply(tf.math.sqrt(diff_3), noise_3) # A3 dims are (batchsize x n_samples x 1)

        #A3 = tf.reduce_sum(tf.reshape(h1, shape = [ batch_size, n_samples, n_units1, 1 ]) * W2, axis = 2) + bias_A3

    # MLE solution estimate for the sampled functions
    means_fx = tf.reduce_mean( A3, axis = [1, 2]) # Means of the functions evaluated at the points of the batch
    delta_s_fx = A3[:,:, 0 ] - means_fx # Shape of delta_s_fx is (batch_size x n_samples)

    # import pdb; pdb.set_trace()

    return means_fx, delta_s_fx



# Calculate the covariance matrix given two delta matrices
def calculate_covariances(delta_1, delta_2):

    # Check whether the number of samples for each delta is the same
    if tf.cast(tf.shape(delta_1)[ 1 ], tf.float32) != tf.cast(tf.shape(delta_2)[ 1 ], tf.float32):
        print("Number of samples on the inputs do not coincide")

    # Matrix multiplications averaging out in sample size
    K_ff = (1 / tf.cast(tf.shape(delta_1)[ 1 ], tf.float32) ) * tf.matmul( delta_1, tf.transpose( delta_2 ) )

    return K_ff # dims are (delta_1 batchsize x delta_2 batchsize)

###############################################################################
###############################################################################
###############################################################################

def main(permutation, split, alpha, layers):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # We load the original dataset

    data = np.loadtxt(original_file)

    # =========================================================================
    #  Parameters of the complete system
    # =========================================================================

    # We obtain the features and the targets

    X = data[ :, range(data.shape[ 1 ] - 1) ]
    y = data[ :, data.shape[ 1 ] - 1 ]

    # We create the train and test sets with 100*ratio % of the data

    data_size = X.shape[ 0 ]
    size_train = int(np.round(data_size * ratio_train))
    total_training_data = size_train

    index_train = permutation[ 0 : size_train ]
    index_test = permutation[ size_train : ]

    X_train = X[ index_train, : ]
    y_train = np.vstack(y[ index_train ])
    X_test = X[ index_test, : ]
    y_test = np.vstack(y[ index_test ])

    # Normalizamos los argumentos

    meanXTrain = np.mean(X_train, axis = 0)
    stdXTrain = np.std(X_train, axis = 0)

    meanyTrain = np.mean(y_train)
    stdyTrain = np.std(y_train)

    X_train = (X_train - meanXTrain) / stdXTrain
    X_test = (X_test - meanXTrain) / stdXTrain
    y_train = (y_train - meanyTrain) / stdyTrain


    # Create the model

    dim_data = X_train.shape[ 1 ]

    # Placeholders for data and number of samples

    x = tf.placeholder(tf.float32, [ None, dim_data ])
    y_ = tf.placeholder(tf.float32, [ None, 1 ])
    n_samples = tf.placeholder(tf.int32, [ 1 ])[ 0 ]
    kl_factor_ = tf.placeholder(tf.float32, [ 1 ])[ 0 ]

    n_layers_nn = n_layers_gen = n_layers_disc = layers

    total_weights = 0
    extended_main_structure = []
    extended_main_structure = main_structure[:]
    extended_main_structure.insert(0, dim_data)

    for i in (range(layers)):
        total_weights += extended_main_structure[ i ] * extended_main_structure[ i+1 ]

    total_weights += extended_main_structure[ layers ]  # Total number of weights used in the main NN

    # import pdb; pdb.set_trace()

    # Update arrays which contain the structure of all the networks needed
    generator_structure.append(total_weights)

    #generator = create_generator(generator_structure, noise_comps_gen, n_layers_gen)
    generator = create_new_generator(generator_structure, noise_comps_gen, n_layers_gen)
    discriminator = create_discriminator(discriminator_structure, n_layers_disc, total_weights )
    main_NN = create_main_NN(main_structure, layers)

    #weights = compute_output_generator(generator, tf.shape(x)[ 0 ], n_samples, noise_comps_gen)
    weights = compute_new_output_generator(generator, noise_comps_gen, n_samples)

    # Obtain the moments of the weights and pass the values through the disc

    #mean_w , var_w = tf.nn.moments(weights, axes = [0, 1])
    # mean_w = weights[:,:, : (total_weights)]
    # log_sigma2_weights = weights[:,:, (total_weights) :]
    # var_w = tf.exp( log_sigma2_weights )

    # mean_w = tf.stop_gradient(mean_w)
    # var_w = tf.stop_gradient(var_w)

    # Normalize real weights

#    norm_weights = (weights - mean_w) / tf.sqrt(var_w)     # There is no need for this since the sampled are moments

    # Generate samples of a normal distribution with the moments of the weights

    # w_gaussian = tf.random_normal(shape = tf.shape(weights), mean = 0, stddev = 1, seed = seed)

    # Obtain the T(z,x) for the real and the sampled weights

#    T_real = compute_output_discriminator(discriminator, norm_weights, layers)
#    T_sampled = compute_output_discriminator(discriminator, w_gaussian, layers)

    # Calculate the cross entropy loss for the discriminator

#    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_real, labels=tf.ones_like(T_real)))
#    d_loss_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_sampled, labels=tf.zeros_like(T_sampled)))

#    cross_entropy_per_point = (d_loss_real + d_loss_sampled) / 2.0

    # Obtain the KL and ELBO

#    logr = -0.5 * tf.reduce_sum(norm_weights**2 + tf.log(var_w) + np.log(2.0 * np.pi), [ 2 ])
#    logz = -0.5 * tf.reduce_sum((weights)**2 / tf.exp(main_NN['log_vars_prior']) + main_NN['log_vars_prior'] + np.log(2.0 * np.pi), [ 2 ])
#    KL = (T_real + logr - logz)

    #res_train, squared_error, log_prob_data = compute_outputs_main_NN(main_NN, x, y_, weights, \
    #    alpha, n_samples, dim_data, meanyTrain, stdyTrain)

    means_fx, delta_s_fx = compute_res_new_NN(main_NN, x, weights, alpha, n_samples, dim_data, total_weights)
    #res_train, squared_error, log_prob_data = compute_base_res_NN(main_NN, x, y_, weights, \
    #    alpha, n_samples, dim_data) #, meanyTrain, stdyTrain)


    # Make the estimates of the ELBO for the primary classifier

    ELBO = (tf.reduce_sum(res_train) - kl_factor_ * tf.reduce_mean(KL) * tf.cast(tf.shape(x)[ 0 ], tf.float32) / \
        tf.cast(total_training_data, tf.float32)) * tf.cast(total_training_data, tf.float32) / tf.cast(tf.shape(x)[ 0 ], tf.float32)

    neg_ELBO = -ELBO
    main_loss = neg_ELBO
    mean_ELBO = ELBO

    # KL y res_train have shape batch_size x n_samples

    mean_KL = tf.reduce_mean(KL)

    # Create the variable lists to be updated

    vars_primal = get_variables_generator(generator) + get_variables_main_NN(main_NN)
    vars_dual = get_variables_discriminator(discriminator)

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(main_loss, var_list = vars_primal)
    train_step_dual = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy_per_point, var_list = vars_dual)

    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
        allow_soft_placement=True, device_count = {'CPU': 1})

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())

        total_ini = time.time()

        # Change the value of alpha to begin exploring using the second value given

        for epoch in range(n_epochs):

            L = 0.0
            ce_estimate = 0.0
            kl = 0.0

            kl_factor = np.minimum(1.0 * epoch / kl_factor_limit, 1.0)

            n_batches_train = int(np.ceil(size_train / n_batch))
            for i_batch in range(n_batches_train):

                ini = time.clock()
                ini_ref = time.time()
                ini_train = time.clock()

                last_point = np.minimum(n_batch * (i_batch + 1), size_train)

                batch = [ X_train[ i_batch * n_batch : last_point, : ] , y_train[ i_batch * n_batch : last_point, ] ]

                sess.run(train_step_dual, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})
                sess.run(train_step_primal, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})

                L += sess.run(mean_ELBO, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})
                kl += sess.run(mean_KL, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})
                ce_estimate += sess.run(cross_entropy_per_point, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})

                sys.stdout.write('.')
                sys.stdout.flush()

                fini_train = time.clock()

            # Test Evaluation

            sys.stdout.write('\n')
            ini_test = time.time()

            # We do the test evaluation RMSE

            errors = 0.0
            LL  = 0.0
            n_batches_to_process = int(np.ceil(X_test.shape[ 0 ] / n_batch))
            for i in range(n_batches_to_process):

                last_point = np.minimum(n_batch * (i + 1), X_test.shape[ 0 ])

                batch = [ X_test[ i * n_batch : last_point, : ] , y_test[ i * n_batch : last_point, ] ]

                errors += sess.run(squared_error, feed_dict={x: batch[0], y_: batch[1], n_samples: n_samples_test}) / batch[ 0 ].shape[ 0 ]
                LL += sess.run(log_prob_data, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_test}) / batch[ 0 ].shape[ 0 ]

            # error_class = errors / float(X_test.shape[ 0 ])
            RMSE = np.sqrt(errors / n_batches_to_process)
            TestLL = LL / n_batches_to_process


            fini_test = time.time()
            fini = time.clock()
            fini_ref = time.time()
            total_fini = time.time()

            string = ('alpha %g batch %g datetime %s epoch %d ELBO %g CROSS-ENT %g KL %g real_time %g cpu_time %g ' + \
                'train_time %g test_time %g total_time %g KL_factor %g LL %g RMSE %g') % \
                (alpha, i_batch, str(datetime.now()), epoch, \
                L / n_batches_train, ce_estimate / n_batches_train, kl / n_batches_train, (fini_ref - \
                ini_ref), (fini - ini), (fini_train - ini_train), (fini_test - ini_test), (total_fini - total_ini), \
                kl_factor, TestLL, RMSE)
            print(string)
            sys.stdout.flush()

            L = 0.0
            ce_estimate = 0.0
            kl = 0.0

        np.savetxt('res_alpha/' + str(alpha) + 'results_error_' + str(split) + '_1layer.txt', [ RMSE ])
        np.savetxt('res_alpha/' + str(alpha) + 'results_ll_' + str(split) + '1layer.txt', [ TestLL ])


if __name__ == '__main__':

    split = int(sys.argv[1])
    alpha = np.float(sys.argv[2])
    layers = int(sys.argv[3])

    # Load the permutation to be used

    available_perm = np.loadtxt('permutations_' + original_file, delimiter = ",", dtype = int)

    # Create the folder to save all the results
    if not os.path.isdir("res_alpha"):
        os.makedirs("res_alpha")

    main(available_perm[split,], split, alpha, layers)
