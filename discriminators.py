

################################################################################
################################################################################
# Here we define the DISCRIMINATOR NETWORK that will classify samples from the
# approximating distributions and the priors
################################################################################
################################################################################


# We will need TWO DISCRIMINATOR NNs in order to apply twice adaptive contrast
# to samples both from the approximating distribution q(·) and the prior p(·)


from test_code import w_variable_mean, w_variable_variance

import tensorflow as tf
import numpy as np

################################################################################
# Discriminator for samples of the approximate distribution q(·)
################################################################################

def create_discriminator_approx(disc_structure_approx, total_sampled_values, n_layers_disc_approx):

    W1_disc_approx = w_variable_mean([ total_sampled_values, disc_structure_approx[ 0 ]])
    bias1_disc_approx = w_variable_mean([ disc_structure_approx[ 0 ] ])

    W2_disc_approx = w_variable_mean([ disc_structure_approx[ 0 ], disc_structure_approx[ 1 ] ])
    bias2_disc_approx = w_variable_mean([ disc_structure_approx[ 1 ] ])

    W3_disc_approx = w_variable_mean([ disc_structure_approx[ 1 ] ,  1 ])
    bias3_disc_approx = w_variable_mean([ 1 ])

    return {'W1_disc_approx': W1_disc_approx, 'bias1_disc_approx': bias1_disc_approx, 'W2_disc_approx': W2_disc_approx, \
        'bias2_disc_approx': bias2_disc_approx, 'W3_disc_approx': W3_disc_approx, 'bias3_disc_approx': bias3_disc_approx, 'n_layers_disc_approx': n_layers_disc_approx,
        'total_sampled_values': total_sampled_values}

def get_variables_discriminator_approx(discriminator_approx):
    return [ discriminator_approx['W1_disc_approx'], discriminator_approx['W2_disc_approx'], discriminator_approx['W3_disc_approx'], discriminator_approx['bias1_disc_approx'], \
        discriminator_approx['bias2_disc_approx'], discriminator_approx['bias3_disc_approx'] ]

def compute_output_discriminator_approx(discriminator_approx, sampled_values, n_layers):

    total_sampled_values = discriminator_approx['total_sampled_values']
    W1_disc_approx = discriminator_approx['W1_disc_approx']
    W2_disc_approx = discriminator_approx['W2_disc_approx']
    W3_disc_approx = discriminator_approx['W3_disc_approx']

    bias1_disc_approx = discriminator_approx['bias1_disc_approx']
    bias2_disc_approx = discriminator_approx['bias2_disc_approx']
    bias3_disc_approx = discriminator_approx['bias3_disc_approx']

    A1_disc_approx = tf.tensordot(sampled_values, W1_disc_approx, axes = [[2], [0]]) + bias1_disc_approx
    h1_disc_approx = tf.nn.leaky_relu(A1_disc_approx)

    if discriminator_approx['n_layers_disc_approx'] == 2:
        A2_disc_approx = tf.tensordot(h1_disc_approx, W2_disc_approx, axes = [[2], [0]]) + bias2_disc_approx
        h2_disc_approx = tf.nn.leaky_relu(A2_disc_approx)

        A3_disc_approx = tf.tensordot(h2_disc_approx, W3_disc_approx, axes = [[2], [0]]) + bias3_disc_approx
    else:
        A3_disc_approx = tf.tensordot(h1_disc_approx, W3_disc_approx, axes = [[2], [0]]) + bias3_disc_approx

    return A3_disc_approx[ :, :, 0 ]



################################################################################
# Discriminator for samples of the prior distribution p(·)
################################################################################

def create_discriminator_prior(disc_structure_prior, total_sampled_values, n_layers_disc_prior):

    W1_disc_prior = w_variable_mean([ total_sampled_values, disc_structure_prior[ 0 ]])
    bias1_disc_prior = w_variable_mean([ disc_structure_prior[ 0 ] ])

    W2_disc_prior = w_variable_mean([ disc_structure_prior[ 0 ], disc_structure_prior[ 1 ] ])
    bias2_disc_prior = w_variable_mean([ disc_structure_prior[ 1 ] ])

    W3_disc_prior = w_variable_mean([ disc_structure_prior[ 1 ] ,  1 ])
    bias3_disc_prior = w_variable_mean([ 1 ])

    return {'W1_disc_prior': W1_disc_prior, 'bias1_disc_prior': bias1_disc_prior, 'W2_disc_prior': W2_disc_prior, \
        'bias2_disc_prior': bias2_disc_prior, 'W3_disc_prior': W3_disc_prior, 'bias3_disc_prior': bias3_disc_prior, 'n_layers_disc_prior': n_layers_disc_prior,
        'total_sampled_values': total_sampled_values}

def get_variables_discriminator_prior(discriminator_prior):
    return [ discriminator_prior['W1_disc_prior'], discriminator_prior['W2_disc_prior'], discriminator_prior['W3_disc_prior'], discriminator_prior['bias1_disc_prior'], \
        discriminator_prior['bias2_disc_prior'], discriminator_prior['bias3_disc_prior'] ]

def compute_output_discriminator_prior(discriminator_prior, sampled_values, n_layers):

    total_sampled_values = discriminator_prior['total_sampled_values']
    W1_disc_prior = discriminator_prior['W1_disc_prior']
    W2_disc_prior = discriminator_prior['W2_disc_prior']
    W3_disc_prior = discriminator_prior['W3_disc_prior']

    bias1_disc_prior = discriminator_prior['bias1_disc_prior']
    bias2_disc_prior = discriminator_prior['bias2_disc_prior']
    bias3_disc_prior = discriminator_prior['bias3_disc_prior']

    A1_disc_prior = tf.tensordot(sampled_values, W1_disc_prior, axes = [[2], [0]]) + bias1_disc_prior
    h1_disc_prior = tf.nn.leaky_relu(A1_disc_prior)

    if discriminator_prior['n_layers_disc_prior'] == 2:
        A2_disc_prior = tf.tensordot(h1_disc_prior, W2_disc_prior, axes = [[2], [0]]) + bias2_disc_prior
        h2_disc_prior = tf.nn.leaky_relu(A2_disc_prior)

        A3_disc_prior = tf.tensordot(h2_disc_prior, W3_disc_prior, axes = [[2], [0]]) + bias3_disc_prior
    else:
        A3_disc_prior = tf.tensordot(h1_disc_prior, W3_disc_prior, axes = [[2], [0]]) + bias3_disc_prior

    return A3_disc_prior[ :, :, 0 ]
