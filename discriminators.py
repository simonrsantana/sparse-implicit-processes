

################################################################################
################################################################################
# Here we define the DISCRIMINATOR NETWORK that will classify samples from the
# approximating distribution and the prior against standard gaussians
################################################################################
################################################################################


from test_code import w_variable_mean, w_variable_variance

import tensorflow as tf
import numpy as np

################################################################################
# Discriminator for samples from functions in the main code
################################################################################

def create_discriminator_fs(disc_structure_fs, total_sampled_values, n_layers_disc_fs):

    W1_disc_fs = w_variable_mean([ total_sampled_values, disc_structure_fs[ 0 ]])
    bias1_disc_fs = w_variable_mean([ disc_structure_fs[ 0 ] ])

    W2_disc_fs = w_variable_mean([ disc_structure_fs[ 0 ], disc_structure_fs[ 1 ] ])
    bias2_disc_fs = w_variable_mean([ disc_structure_fs[ 1 ] ])

    W3_disc_fs = w_variable_mean([ disc_structure_fs[ 1 ] ,  1 ])
    bias3_disc_fs = w_variable_mean([ 1 ])

    return {'W1_disc_fs': W1_disc_fs, 'bias1_disc_fs': bias1_disc_fs, 'W2_disc_fs': W2_disc_fs, \
        'bias2_disc_fs': bias2_disc_fs, 'W3_disc_fs': W3_disc_fs, 'bias3_disc_fs': bias3_disc_fs, 'n_layers_disc_fs': n_layers_disc_fs,
        'total_sampled_values': total_sampled_values}

def get_variables_discriminator_fs(discriminator_fs):
    return [ discriminator_fs['W1_disc_fs'], discriminator_fs['W2_disc_fs'], discriminator_fs['W3_disc_fs'], discriminator_fs['bias1_disc_fs'], \
        discriminator_fs['bias2_disc_fs'], discriminator_fs['bias3_disc_fs'] ]

def compute_output_discriminator_fs(discriminator_fs, sampled_values, n_layers):

    total_sampled_values = discriminator_fs['total_sampled_values']
    W1_disc_fs = discriminator_fs['W1_disc_fs']
    W2_disc_fs = discriminator_fs['W2_disc_fs']
    W3_disc_fs = discriminator_fs['W3_disc_fs']

    bias1_disc_fs = discriminator_fs['bias1_disc_fs']
    bias2_disc_fs = discriminator_fs['bias2_disc_fs']
    bias3_disc_fs = discriminator_fs['bias3_disc_fs']

    A1_disc_fs = tf.tensordot(sampled_values, W1_disc_fs, axes = [[2], [0]]) + bias1_disc_fs
    h1_disc_fs = tf.nn.leaky_relu(A1_disc_fs)

    if discriminator_fs['n_layers_disc_fs'] == 2:
        A2_disc_fs = tf.tensordot(h1_disc_fs, W2_disc_fs, axes = [[2], [0]]) + bias2_disc_fs
        h2_disc_fs = tf.nn.leaky_relu(A2_disc_fs)

        A3_disc_fs = tf.tensordot(h2_disc_fs, W3_disc_fs, axes = [[2], [0]]) + bias3_disc_fs
    else:
        A3_disc_fs = tf.tensordot(h1_disc_fs, W3_disc_fs, axes = [[2], [0]]) + bias3_disc_fs

    return A3_disc_fs[ :, :, 0 ]
