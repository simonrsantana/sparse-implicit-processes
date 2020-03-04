###############################################################################
############### AADM code modified to employ Implicit Processes ###############
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


# =============================================================================
# We define the following two functions to simplify the rest of the code
# =============================================================================

def w_variable_mean(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = seed) # mean 0 stddev 1
  return tf.Variable(initial)

def w_variable_variance(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = seed) - 5.0 # mean 0 stddev 1
  return tf.Variable(initial)


# Import relevant functions concerning the different distributions of the model
from aux_functions import *         # Functions that calculate moments given sampled values
from BNN_prior import *             # Prior BNN model functions
from neural_sampler import *        # Neural sampler that draws instances from q(·)
from discriminators import *        # NNs that discriminate samples between distributions

import os
os.chdir(".")

seed = 123

# =============================================================================
# Complete system parameters
# =============================================================================

# File to be analyzed
original_file = sys.argv[ 4 ]

# This is the total number of training/test samples, epochs and batch sizes
n_samples_train = 20
n_samples_test = 100

n_batch = 10
n_epochs = 100

ratio_train = 0.9 # Percentage of the data devoted to train

# Parameters concerning the annealing factor of the KL divergence
kl_factor_limit = int(n_epochs / 10)

# Learning rates
primal_rate = 1e-4 # Main BNN and neural sampler parameters
dual_rate = 1e-3   # Discriminators


# STRUCTURE OF ALL THE NNs IN THE SYSTEM

# Structural parameters of the BNN
bnn_structure = [50, 50, 1]                     # Structure of the BNN
n_layers_bnn = len(bnn_structure)               # Number of layers in the NN

# Structure of the neural sampler
neural_sampler_structure = [50, 50, 1]          # Structure of the neural sampler
n_layers_ns = len(neural_sampler_structure)     # Number of layers in the NS
noise_comps_ns = 100                            # Number of gaussian variables used as input in the NS
n_samples_qu = 20                               # ????????????????????????? TBD

total_sampled_values_prior = 20                 # ????????????????????????? TBD
total_sampled_values_approx = 20                # ????????????????????????? TBD


# Structure of the discriminators
discriminator_structure_approx = [50, 50]                   # Structure of the discriminator for the approx. distribution
n_layers_disc_approx = len(discriminator_structure_approx)  # Number of layers in the discriminator

discriminator_structure_prior = [50, 50]                    # Structure of the discriminator for the prior distribution
n_layers_disc_prior = len(discriminator_structure_prior)    # Number of layers in the discriminator


###############################################################################
##########################   Network structure  ###############################
##############################################################################a

############################### GENERATOR ######################################


# THERE IS NO NEED FOR A GENERATOR AS IS. ITS FUNCTION IS COVERED BY THE CODE IN
# FILE neural_sampler.py WHERE SAMPLES FROM THE APPROXIMATING q(·) DISTRIBUTION
# ARE OBTAINED


############################# DISCRIMINATOR ####################################


# THE DISCRIMINATORS ARE NOW INCLUDED IN THE disc_NN.py FILE, WHERE THERE ARE NOW
# TWO OF THEM: ONE FOR THE APPROXIMATE DISTRIBUTION SAMPLES AND ANOTHER ONE FOR
# SAMPLES FROM THE PRIOR DISTRIBUTION


############################ MAIN NEURAL NETWORK ###############################


#  NOW THIS IS DONE THROUGH THE BNN_prior.py CODE, CREATING THE NETWORK AND
# SAMPLING FROM IT


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

    x = tf.placeholder(tf.float32, [ None, dim_data ]) ################################ DEFINE THE INDUCING POINTS Z
    z = tf.placeholder(tf.float32, [ None, dim_data ])                          # PLACEHOLDER FOR THE I.P.
    y_ = tf.placeholder(tf.float32, [ None, 1 ])
    n_samples = tf.placeholder(tf.int32, [ 1 ])[ 0 ]
    kl_factor_ = tf.placeholder(tf.float32, [ 1 ])[ 0 ]

    n_layers_bnn = n_layers_ns = n_layers_disc_prior = n_layers_disc_approx = layers

    # Estimate the total number of weights needed in the BNN
    total_weights_bnn = 0
    extended_main_structure = []
    extended_main_structure = bnn_structure[:]
    extended_main_structure.insert(0, dim_data)

    for i in (range(layers)):
        total_weights_bnn += extended_main_structure[ i ] * extended_main_structure[ i+1 ]

    total_weights_bnn += extended_main_structure[ layers ]  # Total number of weights used in the BNN


    # Create arrays that will contain the structure of all the components needed
    neural_sampler = create_neural_sampler(neural_sampler_structure, noise_comps_ns, n_layers_ns)
    discriminator_prior = create_discriminator_prior(discriminator_structure_prior, total_sampled_values_prior, n_layers_disc_prior)
    discriminator_approx = create_discriminator_approx(discriminator_structure_approx, total_sampled_values_approx, n_layers_disc_approx)
    bnn = create_bnn(dim_data, bnn_structure, n_layers_bnn)


    # Obtain values for the functions sampled at the points
    fx = compute_samples_bnn(bnn, n_layers_bnn, x, n_samples, dim_data, bnn_structure)
    fz = compute_samples_bnn(bnn, n_layers_bnn, z, n_samples, dim_data, bnn_structure)

    #########################################################################################
    # THE INDUCING POINTS REMAIN TO BE DETERMINED STILL - SUBSET OF X (PLACEHOLDER FOR NOW) #
    #########################################################################################

    # Means
    m_fx = mean_f(fx)
    m_fz = mean_f(fz)

    # Deviations
    delta_fx = deltas_f(fx, m_fx)
    delta_fz = deltas_f(fz, m_fz)

    # Covariance functions
    K_xx = calculate_covariances(delta_fx, delta_fx)
    K_xz = calculate_covariances(delta_fx, delta_fz)    # dim = (batchsize(x), batchsize(z))
    K_zz = calculate_covariances(delta_fz, delta_fz)

    # Estimate the moments of the p(f(x)|f(z))
    log_sigma2_gp = w_variable_variance([ 1 ])
    inv_term = tf.linalg.inv( K_zz + tf.eye(tf.shape(K_zz)[ 0 ]) * tf.exp(log_sigma2_gp))
    cov_product = tf.matmul(K_xz, inv_term)

    mean_est = cov_product * (y_ - tf.expand_dims(m_fx, -1))                    # Missing the mean of the evaluated points thus far
    cov_est = K_xx - tf.matmul( cov_product, K_xz, transpose_b = True )

    # ALL THAT IS LEFT HERE IS TO COMPUTE THE FINAL EXPECTED VALUE FOR THE LOSS

    # Obtain samples from the approximating distribution
    samples_qu = compute_output_ns(neural_sampler, n_samples_qu, noise_comps_ns)    # right now, dims are (20(=n_samples_qu) x 1(=ns_struc[2]))

    # (?) SAMPLES of q(u): They should be of shape (batchsize(z), n_samples_train) (?)
    import pdb; pdb.set_trace()

    # Obtain the moments of the weights and pass the values through the disc

    #weights = compute_output_generator(generator, tf.shape(x)[ 0 ], n_samples, noise_comps_gen)

    # mean_w , var_w = tf.nn.moments(weights, axes = [0, 1])
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
