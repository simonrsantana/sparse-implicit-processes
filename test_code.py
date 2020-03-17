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
import random

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

n_batch = 100
n_epochs = 100

ratio_train = 0.9 # Percentage of the data devoted to train

# Number of inducing points
number_IP = 50

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
neural_sampler_structure = [50, 50, number_IP]  # Structure of the neural sampler
n_layers_ns = len(neural_sampler_structure)     # Number of layers in the NS
noise_comps_ns = 100                            # Number of gaussian variables used as input in the NS



# Structure of the discriminators
discriminator_structure = [50, 50]                  # Structure of the discriminator for the approx. distribution
n_layers_disc = len(discriminator_structure)        # Number of layers in the discriminator


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

    data = np.loadtxt(original_file).astype(np.float32)

    # =========================================================================
    #  Parameters of the complete system
    # =========================================================================

    # We obtain the features and the targets

    X = data[ :, range(data.shape[ 1 ] - 1) ]
    y = data[ :, data.shape[ 1 ] - 1 ]

    # We create the train and test sets with 100*ratio % of the data

    data_size = X.shape[ 0 ]
    size_train = int(np.round(data_size * ratio_train))

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

    dim_data = X_train.shape[ 1 ]


    # Placeholders for data and number of samples
    x = tf.placeholder(tf.float32, [ None, dim_data ])
    y_ = tf.placeholder(tf.float32, [ None, 1 ])
    n_samples = tf.placeholder(tf.int32, [ 1 ])[ 0 ]
    kl_factor_ = tf.placeholder(tf.float32, [ 1 ])[ 0 ]


    # Introduce the inducing points as variables initialized in a random subset of X
    index_shuffle = [x for x in range(size_train)]
    random.shuffle(index_shuffle)
    index_IP = index_shuffle[ : number_IP ]
    z = tf.Variable(X_train[ index_IP, : ])         # Initialize the inducing points at random values of X_train

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
    discriminator_prior = create_discriminator(discriminator_structure, number_IP, n_layers_disc)
    discriminator_approx = create_discriminator(discriminator_structure, number_IP, n_layers_disc)
    bnn = create_bnn(dim_data, bnn_structure, n_layers_bnn)


    # Obtain values for the functions sampled at the points
    fx = compute_samples_bnn(bnn, n_layers_bnn, x, n_samples, dim_data, bnn_structure)
    fz = compute_samples_bnn(bnn, n_layers_bnn, z, n_samples, dim_data, bnn_structure)

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

    additive_factor = 1e-5
    inv_K_zz = K_zz + tf.eye(tf.shape(K_zz)[ 0 ]) * additive_factor   # To ensure it is invertible we add a small noise

    # Obtain samples from the approximating distribution
    samples_qu = compute_output_ns(neural_sampler, n_samples, noise_comps_ns)    # right now, dims are (number_IP, n_samples)

    #################
    ### LOSS TERM ###
    #################

    # Estimate the moments of the p(f(x)|f(z))
    inv_term = tf.linalg.inv( inv_K_zz )
    cov_product = tf.matmul(K_xz, inv_term)

    #### WE ARE DOING THE MEAN THROUGH THE SAMPLES OF f(z) (= u)
    # mean_est = tf.reduce_mean( tf.expand_dims(m_fx, -1) + tf.tensordot(cov_product,  (samples_qu - tf.expand_dims(m_fz, -1)), axes = [[1], [0]]), axis = 1)
    # Instead of using u from the prior p(·), we use u from the approximate distribution q(·) for the differences in the expression
    mean_est = tf.expand_dims(m_fx, -1) + tf.tensordot(cov_product,  (samples_qu - tf.expand_dims(m_fz, -1)), axes = [[1], [0]])     # Missing the mean of the evaluated points thus far
    cov_est = K_xx - tf.matmul( cov_product, K_xz, transpose_b = True )

    sample_pf_noise = tf.random_normal(shape = [ tf.shape(x)[0], n_samples ] )
    # samples_pf = mean_est +  tf.tensordot(cov_est, sample_pf_noise, axes = [[1], [0]])
    samples_pf = mean_est +  tf.matmul(cov_est, sample_pf_noise)                # Shape is (batchsize, n_samples)

    log_sigma2_noise = tf.Variable(tf.cast(1.0 / 100.0, dtype = tf.float32))

    # Final loss for the data term
    loss_train = (1.0/alpha) * (-tf.log(tf.cast(n_samples, tf.float32 )) + tf.reduce_logsumexp( -0.5 * alpha * (np.log( 2 * np.pi ) + log_sigma2_noise  + (samples_pf - y_)**2 / tf.exp(log_sigma2_noise) ), axis = [ 1 ]))


    ###############
    ### KL TERM ###
    ###############

    # Estimate the means and variances of the samples of p(u) and q(u) to feed normalized samples to the discriminators
    mean_p, var_p = tf.nn.moments(fz, axes = [ 1 ])
    mean_q, var_q = tf.nn.moments(samples_qu, axes = [ 1 ])

    tf.stop_gradient(mean_p); tf.stop_gradient(var_p);
    tf.stop_gradient(mean_q); tf.stop_gradient(var_q);

    # Normalize the samples
    norm_p = (tf.transpose(fz) - mean_p) / var_p
    norm_q = (tf.transpose(samples_qu) - mean_q) / var_q

    # Construct the auxiliar gaussian distributions
    samples_p_gaussian = tf.random_normal(shape = tf.shape(norm_p), mean = 0, stddev = 1)  #, seed = seed)
    samples_q_gaussian = tf.random_normal(shape = tf.shape(norm_q), mean = 0, stddev = 1)  #, seed = seed)

    # Obtain the results from the discriminators
    T_real_p = compute_output_discriminator(discriminator_prior, norm_p, layers)
    T_sampled_p = compute_output_discriminator(discriminator_prior, samples_p_gaussian, layers)

    T_real_q = compute_output_discriminator(discriminator_approx, norm_q, layers)
    T_sampled_q = compute_output_discriminator(discriminator_approx, samples_q_gaussian, layers)

    # Obtain the cross entropy for the results of each discriminator
    p_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_real_p, labels=tf.ones_like(T_real_p)))
    p_loss_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_sampled_p, labels=tf.zeros_like(T_sampled_p)))

    q_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_real_q, labels=tf.ones_like(T_real_q)))
    q_loss_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_sampled_q, labels=tf.zeros_like(T_sampled_q)))

    # CE per point
    cross_entropy_p = (p_loss_real + p_loss_sampled) / 2.0
    cross_entropy_q = (q_loss_real + q_loss_sampled) / 2.0

    # Calculate the rest of the KL terms
    log_p_gaussian = -0.5 * tf.reduce_mean(np.log(2 * np.pi) + tf.log(var_p) + norm_p**2, axis = [ 1 ])
    log_q_gaussian = -0.5 * tf.reduce_mean(np.log(2 * np.pi) + tf.log(var_q) + norm_q**2, axis = [ 1 ])

    ##########################
    #### MEAN OR SUM ???? ####
    ##########################

    KL = T_real_p - T_real_q + log_p_gaussian - log_q_gaussian


    ######################
    # Calculate the ELBO #
    ######################


    ELBO = tf.reduce_sum( loss_train ) - kl_factor_ * tf.reduce_mean( KL ) * tf.cast(tf.shape(x)[ 0 ], tf.float32) / tf.cast(size_train, tf.float32)

    neg_ELBO = -ELBO
    mean_ELBO = ELBO

    mean_KL = tf.reduce_mean(KL)

    vars_primal = get_variables_ns(neural_sampler) + get_variables_bnn(bnn) + [ log_sigma2_noise ]
    vars_disc_prior = get_variables_discriminator(discriminator_prior)
    vars_disc_approx = get_variables_discriminator(discriminator_approx)

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(neg_ELBO, var_list = vars_primal)
    train_step_disc_prior = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy_p, var_list = vars_disc_prior)
    train_step_disc_approx = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy_q, var_list = vars_disc_approx)

    # HAY ALGO MAL EN LOS SIGNOS Y LAS SUMAS DE LA ELBO, NECESITA REVISIÓN


    # import pdb; pdb.set_trace()


    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
        allow_soft_placement=True, device_count = {'CPU': 1})

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())

        total_ini = time.time()

        # Change the value of alpha to begin exploring using the second value given

        for epoch in range(n_epochs):

            # Initialize the containers for the needed quantities
            L = 0.0
            ce_estimate_prior = 0.0
            ce_estimate_approx = 0.0
            kl = 0.0
            loss = 0.0

            # Annealing factor for the KL term
            kl_factor = np.minimum(1.0 * epoch / kl_factor_limit, 1.0)

            ini = time.clock()
            ini_ref = time.time()
            ini_train = time.clock()

            # Train the model
            n_batches_train = int(np.ceil(size_train / n_batch))
            for i_batch in range(n_batches_train):

                kl_factor = 1

                last_point = np.minimum(n_batch * (i_batch + 1), size_train)

                batch = [ X_train[ i_batch * n_batch : last_point, : ] , y_train[ i_batch * n_batch : last_point, ] ]

                # import pdb; pdb.set_trace()

                sess.run(train_step_disc_prior, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})
                sess.run(train_step_disc_approx, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})
                sess.run(train_step_primal, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})

                # Overwrite the important quantities for the printed results
                L += sess.run(ELBO, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, kl_factor_: kl_factor})
                loss += sess.run(tf.reduce_sum(loss_train), feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, kl_factor_: kl_factor})
                kl += sess.run(mean_KL, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})
                ce_estimate_prior += sess.run(cross_entropy_p, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})
                ce_estimate_approx += sess.run(cross_entropy_q, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})

                sys.stdout.write('.')
                sys.stdout.flush()

                fini_train = time.clock()

            fini = time.clock()
            fini_ref = time.time()

            sys.stdout.write('\n')
            sys.stdout.flush()

            # Store the training results while running
            with open("prints/print_IPs_" + str(alpha) + "_" + str(split) + "_" +  original_file, "a") as res_file:
                res_file.write('alpha %g datetime %s epoch %d ELBO %g Loss %g KL %g real_time %g cpu_train_time %g annealing_factor %g' % (alpha, str(datetime.now()), epoch, L, loss, kl, (fini_ref - ini_ref), (fini - ini), kl_factor) + "\n")

        import pdb; pdb.set_trace()

        # Test Evaluation
        sys.stdout.write('\n')
        # ini_test = time.time()

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

        # fini_test = time.time()

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

    # Create a folder to store the screen prints
    if not os.path.isdir("prints"):
        os.makedirs("prints")

    # Create a file to store the results of the run (or empty the previously existing one)
    if os.path.isfile("prints/print_IPs_" + str(alpha) + "_" + str(split) + "_" +  original_file):
        with open("prints/print_IPs_" + str(alpha) + "_" + str(split) + "_" +  original_file, "w") as res_file:
           res_file.close()

    # Create the folder to save all the results
    if not os.path.isdir("res_alpha"):
        os.makedirs("res_alpha")

    main(available_perm[split,], split, alpha, layers)
