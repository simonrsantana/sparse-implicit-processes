
# Experimental prior neural sampler architecture

import numpy as np
import tensorflow as tf
import pdb
from aux_functions import calculate_covariances


from AIP_ns import w_variable_mean

'''
#### function to declare tf variables
def ns_variables(shape):
    initial = tf.random.normal(shape = shape, mean = 0.0, stddev = 0.1)
    return tf.Variable(initial)
'''

#### Function: create_neural_sampler
#### this func. create the tf variables of the model
def create_posterior(dim_data, posterior_ns_structure, n_layers_ns):
    
    # creating parameters of the neural net for the stochastic components
    # m is both the dim. of the noise input and the output of this net
    m = posterior_ns_structure[-1]
    
    W1_posterior_noise = w_variable_mean([m, posterior_ns_structure[0]])
    bias1_posterior_noise = w_variable_mean([posterior_ns_structure[0]])

    W2_posterior_noise = w_variable_mean([posterior_ns_structure[0], posterior_ns_structure[1]])
    bias2_posterior_noise = w_variable_mean([posterior_ns_structure[1]])

    W3_posterior_noise = w_variable_mean([posterior_ns_structure[1], posterior_ns_structure[2]])
    bias3_posterior_noise = w_variable_mean([posterior_ns_structure[2]])

    # creating parameters of the neural net for the input
    W1_posterior_in = w_variable_mean([dim_data, posterior_ns_structure[0]])
    bias1_posterior_in = w_variable_mean([posterior_ns_structure[0]])

    W2_posterior_in = w_variable_mean([posterior_ns_structure[0], posterior_ns_structure[1]])
    bias2_posterior_in = w_variable_mean([posterior_ns_structure[1]])

    W3_posterior_in = w_variable_mean([posterior_ns_structure[1], posterior_ns_structure[2]])
    bias3_posterior_in = w_variable_mean([posterior_ns_structure[2]])

    return {'W1_posterior_noise': W1_posterior_noise, 'bias1_posterior_noise':bias1_posterior_noise,
            'W2_posterior_noise': W2_posterior_noise, 'bias2_posterior_noise':bias2_posterior_noise,
            'W3_posterior_noise': W3_posterior_noise, 'bias3_posterior_noise':bias3_posterior_noise,
            'W1_posterior_in': W1_posterior_in, 'bias1_posterior_in':bias1_posterior_in,
            'W2_posterior_in': W2_posterior_in, 'bias2_posterior_in':bias2_posterior_in,
            'W3_posterior_in': W3_posterior_in, 'bias3_posterior_in':bias3_posterior_in}



# Extract the variables employed in the neural sampler
# Inputs:
#       ns                      :   network created previously for the sampler
# Outputs:
#       []                      :   list of variables
def get_variables_ns_posterior(ns):

    # Extract the variables from the neural sampler as a list
    return [ ns["W1_posterior_noise"], ns["bias1_posterior_noise"], \
    	ns["W2_posterior_noise"], ns["bias2_posterior_noise"], \
        ns["W3_posterior_noise"], ns["bias3_posterior_noise"], \
        ns["W1_posterior_in"], ns["bias1_posterior_in"], \
        ns["W2_posterior_in"], ns["bias2_posterior_in"], \
        ns["W3_posterior_in"], ns["bias3_posterior_in"] ]




#### function to compute the 2 different outputs of the neural sampler
def compute_samples_posterior(ns_posterior, n_layers_ns_posterior, X, n_samples, dim_data,
                             posterior_structure):

    #X=X.astype(np.float32)
    #Z=Z.astype(np.float32)
    
    #pdb.set_trace()
    m = posterior_structure[-1]
    #these are the vectors "epsilon", dim=m, unique for all data points
    # but there will be a number n_samples of them 
    noise = tf.random_normal(shape = [n_samples, m], dtype = tf.float64)

    # loading parameters of the neural net for the stochastic components
    W1_noise = ns_posterior['W1_posterior_noise']
    bias1_noise = ns_posterior['bias1_posterior_noise']
    W2_noise = ns_posterior['W2_posterior_noise']
    bias2_noise = ns_posterior['bias2_posterior_noise']
    W3_noise = ns_posterior['W3_posterior_noise']
    bias3_noise = ns_posterior['bias3_posterior_noise']

    # loading parameters of the neural net for the input
    W1_in = ns_posterior['W1_posterior_in']
    bias1_in = ns_posterior['bias1_posterior_in']
    W2_in = ns_posterior['W2_posterior_in']
    bias2_in = ns_posterior['bias2_posterior_in']
    W3_in = ns_posterior['W3_posterior_in']
    bias3_in = ns_posterior['bias3_posterior_in']

    # computing the output of the stochastic network
    # dimension = n_samples x m
    A1_noise = tf.matmul(noise, W1_noise) +  bias1_noise
    h1_noise = tf.nn.leaky_relu(A1_noise)
    A2_noise = tf.matmul(h1_noise, W2_noise) + bias2_noise
    h2_noise = tf.nn.leaky_relu(A2_noise)
    output_net_noise = tf.matmul(h2_noise, W3_noise) + bias3_noise
    
    
    #computing the outputs of the network of inputs
    # for X as well as for inducing points Z
    # dimension = (batch_size x m) or (number_IP x m)
    #global e_X, e_w1, e_bias
    A1_in = tf.matmul(X, W1_in) +  bias1_in
    #e_X = X
    #e_w1 = W1_in
    #e_bias = bias1_in
    h1_in = tf.nn.leaky_relu(A1_in)
    A2_in = tf.matmul(h1_in, W2_in) + bias2_in
    h2_in = tf.nn.leaky_relu(A2_in)
    output_net_in = tf.matmul(h2_in, W3_in) + bias3_in

    # v_i and a_i in the notation of AVB paper (expr. III.1)
    # dimensions:
    # dim(v_i) = N_samples x m,
    # dim(a_i_x) = batch_size x m,
    # dim(a_i_z) = (number_IP x m) 

    fx = tf.matmul(output_net_in, output_net_noise, transpose_b = True)

    return fx

'''

#### function to compute the moments of the implicit processes
def compute_moments_ns_prior(outputs, m):

    vi = outputs['v_i']
    ai_x = outputs['a_i_x']
    ai_z = outputs['a_i_z']

    # compute the combined output of the total network
    fx = tf.matmul(ai_x, vi, transpose_b = True)
    fz = tf.matmul(ai_z, vi, transpose_b = True)
    
    # 1st and 2nd moments across samples only concern the stochastic part
    # dimension = [m,1]
    mean_vi = tf.reduce_mean(vi, axis=[0])
    cov_vi = calculate_covariances(tf.transpose(vi-mean_vi),
                                   tf.transpose(vi-mean_vi))
    mean_vi = tf.reshape(mean_vi, shape=[m,1])

    mean_fx = tf.matmul(ai_x, mean_vi)
    mean_fz = tf.matmul(ai_z, mean_vi)

    cov_xx = tf.matmul(ai_x,tf.matmul(cov_vi,ai_x, transpose_b=True))
    cov_zz = tf.matmul(ai_z,tf.matmul(cov_vi,ai_z, transpose_b=True))
    cov_xz = tf.matmul(ai_x,tf.matmul(cov_vi,ai_z, transpose_b=True))
    cov_zx = tf.matmul(ai_z,tf.matmul(cov_vi,ai_x, transpose_b=True))
    
    return {'fx':fx, 'fz':fz, 'mean_fx':mean_fx, 'mean_fz':mean_fz,
            'cov_xx':cov_xx, 'cov_xz':cov_xz, 'cov_zx':cov_zx, 'cov_zz':cov_zz}

''' 

    
    


    
