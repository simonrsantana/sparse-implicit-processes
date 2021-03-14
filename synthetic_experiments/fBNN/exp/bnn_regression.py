import os.path as osp
import sys
import os
import pandas as pd
import random as rd

# import pdb; pdb.set_trace()

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
# sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))) + '/utils')

import tensorflow as tf
import numpy as np

import gpflowSlim as gfs
float_type = gfs.settings.tf_float
import argparse

# from utils.logging import get_logger
# from core.fvi import EntropyEstimationFVI
from core.fvi import KLEstimatorFVI

from utils.nets import get_posterior
# from data import uci_woval
from utils.utils import median_distance_local

from utils.bnn_prior import *


# SRS
from properscoring._crps import *

seed = 123


# SRS New model
parser = argparse.ArgumentParser('Implicit Process prior model') #  'Regression')

# SRS # Changes to the args parser
# parser.add_argument('-d', '--dataset', type=str, default='yacht')
parser.add_argument('-d', '--dataset', type=str) # , default='boston_housing.txt')
parser.add_argument('-s', '--split', type=int) # , default=0)
parser.add_argument('-il', '--init_logstd', type=float, default=-5.)



parser.add_argument('-in', '--injected_noise', type=float, default=0.01)
parser.add_argument('-r', '--rand', type=str, default='uniform')
parser.add_argument('-na', '--n_rand', type=int, default=5)
parser.add_argument('-nh', '--n_hidden', type=int, default=2)
parser.add_argument('-nu', '--n_units', type=int, default=50)
parser.add_argument('-bs', '--batch_size', type=int, default=10)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3) # 0.001)
parser.add_argument('-e', '--epochs', type=int, default=2000)
parser.add_argument('--n_eigen_threshold', type=float, default=0.99)
parser.add_argument('--train_samples', type=int, default=100)

parser.add_argument('--test_samples', type=int, default=500)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=100)

parser.add_argument('--number_ips', type=int, default=50)


args = parser.parse_args()
# logger = get_logger(args.dataset, 'results/regression/%s/'%args.dataset, __file__)
# print = logger.info


# SRS # Control variables

n_epochs = args.epochs
ratio_train = 0.9 # Split the data 90-10 (train-test)

split = args.split  # int( sys.argv[ 1 ] )
original_file = args.dataset # sys.argv[ 2 ]


def run(seed, permutation):
    tf.reset_default_graph()

    ############################## load and normalize data ##############################
    
    # SRS # Use our splits for the datasets
    
    # dataset = uci_woval(args.dataset, seed=seed)
    data = np.loadtxt(original_file).astype(np.float32)  # data_type_numpy)     

    # We obtain the features and the targets

    X = data[ :, range(data.shape[ 1 ] - 1) ]
    y = data[ :, data.shape[ 1 ] - 1 ]

    # We create the train and test sets with 100*ratio % of the data

    data_size = X.shape[ 0 ]
    size_train = int(np.round(data_size * ratio_train))

    index_train = permutation[ 0 : size_train ]
    index_test = permutation[ size_train : ]

    train_x = X[ index_train, : ]
    train_y = np.vstack(y[ index_train ])
    
    # SRS: Randomized test set
    # x_test = x[ index_test, : ]
    # y_test = np.vstack(y[ index_test ])

    # SRS: Predetermined test set (to generate fuller figures)
    data_test = np.loadtxt( 'test_' + original_file ).astype(np.float32)
    test_x = data_test[ :, range(data_test.shape[ 1 ] - 1) ]
    test_y = np.vstack( data_test[ :, data_test.shape[ 1 ] - 1 ])


    #Normalize the input values
    meanXTrain = np.mean(train_x, axis = 0)
    stdXTrain = np.std(train_x, axis = 0)

    # import pdb; pdb.set_trace()

    meanyTrain = np.mean(train_y)
    stdyTrain = np.std(train_y)

    train_x = (train_x - meanXTrain) / stdXTrain
    test_x = (test_x - meanXTrain) / stdXTrain
    train_y = ((train_y - meanyTrain) / stdyTrain)[:,0]
    test_y = ((test_y - meanyTrain) / stdyTrain)[:,0]

    # SRS # Return to the fBNN code

    # import pdb; pdb.set_trace()

    # train_x, test_x, train_y, test_y = dataset.x_train, dataset.x_test, dataset.y_train, dataset.y_test
    std_y_train = stdyTrain # dataset.std_y_train[0]
    N, input_dim = train_x.shape

    lower_ap = np.min(train_x) # np.minimum(np.min(train_x), np.min(test_x))
    upper_ap = np.max(train_x) # np.maximum(np.max(train_x), np.max(test_x))
    mean_x_train, std_x_train = np.mean(train_x, 0), np.std(train_x, 0)

    ############################## setup FBNN model ##############################
    
    # with tf.variable_scope('prior'):
    #     ls = median_distance_local(train_x).astype('float32')
    #     ls[abs(ls) < 1e-6] = 1.
    #     prior_kernel = gfs.kernels.RBF(input_dim=input_dim, name='rbf', lengthscales=ls, ARD=True)

    # with tf.variable_scope('likelihood'):
    #     obs_log1p = tf.get_variable('obs_log1p', shape=[], 
    #                                 initializer=tf.constant_initializer(np.log(np.exp(0.5) - 1.)))
    #     obs_var = tf.nn.softplus(obs_log1p)**2.

    # def rand_generator(*arg):
    #      if args.rand == 'uniform':
    #         return tf.random_uniform(shape=[args.n_rand, input_dim], minval=lower_ap, maxval=upper_ap)
    #     elif args.rand == 'normal':
    #         return mean_x_train + std_x_train * tf.random_normal(shape=[args.n_rand, input_dim])
    #     else:
    #         raise NotImplementedError

    # import pdb; pdb.set_trace()

    # location_inducing_points  = rd.sample(range(0, train_x.shape[ 0 ]),  args.number_ips)
    # inducing_points_initial = train_x[ location_inducing_points, ]


    y_std = stdyTrain
    x_std = stdXTrain

    # Define the function that samples from the prior

    # import pdb; pdb.set_trace()

    prior_generator = bnn_prior_sample( input_dim ) # sample_from_prior


    # SRS New version for the implicit prior
    def rand_generator(*arg):
        return tf.random_uniform(shape=[args.n_rand, 1], minval=np.min(train_x), maxval=np.max(train_x))


    layer_sizes = [input_dim] + [args.n_units] * args.n_hidden + [1]

    # import pdb; pdb.set_trace()

    model = KLEstimatorFVI(
        prior_generator, get_posterior('bnn')(layer_sizes, args.init_logstd), rand_generator=rand_generator,
        obs_var=y_std**2., input_dim=input_dim, n_rand=args.n_rand, injected_noise=args.injected_noise,
        n_eigen_threshold=args.n_eigen_threshold)

    # import pdb; pdb.set_trace()

    # sess = tf.Session()


    # model = EntropyEstimationFVI(
    #     prior_kernel, get_posterior('bnn')(layer_sizes, logstd_init=-2.), rand_generator=rand_generator,
    #     obs_var=obs_var, input_dim=input_dim, n_rand=args.n_rand, injected_noise=args.injected_noise)
    # model.build_prior_gp(init_var=0.1, inducing_points = inducing_points_initial)
    # update_op = tf.group(model.infer_latent, model.infer_likelihood)
    # with tf.control_dependencies([update_op]):
    #     train_op = tf.assign(obs_log1p, tf.maximum(tf.maximum(
    #         tf.to_float(tf.log(tf.exp(model.gp_var**0.5) - 1.)), obs_log1p), tf.log(tf.exp(0.05) - 1.)))

    ############################## training #######################################
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # SRS
    # gp_epochs = 5000 # SRS: They have this pretraining procedure (maybe we should shorten it to make it fair)
    # for epoch in range(gp_epochs):
    #     feed_dict = {model.x_gp:train_x, model.y_gp:train_y, model.learning_rate_ph: 0.003}
    #     _, loss, gp_var = sess.run([model.infer_gp, model.gp_loss, model.gp_var], feed_dict=feed_dict)
    #     if epoch % args.print_interval == 0:
    #         print('>>> Seed {:5d} >>> Pretrain GP Epoch {:5d}/{:5d}: Loss={:.5f} | Var={:.5f}'.format(
    #             seed, epoch, gp_epochs, loss, gp_var))


    epoch_iters = max(N // args.batch_size, 1)
    for epoch in range(1, args.epochs+1):

        # import pdb; pdb.set_trace()

        indices = np.random.permutation(N)
        train_x, train_y = train_x[indices], train_y[indices]
        for iter in range(epoch_iters):
            x_batch = train_x[iter * args.batch_size: (iter + 1) * args.batch_size]
            y_batch = train_y[iter * args.batch_size: (iter + 1) * args.batch_size]

            feed_dict = {model.x: x_batch, model.y: y_batch, model.learning_rate_ph: args.learning_rate,
                         model.n_particles: args.train_samples}
            feed_dict.update(model.default_feed_dict())

            # import pdb; pdb.set_trace()

            # sess.run(train_op, feed_dict=feed_dict)

            _, elbo_sur, kl_sur, logll = sess.run(
                 [model.infer_latent, model.elbo, model.kl_surrogate, model.log_likelihood],
                 feed_dict=feed_dict)

            # import pdb; pdb.set_trace()

            # sess.run(model.infer_latent, feed_dict=feed_dict)


        if epoch % args.test_interval == 0 or epoch == args.epochs:

            feed_dict = {model.x: test_x, model.y: test_y, model.n_particles: args.test_samples}

#             rmse, lld, ov = sess.run([model.eval_rmse, model.eval_lld, obs_var], feed_dict=feed_dict)

            # SRS # Include the CRPS estimation in the metrics through the ENSEMBLE method

            rmse_tmp, lld_tmp, fore = sess.run([model.eval_rmse, model.eval_lld, model.func_x], feed_dict=feed_dict)

            # import pdb; pdb.set_trace()

            rmse = rmse_tmp * std_y_train
            lld = lld_tmp - np.log(std_y_train)

            corrected_y = test_y*stdyTrain + meanyTrain

            crps_estimate = crps_ensemble( corrected_y, np.transpose(fore)*stdyTrain + meanyTrain )
            mean_crps = np.mean( crps_estimate ) 

            with open("prints/print_fbnn_" + str(split) + "_" +  original_file, "a") as res_file:
                res_file.write('>>> Seed %g >>> Epoch %g/%g | rmse=%g | lld=%g | crps=%g' % (seed, epoch, args.epochs, rmse, lld, mean_crps) + "\n")

            # '''
            print('>>> Seed {:5d} >>> Epoch {:5d}/{:5d} | rmse={:.5f} | lld={:.5f} | crps={:.5f}'.format(
                seed, epoch, args.epochs, rmse, lld, mean_crps))
            # '''

            if epoch == args.epochs:

                # import pdb; pdb.set_trace() # XXX XXX XXX XXX XXX 

                prior_samples = sess.run(model.func_x_pred_prior, feed_dict = \
                    {model.x_pred_prior:test_x, model.n_particles: args.test_samples}) # , model.y_gp:test_y, model.x_gp:test_x})

                merge_y_pred = pd.concat([ pd.DataFrame(test_x), pd.DataFrame(test_y), pd.DataFrame( np.transpose(fore)*stdyTrain + meanyTrain) ], axis = 1) 
                merge_prior_samples = pd.concat([ pd.DataFrame(test_x), pd.DataFrame(test_y), pd.DataFrame(prior_samples.T) ], axis = 1)
                merge_train_data = pd.concat([ pd.DataFrame(train_x), pd.DataFrame(train_y) ], axis = 1)

                merge_y_pred.to_csv('res/fbnn_y_forecasted_' + str(split) + '.txt', index = False, header = False)
                merge_prior_samples.to_csv('res/fbnn_prior_samples_' + str(split) + '.txt', index = False, header = False)
                merge_train_data.to_csv('res/fbnn_train_data_' + str(split) + '.txt', index = False, header = False)



            if epoch == args.epochs:
                return rmse, lld, mean_crps
         

        # np.savetxt('res/fbnn_rmse_' + str(split) + '.txt', [ rmse ])
        # np.savetxt('res/fbnn_ll_' + str(split) + '.txt', [ lld ])
        # np.savetxt('res/fbnn_ll_' + str(split) + '.txt', [ mean_crps ])
 



if __name__ == '__main__':


    # Create a folder to store the screen prints
    if not os.path.isdir("prints"):
        os.makedirs("prints")

    # Create a file to store the results of the run (or empty the previously existing one)
    if os.path.isfile("prints/print_fbnn_" + str(split) + "_" +  original_file):
        with open("prints/print_fbnn_" + str(split) + "_" +  original_file, "w") as res_file:
           res_file.close()

    # Create the folder to save all the results
    if not os.path.isdir("res"):
        os.makedirs("res")

    if not os.path.isdir("res/"):
        os.makedirs("res/")

    n_run = 10
    rmse_results, lld_results, crps_results = [], [], []

    # import pdb; pdb.set_trace()   
 
    # SRS # Read the permutations file only once
    available_perm = np.loadtxt('permutations_' + original_file, delimiter = ",", dtype = int)

    # SRS: THIS VERSION DOES NOT USE DIFFERENT SEEDS

    rmse, ll, crps = run(seed, available_perm[split,:])
    rmse_results = rmse
    lld_results = ll
    crps_results = crps

    print("BNN test rmse = {}".format(rmse_results)) # , rmse_results / n_run ** 0.5))
    print("BNN test log likelihood = {}".format(lld_results)) # , np.std(lld_results) / n_run ** 0.5))
    print("BNN test CRPS = {}".format(crps_results))# , np.std(crps_results) / n_run ** 0.5))
    # print('NOTE: Test result above output mean and std. errors')

    np.savetxt('res/fbnn_rmse_' + str(split) + '.txt', [ rmse_results ]) #, np.std(rmse_results) / n_run ** 0.5  ])
    np.savetxt('res/fbnn_ll_' + str(split) + '.txt', [ lld_results ]) # , np.std(lld_results) / n_run ** 0.5 ])
    np.savetxt('res/fbnn_crps_' + str(split) + '.txt', [ crps_results ]) # , np.std(crps_results) / n_run ** 0.5 ])


    '''
    for seed in range(1, n_run+1):
        rmse, ll, crps = run(seed, available_perm[split,:])
        rmse_results.append(rmse)
        lld_results.append(ll)
        crps_results.append(crps)

    print("BNN test rmse = {}/{}".format(np.mean(rmse_results), np.std(rmse_results) / n_run ** 0.5))
    print("BNN test log likelihood = {}/{}".format(np.mean(lld_results), np.std(lld_results) / n_run ** 0.5))
    print("BNN test CRPS = {}/{}".format(np.mean(crps_results), np.std(crps_results) / n_run ** 0.5))
    print('NOTE: Test result above output mean and std. errors')

    np.savetxt('res/fbnn_final_rmse_' + str(split) + '.txt', [ np.mean(rmse_results), np.std(rmse_results) / n_run ** 0.5  ])
    np.savetxt('res/fbnn_final_ll_' + str(split) + '.txt', [ np.mean(lld_results), np.std(lld_results) / n_run ** 0.5 ])
    np.savetxt('res/fbnn_final_crps_' + str(split) + '.txt', [ np.mean(crps_results), np.std(crps_results) / n_run ** 0.5 ])
    '''

