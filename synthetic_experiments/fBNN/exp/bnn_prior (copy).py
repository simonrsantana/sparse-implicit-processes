import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse

# from utils.logging import get_logger
from core.fvi import KLEstimatorFVI
from utils.nets import get_posterior
from utils.utils import default_plotting_new as init_plotting
# from utils.prior_gen import PiecewiseLinear, PiecewiseConstant
from utils.bnn_prior import *
from utils.data_precision_type import *


# SRS
# This model will be the same as piecewise but adapted to an implicit BNN prior
parser = argparse.ArgumentParser('Implicit')
# parser = argparse.ArgumentParser('Piecewise')

# SRS
parser.add_argument('-d', '--dataset', type=str) # , default='p_lin')  # Now this arg will be the dataset !!!!!
parser.add_argument('-s', '--split', type=int) # , default=0)

parser.add_argument('-m', '--method', type=str, default='bnn') # Add the model default
parser.add_argument('-r', '--rand', type=str, default='uniform')

parser.add_argument('-N', '--N', type=int, default=40) # This is a parameter related to the generation of the data
parser.add_argument('-in', '--injected_noise', type=float, default=0.00)
parser.add_argument('-il', '--init_logstd', type=float, default=-5.)
parser.add_argument('-na', '--n_rand', type=int, default=100)
parser.add_argument('-nh', '--n_hidden', type=int, default=2) # number of hidden layers
parser.add_argument('-nu', '--n_units', type=int, default=50) # number of units in the hidden layers
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

# SRS ¡Hay muchas épocas puestas por defecto!
# Hay que hacer el entrenamiento por batches
parser.add_argument('-e', '--epochs', type=int, default=2000) # !!!!! 30000) # epochs in training (NEEDS TO BE DIVIDED BY BATCHES STILL)
parser.add_argument('--n_eigen_threshold', type=float, default=0.9)

parser.add_argument('--train_samples', type=int, default=100) 

parser.add_argument('--test_samples', type=int, default=500)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=2000)

parser.add_argument('--seed', type=int, default=8)

parser.add_argument('--data_type_numpy', type=str, default = "np.float32")

args = parser.parse_args()

# logger = get_logger(args.dataset, 'results/%s/'%(args.dataset), __file__)
# print = logger.info

# SRS Parametros que necesitamos para el modelo
n_epochs = args.epochs 
ratio_train = 0.9

split = args.split
original_file = args.dataset


tf.set_random_seed(args.seed)
np.random.seed(args.seed)

# SRS Parametros que necesitaban para hacer la evaluación con los procesos implícitos lineales a trozos
# xmin, xmax = 0., 1.
# ymin, ymax = 0., 1.
# lambda_, y_std = 3., 0.02



############################## setup the data for the model ##############################

# SRS Nuestro sampleador requiere de tratar los datos antes para poder pasárselos aquí
# Atendiendo al código de VIP, parece que hay que pasarle todo el rango de definición de las X y las y.
# Por tanto, leemos todos los datos y sampleamos las funciones en todo el conjunto de train

# We obtain the features and the targets

data = np.loadtxt(original_file).astype(data_type_numpy)

X = data[ :, range(data.shape[ 1 ] - 1) ]
y = data[ :, data.shape[ 1 ] - 1 ]

# We create the train and test sets with 100*ratio % of the data

data_size = X.shape[ 0 ]
size_train = int(np.round(data_size * ratio_train))


# Load the permutation to be used

available_perm = np.loadtxt('permutations_' + original_file, delimiter = ",", dtype = int)

permutation = available_perm[ split, ]


# Create the indices for the train and test sets 

index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

X_train = X[ index_train, : ]
y_train = np.vstack(y[ index_train ])

X_test = X[ index_test, : ]
y_test = np.vstack(y[ index_test ])

# If you want to use a predetermined test set, load it here

# data_test = np.loadtxt( 'test_' + original_file ).astype(data_type_numpy)
# X_test = data_test[ :, range(data_test.shape[ 1 ] - 1) ]
# y_test = np.vstack( data_test[ :, data_test.shape[ 1 ] - 1 ])

#Normalize the input values
meanXTrain = np.mean(X_train, axis = 0)
stdXTrain = np.std(X_train, axis = 0)

meanyTrain = np.mean(y_train)
stdyTrain = np.std(y_train)

train_x = (X_train - meanXTrain) / stdXTrain
test_x = (X_test - meanXTrain) / stdXTrain
train_y = (y_train - meanyTrain) / stdyTrain
test_y = (y_test - meanyTrain) / stdyTrain


y_std = stdyTrain
x_std = stdXTrain


input_dim = train_x.shape[ 1 ]


# Define the domain for the X 
lower_ap = np.minimum(np.min(train_x), np.min(test_x))
upper_ap = np.maximum(np.max(train_x), np.max(test_x))



############################## setup FBNN model ##############################

# SRS Ahora construimos un prior que sea una BNN y que solo dependa en la entrada de la entrada y las muestras deseadas

prior_structure = [50, 50, 1]


# def sample_from_prior(x, n_samples):
#     return bnn_prior_sample(x, n_samples)

prior_generator = bnn_prior_sample # sample_from_prior


# prior_generator = dict(p_lin=PiecewiseLinear, p_const=PiecewiseConstant)[args.dataset](
#     lambda_, xmin, xmax, ymin, ymax).sample

# import pdb; pdb.set_trace()

# Original rand_generator
def rand_generator(*arg):
    return tf.random_uniform(shape=[args.n_rand, 1], minval=np.min(train_x), maxval=np.max(train_x))



# SRS From regression.py
# def rand_generator(*arg):
#     if args.rand == 'uniform':
#         return tf.random_uniform(shape=[args.n_rand, input_dim], minval=lower_ap, maxval=upper_ap)
#     elif args.rand == 'normal':
#         return mean_x_train + std_x_train * tf.random_normal(shape=[args.n_rand, input_dim])
#     else:
#         raise NotImplementedError

# For the posterior model
layer_sizes = [1] + [args.n_units] * args.n_hidden + [1]
model = KLEstimatorFVI(
    prior_generator, get_posterior('bnn')(layer_sizes, args.init_logstd), rand_generator=rand_generator,
    obs_var=y_std**2., input_dim=1, n_rand=args.n_rand, injected_noise=args.injected_noise,
    n_eigen_threshold=args.n_eigen_threshold)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

############################## generate and normalize data ##############################

# SRS:  WE HAVE DONE THIS ALREADY 


# train_x = tf.concat([
#     tf.random_uniform(minval=0., maxval=0.2, shape=[args.N // 2, 1]),
#     tf.random_uniform(minval=0.8, maxval=1.0, shape=[args.N // 2, 1])
# ], axis=0)
# test_x = tf.reshape(tf.linspace(xmin, xmax, 2000), [-1, 1])
# train_y = tf.squeeze(prior_generator(tf.concat([train_x, test_x], axis=0), 1))
# train_y, test_y = train_y[:args.N], train_y[args.N:]
# train_x_sample, train_y_sample, test_x_sample, test_y_sample = sess.run([train_x, train_y, test_x, test_y])
# train_y_sample = train_y_sample + y_std * np.random.normal(size=train_y_sample.shape)


############################## training ##############################

for epoch in range(1, 1+args.epochs):
    feed_dict = {model.x: train_x, model.y: train_y, model.learning_rate_ph: args.learning_rate,
                 model.n_particles: args.train_samples}
    feed_dict.update(model.default_feed_dict())

    _, elbo_sur, kl_sur, logll = sess.run(
        [model.infer_latent, model.elbo, model.kl_surrogate, model.log_likelihood],
        feed_dict=feed_dict)

    if epoch % args.print_interval == 0:
        print('>>> Epoch {:5d}/{:5d} | elbo_sur={:.5f} | logLL={:.5f} | kl_sur={:.5f}'.format(
            epoch, args.epochs, elbo_sur, logll, kl_sur))

    if epoch % args.test_interval == 0:
        y_pred = sess.run(model.func_x_pred,
                          feed_dict={model.x_pred: np.reshape(test_x, [-1, 1]),
                                     model.n_particles: args.test_samples})
        mean_y_pred, std_y_pred = np.mean(y_pred, 0), np.std(y_pred, 0)

        plt.clf()
        figure = plt.figure(figsize=(8, 5.5), facecolor='white')
        init_plotting()

        ## plt.plot(test_x_sample.squeeze(), mean_y_pred, 'steelblue', label='Mean function')
        plt.fill_between(test_x.squeeze(),
                         mean_y_pred - 3. * std_y_pred,
                         mean_y_pred + 3. * std_y_pred, alpha=0.2, color='b')
        for id in range(4):
            plt.plot(test_x.squeeze(), y_pred[id], 'g')
        plt.scatter(train_x, train_y, c='tomato', zorder=10, label='Observations')

        plt.grid(True)
        plt.tick_params(axis='both', bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        plt.tight_layout()
        plt.xlim([xmin, xmax])
        # plt.ylim([ymin, ymax])
        plt.tight_layout()

        plt.savefig('results/{}/plot_epoch{}.pdf'.format(args.dataset, epoch))
