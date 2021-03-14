import numpy as np
from matplotlib.pylab import pcolor
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from _crps import *
#### clear variables and ops and everything from TF
tf.reset_default_graph()

n_train = x_train.shape[1]

n_test = x_test.shape[1]

n_dim = x_train.shape[0]



structure = np.concatenate([[n_dim],structure]) # BNN structure


#### create and initialize auxiliary random inputs for BNN
def init_aux_placeholder(structure,n_bt):
    tf.set_random_seed(rs)
    np.random.seed(rs)
    Z_w = []
    Z_b = []
    for j in range(len(structure)-1):
        Z_w.append(tf.placeholder(tf.float32, shape=(n_bt,structure[j+1],structure[j])))
        Z_b.append(tf.placeholder(tf.float32, shape=(n_bt,structure[j+1],1)))

    return Z_w, Z_b

def init_aux_value(structure,n_bt):
    tf.set_random_seed(rs)
    np.random.seed(rs)
    Z_w = []
    Z_b = []
    for j in range(len(structure)-1):
        Z_w.append(np.random.normal(0,1,(n_bt,structure[j+1],structure[j])))
        Z_b.append(np.random.normal(0,1,(n_bt,structure[j+1],1)))

    return Z_w,Z_b


#### construct BNN
def bnn(X,Z_w,Z_b,structure,n_bt,n_train):
    tf.set_random_seed(rs)
    np.random.seed(rs)
    #save

    n_train = X.get_shape()[1].value

    def weight_mean_variable(shape):
      return tf.get_variable("w_m",shape,initializer=tf.random_normal_initializer(mean = 0.0, stddev = 0.05)) # SRS 0.01))

    def weight_std_variable(shape):
      return tf.get_variable("w_std",shape,initializer=tf.random_normal_initializer(mean=0.0, stddev = 0.25)) # SRS mean=0.01,stddev = 0.0))


    def bias_mean_variable(shape):
      return tf.get_variable("b_m",shape,initializer=tf.random_normal_initializer(mean = 0.01,stddev = 0.3 )) # SRS 1))

    def bias_std_variable(shape):
      return tf.get_variable("b_std",shape,initializer=tf.random_normal_initializer(mean=0.0,stddev = 0.0)) # SRS 0.1))

    W_m = []
    W_sig = []
    b_m = []
    b_sig = []
    with tf.name_scope('BayesNN'):
        for l in range(len(structure)-1):
            with tf.variable_scope("layer%(a)d" %{"a":l}):
                W_m.append(weight_mean_variable((structure[l+1],structure[l])))
                W_sig.append(weight_std_variable((structure[l+1],structure[l])))
                b_m.append(bias_mean_variable((structure[l+1],1)))
                b_sig.append(0.0 * bias_std_variable((structure[l+1],1)))

    X = tf.expand_dims(X,0)
    X = tf.tile(X,[n_bt,1,1])
    H = [X]

    # import pdb; pdb.set_trace()


    for l in range(len(structure)-1):
        if l == len(structure)-2:
            S = (tf.einsum('abc,acd->abd',tf.add(W_sig[l]*Z_w[l],W_m[l]),H[l])+ tf.add( tf.multiply(b_sig[l],Z_b[l]),b_m[l] ))
            H.append(S)
        else:
            # import pdb; pdb.set_trace()
            S = tf.nn.tanh(tf.einsum('abc,acd->abd',tf.add(W_sig[l]*Z_w[l],W_m[l]),H[l])+ tf.add( tf.multiply(b_sig[l],Z_b[l]),b_m[l] ))
            H.append(S)

    y = H[len(structure)-1][:,0,:]

    # import pdb; pdb.set_trace()
    y = tf.transpose(y)
    y_orig = y

    # m_y = tf.reshape(tf.reduce_mean(y,1),[n_train,1])
    # SRS Correct the expression
    m_y = tf.expand_dims( tf.reduce_mean(y, 1), -1 )

    # import pdb; pdb.set_trace()

    y = y - m_y
    y = y/np.sqrt(n_bt-1)
    return y, m_y, y_orig


#### VIP wake phase objective
def VIP_loss_marginal(Y,ft_train,m_ft_train,init,alpha,A):
    tf.set_random_seed(rs)
    np.random.seed(rs)
    n_samp = A.get_shape()[1].value
    n_train = tf.cast( tf.shape(ft_train)[ 0 ], tf.float32)
    # n_train = ft_train.get_shape()[0].value
    c = ft_train.get_shape()[1].value
    with tf.variable_scope("output"):
        tau = tf.get_variable("tau",[1],initializer=tf.constant_initializer(init[0]))
        yita = tf.get_variable("yita",[1],initializer=tf.constant_initializer(init[1]))
        SL =  tf.get_variable("SL",(c,c),initializer=tf.random_normal_initializer(mean = 0.0, stddev = 0.1))
        m =  tf.get_variable("m",(c,1),initializer=tf.random_normal_initializer(mean = 0.0, stddev = 0.1))

    S = tf.linalg.inv(tf.matmul(SL,tf.transpose(SL)) + tf.eye(c)/(tau**2) + tf.eye(c)*1e-5) # SRS Introduce jitter to make it invertible
    miu = tf.matmul(S,m)
    lgdetS = tf.linalg.logdet( S )  # tf.log(tf.matrix_determinant(S)) # SRS change to use a more robust expression
    temp = -0.5*(tau**2)*(tf.transpose(Y)-m_ft_train-tf.matmul(ft_train,A))

    # import pdb; pdb.set_trace()

    llh_regularization = 1*tf.reduce_logsumexp( ( tf.reduce_sum(temp**2,axis = 0) - n_train*0.5*tf.log(2*np.pi/(tau**2)) ) ) - np.log(n_samp)
    uc2 = tf.reduce_sum(tf.matmul(ft_train,S)*ft_train,axis=1,keep_dims=True)+(1/(tau**2*alpha**2))

    # instead of using Monte carlo estimation, we analytically compute alpha energy, since likelihood is gaussian
    alpha_energy = 1/alpha*(-0.5*tf.reduce_sum((tf.matmul(ft_train,miu)-tf.transpose(Y))**2/(uc2)) \
                          -0.5*tf.reduce_sum(tf.log(uc2))-0.5*n_train * tf.log(2 * np.pi)) +\
                          1/alpha*n_train*(alpha-1)/2*tf.log(2*np.pi/(tau**2)) + 1/alpha*n_train/2*np.log(alpha)

    # import pdb; pdb.set_trace()

    KL_term = 0.5*( tf.matmul(tf.transpose(miu),miu) + tf.linalg.trace(S)-lgdetS - c)
    llh = alpha_energy  -KL_term - llh_regularization
    nllh=-llh/n_train

    # import pdb; pdb.set_trace()
    return nllh, tau, yita, S

### generate VIP prediction
def VIP_pred(ft_train,Y,ft_test,m_ft_train,m_ft_test,tau,yita):
    tf.set_random_seed(rs)
    np.random.seed(rs)
    sig = 1/tau

    # import pdb; pdb.set_trace()

    c = ft_train.get_shape()[1].value

    Kinv_est = tf.linalg.inv(tf.matmul(tf.transpose(ft_train),ft_train) + tf.eye(c)*(sig**2+1/(yita**2))) # + tf.eye(c)*1e-5) # SRS add jitter to make it invertible
    pred_projection = tf.matmul(Kinv_est,tf.matmul(tf.transpose(ft_train),tf.transpose(Y)-m_ft_train))
    y_pred = tf.matmul(ft_test,pred_projection)+m_ft_test

    # SRS change in the following line 
    n_test = y_pred.get_shape()[0]
    uc2 = tf.reduce_sum(tf.matmul(ft_test,Kinv_est)*ft_test,axis=1,keep_dims=True)*(sig**2+1/(yita**2))+(1*(sig**2))

    # import pdb; pdb.set_trace()

    return y_pred,uc2

### prediction loss (not training loss)
def VIP_loss(y_pred,uc2,y_test):
    tf.set_random_seed(rs)
    np.random.seed(rs)
    n_test = y_pred.get_shape()[0].value

    # SRS change
    n_test = tf.cast(tf.shape(y_pred)[ 0 ], tf.float32)

    # import pdb; pdb.set_trace()
    
    # XXX Correction of the llh expression  

    # llh = (-0.5 * tf.reduce_sum((y_pred-tf.transpose(y_test))**2/(uc2)) -0.5 * tf.reduce_sum(tf.log(uc2))- n_test * np.log(2 * np.pi) * 0.5) / n_test

    llh = (-0.5 * tf.reduce_sum((y_pred-tf.transpose(y_test))**2/(uc2)) -0.5 * tf.reduce_sum(tf.log(uc2))- 0.5 * n_test * np.log(2 * np.pi)) / n_test





    nllh=-llh
    return nllh


### input placeholder setup
X_train = tf.placeholder(tf.float32, shape=(n_dim, None ))    # SRS Change to accept the size of the batch
Y_train = tf.placeholder(tf.float32, shape=(1, None))         # SRS Change to accept the size of the batch
X_test = tf.placeholder(tf.float32, shape=(n_dim,n_test))
Y_test = tf.placeholder(tf.float32, shape=(1,n_test))
A = tf.placeholder(tf.float32, shape=(n_bt,n_mc))
tau_va = tf.placeholder(tf.float32)
yita_va = tf.placeholder(tf.float32)
with tf.variable_scope("VIP") as scope:
    tf.set_random_seed(rs)
    np.random.seed(rs)
    Z_w,Z_b = init_aux_placeholder(structure,n_bt)
    Z_w_samp,Z_b_samp = init_aux_value(structure,n_bt)
    A_samp = np.random.normal(0,1,(n_bt,n_mc))

    ## training
    ft_train,m_ft_train, y_orig_train = bnn(X_train,Z_w,Z_b,structure,n_bt,n_train)
    loss_train,tau,yita,S_val = VIP_loss_marginal(Y_train,ft_train,m_ft_train,init,alpha,A)

    scope.reuse_variables() ### share weights between different call of functions

    ## validation of hyperparameters
    ft_va,m_ft_va, y_orig_validation = bnn(X_test,Z_w,Z_b,structure,n_bt,n_train)
    y_pred_va,uc2_va= VIP_pred(ft_train,Y_train,ft_va,m_ft_train,m_ft_va,tau_va,yita_va)
    loss_va = VIP_loss(y_pred_va,uc2_va,Y_test)

    ## test
    ft_test,m_ft_test, prior_orig_test = bnn(X_test,Z_w,Z_b,structure,n_bt,n_train)
    y_pred_test,uc2_test = VIP_pred(ft_train,Y_train,ft_test,m_ft_train,m_ft_test,tau,yita)
    loss_test = VIP_loss(y_pred_test,uc2_test,Y_test)

    # import pdb; pdb.set_trace()

    aaaaaa = 0

opt_operation = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_train) # SRS learning rate hyper-parameter optimization phase


### arrays to store results
# llh_test_hist_VIP = np.zeros((n_optimize))
# llh_train_hist_VIP = np.zeros((n_optimize))
# rmse_test_hist_VIP = np.zeros((n_optimize))

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess: # True
 
    tf.set_random_seed(rs)
    np.random.seed(rs)
    # Initialize Variables in graph
    sess.run(tf.global_variables_initializer())


    # SRS Introduce batch training 
    # Define batch size

    # import pdb; pdb.set_trace()

    size_data = x_train.shape[ 1 ]

    n_batches = int(np.ceil(size_data / n_batch))

    perm = np.random.permutation(np.arange(0, size_data))

    if mode == 0:
   	    
        prior_samples_initial = sess.run(ft_test, feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})

        merge_prior_samples_initial = pd.concat([ pd.DataFrame(x_test.T * std_x + mean_x), pd.DataFrame(y_test.T * std_y + mean_y), pd.DataFrame(prior_samples_initial * std_y + mean_y)], axis = 1)
        merge_prior_samples_initial.to_csv('res/vip_initial_prior_samples_' + str(split) + '.txt', index = False, header = False)

        print("\n Initial prior functions sampled! \n")


    # train VIP (only wake phase here, since sleep phase has analytic solution)
    t = time.clock()
    for r in range(n_optimize):

        print("Training... Epoch: ", r, "/", n_optimize)

        for i_batch in range(n_batches):

            last_point = np.minimum(n_batch * (i_batch + 1), size_data)

            # import pdb; pdb.set_trace()

            batch = [ x_train[ :, perm[ i_batch * n_batch : last_point ] ] , y_train[ :, perm[ i_batch * n_batch : last_point ],  ] ]

            # if (i_batch % 60) == 0: 
            #     print("Epoch: ", r , "; batch: ", i_batch, "/", n_batches)

        
            # SRS XXX XXX XXX XXX XXX
            # import pdb; pdb.set_trace()
            sess.run([opt_operation], feed_dict={A:A_samp, X_train:batch[ 0 ], Y_train:batch[ 1 ], Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})


    # SRS
    # En lo siguiente se hace el grid search para buscar los mejores valores de Tau y Yita 

     #grid search using validation set
    loss_va_val = np.zeros((grid1.shape[0],grid2.shape[0]))
    for l in range(grid1.shape[0]):
        for s in range(grid2.shape[0]):
            loss_va_val[l,s] = sess.run(loss_va, feed_dict={tau_va: grid1[l], yita_va: 1/grid2[s], A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train,Y_train:y_train, Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})
    best_init = np.unravel_index(loss_va_val.argmin(), loss_va_val.shape)
    tau_best = grid1[best_init[0]]
    yita_best = grid2[best_init[1]]
    loss_best = loss_va_val.min()

    if mode ==0:
        print("validation on split {:.0f} completed.".format(split+1))
        print("Best tau on validation: {:.4f}.".format(tau_best))
        print("Best yita on validation: {:.4f}.".format(yita_best))
    best_grid[split,0] = tau_best
    best_grid[split,1] = yita_best


    t_VIP = time.clock()-t
    # compute variables of interest
  

    if testing_phase == True:
        validation_test = True
    else:
        validation_train = True


    feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]}


    # SRS
    # XXX XXX XXX XXX XXX

    # if mode == 1: 

    #     import pdb; pdb.set_trace()

    #     y_tmp = sess.run(y_pred_test, feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})



    # Predictions
    y_pred_test_val_VIP = sess.run(y_pred_test, feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})

    uc2_test_val_VIP = sess.run(uc2_test, feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})

    # import pdb; pdb.set_trace()

    # Prior samples
    prior_samples, prior_orig = sess.run([ft_test, prior_orig_test], feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})

    # import pdb; pdb.set_trace()

    llh_test_VIP = -sess.run(loss_test, feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})
    rmse_test_VIP = np.sqrt(((y_test.T-y_pred_test_val_VIP)**2).mean())

    # predictive mean
    m_ft_train_val_VIP = sess.run(m_ft_train, feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})
    tau_val = sess.run(tau, feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})
    yita_val = sess.run(yita, feed_dict={A:A_samp, X_test:x_test,Y_test:y_test, X_train:x_train, Y_train:y_train,Z_w[0]:Z_w_samp[0], Z_w[1]:Z_w_samp[1],Z_w[2]:Z_w_samp[2], Z_b[0]:Z_b_samp[0], Z_b[1]:Z_b_samp[1],Z_b[2]:Z_b_samp[2]})


    # predictive uncertainty
    uc_test_VIP_original = np.sqrt(uc2_test_val_VIP)

    # CRPS metric
    crps_test_VIP = np.mean( crps_gaussian(y_test[0,:], y_pred_test_val_VIP[:,0], uc_test_VIP_original[:,0]) )


    # import pdb; pdb.set_trace()

    aaaaaa = 0

uc_test_VIP = 2*uc_test_VIP_original


