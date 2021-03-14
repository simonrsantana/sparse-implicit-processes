### VIP test mode
mode = 1
print("Testing model")

# SRS Remove the for loop running on splits (now one by one)

x = Data[:,0:size_feature]
y = Data[:,size_feature]

# std_y = y.std()
# mean_y = y.mean()

# y = preprocessing.scale(y) # standardization
y = y.reshape(y.shape[0],1)

#    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=38+split)

# SRS

permutation = available_perm[split,]

# import pdb; pdb.set_trace()

data_size = x.shape[ 0 ]
size_train = int(np.round(data_size * ratio_train))

index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

x_train = x[ index_train, : ]
y_train = np.vstack(y[ index_train ])


# SRS: Randomized test set
# x_test = x[ index_test, : ]
# y_test = np.vstack(y[ index_test ])

# SRS: Predetermined test set (to generate fuller figures)
data_test = np.loadtxt( 'test_' + original_file ).astype(np.float32)
x_test = data_test[ :, range(data_test.shape[ 1 ] - 1) ]
y_test = np.vstack( data_test[ :, data_test.shape[ 1 ] - 1 ])


# ----

x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

# preprocess data
mean_x = (x_train.mean(1)).reshape(x_train.shape[0],1)
std_x = x_train.std(1).reshape(x_train.shape[0],1)

mean_y = y_train.mean()
std_y = y_train.std()

x_train = (x_train - mean_x)/std_x
x_test = (x_test - mean_x)/std_x

y_train = (y_train - mean_y)/std_y
y_test = (y_test - mean_y)/std_y


# import pdb; pdb.set_trace()


# SRS
# Check whether the code entered here already
testing_phase = True



# use best hyper parameters from validation
init = [best_grid[split,0],1/best_grid[split,1]]
# train the model on the full training set (during validation phase, the model is only trained on part of the training set,
# therefore we train the VIP on the full training set under the best hyper parameters from validation
structure = [50,50,1] # BNN structure # SRS estaba en 10 10 
exec(open("VIP_fun_speedup_recon_alpha_classic_validate.py").read())

# import pdb; pdb.set_trace()

crps_VIP_agg[split] = crps_test_VIP*std_y
llh_VIP_agg[split] = llh_test_VIP-np.log(std_y)
rmse_VIP_agg[split] = rmse_test_VIP*std_y
print("Train and test on split {:.0f} completed.".format(split))
print("Test llh of this split: {:.4f}.".format(llh_test_VIP-np.log(std_y)))
print("Test rmse of this split: {:.4f}.".format(rmse_test_VIP*std_y))

np.savetxt('res/vip_rmse_' + str(split) + '.txt', [ rmse_VIP_agg[ split ]  ])
np.savetxt('res/vip_ll_' + str(split) + '.txt', [ llh_VIP_agg[ split ]  ])
np.savetxt('res/vip_crps_' + str(split) + '.txt', [ crps_VIP_agg[ split ] ])

# SRS:  Store the final predictions and the prior samples
# predictions_agg.append( y_pred_test_val_VIP  )
# prior_samples_agg.append( prior_samples  )

# import pdb; pdb.set_trace()

merge_y_pred = pd.concat([ pd.DataFrame(x_test.T * std_x + mean_x), pd.DataFrame(y_test.T * std_y + mean_y), pd.DataFrame(y_pred_test_val_VIP * std_y + mean_y), pd.DataFrame( uc_test_VIP_original * std_y ) ], axis = 1)

merge_prior_samples = pd.concat([ pd.DataFrame(x_test.T * std_x + mean_x), pd.DataFrame(y_test.T * std_y + mean_y), pd.DataFrame(prior_samples * std_y + mean_y)], axis = 1)
merge_prior_samples_orig = pd.concat([ pd.DataFrame(x_test.T * std_x + mean_x), pd.DataFrame(y_test.T * std_y + mean_y), pd.DataFrame(prior_orig * std_y + mean_y)], axis = 1)

merge_train_data = pd.concat([ pd.DataFrame(x_train.T), pd.DataFrame(y_train.T) ], axis = 1)


# import pdb; pdb.set_trace()

merge_y_pred.to_csv('res/vip_y_forecasted_' + str(split) + '.txt', index = False, header = False)
merge_prior_samples.to_csv('res/vip_prior_samples_' + str(split) + '.txt', index = False, header = False)
merge_prior_samples_orig.to_csv('res/vip_prior_samples_orig_' + str(split) + '.txt', index = False, header = False)
merge_train_data.to_csv('res/vip_train_data_' + str(split) + '.txt', index = False, header = False)


print("negative log likelihood averaged on all split {:.4f} + {:.4f}. RMSE averaged on all split {:.4f}+{:.4f}.".format(-llh_VIP_agg.mean(), llh_VIP_agg.std()/np.sqrt(n_split),rmse_VIP_agg.mean(),rmse_VIP_agg.std()/np.sqrt(n_split), crps_VIP_agg.mean(),crps_VIP_agg.std()/np.sqrt(n_split)))

# np.savetxt('res/vip_final_rmse.txt', [ rmse_VIP_agg.mean(),rmse_VIP_agg.std()/np.sqrt(n_split) ])
# np.savetxt('res/vip_final_ll.txt', [ -llh_VIP_agg.mean(), llh_VIP_agg.std()/np.sqrt(n_split) ])
# np.savetxt('res/vip_final_crps.txt', [ crps_VIP_agg.mean(), crps_VIP_agg.std()/np.sqrt(n_split) ])

