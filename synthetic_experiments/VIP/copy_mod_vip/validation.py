### VIP validation mode

mode =0
print("validation phase")
best_grid = np.zeros((n_split,2)) # array for storing best hyper parameters after validation.

available_perm = np.loadtxt('permutations_' + original_file, delimiter = ",", dtype = int)

for split in range(0,n_split,1):
    x = Data[:,0:size_feature]
    y = Data[:,size_feature]
    std_y = y.std()

    y = preprocessing.scale(y) # standardization
    y = y.reshape(y.shape[0],1)

    # import pdb; pdb.set_trace()

    # SRS We are NOT re-splitting after the first time for validation (sames as in our code)

    permutation = available_perm[split,]

    data_size = x.shape[ 0 ]
    size_train = int(np.round(data_size * ratio_train))

    index_train = permutation[ 0 : size_train ]
    index_test = permutation[ size_train : ]

    x_train = x[ index_train, : ]
    y_train = np.vstack(y[ index_train ])

    x_test = x[ index_test, : ]
    y_test = np.vstack(y[ index_test ])

    # ---


    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=rs+split)
    # during validation, training set is again divided for validation phase, to make sure the model is not trained on test set.
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=38+split)



    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    # preprocess data
    mean_x = (x_train.mean(1)).reshape(x_train.shape[0],1)
    std_x = x_train.std(1).reshape(x_train.shape[0],1)
    x_train = (x_train - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x



    # initialize hyper parameter
    init = [1000,1]
    structure = [10,10,1] # BNN structure


    exec(open("VIP_fun_speedup_recon_alpha_classic_validate.py").read())
