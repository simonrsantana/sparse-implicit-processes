### Code for VIP modified 

#### Structure of the code

* **test_code.py**: Main body of the code. Includes calls to other functions to make the calculations needed in the model.

* **aux_functions.py**: Create auxiliary functions that estimate moments from given samples of functions evaluated at selected points (mean, deviation and covariance between two collections of samples).

* **BNN_prior.py**: Creates the BNN that samples values from prior on functions. Weight parameters (means and variances) are created and imported into the main NN. This, given an input (x), outputs f_S(x) according to the prior.

* **disc_NNs.py**: Creates and computes the output of both discriminators now needed in the procedure (since now both the approximating distribution and the prior are defined implicitly). There is a discriminator for each distribution samples (prior <em>p(·)</em> and approximating distribution <em>q(·)</em>).

* **neural_sampler.py**: Creates the NS that will output samples from the approximating distribution. Using gaussian noise as input and processing it through a NN, samples of (u) are given (with shape defined through the calling of the function)


Moreover, the code for AADM_class.py and first_test_code.py is included just for development issues, since they represent previous work on which the new code is based on. 
