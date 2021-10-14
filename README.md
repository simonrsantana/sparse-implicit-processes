## Sparse Implicit Processes

In this repository we include the two different versions of the SIP code developed for the article __Sparse Implicit Processes for Approximate Inference__. 

### Structure of the code

We have two setups for the model: one that makes use of BNNs to model the implicit prior distribution, and another that uses Neural Samplers to do so. The BNN-based can be found in [VIIP-BNN_prior](VIIP-BNN_prior), and the NS-based on [VIIP-NS_prior](VIIP-NS_prior). For each of them, the structure of the code goes as follows: 

* **SIP_**(·)**.py**: Main body of the code. Includes calls to other functions to make the calculations needed in the model.

* **aux_functions.py**: Create auxiliary functions that estimate moments from given samples of functions evaluated at selected points (mean, deviation and covariance between two collections of samples).

* **prior_BNN.py** or **prior_neural_sampler.py**: Creates the prior samples of functions. Weight parameters (means and variances) are created and imported into the main NN. This, given an input (x), outputs f_S(x) according to the prior, which can be parametrized either as a BNN or a NS, depending on the selected folder. 

* **discriminators.py**: Creates and computes the output of both discriminators now needed in the procedure (since now both the approximating distribution and the prior are defined implicitly). There is a discriminator for each distribution samples (prior <em>p(·)</em> and approximating distribution <em>q(·)</em>).

* **posterior_neural_sampler.py**: Creates the NS that will output samples from the approximating distribution. Using gaussian noise as input and processing it through a NN, samples of (u) are given (with shape defined through the calling of the function). The posterior is the same for both models implemented.

The rest of the python codes present in each folder serve to control the random variables, as well as their precision. 	

**To run the code, do:** python SIP_main.py [split] [alpha value] [n_layers (1 or 2)] [dataset as .txt]

The rest of the codes included here, and additional folders that may appear are just auxiliary and extra material that aid to develop the main code, on which it is based on. 

