import numpy as np
from matplotlib.pylab import pcolor
import pandas as pd
from sklearn.datasets import load_boston, load_diabetes
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
from tensorflow.contrib.distributions import Bernoulli
import matplotlib.pyplot as plt

# SRS
import sys
from _crps import crps_gaussian 
