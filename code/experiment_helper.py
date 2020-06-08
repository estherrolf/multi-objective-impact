import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import warnings

class Population:
    """A class for simulating a population with profit and welfare labels.

    The population features are distributed as N(mean, cov)
    The labels are set as:
        p_i = v_p^T x_i + eps
        delta_i = v_d^T x_i + eps

    Attributes:
        cov, mean: of Gaussian feature distribution
        v_p, v_d: profit and delta parameters
        sd_p, sd_d: standard deviation of noisiness of linear fit
    """
    def __init__(self, d, noise=(0.1, 0.1), correlation=0, 
                 cov=None, mean=None, seed=0,
                 thetas=None):
        ''' Initializing Population Attributes '''
        np.random.seed(0)
        # setting populating parameters
        self.d = d
        self.cov = np.eye(d) if cov is None else cov
        self.mean = np.zeros(d) if mean is None else mean

        # setting label parameters
        if thetas is None:
            self.v_p, self.v_d = generate_thetas(d, correlation, seed=seed)
        else:
            self.v_p, self.v_d = thetas

        # setting label noise parameters
        self.sd_p = noise[0]
        self.sd_d = noise[1]
        
    def get_samples(self, n=1, noise=None, welf_corr_with=None, welf_var_with=None):
        ''' Returns sample (x_i,p_i,d_i) of size n 
            noise argument overrides the population noise

            we have
                welf_corr_with = (theta_alpha, corr)
                sets welfare noise to be corr correlated with the score <theta_alpha, x>
                welf_var_with = theta_alpha
                sets welfare noise have variance proportional to (<theta_alpha, x>)^2
        '''
        # TODO: add functionality for mismatched populations?
        X = np.random.multivariate_normal(self.mean, self.cov, size=(n))
        
        sd_p = self.sd_p if noise is None else noise[0]
        p = X.dot(self.v_p)
        p = add_noise(X, p, sd_p)

        
        sd_d = self.sd_d if noise is None else noise[1]
        delta = X.dot(self.v_d) 
        delta = add_noise(X, delta, sd_d, corr_with=welf_corr_with, var_with=welf_var_with)

        data_dict = {'X':X, 'yw':delta, 'yp':p}
        return data_dict

def add_noise(X, y, sd, corr_with=None, var_with=None):
    ''' Returns sample (x_i,p_i,d_i) of size n 
            noise argument overrides the population noise

            we have
                welf_corr_with = (theta_alpha, corr)
                welf_var_with = theta_alpha
                sets welfare noise have variance proportional to (<theta_alpha, x>)^2
        '''
        # TODO: add functionality for mismatched populations?
    y = y.reshape(1, -1).T
    if corr_with is None and var_with is None:
        y_noise = y + np.random.normal(scale=sd, size=y.shape)
    else:
        if corr_with is not None:
            theta1, corr = corr_with
        else:
            theta1 = None; corr = None
        theta2 = var_with
        c1, c2 = get_noise_model(sd**2, corr, theta1, theta2)
        noise_ind = np.random.normal(scale=1, size=y.shape)
        y_noise = y + X.dot(c1) + X.dot(c2) * noise_ind
    return y_noise

def generate_thetas(d, corr, seed=0):
    np.random.seed(seed)
    theta_p = np.random.randn(d,1)
    theta_p = theta_p / np.linalg.norm(theta_p)
    # works as long as X covariance is c*I
    theta_w =  rand_cos_sim(theta_p.flatten(), corr) #
    theta_w = theta_w / np.linalg.norm(theta_w)
    theta_w = theta_w.reshape(theta_p.shape)
    return theta_p, theta_w


def rand_cos_sim(vector, costheta):
    '''
    Generate a random vector with a fixed cosine similarity wrt another vector
    Method from https://stackoverflow.com/questions/52916699/create-random-vector-given-cosine-similarity
    Args:
        v: a fixed vector
        costheta: the cosine similarity
    Returns:
        w: random vector such that <v,w> = costheta
    '''
    # Form the unit vector parallel to v:
    v = vector.flatten()
    u = v / np.linalg.norm(v)

    # Pick a random vector:
    r = np.random.multivariate_normal(np.zeros_like(v.flatten()), np.eye(len(v)))

    # Form a vector perpendicular to v:
    uperp = r - r.dot(u)*u

    # Make it a unit vector:
    uperp = uperp / np.linalg.norm(uperp)

    # w is the linear combination of u and uperp with coefficients costheta
    # and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = costheta*u + np.sqrt(1 - costheta**2)*uperp

    return w.reshape(vector.shape)
