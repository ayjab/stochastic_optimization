from numpy.linalg import norm
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz
from numpy.random import randn
import numpy as np

class ModelLinReg:
    """A class giving first order information for linear regression
    with least-squares loss
    
    Parameters
    ----------
    w0 : `numpy.array`, shape=(n_features,)
        Model weights
    
    n_samples : `int`, default=1000
        Number of samples to simulate
    
    corr : `float`, default=0.5
        Correlation of the features
    
    std : `float`, default=0.5
        Standard deviation of the noise
    
    strength : `float`
        The strength of ridge penalization
    """    
    def __init__(self, w0, n_samples, strength, corr=0.5, std=0.5, X=None, y=None):
        self.w0 = w0
        self.n_samples = n_samples
        self.strength = strength
        self.corr = corr
        self.std = std
        self.X, self.y = self.simu_linreg()
    
    def simu_linreg(self):
        w0, n_samples = self.w0, self.n_samples
        n_features = w0.shape[0]
        cov = toeplitz(self.corr ** np.arange(0, n_features))
        X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
        y = X.dot(w0) + self.std * randn(n_samples)
        return X, y

    def loss(self, w):
        """
        Computes f(w)
        """
        X, y, n_samples, strength = self.X, self.y, self.n_samples, self.strength
        return 0.5 * norm(y - X.dot(w)) ** 2 / n_samples + strength * norm(w) ** 2 / 2 
        
    def grad(self, w):
        """
        Computes the gradient of f at w
        """
        X, y, n_samples, strength = self.X, self.y, self.n_samples, self.strength
        return X.T.dot(X.dot(w) - y) / n_samples + strength * w
        
    def grad_i(self, i, w):
        """
        Computes the gradient of f_i at w
        """
        X, y, n_samples, strength = self.X, self.y, self.n_samples, self.strength
        x_i = self.X[i]
        y_i = self.y[i]
        return (np.dot(x_i, w) - y_i) * x_i / n_samples + strength * w
    
    def grad_coordinate(self, j, w):
        """
        Computes the partial derivative of f with respect to the j-th coordinate
        """
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        return X[:, j].dot(X.dot(w) - y) / n_samples + strength * w[j]
        
    def lip(self):
        """
        Computes the Lipschitz constant of grad f
        """
        X, n_samples, strength = self.X, self.n_samples, self.strength
        return norm(X.T.dot(X), 2) / n_samples + strength
        
    def lip_coordinates(self):
        """
        Computes the Lipschitz constant of the derivative of f^j (f with respect to the j-th coordinate)
        """
        X, n_samples = self.X, self.n_samples
        return (X ** 2).sum(axis=0) / n_samples + self.strength
    
    def mu(self):
        """
        Computes the strong convexity param of f
        """
        X, n_samples, strength = self.X, self.n_samples, self.strength
        return norm(X.T.dot(X), -2) / n_samples + strength
    
    def lip_max(self):
        """
        Computes the maximum of the lipschitz constants of f_i
        """
        X, n_samples, strength = self.X, self.n_samples, self.strength
        return ((X ** 2).sum(axis=1)+ n_samples * strength).max() 
