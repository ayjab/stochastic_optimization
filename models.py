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
        
    def lip_coordinates(self, j):
        """
        Computes the Lipschitz constant of the derivative of f^j (f with respect to the j-th coordinate)
        """
        X, n_samples = self.X, self.n_samples
        return norm(X[:, j]) ** 2 / n_samples + self.strength
    
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
        return ((X ** 2).sum(axis=1) + n_samples * strength).max() 
    
class ModelLogReg:
    """A class giving first order information for logistic regression
    
    Parameters
    ----------
    X : `numpy.array`, shape=(n_samples, n_features)
        The features matrix
    
    y : `numpy.array`, shape=(n_samples,)
        The vector of labels
    
    strength : `float`
        The strength of ridge penalization
    """    
    def __init__(self, X, y, strength):
        self.X = X
        self.y = y
        self.strength = strength
        self.n_samples, self.n_features = X.shape
    
    def loss(self, w):
        """
        Computes f(w)
        """
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        return 1/n_samples * np.sum([np.log(1 + np.exp(-y[i]*(X[i].T.dot(w)))) for i in range(n_samples)]) + strength * norm(w) ** 2 / 2 
       
    def grad(self, w):
        """
        Computes the gradient of f at w
        """
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        return -np.sum([y[i] / (1 + np.exp(y[i] * X[i].dot(w))) * X[i] for i in range(n_samples)], axis=0) / n_samples + strength * w

    def grad_i(self, i, w):
        """
        Computes the gradient of f_i at w
        """
        x_i, y_i, strength = self.X[i], self.y[i], self.strength
        return -y_i / (1 + np.exp(y_i * x_i.dot(w))) * x_i + strength * w

    def grad_coordinate(self, j, w):
        """
        Computes the partial derivative of f with respect to the j-th coordinate
        """
        y, X, n_samples, strength = self.y, self.X, self.n_samples, self.strength
        return -np.sum([y[i] * X[i, j] / (1 + np.exp(y[i] * X[i].dot(w))) for i in range(n_samples)], axis=0) / n_samples + strength * w[j]

    def lip(self):
        """
        Computes the Lipschitz constant of grad f
        """
        X, n_samples = self.X, self.n_samples
        return norm(X, 2) ** 2 / (4 * n_samples)
    
    def mu(self):
        """
        Computes the strong convexity param of f
        """
        X, n_samples, strength = self.X, self.n_samples, self.strength
        p = 1 / (1 + np.exp(-X.dot(w)))
        diag_weights = p * (1 - p)
        A = X.T @ np.diag(diag_weights) @ X / n_samples + strength * np.eye(self.n_features)
        return np.min(np.linalg.eigvals(A))

#    def lip_coordinates(self):
#        """Computes the Lipschitz constant of the derivative of f^j (f with respect to 
#        the j-th coordinate)"""
#        X, n_samples = self.X, self.n_samples
#        ### TODO
#
#        ### END TODO

    def lip_max(self):
        """Computes the maximum of the lipschitz constants of f_i"""
        X, n_samples = self.X, self.n_samples
        ### TODO

        ### END TODO
