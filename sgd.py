import numpy as np

class SGD:
    def __init__(self, model, w0, n_iter, verbose=True):
        self.w0 = w0
        self.n_iter = n_iter
        self.model = model
        self.verbose = verbose

    def sgd(self, callback, idx_samples, step):
        """
        Stochastic gradient descent without Polyak-Ruppert averaging
        """
        model = self.model
        w0 = self.w0
        n_iter = self.n_iter

        w = w0.copy()
        callback(w)
        n_samples = model.n_samples

        for idx in range(n_iter):
            i = idx_samples[idx]
            step_i = 1/np.sqrt(idx+1)
            # Compute the gradient of the i-th term
            gradient_i = model.grad_i(i, w)
            # SGD update step
            w = w - (step) * gradient_i
            if idx % n_samples == 0:
                callback(w)
        return w

    def sgd_polyak_ruppert(self, callback, idx_samples, step):
        """
        Stochastic gradient descent with Polyak-Ruppert averaging
        """
        model = self.model
        w0 = self.w0
        n_iter = self.n_iter

        w = w0.copy()
        avg_w = w0.copy() 
        callback(w)
        n_samples = model.n_samples

        for idx in range(n_iter):
            i = idx_samples[idx]

            gradient_i = model.grad_i(i, w)
            w = w - step * gradient_i
            avg_w = (avg_w * idx + w) / (idx + 1)

            if idx % n_samples == 0:
                callback(avg_w)
        return avg_w, w
    
    def svrg(self, callback, idx_samples, step):
        """
        Stochastic variance reduced gradient descent
        """
        model = self.model
        w0 = self.w0
        n_iter = self.n_iter

        w = w0.copy()
        w_old = w.copy()
        n_samples = model.n_samples
        callback(w)
        for idx in range(n_iter):        
            if idx % n_samples == 0:
                w_old[:] = w
                mu = model.grad(w)        
            i = idx_samples[idx]
            z_new = model.grad_i(i, w)
            z_old = model.grad_i(i, w_old)
            w -= step * (z_new - z_old + mu)
            if idx % n_samples == 0:
                callback(w)
        return w