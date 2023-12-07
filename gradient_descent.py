import numpy as np

class GradientDescent:
    def __init__(self, model, w0, n_iter, verbose=True):
        self.w0 = w0
        self.n_iter = n_iter
        self.model = model
        self.verbose = verbose
    
    def hb(self, callback):
        """
        Accelerated gradient descent via Heavy Ball
        """
        #step = 4. / (model.lip()*(1+model.mu()/model.lip())**2)
        #alpha = 2*np.sqrt(model.mu())/(1+np.sqrt(model.mu()/model.lip()))*step
        
        #step = 1./model.lip()
        #alpha = np.sqrt(step)
        
        # Polyak 84
        model = self.model
        w0 = self.w0
        verbose = self.verbose
        n_iter = self.n_iter

        kappa = model.mu()/model.lip()
        step = 4./(model.lip()*(1+np.sqrt(kappa))**2)
        alpha = (1-np.sqrt(kappa))/(1+np.sqrt(kappa))**2
        
        w = w0.copy()
        w_old = w0.copy()
        w_new = w0.copy()
        # An extra variable is required for acceleration, we call it z
        z = w0.copy()   
        if verbose:
            print("Lauching HB solver...")
        callback(w)
        for k in range(n_iter + 1):
            z[:] = w + (1-alpha) * (w - w_old)
            w_old[:]= w
            w[:] = z - step*model.grad(w)
            callback(w)
        return w
    
    def gd(self, callback):
        """
        Gradient descent
        """
        model = self.model
        w0 = self.w0
        verbose = self.verbose
        n_iter = self.n_iter

        step = 1 / model.lip()
        w = w0.copy()
        w_new = w0.copy()
        if verbose:
            print("Lauching GD solver...")
        callback(w)
        for k in range(n_iter + 1):
            w_new[:] = w - step * model.grad(w)
            w[:] = w_new
            callback(w)
        return w
    
    def agd(self, callback):
        """
        Accelerated gradient descent
        """
        model = self.model
        w0 = self.w0
        verbose = self.verbose
        n_iter = self.n_iter

        step = 1 / model.lip()
        w = w0.copy()
        w_new = w0.copy()
        # An extra variable is required for acceleration, we call it z
        z = w0.copy()
        t = 1.
        t_new = 1.    
        if verbose:
            print("Lauching AGD solver...")
        callback(w)
        for k in range(n_iter + 1):
            w_new[:] = z - step * model.grad(z)
            t_new = 1./2. + (1. + 4. * t * t) ** (0.5) / 2.
            z[:] = w_new + (t - 1) / t_new * (w_new - w)
            t = t_new
            w[:] = w_new
            callback(w)
        return w

    def cgd(self, callback):
        """
        Coordinate gradient descent
        """
        model = self.model
        w0 = self.w0
        verbose = self.verbose
        n_iter = self.n_iter

        w = w0.copy()
        n_features = w0.shape[0]
        steps = 1 / model.lip_coordinates()
        if verbose:
            print("Lauching CGD solver...")
        callback(w)
        for k in range(n_iter + 1):
            for j in range(n_features):
                w[j] -= steps[j] * model.grad_coordinate(j, w) 
            callback(w)
        return w