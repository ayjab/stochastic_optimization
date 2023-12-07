
from models import ModelLinReg
from gradient_descent import GradientDescent
import numpy as np
from utils import inspector
import matplotlib.pyplot as plt

n_samples = 1000	
n_features = 50
corr, std = 0.6, 0.5
n_iter = 50
strength = 1e-3
w0 = np.zeros(n_features)
n_samples = 500

nnz = 20
idx = np.arange(n_features)
w_true = (-1) ** idx * np.exp(-idx / 10.)
w_true[nnz:] = 0.

print("Creating model...")
model = ModelLinReg(n_samples=n_samples, corr=corr, std=std, strength=strength, w0=w_true)
print("Simulating data...")
X, y = model.simu_linreg()

gradient_descent = GradientDescent(w0=w0, n_iter=n_iter, model=model, verbose=True)
callback_gd = inspector(model, n_iter=n_iter)
w_gd = gradient_descent.gd(callback=callback_gd)

callback_agd = inspector(model, n_iter=n_iter)
w_agd = gradient_descent.agd(callback=callback_agd)

callback_hb = inspector(model, n_iter=n_iter)
w_hb = gradient_descent.hb(callback=callback_hb)
