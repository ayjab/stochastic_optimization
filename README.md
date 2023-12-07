# Stochastic optimization
This repository contains implementations of various stochastic gradient descent algorithms and gredient descent algorithms.

### Content 

- `models.py`: Contains the implementation of the linear model.
- `sgd.py`: Contains the implementation of the Stochastic Gradient Descent (SGD) algorithms:
    - SGD
    - SGD with Polyak-Ruppert averaging
    - Stochastic variance reduced gradient descent
- `gradient_descent.py`: Contains the implementation of the Gradient Descent (GD) algorithms:
    - Gradient descent
    - Accelerated gradient descent via Heavy Ball
    - Accelerated gradient descent
    - Coordinate gradient descent (not a real GD in a sense that it doesn't update all parameters at once but only one parameter per iteration)
- `utils.py`: Contains some useful utility functions.
- `test.ipynb`: A Jupyter notebook for testing the implemented algorithms.

### Dependencies
```
pip install -r requirements.txt
```
