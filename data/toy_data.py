import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = list(sns.color_palette())


def squared_exponential(x1, x2, h=0.5):
    return np.exp(-0.5 * (x1 - x2) ** 2 / h ** 2)

def task_f(x, z):
    f_z = z
    return f_z

def create_data(num_tasks, num_data, x_lim=(-5., 5.), noise_std=0.1):

    x = np.linspace(x_lim[0], x_lim[1], num_data)[:, None]

    mu = np.zeros(len(x))
    C = squared_exponential(x[:, 0], x)
    f_global = np.random.multivariate_normal(mu, C, 1).T
    inds = np.arange(f_global.shape[0])

    X = []
    F = []
    Y = []
    Z = []

    f, ax = plt.subplots(1)

    for p in range(num_tasks):

        np.random.shuffle(inds)
        z = np.random.randn(1,) * 3.
        f = f_global + task_f(x, z)
        y = f + noise_std * np.random.randn(f.shape[0], f.shape[1])

        X.append(x[None, :])
        F.append(f[None, :])
        Y.append(y[None, :])
        Z.append(z[None])
    
    X = np.vstack(X)
    F = np.vstack(F)
    Y = np.vstack(Y)
    Z = np.vstack(Z)

    return X, F, Y, Z
