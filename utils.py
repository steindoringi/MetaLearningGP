import numpy as np


def mu_std(X, flatten=True):

    if len(X.shape) == 3:
        D = X.shape[2]
    else:
        D = 1

    if flatten:
        X = X.reshape(-1, D)

    return np.mean(X, axis=0), np.std(X, axis=0)

def norm(X, mu, std):
    return (X - mu) / std

def rbf_kernel(x1, x2, ls=1., var=1.):
    return var*np.exp(-((x1-x2) ** 2.) / (2.*ls))

def gram_matrix(xs):
    return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]

def generate_toy_data(seed, n_tasks, n_data, noise_var=0.1):

    np.random.seed(seed)

    # True functions
    xs = np.arange(-8, 8, 0.05)
    mean = np.float32([0 for x in xs])
    gram = np.float32(gram_matrix(xs))
    fs = np.random.multivariate_normal(mean, gram)

    # Sample data
    offset_range = [-3., 3.]
    offset_size = np.sum(np.abs(offset_range))
    offsets = np.arange(
        offset_range[0], offset_range[1],
        offset_size/n_tasks)
    inds = np.arange(xs.shape[0])
    start_ind = 15
    inds = inds[::6][start_ind:start_ind+n_data]
    X = xs[inds]
    sorted_ind = np.argsort(X)
    X = X[sorted_ind]
    F = []
    Y = []
    for task in range(n_tasks):
        f = fs + offsets[task]
        y = f[inds][sorted_ind] + np.random.randn(n_data) * np.sqrt(noise_var)
        F.append(np.expand_dims(f, 0))
        Y.append(np.expand_dims(y, 0))
    X = np.expand_dims(np.tile(X, [n_tasks, 1]), 2)
    F = np.expand_dims(np.vstack(F), 2)
    Y = np.expand_dims(np.vstack(Y), 2)

    return xs.reshape(-1, 1), X, F, Y
