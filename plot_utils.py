import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_grid(N):

    H = int(np.sqrt(N))
    W = H + N - (H**2)

    G = np.empty(shape=(N, 2), dtype=np.int32)
    p = 0
    for w in range(W):
        for h in range(H):
            G[p, 0] = h
            G[p, 1] = w
            p += 1

    return H, W, G

def plot_predictions(X_pred, Y_mu, Y_var, Y, num_train, plot_dir):

    colors = list(sns.color_palette("hls", 20))

    H, W, G = create_grid(X_pred.shape[0])
    f, ax = plt.subplots(H, W, figsize=(W*2, H*2))

    for p in range(X_pred.shape[0]):

        xp = X_pred[p]
        yp = Y[p]

        xp_pred = X_pred[p]
        yp_mu = Y_mu[p]
        yp_var = Y_var[p]
        yp_err = 2*np.sqrt(yp_var)
        yp_min = yp_mu - yp_err
        yp_max = yp_mu + yp_err

        if p < num_train:
            dat_c = "black"
            title = "train {}".format(p)
        else:
            dat_c = "red"
            title = "test {}".format(p)

        r = G[p, 0]
        c = G[p, 1]

        ax[r, c].scatter(xp, yp, color=dat_c)
        ax[r, c].plot(xp_pred, yp_mu, color=colors[p%20], label="task {}".format(p))
        ax[r, c].fill_between(xp_pred.reshape(-1), yp_min.reshape(-1), yp_max.reshape(-1), alpha=0.33, color=colors[p%20])
        ax[r, c].set_title(title)
    
    plt.savefig(plot_dir + 'predictions.png', dpi=300)