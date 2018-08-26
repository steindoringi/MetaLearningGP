import numpy as np
from utils import mu_std, norm


class MultiTaskData:

    def __init__(
        self, X, Y, D, n_data, n_tasks, n_train_tasks,
            dim_in, dim_out, normalize=True):

        assert X.shape[0] == n_tasks
        assert X.shape[1] == n_data
        assert X.shape[2] == dim_in
        assert Y.shape[0] == n_tasks
        assert Y.shape[1] == n_data
        assert Y.shape[2] == dim_out

        self.data = {
            "training": {"X": None, "Y": None, "ids": None},
            "test": {"X": None, "Y": None, "ids": None},
            "domain": D}

        self.n_data = n_data
        self.n_train_data = n_train_tasks * n_data
        self.n_tasks = n_tasks
        self.n_train_tasks = n_train_tasks
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.normalized = normalize

        self._prepare_data(X, Y, n_train_tasks, normalize=normalize)

    def _prepare_data(self, X, Y, n_train, normalize=True):

        X_train, Y_train = X[:n_train], Y[:n_train]
        X_test, Y_test = X[n_train:], Y[n_train:]

        if normalize:
            X_mu, X_std = mu_std(X_train)
            Y_mu, Y_std = mu_std(Y_train)

            X_train = norm(X_train, X_mu, X_std)
            X_test = norm(X_test, X_mu, X_std)
            Y_train = norm(Y_train, Y_mu, Y_std)
            Y_test = norm(Y_test, Y_mu, Y_std)

        #ids = np.int32(np.arange(self.n_tasks))
        #np.random.shuffle(ids)
        #train_ids = ids[:n_train]
        #test_ids = ids[n_train:]

        train_ids = np.int32(list(range(n_train)))
        test_ids = np.int32(list(range(n_train, X.shape[0])))

        self.data["training"]["X"] = X_train
        self.data["training"]["Y"] = Y_train
        self.data["training"]["ids"] = train_ids
        self.data["test"]["X"] = X_test
        self.data["test"]["Y"] = Y_test
        self.data["test"]["ids"] = test_ids

        self.X_mu = X_mu
        self.X_std = X_std
        self.Y_mu = Y_mu
        self.Y_std = Y_std

    def get_domain(self, n_tasks):

        D = np.expand_dims(self.data["domain"], 0)
        if self.normalized:
            D = norm(D, self.X_mu, self.X_std)
        return np.tile(D, [n_tasks, 1, 1])

    def get_raw_data(self, set):

        X = self.data[set]["X"]
        Y = self.data[set]["Y"]
        if self.normalized:
            X = (X * self.X_std) + X_mu
            Y = (Y * self.Y_std) + Y_mu

        return X, Y

    def get_batch(self, seq, si, ei, set="training"):

        X_b = self.data[set]["X"][seq][si:ei].reshape(-1, self.dim_in)
        Y_b = self.data[set]["Y"][seq][si:ei].reshape(-1, self.dim_out)
        ids_b = self.data[set]["ids"][seq]

        num_steps = self.n_data
        num_tasks = len(ids_b)
        data_scale = (num_steps * num_tasks) / self.n_train_data
        task_scale =  num_tasks / self.n_train_tasks

        return X_b, Y_b, ids_b, num_steps, data_scale, task_scale

    def get_task(self, task, num_steps, set="training"):

        X_b = self.data[set]["X"][task].reshape(-1, self.dim_in)[:num_steps]
        Y_b = self.data[set]["Y"][task].reshape(-1, self.dim_out)[:num_steps]

        data_scale = num_steps / self.n_train_data
        task_scale =  1. / self.n_train_tasks

        return X_b, Y_b, data_scale, task_scale
