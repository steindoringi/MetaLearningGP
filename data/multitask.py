import tensorflow as tf
import numpy as np


class Dataset(object):

    def __init__(self,
        X=None, Y=None, standardise=False):

        self.X = X # Input data [N x num_tasks x D_in]
        self.Y = Y # Output data [N x num_tasks x D_out]

        self.standardise = standardise

    @property
    def dim_in(self):
        return self.X.shape[-1]

    @property
    def dim_out(self):
        return self.Y.shape[-1]

    @property
    def num_obs(self):
        return self.X.shape[0]*self.X.shape[1]

    @property
    def num_tasks(self):
        return self.X.shape[0]

    def add_observations(self, X_obs, Y_obs):

        if self.X is None:
            self.X = X_obs
        else:
            self.X = np.vstack([self.X, X_obs])
        
        if self.Y is None:
            self.Y = Y_obs
        else:
            self.Y = np.vstack([self.Y, Y_obs])

    def create_tf_dataset(self, batch_size, reshuffle, shuffle_buffer,
            p=None, num_obs=None):

        if p is None:
            p = np.int32(np.arange(self.num_tasks))
        if num_obs is None:
            num_obs = self.num_obs

        X_inp = self.X[p, :num_obs]
        p_inp = np.tile(p[:, None, None], [1, X_inp.shape[1], 1])

        inputs = {
            "X": X_inp.reshape(-1, self.dim_in),
            "p": p_inp.reshape(-1, 1)
            }

        Y_out = self.Y[p, :num_obs]
        #NOTE: Dictionary outputs not working for subclassed models: #25299
        Yp = np.concatenate([Y_out, p_inp], axis=2)
        outputs = Yp.reshape(-1, self.dim_out+1)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        dataset = dataset.shuffle(
            shuffle_buffer, reshuffle_each_iteration=reshuffle)
        dataset = dataset.batch(batch_size)

        return dataset