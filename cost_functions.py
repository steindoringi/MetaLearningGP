import tensorflow as tf
import numpy as np

from gpflow import settings
from gpflow.params import Parameter, Parameterized
from gpflow.decors import params_as_tensors


class MultiTaskCost(Parameterized):

    def __init__(self, dim_states, target_state, **kwargs):

        super(MultiTaskCost, self).__init__(**kwargs)
        self.dim_states = dim_states
        self.target_state = tf.constant(
            target_state.reshape(1, -1), dtype=settings.float_type)

    def expected_costs(self, states_mu, states_var):
        raise NotImplementedError()


class SquaredCost(MultiTaskCost):

    def __init__(self, dim_states, target_state):

        super(SquaredCost, self).__init__(
            dim_states=dim_states, target_state=target_state)

        self.W = tf.eye(self.dim_states, dtype=settings.float_type)

    def expected_costs(self, states_mu, states_var):

        N = tf.shape(states_mu)[0]
        W_tile = tf.tile(self.W[None, :, :], [N, 1, 1])
        states_mu = states_mu[:, :self.dim_states]
        states_var = states_var[:, :self.dim_states, :self.dim_states]

        var_term = tf.trace(tf.matmul(W_tile, states_var))
        diff = (states_mu - self.target_state)[:, None, :]
        diff_W = tf.matmul(diff, W_tile)
        squared_dist = tf.matmul(diff_W, tf.transpose(diff, [0, 2, 1]))[:, 0, 0]

        return var_term + squared_dist


class PendulumSquaredCost(SquaredCost):

    def __init__(self, dim_states, target_state):

        super(PendulumSquaredCost, self).__init__(
            dim_states=dim_states, target_state=target_state)

        self.length = tf.placeholder(settings.float_type, [])

        self.W = tf.eye(self.dim_states, dtype=settings.float_type)
        self.W = self.length**2 * self.W


class CartpoleSquaredCost(SquaredCost):

    def __init__(self, dim_states, target_state):

        super(CartpoleSquaredCost, self).__init__(
            dim_states=dim_states, target_state=target_state)

        self.length = tf.placeholder(settings.float_type, [])

        zero = tf.zeros(shape=(1, 1), dtype=settings.float_type)
        one = tf.ones(shape=(1, 1), dtype=settings.float_type)
        C1 = tf.concat([tf.reshape(self.length, [1, 1]), zero, zero], 1)
        C2 = tf.concat([zero, tf.reshape(self.length, [1, 1]), one], 1)
        C = tf.concat([C1, C2], 0)
        self.W = tf.matmul(C, C, transpose_a=True)
