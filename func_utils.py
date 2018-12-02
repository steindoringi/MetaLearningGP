import tensorflow as tf
import numpy as np
from gpflow import settings


def mu_std(X):

    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mu, std

def covariance_scale(var, scale):

    scale = tf.matmul(scale, scale, transpose_a=True)
    return var * scale

def block_diag(M1, M2):

    D1 = tf.shape(M1)[0]
    D2 = tf.shape(M2)[0]
    UR = tf.zeros((D1, D2), dtype=settings.float_type)
    LL = tf.transpose(UR)
    U = tf.concat([M1, UR], 1)
    L = tf.concat([LL, M2], 1)
    return tf.concat([U, L], 0)


def vec_to_matsum(v, op):

    d = tf.shape(v)[1]
    v_tile = tf.tile(v[:, :, None], [1, 1, d])
    if op == "sum":
        v_sum = v_tile + v[:, None, :]
    else:
        v_sum = v_tile - v[:, None, :]
    return v_sum


def angular_transform(state_mu, state_var, dim_theta):

    n = tf.shape(state_mu)[0]
    d = tf.shape(state_mu)[1]
    d_diff = d - dim_theta
    new_d = 2*dim_theta + d_diff
    theta_mu = state_mu[:, :dim_theta]
    theta_var = state_var[:, :dim_theta, :dim_theta]
    theta_var = tf.matrix_diag_part(theta_var)

    exp_theta_var = tf.exp(-theta_var/2.)
    cos_theta_mu = tf.cos(theta_mu)
    sin_theta_mu = tf.sin(theta_mu)

    cos_mu = exp_theta_var * cos_theta_mu
    sin_mu = exp_theta_var * sin_theta_mu

    theta_mu_sum = vec_to_matsum(theta_mu, "sum")
    theta_mu_sub = vec_to_matsum(theta_mu, "sub")

    theta_var_sum = vec_to_matsum(theta_var, "sum")
    theta_var_sum = -theta_var_sum / 2.
    exp_theta_var_sum = tf.exp(theta_var_sum)
    exp_term_sum = tf.exp(theta_var_sum + theta_var) - exp_theta_var_sum
    exp_term_sub = tf.exp(theta_var_sum - theta_var) - exp_theta_var_sum

    U1 = exp_term_sum * tf.sin(theta_mu_sub)
    U2 = exp_term_sub * tf.sin(theta_mu_sum)
    U3 = exp_term_sum * tf.cos(theta_mu_sub)
    U4 = exp_term_sub * tf.cos(theta_mu_sum)

    cos_var = U3 + U4
    sin_var = U3 - U4
    cos_sin_cov = U1 + U2
    sin_cos_cov = tf.transpose(cos_sin_cov, [0, 2, 1])

    new_theta_mu = tf.concat([cos_mu, sin_mu], 1)
    new_cos_var = tf.concat([cos_var, cos_sin_cov], 2)
    new_sin_var = tf.concat([sin_cos_cov, sin_var], 2)
    new_theta_var = tf.concat([new_cos_var, new_sin_var], 1)

    new_theta_var = new_theta_var / 2.

    cos_mu_diag = tf.matrix_diag(cos_mu)
    sin_mu_diag = -tf.matrix_diag(sin_mu)
    C = tf.concat([sin_mu_diag, cos_mu_diag], 2)
    C = tf.concat([C, tf.zeros(
        (n, d_diff, 2*dim_theta), dtype=state_mu.dtype)], 1)

    inp_out_cov = tf.matmul(state_var, C)
    new_old_cov = inp_out_cov[:, dim_theta:]
    old_var = state_var[:, dim_theta:, dim_theta:] / 2.

    lower = tf.concat([new_old_cov, old_var], 2)
    right = tf.concat(
        [tf.transpose(new_old_cov, [0, 2, 1]), old_var], 1)

    zeros = tf.zeros((n, new_d, 2*dim_theta), dtype=state_mu.dtype)
    lower = tf.concat([tf.transpose(zeros, [0, 2, 1]), lower], 1)
    right = tf.concat([zeros, right], 2)

    zeros = tf.zeros((n, new_d, new_d), dtype=state_mu.dtype)
    new_theta_var = tf.concat([new_theta_var, zeros[:, :d_diff, :2*dim_theta]], 1)
    new_theta_var = tf.concat([new_theta_var, zeros[:, :, :d_diff]], 2)

    new_state_mu = tf.concat([new_theta_mu, state_mu[:, dim_theta:]], 1)
    new_state_var = new_theta_var + lower + right

    old_diff_cov = state_var[:, :, -d_diff:]
    inp_out_cov = tf.concat([inp_out_cov, old_diff_cov], 2)

    return new_state_mu, new_state_var, inp_out_cov
