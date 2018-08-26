import tensorflow as tf
import numpy as np
from tensorflow.contrib import distributions as dist

from gpflow.models.svgp import SVGP
from gpflow import settings, features
from gpflow.params import Parameter
from gpflow.decors import params_as_tensors
from gpflow_mod.conditionals import conditional, uncertain_conditional


class MLSVGP(SVGP):

    def __init__(self,
            dim_in, dim_out,
            kern, likelihood,
            dim_h, num_h,
            feat=None,
            mean_function=None,
            num_latent=None,
            q_diag=False,
            whiten=True,
            minibatch_size=None,
            Z=None,
            num_data=None,
            max_lik_h=False,
            **kwargs):

        # Only used to initialize the SVGP class
        X_init = np.zeros(shape=(1, dim_in))
        Y_init = np.zeros(shape=(1, dim_out))

        SVGP.__init__(
            self, X=X_init, Y=Y_init,
            kern=kern, likelihood=likelihood,
            feat=feat,
            mean_function=mean_function,
            num_latent=num_latent,
            q_diag=q_diag,
            whiten=whiten,
            minibatch_size=minibatch_size,
            Z=Z,
            num_data=num_data,
            **kwargs)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_latent = dim_out
        self.dim_h = dim_h
        self.num_h = num_h
        self.max_lik_h = max_lik_h

        # Initialize task variables
        h_mu = np.zeros((1, dim_h))
        h_var = np.log(np.ones_like(h_mu) * 0.1)
        H_init = np.hstack([h_mu, h_var])
        for h in range(num_h):
            setattr(self, "H_{}".format(h), Parameter(H_init, dtype=settings.float_type))

        # Create placeholders
        self.X_ph = tf.placeholder(settings.float_type, [None, dim_in])
        self.X_var_ph = tf.placeholder(settings.float_type, [None, dim_in])
        self.Y_ph = tf.placeholder(settings.float_type, [None, dim_out])
        self.H_ids_ph = tf.placeholder(tf.int32, [None])
        self.num_steps = tf.placeholder(tf.int32, [])
        self.data_scale = tf.placeholder(settings.float_type, [])
        self.task_scale = tf.placeholder(settings.float_type, [])

    @params_as_tensors
    def _build_likelihood(self):

        # Get prior KL.
        KL = self.build_prior_KL()

        H = [getattr(self, "H_{}".format(h)) for h in range(self.num_h)]
        H = tf.concat(H, 0)
        H = tf.gather(H, self.H_ids_ph)

        H_sample, KL_H = self.sample_qH(H)
        H_tiled = tf.tile(tf.reshape(H_sample, [-1, 1, self.dim_h]), [1, self.num_steps, 1])
        H_flat = tf.reshape(H_tiled, [-1, self.dim_h])
        XH = tf.concat([self.X_ph, H_flat], 1)

        # Get conditionals
        fmean, fvar = self._build_predict(XH, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y_ph)

        lik_term = tf.reduce_sum(var_exp) * self.data_scale

        if self.max_lik_h:
            likelihood = lik_term - KL
        else:
            likelihood = lik_term - KL - self.task_scale * KL_H

        return likelihood

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(Xnew, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
                              white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var

    @params_as_tensors
    def _build_predict_uncertain(self, Xnew_mu, Xnew_var, full_cov=False, Luu=None):
        mu, var = uncertain_conditional(
            Xnew_mu=Xnew_mu, Xnew_var=Xnew_var, feat=self.feature, kern=self.kern,
            q_mu=self.q_mu, q_sqrt=self.q_sqrt, Luu=Luu,
            mean_function=self.mean_function,
            full_cov=full_cov, white=self.whiten)
        return mu, var

    def sample_qH(self, H):
        h_mu = H[:, :self.dim_h]
        h_var = tf.exp(H[:, self.dim_h:])
        qh = dist.Normal(h_mu, tf.sqrt(h_var))
        ph = dist.Normal(tf.zeros_like(h_mu), tf.ones_like(h_var))
        kl_h = tf.reduce_sum(dist.kl_divergence(qh, ph))
        h_sample = qh.sample()

        return h_sample, kl_h

    @params_as_tensors
    def predict_y(self, full_cov=False, use_var=True):

        H = [getattr(self, "H_{}".format(h)) for h in range(self.num_h)]
        H = tf.concat(H, 0)
        H = tf.gather(H, self.H_ids_ph)

        hmu = H[:, :self.dim_h]
        hvar = tf.exp(H[:, self.dim_h:])

        if use_var:
            H_tiled = tf.tile(tf.reshape(H, [-1, 1, 2*self.dim_h]), [1, self.num_steps, 1])
            H_flat = tf.reshape(H_tiled, [-1, 2*self.dim_h])
            H_mu = H_flat[:, :self.dim_h]
            H_var = tf.exp(H_flat[:, self.dim_h:])
            XH_mu = tf.concat([self.X_ph, H_mu], 1)
            XH_var = tf.concat([self.X_var_ph, H_var], 1)
            fmean, fvar = self._build_predict_uncertain(XH_mu, XH_var, full_cov=full_cov)
        else:
            H_sample, _ = self.sample_qH(H)
            H_tiled = tf.tile(tf.reshape(H_sample, [-1, 1, self.dim_h]), [1, self.num_steps, 1])
            H_mu = tf.reshape(H_tiled, [-1, self.dim_h])
            XH = tf.concat([self.X_ph, H_mu], 1)
            fmean, fvar = self._build_predict(XH, full_cov=full_cov)

        ymu, yvar = self.likelihood.predict_mean_and_var(fmean, fvar)

        return ymu, yvar, hmu, hvar
