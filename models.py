import tensorflow as tf
import numpy as np
from tensorflow.contrib import distributions as dist

from gpflow_mod.svgp import SVGP
from gpflow import settings, features
from gpflow.params import Parameter, Parameterized
from gpflow.decors import params_as_tensors
from gpflow_mod.conditionals import conditional, uncertain_conditional

from func_utils import block_diag


class BASESVGP(SVGP):

    def __init__(self,
            dim_in, dim_out,
            kern, likelihood,
            feat=None,
            mean_function=None,
            q_diag=False,
            whiten=True,
            Z=None,
            num_data=None,
            **kwargs):

        super(BASESVGP, self).__init__(
            dim_in=dim_in, dim_out=dim_out,
            kern=kern, likelihood=likelihood,
            feat=feat,
            mean_function=mean_function,
            q_diag=q_diag,
            whiten=whiten,
            Z=Z,
            num_data=num_data,
            **kwargs)

    @params_as_tensors
    def _build_predict_uncertain(self, Xnew_mu, Xnew_var,
                    full_cov=False, full_output_cov=False, Luu=None):

        mu, var, inp_out_cov = uncertain_conditional(
            Xnew_mu=Xnew_mu, Xnew_var=Xnew_var, feat=self.feature, kern=self.kern,
            q_mu=self.q_mu, q_sqrt=self.q_sqrt, Luu=Luu,
            mean_function=self.mean_function,
            full_cov=full_cov, full_output_cov=full_output_cov, white=self.whiten)

        return mu, var, inp_out_cov

    @params_as_tensors
    def _compute_Luu(self):
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        return tf.cholesky(Kuu)

    def get_model_param(self, session):
        lik_noise = self.likelihood.variance.read_value(session=session)
        kern_var = self.kern.variance.read_value(session=session)
        kern_ls = self.kern.lengthscales.read_value(session=session)
        return lik_noise, kern_var, kern_ls


class MLSVGP(BASESVGP):

    def __init__(self,
            dim_in, dim_out,
            kern, likelihood,
            dim_h, num_h,
            feat=None,
            mean_function=None,
            q_diag=False,
            whiten=True,
            Z=None,
            num_data=None,
            **kwargs):

        super(MLSVGP, self).__init__(
            dim_in=dim_in, dim_out=dim_out,
            kern=kern, likelihood=likelihood,
            feat=feat,
            mean_function=mean_function,
            q_diag=q_diag,
            whiten=whiten,
            Z=Z,
            num_data=num_data,
            **kwargs)

        self.dim_h = dim_h
        self.num_h = num_h

        # Initialize task variables
        H_mu = np.random.randn(num_h, dim_h)
        H_var = np.log(np.ones_like(H_mu) * 0.1)
        H_init = np.hstack([H_mu, H_var])
        self.H = Parameter(H_init, dtype=settings.float_type, name="H")

        # Create placeholders
        self.H_ids_ph = tf.placeholder(tf.int32, [None])
        self.H_unique_ph = tf.placeholder(tf.int32, [None])
        self.H_scale = tf.placeholder(settings.float_type, [])

    @params_as_tensors
    def _build_likelihood(self):

        # Get prior KL.
        KL_U = self.build_prior_KL()

        H_sample, KL_H = self.sample_qH(self.H)
        KL_H = tf.reduce_sum(tf.gather(KL_H, self.H_unique_ph))
        KL_H *= self.H_scale

        H_sample = tf.gather(H_sample, self.H_ids_ph)
        XH = tf.concat([self.X_mu_ph, H_sample], 1)

        # Get conditionals
        fmean, fvar = self._build_predict(XH, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y_ph)

        lik_term = tf.reduce_sum(var_exp) * self.data_scale

        likelihood = lik_term - KL_U - KL_H

        return likelihood

    @params_as_tensors
    def _build_predict(self, XHnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(XHnew, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
                              white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(XHnew), var

    @params_as_tensors
    def _build_predict_uncertain(self, Xnew_mu, Xnew_var,
                    full_cov=False, full_output_cov=False, Luu=None):

        H = tf.gather(self.H, self.H_ids_ph)
        H_mu = H[:, :self.dim_h]
        H_var = tf.matrix_diag(tf.exp(H[:, self.dim_h:]))

        XH_mu = tf.concat([Xnew_mu, H_mu], 1)
        XH_var = block_diag(Xnew_var[0], H_var[0])[None, :, :]

        mu, var, inp_out_cov = uncertain_conditional(
            Xnew_mu=XH_mu, Xnew_var=XH_var, feat=self.feature, kern=self.kern,
            q_mu=self.q_mu, q_sqrt=self.q_sqrt, Luu=Luu,
            mean_function=self.mean_function,
            full_cov=full_cov, full_output_cov=full_output_cov, white=self.whiten)

        return mu, var, inp_out_cov[:, :-self.dim_h]

    @params_as_tensors
    def sample_qH(self, H):
        h_mu = H[:, :self.dim_h]
        h_var = tf.exp(H[:, self.dim_h:])
        qh = dist.Normal(h_mu, tf.sqrt(h_var))
        ph = dist.Normal(tf.zeros_like(h_mu), tf.ones_like(h_var))
        kl_h = dist.kl_divergence(qh, ph)
        h_sample = qh.sample()

        return h_sample, kl_h

    @params_as_tensors
    def compute_Luu(self):
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        return tf.cholesky(Kuu)

    def get_H_space(self, session):
        H = self.H.read_value(session=session)
        H_mu = H[:, :self.dim_h]
        H_var = np.exp(H[:, self.dim_h:])
        return H_mu, H_var

    def get_model_param(self, session):
        lik_noise = self.likelihood.variance.read_value(session=session)
        kern_var = self.kern.variance.read_value(session=session)
        kern_ls = self.kern.lengthscales.read_value(session=session)
        return lik_noise, kern_var, kern_ls
