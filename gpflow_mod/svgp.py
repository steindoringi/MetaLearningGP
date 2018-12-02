# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import tensorflow as tf

from gpflow import kullback_leiblers, features
from gpflow import settings
from gpflow import transforms
from gpflow.mean_functions import Zero
from gpflow.conditionals import conditional, Kuu
from gpflow.decors import params_as_tensors
from gpflow.params import Parameter, Parameterized


class SVGP(Parameterized):

    def __init__(self, dim_in, dim_out,
                 kern, likelihood,
                 feat=None,
                 mean_function=None,
                 q_diag=False,
                 whiten=True,
                 Z=None,
                 num_data=None,
                 q_mu=None,
                 q_sqrt=None,
                 **kwargs):

        super(SVGP, self).__init__(**kwargs)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_latent = dim_out
        self.mean_function = mean_function or Zero(output_dim=self.num_latent)
        self.kern = kern
        self.likelihood = likelihood

        self.q_diag, self.whiten = q_diag, whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

        # Create placeholders
        self.X_mu_ph = tf.placeholder(settings.float_type, [None, dim_in])
        self.X_var_ph = tf.placeholder(settings.float_type, [None, dim_in, dim_in])
        self.Y_ph = tf.placeholder(settings.float_type, [None, dim_out])
        self.data_scale = tf.placeholder(settings.float_type, [])

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):

        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = Kuu(self.feature, self.kern, jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def _build_likelihood(self):

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X_mu_ph, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y_ph)

        return tf.reduce_sum(var_exp) * self.data_scale - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(Xnew, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
                              white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var
