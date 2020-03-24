import numpy as np

import tensorflow as tf
from tensorflow import keras as tfk
tfk.backend.set_floatx('float64')
from tensorflow_probability import distributions as tfd

#from GPflow import gpflow as gpf
import gpflow as gpf
gpf.config.set_default_float(np.float64)


class TFK_GPBASE(tfk.Model):

    def __init__(self, dim_in, dim_out, num_inducing, multi_output=False, **kwargs):

        super(TFK_GPBASE, self).__init__(**kwargs)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_inducing = num_inducing
        self.multi_output = multi_output

        if self.multi_output:
            self.dim_gp_out = 1
        else:
            self.dim_gp_out = self.dim_out

        self.GP = self._create_svgp()
        self._build()
        
    def _build(self):
        pass

    def _init_inducing(self):

        Z = np.linspace(-3., 3., self.num_inducing)[:, None]
        Z = np.tile(Z, [1, self.dim_in])
        Z = gpf.inducing_variables.InducingPoints(Z)
        return Z

    def _create_svgp(self):

        ls = tf.constant(np.array(self.dim_in*[1.]), dtype=tfk.backend.floatx())
        var = 1.0

        if self.multi_output:
            raise NotImplementedError()
        else:
            kernel = gpf.kernels.RBF(variance=var, lengthscale=ls)
        
        lik = gpf.likelihoods.Gaussian()
        Z = self._init_inducing()
        #A = np.ones((self.dim_in, self.dim_gp_out))
        #b = np.zeros((self.dim_gp_out,))
        #mean_func = gpf.mean_functions.Linear(A=A, b=b)
        mean_func = None

        svgp = gpf.models.SVGP(kernel, lik, Z, mean_function=mean_func, num_latent=self.dim_gp_out)

        #NOTE: Quick hack to assign GPflow vars to tfk.Model
        for v, var in enumerate(svgp.trainable_variables):
            setattr(self, "var_{}".format(v), var)
        
        return svgp

    def predict_F_uncertain(self, X, X_var):

        q_mu = self.GP.q_mu
        q_sqrt = self.GP.q_sqrt
        mu, var = gpf.conditionals.uncertain_conditionals.uncertain_conditional(
            X,
            X_var,
            self.GP.inducing_variable,
            self.GP.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=False,
            white=self.GP.whiten,
            full_output_cov=False)

        return mu + self.GP.mean_function(X), var

    def predict_F(self, X, X_var=None):

        X = tf.reshape(X, [-1, self.dim_in])

        if X_var is None:
            F_mu, F_var = self.GP.predict_f(X)
        else:
            X_var = tf.reshape(X_var, [-1, self.dim_in, self.dim_in])
            F_mu, F_var = self.predict_F_uncertain(X, X_var)

        #F_mu = tf.reshape(F_mu, [-1, tf.shape(X)[0], self.dim_out])
        #F_var = tf.reshape(F_var, [-1, tf.shape(X)[0], self.dim_out])
        return tf.concat([F_mu, F_var], 1)

    def predict_Y(self, X, X_var=None):

        F = self.predict_F(X, X_var)
        F = tf.reshape(F, [-1, 2*self.dim_out])
        F_mu, F_var = tf.split(F, 2, axis=-1)
        Y_mu, Y_var = self.GP.likelihood.predict_mean_and_var(F_mu, F_var)
        #Y_mu = tf.reshape(Y_mu, [tf.shape(X)[0], tf.shape(X)[1], self.dim_out])
        #Y_var = tf.reshape(Y_var, [tf.shape(X)[0], tf.shape(X)[1], self.dim_out])
        return Y_mu, Y_var

    def kl_U(self):
        return self.GP.prior_kl()

    def log_likelihood(self, Y, F):

        Y = tf.reshape(Y, [-1, self.dim_out])
        F = tf.reshape(F, [-1, 2*self.dim_out])
        F_mu, F_var = tf.split(F, 2, axis=-1)
    
        log_pY = self.GP.likelihood.variational_expectations(F_mu, F_var, Y)
        log_pY = tf.reduce_sum(log_pY)
        return log_pY


class TFSVGP(TFK_GPBASE):

    def _build(self):

        X = tfk.Input(shape=[None, self.dim_in], name="X", dtype=tfk.backend.floatx())
        inputs = {"X": X}
        F = self.call(inputs)

        self._set_input_attrs(inputs)
        #NOTE: Dictionary outputs not working for subclassed models: #25299
        self._set_output_attrs(F)

    def call(self, inputs):
        X = inputs["X"]
        return self.predict_F(X)

    def predict(self, inputs):
        X = inputs["X"]
        return self.predict_Y(X)

    def objective(self, Y, F, num_data):

        num_data = tf.cast(num_data, tfk.backend.floatx())
        minibatch_size = tf.cast(tf.shape(Y)[0]*tf.shape(Y)[1], tfk.backend.floatx())
        Y_scale = num_data / minibatch_size

        log_pY = self.log_likelihood(Y, F)
        KL_U = self.GP.prior_kl()
        return -(Y_scale * log_pY) + KL_U


class MLGP(tfk.Model):

    def __init__(self, dim_in, dim_out, dim_latent, num_latent, num_inducing,
        multi_output=False, **kwargs):

        super(MLGP, self).__init__(**kwargs)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_latent = dim_latent
        self.num_latent = num_latent
        self.inf_network = tfk.Sequential([
            tfk.layers.Dense(2*dim_latent, use_bias=False)
        ])
        self.dim_gp_in = dim_in + dim_latent
        self.GP = TFK_GPBASE(self.dim_gp_in, dim_out, num_inducing, multi_output=multi_output)

        self._build()


    def _build(self):

        X = tfk.Input(shape=[self.dim_in], name="X", dtype=tfk.backend.floatx())
        p = tfk.Input(shape=[1], name="p", dtype=tf.int32)
        inputs = {"X": X, "p": p}
        F = self.call(inputs)
        outputs = F

        self._set_input_attrs(inputs)
        #NOTE: Dictionary outputs not working for subclassed models: #25299
        self._set_output_attrs(outputs)

    def encode_p(self, p):

        p = tf.cast(p, tf.int32)
        p_inp = tf.one_hot(p, self.num_latent)[:, 0]
        H_param = self.inf_network(p_inp)
        H_mu = H_param[:, :self.dim_latent]
        H_var = tf.nn.softplus(H_param[:, self.dim_latent:])
        return H_mu, H_var

    def kl_H(self, p):

        H_mu, H_var = self.encode_p(p)
        qH = tfd.Normal(H_mu, H_var)
        pH = tfd.Normal(tf.zeros_like(H_mu), tf.ones_like(H_var))
        kl_H = tfd.kl_divergence(qH, pH)
        kl_H = tf.reduce_sum(kl_H)
        return kl_H

    def make_XH_var(self, X, H_var):

        batch_shape = [tf.shape(X)[0], tf.shape(X)[1]]
        X_var_shape = batch_shape + [self.dim_in, self.dim_in]
        UR_shape = batch_shape + [self.dim_in, self.dim_latent]
        LL_shape = batch_shape + [self.dim_latent, self.dim_in]

        X_var = tf.zeros(shape=X_var_shape, dtype=tfk.backend.floatx())
        UR = tf.zeros(shape=UR_shape, dtype=tfk.backend.floatx())
        LL = tf.zeros(shape=LL_shape, dtype=tfk.backend.floatx())

        H_var = tf.linalg.diag(H_var)[:, None]
        H_var = tf.tile(H_var, [1, tf.shape(X)[1], 1, 1])

        XHU_var = tf.concat([X_var, UR], 2)
        XHL_var = tf.concat([LL, H_var], 2)
        XH_var = tf.concat([XHU_var, XHL_var], 1)
        return XH_var

    def make_XH(self, inputs, sample=True):

        X = inputs["X"]
        p = inputs["p"]

        H_mu, H_var = self.encode_p(p)
        qH = tfd.Normal(H_mu, H_var)
        if sample:
            H_sample = qH.sample()
            #H_sample = tf.tile(H_sample[:, None], [1, tf.shape(X)[1], 1])
            XH = tf.concat([X, H_sample], 1)
            return XH
        else:
            XH_var = self.make_XH_var(X, H_var)
            #H_mu = tf.tile(H_mu[:, None], [1, tf.shape(X)[1], 1])
            XH_mu = tf.concat([X, H_mu], 1)
            return XH_mu, XH_var

    def call(self, inputs):
        XH = self.make_XH(inputs)
        return self.GP.predict_F(XH)

    def predict(self, inputs, sample=True):
        if sample:
            XH_mu = self.make_XH(inputs, sample=sample)
            XH_var = None
        else:
            XH_mu, XH_var = self.make_XH(inputs, sample=sample)

        return self.GP.predict_Y(XH_mu, XH_var)

    def objective(self, Yp, F, num_data, num_tasks):

        Y = Yp[:, :self.dim_out]
        p = Yp[:, self.dim_out:]

        num_data = tf.cast(num_data, tfk.backend.floatx())
        #minibatch_size = tf.cast(tf.shape(Y)[0]*tf.shape(Y)[1], tfk.backend.floatx())
        minibatch_size = tf.cast(tf.shape(Y)[0], tfk.backend.floatx())
        Y_scale = num_data / minibatch_size

        num_tasks = tf.cast(num_tasks, tfk.backend.floatx())
        #task_minibatch_size = tf.cast(tf.shape(Y)[0], tfk.backend.floatx())
        H_scale = num_tasks / minibatch_size

        log_pY = self.GP.log_likelihood(Y, F)
        KL_U = self.GP.kl_U()
        KL_H = self.kl_H(p)

        return -(Y_scale * log_pY) + KL_U + (H_scale * KL_H)


class MLGP_OLD(tfk.Model):

    def __init__(self, dim_in, dim_out, dim_latent, num_latent, num_inducing,
        multi_output=False, **kwargs):

        super(MLGP, self).__init__(**kwargs)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_latent = dim_latent
        self.num_latent = num_latent
        self.H = self._create_H_param(num_latent)
        self.dim_gp_in = dim_in + dim_latent
        self.GP = TFK_GPBASE(self.dim_gp_in, dim_out, num_inducing, multi_output=multi_output)

        self._build()


    def _build(self):

        X = tfk.Input(shape=[None, self.dim_in], name="X", dtype=tfk.backend.floatx())
        p = tfk.Input(shape=[1], name="p", dtype=tf.int32)
        inputs = {"X": X, "p": p}
        F = self.call(inputs)

        self._set_input_attrs(inputs)
        #NOTE: Dictionary outputs not working for subclassed models: #25299
        self._set_output_attrs(F)

    def _create_H_param(self, num_var):
        var_name = "task_vars_{}-{}".format(self.num_latent, self.num_latent+num_var)
        H = tf.constant(np.random.randn(num_var, 2*self.dim_latent))
        return tf.Variable(H, dtype=tfk.backend.floatx(), name=var_name)

    def H_param(self, p=None):

        if p is None:
            H_mu = self.H[:, :self.dim_latent]
            H_var = tf.nn.softplus(self.H[:, self.dim_latent:])
        else:
            H_p = tf.gather(self.H, tf.squeeze(p, axis=1))
            H_mu = H_p[:, :self.dim_latent]
            H_var = tf.nn.softplus(H_p[:, self.dim_latent:])
        return H_mu, H_var

    def kl_H(self, p=None):

        H_mu, H_var = self.H_param(p=p)
        qH = tfd.Normal(H_mu, H_var)
        pH = tfd.Normal(tf.zeros_like(H_mu), tf.ones_like(H_var))
        kl_H = tfd.kl_divergence(qH, pH)
        return kl_H

    def make_XH_var(self, X, H_var):

        batch_shape = [tf.shape(X)[0], tf.shape(X)[1]]
        X_var_shape = batch_shape + [self.dim_in, self.dim_in]
        UR_shape = batch_shape + [self.dim_in, self.dim_latent]
        LL_shape = batch_shape + [self.dim_latent, self.dim_in]

        X_var = tf.zeros(shape=X_var_shape, dtype=tfk.backend.floatx())
        UR = tf.zeros(shape=UR_shape, dtype=tfk.backend.floatx())
        LL = tf.zeros(shape=LL_shape, dtype=tfk.backend.floatx())

        H_var = tf.linalg.diag(H_var)[:, None]
        H_var = tf.tile(H_var, [1, tf.shape(X)[1], 1, 1])

        XHU_var = tf.concat([X_var, UR], 2)
        XHL_var = tf.concat([LL, H_var], 2)
        XH_var = tf.concat([XHU_var, XHL_var], 1)
        return XH_var

    def make_XH(self, inputs, sample=True):

        X = inputs["X"]
        p = inputs["p"]

        H_mu, H_var = self.H_param(p)
        qH = tfd.Normal(H_mu, H_var)
        if sample:
            H_sample = qH.sample()
            H_sample = tf.tile(H_sample[:, None], [1, tf.shape(X)[1], 1])
            XH = tf.concat([X, H_sample], 2)
            return XH
        else:
            XH_var = self.make_XH_var(X, H_var)
            H_mu = tf.tile(H_mu[:, None], [1, tf.shape(X)[1], 1])
            XH_mu = tf.concat([X, H_mu], 2)
            return XH_mu, XH_var

    def call(self, inputs):
        XH = self.make_XH(inputs)
        return self.GP.predict_F(XH)

    def predict(self, inputs, sample=True):
        if sample:
            XH_mu = self.make_XH(inputs, sample=sample)
            XH_var = None
        else:
            XH_mu, XH_var = self.make_XH(inputs, sample=sample)

        return self.GP.predict_Y(XH_mu, XH_var)

    def objective(self, Y, F, num_data, num_tasks):

        num_data = tf.cast(num_data, tfk.backend.floatx())
        minibatch_size = tf.cast(tf.shape(Y)[0]*tf.shape(Y)[1], tfk.backend.floatx())
        Y_scale = num_data / minibatch_size

        #NOTE: Doing KL over full task space, so no scaling.
        #NOTE: Make more efficient later (need task indices for H)
        #num_tasks = tf.cast(num_tasks, tfk.backend.floatx())
        #task_minibatch_size = tf.cast(tf.shape(Y)[0], tfk.backend.floatx())
        #H_scale = num_tasks / task_minibatch_size

        log_pY = self.GP.log_likelihood(Y, F)
        KL_U = self.GP.kl_U()
        KL_H = self.kl_H()
        return -(Y_scale * log_pY) + KL_U + KL_H


