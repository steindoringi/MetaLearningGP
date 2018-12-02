import tensorflow as tf
import numpy as np

from gpflow import settings
from gpflow.params import Parameter, Parameterized
from gpflow.decors import params_as_tensors

from func_utils import block_diag, angular_transform, covariance_scale


class MultiAgentMPC(Parameterized):

    def __init__(self, model, cost, n_agents, dim_states, dim_actions, dim_angles,
                        episode_length, planning_horizon, **kwargs):

        super(MultiAgentMPC, self).__init__(**kwargs)

        self.model = model
        self.cost = cost
        self.n_agents = n_agents
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.dim_angles = dim_angles
        self.dim_states_tf = dim_states + dim_angles
        self.episode_length = episode_length
        self.planning_horizon = planning_horizon

        init_policy = np.random.randn(
            n_agents, episode_length+planning_horizon, dim_actions)
        self.policy = Parameter(init_policy, dtype=settings.float_type)

        # Create placeholders
        self.state_mu_ph = tf.placeholder(
            settings.float_type, [1, dim_states])
        self.state_var_ph = tf.placeholder(
            settings.float_type, [1, dim_states, dim_states])
        self.current_step_ph = tf.placeholder(tf.int32, [])
        self.agent_id_ph = tf.placeholder(tf.int32, [])
        self.policy_ph = tf.placeholder(
            settings.float_type, [1, episode_length+planning_horizon, dim_actions])

        self.inp_mu = tf.placeholder(settings.float_type, [1, model.dim_in])
        self.inp_std = tf.placeholder(settings.float_type, [1, model.dim_in])
        self.out_mu = tf.placeholder(settings.float_type, [1, model.dim_out])
        self.out_std = tf.placeholder(settings.float_type, [1, model.dim_out])


    @params_as_tensors
    def _build_objective(self):

        state_mus, state_vars, state_mus_tf, state_vars_tf =\
            self._propagate(self.policy)
        costs = self.cost.expected_costs(state_mus_tf, state_vars_tf)

        return tf.reduce_sum(costs)

    @params_as_tensors
    def _propagate(self, policy):

        agent_policy = tf.gather(policy, [self.agent_id_ph])
        action_var_i = tf.zeros(
            (self.dim_actions, self.dim_actions), dtype=settings.float_type)
        Luu = self.model._compute_Luu()
        steps_left = tf.add(self.episode_length, -self.current_step_ph)
        planning_horizon = tf.cond(
            self.planning_horizon < steps_left,
            lambda: self.planning_horizon,
            lambda: steps_left)


        def loop_cond(i, state_mus, state_vars, state_mus_tf, state_vars_tf, inp_tf_cov):
            return i < planning_horizon

        def loop_body(i, state_mus, state_vars, state_mus_tf, state_vars_tf, inp_tf_cov):

            state_mu_i = state_mus[-1][None, :]
            state_var_i = state_vars[-1][None, :, :]
            state_mu_tf = state_mus_tf[-1][None, :]
            state_var_tf = state_vars_tf[-1][None, :, :]

            # Create state_action input
            action_mu_i = agent_policy[:, self.current_step_ph + i]
            X_mu_i = tf.concat([state_mu_tf, action_mu_i], 1)
            X_var_i = block_diag(state_var_tf[0], action_var_i)[None, :, :]

            # Normalize inputs
            X_mu_i = (X_mu_i - self.inp_mu) / self.inp_std
            X_var_i = covariance_scale(X_var_i, (1./self.inp_std))

            # Predict outputs
            delta_mu, delta_var, C = self.model._build_predict_uncertain(
                X_mu_i, X_var_i,
                Luu=Luu, full_cov=False, full_output_cov=True)

            # Re-normalize outputs
            delta_mu = (delta_mu * self.out_std) + self.out_mu
            delta_var = covariance_scale(delta_var, self.out_std)

            # Input-Output Covariance
            inp_tf_cov = tf.concat([
                inp_tf_cov, tf.zeros(
                    (1, self.dim_states, self.dim_actions),
                    dtype=settings.float_type)], 2)

            inp_tf_cov = inp_tf_cov * (1./self.inp_std[None, :, :])
            inp_out_cov = tf.matmul(inp_tf_cov, C)
            inp_out_cov = inp_out_cov * self.out_std[None, :, :]
            inp_out_cov = inp_out_cov + tf.transpose(inp_out_cov, [0, 2, 1])

            # New state
            new_state_mu = state_mu_i + delta_mu
            new_state_var = state_var_i + delta_var + inp_out_cov

            new_mu_tf, new_var_tf, inp_tf_cov = angular_transform(
                new_state_mu, new_state_var, self.dim_angles)

            state_mus = tf.concat([state_mus, new_state_mu], 0)
            state_vars = tf.concat([state_vars, new_state_var], 0)
            state_mus_tf = tf.concat([state_mus_tf, new_mu_tf], 0)
            state_vars_tf = tf.concat([state_vars_tf, new_var_tf], 0)

            i += 1

            return i, state_mus, state_vars, state_mus_tf, state_vars_tf, inp_tf_cov

        loop_step = tf.constant(0, tf.int32)
        init_mus_tf, init_vars_tf, inp_tf_cov = angular_transform(
            self.state_mu_ph, self.state_var_ph, self.dim_angles)

        loop_vars = [
            loop_step,
            self.state_mu_ph,
            self.state_var_ph,
            init_mus_tf,
            init_vars_tf,
            inp_tf_cov]

        shapes = [
            loop_step.get_shape(),
            tf.TensorShape([None, self.dim_states]),
            tf.TensorShape([None, self.dim_states, self.dim_states]),
            tf.TensorShape([None, self.dim_states_tf]),
            tf.TensorShape([None, self.dim_states_tf, self.dim_states_tf]),
            inp_tf_cov.get_shape()]

        _, state_mus, state_vars, state_mus_tf, state_vars_tf, inp_tf_cov = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=loop_vars,
            shape_invariants=shapes)

        return state_mus[1:], state_vars[1:], state_mus_tf[1:], state_vars_tf[1:]
