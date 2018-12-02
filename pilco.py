import tensorflow as tf
import numpy as np
import gpflow
import gym
import data
import sys, argparse, os

from tqdm import tqdm

from gpflow import settings
from gpflow import kernels
from gpflow_mod import likelihoods
from gpflow.params import Parameter
from gpflow.decors import params_as_tensors_for

from training import initialize_model, initialize_policy

from gpflow.logdensities import gaussian
import pylab


class PILCO(object):

    def __init__(self, model, agent, dataset, **kwargs):

        self.kwargs = kwargs
        self.rng = np.random.RandomState(kwargs["seed"])
        self.session = tf.Session()

        self.model = model
        self.agent = agent
        self.dataset = dataset

        self._build_model_graph()
        self._build_policy_graph()

        self.n_iters = 0
        self.n_active_tasks = kwargs["n_active_tasks"]

    def _build_model_graph(self):

        self.model_objective = -self.model._build_likelihood()
        self.model_train_step, self.model_infer_step, self.model_optimizer =\
            initialize_model(self.model, self.model_objective,
            self.session, self.kwargs["model_learning_rate"])

    def _build_policy_graph(self):

        self.policy_objective = self.agent._build_objective()
        self.policy_optimizers = initialize_policy(
            self.agent, self.policy_objective, self.session)

    def _set_inducing(self):

        n_data = self.dataset.data["n_data"]
        n_inducing = self.kwargs["n_inducing"]
        X = np.vstack(self.dataset.data["inputs"])
        Z = self.model.feature.Z.read_value(session=self.session)

        diff = n_inducing - n_data
        if diff >= 0:
            Z[:n_data, :X.shape[1]] = X
        else:
            seq = np.arange(n_data)
            self.rng.shuffle(seq)
            Z[:, :X.shape[1]] = X[seq[:n_inducing]]

        self.model.feature.Z = Z

    def reset(self):

        self.session.close()
        self.session = tf.Session()
        self.model_train_step, self.model_infer_step, self.model_optimizer =\
            initialize_model(self.model, self.model_objective,
            self.session, self.kwargs["model_learning_rate"])
        self.policy_optimizers = initialize_policy(
            self.agent, self.policy_objective, self.session)
        self.dataset.__init__()

    def train_model(self):

        dataset = self.dataset
        dataset.prepare_data()
        kwargs = self.kwargs
        n_data = dataset.data["n_data"]
        n_inducing = kwargs["n_inducing"]
        n_episodes = dataset.data["n_episodes"]
        batch_size = kwargs["batch_size"]
        num_batches = max(int((n_episodes / batch_size)), 1)
        seq = np.arange(n_episodes)

        if (self.n_iters == 1) or (n_data <= n_inducing):
            self._set_inducing()

        for step in tqdm(range(kwargs["model_train_steps"])):
            all_obj = []
            self.rng.shuffle(seq)
            for b in range(int(num_batches)):
                si = b * batch_size
                ei = si + batch_size

                X_b, Y_b, ids_b, ids_unique = dataset.get_seq_batch(seq, si, ei)
                data_scale = n_data / X_b.shape[0]
                H_scale = self.n_active_tasks / ids_unique.shape[0]

                feed_dict = {
                    self.model.X_mu_ph: X_b,
                    self.model.Y_ph: Y_b,
                    self.model.data_scale: data_scale
                }

                if self.model.name == "MLSVGP":
                    feed_dict[self.model.H_ids_ph] = ids_b
                    feed_dict[self.model.H_unique_ph] = ids_unique
                    feed_dict[self.model.H_scale] = H_scale

                _, obj = self.session.run(
                    [self.model_train_step, self.model_objective],
                    feed_dict=feed_dict)

                all_obj.append(obj)

            mobj = np.mean(all_obj)
            #print("Step {}/{} :: {:.2f}".format(
            #    step+1, kwargs["model_train_steps"], mobj))

    def infer_task_variable(self, env_id, states, actions):

        dataset = self.dataset
        kwargs = self.kwargs
        n_data = dataset.data["n_data"]

        norm = lambda inp, mu, std: (inp - mu) / std
        states = np.vstack(states)
        actions = np.vstack(actions)
        inputs, outputs = dataset.get_inputs_outputs(states, actions)
        inp_norm = norm(inputs, dataset.data["inp_mu"], dataset.data["inp_std"])
        out_norm = norm(outputs, dataset.data["out_mu"], dataset.data["out_std"])
        batch_size = inputs.shape[0]
        ids = np.int32(batch_size * [env_id])
        data_scale = n_data / batch_size

        for step in range(kwargs["model_infer_steps"]):

            feed_dict = {
                self.model.X_mu_ph: inp_norm,
                self.model.Y_ph: out_norm,
                self.model.data_scale: data_scale,
                self.model.H_ids_ph: ids,
                self.model.H_unique_ph: [env_id],
                self.model.H_scale: self.n_active_tasks
            }

            _, obj = self.session.run(
                [self.model_infer_step, self.model_objective],
                feed_dict=feed_dict)

            #print("Step {}/{} :: {:.2f}".format(
            #    step+1, kwargs["model_infer_steps"], obj))


    def run_mpc(self, state, current_step, **mpc_kwargs):

        agent_id = mpc_kwargs["agent_id"]
        optimizer = self.policy_optimizers
        state_var = np.zeros((1, self.agent.dim_states, self.agent.dim_states))

        feed_dict = {
            self.agent.state_mu_ph: state,
            self.agent.state_var_ph: state_var,
            self.agent.current_step_ph: current_step,
            self.agent.agent_id_ph: agent_id,
            self.agent.inp_mu: mpc_kwargs["inp_mu"],
            self.agent.inp_std: mpc_kwargs["inp_std"],
            self.agent.out_mu: mpc_kwargs["out_mu"],
            self.agent.out_std: mpc_kwargs["out_std"],
            self.agent.cost.length: mpc_kwargs["length"]}

        if self.model.name == "MLSVGP":
            feed_dict[self.agent.model.H_ids_ph] = [agent_id]

        optimizer.minimize(
            self.session,
            feed_dict=feed_dict)

        policy = self.agent.policy.read_value(session=self.session)
        action = policy[agent_id, current_step]
        return action.reshape(-1)


    def execute_policy(self, env, env_id, random=False, live_inference=False):

        _ = env.reset()

        states = [env.state.reshape(1, -1)]
        actions = []
        rewards = []

        current_step = 0
        t = 0
        episode_length = self.kwargs["episode_length"]
        pbar = tqdm(total=episode_length)
        while t < episode_length:

            if live_inference:
                if t > 0:
                    self.infer_task_variable(env_id, states, actions)
                self.plot_live_inference(env_id, t)

            if random:
                action = np.float32([np.random.uniform(low=-1., high=1.)])
            else:
                mpc_kwargs = {
                    "agent_id": env_id,
                    "inp_mu": self.dataset.data["inp_mu"].reshape(1, -1),
                    "inp_std": self.dataset.data["inp_std"].reshape(1, -1),
                    "out_mu": self.dataset.data["out_mu"].reshape(1, -1),
                    "out_std": self.dataset.data["out_std"].reshape(1, -1),
                    "length": env.l
                }

                action = self.run_mpc(states[-1], current_step, **mpc_kwargs)

            stop, reward = env.step(action)
            #env.render()

            if stop:
                print("# WARNING: Agent out of bounds, resetting env.")
                states = np.vstack(states)
                actions = np.vstack(actions)
                self.dataset.add_observations(states, actions, env_id)
                _ = env.reset()
                states = [env.state.reshape(1, -1)]
                actions = []
                current_step = 0
                continue

            states.append(env.state.reshape(1, -1))
            actions.append(action.reshape(1, -1))
            rewards.append(reward.reshape(1, 1))

            current_step += 1
            t += 1
            pbar.update(1)

        pbar.close()

        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        self.dataset.add_observations(states, actions, env_id)
        if env_id in self.dataset.rewards:
            self.dataset.rewards[env_id].append(rewards)
        else:
            self.dataset.rewards[env_id] = [rewards]

        return states, actions

    def plot_rewards(self, env_id, target):

        rewards = self.dataset.rewards[env_id][-1]
        episode = len(self.dataset.rewards[env_id])
        target = np.float32(rewards.shape[0] * [target])
        img_path = self.kwargs["experiment_path"]
        img_path += "/img/rewards/task={}_episode={}-rewards.png".format(
            env_id, episode)
        pylab.plot(rewards.reshape(-1))
        pylab.plot(target)
        pylab.savefig(img_path)
        pylab.clf()
        pylab.close("all")

    def plot_H_space(self, xmax=3, ymax=3):

        img_path = self.kwargs["experiment_path"]
        img_path += "/img/H_space/iter={}.png".format(self.n_iters)
        H_mu, H_var = self.model.get_H_space(session=self.session)
        H_err = 2*np.sqrt(H_var)
        for h in range(self.n_active_tasks):
            pylab.errorbar(
                H_mu[h, 0], H_mu[h, 1],
                xerr=H_err[h, 0], yerr=H_err[h, 1], fmt="o",
                label="Task {}".format(h))

        pylab.xlim(-xmax, xmax)
        pylab.ylim(-ymax, ymax)
        pylab.legend()
        pylab.savefig(img_path)
        pylab.clf()
        pylab.close("all")

    def plot_live_inference(self, env_id, step, xmax=3, ymax=3):

        img_path = self.kwargs["experiment_path"]
        img_path += "/img/H_space/live_inference/task={}_step={}.png".format(
            env_id, step)
        labels = self.kwargs["labels"]
        if labels is None:
            labels = ["Task {}".format(h) for h in range(self.n_active_tasks)]
        H_mu, H_var = self.model.get_H_space(session=self.session)
        H_err = 2*np.sqrt(H_var)
        for h in range(self.n_active_tasks):
            pylab.errorbar(
                H_mu[h, 0], H_mu[h, 1],
                xerr=H_err[h, 0], yerr=H_err[h, 1], fmt="o",
                label=labels[h])

        pylab.xlim(-xmax, xmax)
        pylab.ylim(-ymax, ymax)
        pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        pylab.tight_layout()
        pylab.savefig(img_path)
        pylab.clf()
        pylab.close("all")
