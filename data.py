import numpy as np
from func_utils import mu_std


class MultiEnvData:

    def __init__(self):

        self.trajectories = {}
        self.rewards = {}
        self.data = {}
        self.num_envs = 0

    def add_observations(self, states, actions, id):

        trajectory_data = {}
        trajectory_data["states"] = states
        trajectory_data["actions"] = actions

        if id in self.trajectories:
            self.trajectories[id].append(trajectory_data)
        else:
            self.trajectories[id] = [trajectory_data]
            self.num_envs += 1

    def get_inputs_outputs(self, states, actions):
        states_transf = self.state_transform(states)
        inputs = np.hstack([states_transf[:-1], actions])
        outputs = states[1:] - states[:-1]
        return inputs, outputs

    def prepare_data(self):

        all_input_traj = []
        all_output_traj = []
        all_ids_traj = []
        n_episodes = 0
        for eid in self.trajectories:
            episodes = self.trajectories[eid]
            for ep, ep_data in enumerate(episodes):
                states = ep_data["states"]
                actions = ep_data["actions"]
                inputs, outputs = self.get_inputs_outputs(states, actions)
                ids = np.int32(inputs.shape[0] * [eid])

                all_input_traj.append(inputs)
                all_output_traj.append(outputs)
                all_ids_traj.append(ids.reshape(-1, 1))
                n_episodes += 1

        all_inputs = np.vstack(all_input_traj)
        all_outputs = np.vstack(all_output_traj)

        inp_mu, inp_std = mu_std(all_inputs)
        out_mu, out_std = mu_std(all_outputs)

        norm = lambda inp, mu, std: (inp - mu) / std
        inp_norm = [norm(inp, inp_mu, inp_std) for inp in all_input_traj]
        out_norm = [norm(out, out_mu, out_std) for out in all_output_traj]

        n_data = all_inputs.shape[0]

        self.data["n_data"] = n_data
        self.data["n_episodes"] = n_episodes
        self.data["inputs"] = inp_norm
        self.data["outputs"] = out_norm
        self.data["ids"] = all_ids_traj
        self.data["inp_mu"] = inp_mu
        self.data["inp_std"] = inp_std
        self.data["out_mu"] = out_mu
        self.data["out_std"] = out_std

    def get_seq_batch(self, seq, si, ei):

        inp_seq = self.data["inputs"]
        out_seq = self.data["outputs"]
        ids_seq = self.data["ids"]
        inp_seq = [inp_seq[i] for i in seq]
        out_seq = [out_seq[i] for i in seq]
        ids_seq = [ids_seq[i] for i in seq]
        D = inp_seq[0].shape[1]
        E = out_seq[0].shape[1]

        X_b = np.vstack(inp_seq[si:ei]).reshape(-1, D)
        Y_b = np.vstack(out_seq[si:ei]).reshape(-1, E)
        ids_b = np.vstack(ids_seq[si:ei]).reshape(-1)
        ids_unique = np.unique(ids_b)

        return X_b, Y_b, ids_b, ids_unique

    def state_transform(self, states):
        return states


class MultiEnvData_Pendulum(MultiEnvData):

    def state_transform(self, states):

        states_transf = np.zeros((states.shape[0], states.shape[1]+1))
        states_transf[:, 0] = np.cos(states[:, 0])
        states_transf[:, 1] = np.sin(states[:, 0])
        states_transf[:, 2] = states[:, 1]
        return states_transf


class MultiEnvData_Cartpole(MultiEnvData):

    def state_transform(self, states):

        states_transf = np.zeros((states.shape[0], states.shape[1]+1))
        states_transf[:, 0] = np.cos(states[:, 0])
        states_transf[:, 1] = np.sin(states[:, 0])
        states_transf[:, 2] = states[:, 1]
        states_transf[:, 3] = states[:, 2]
        states_transf[:, 4] = states[:, 3]
        return states_transf
