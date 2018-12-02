import numpy as np
import argparse, os
import json
from pilco import PILCO

from gpflow import kernels
from gpflow_mod import likelihoods

from models import MLSVGP, BASESVGP
from agents import MultiAgentMPC

from env_utils import get_envs
from model_rl import independent_loop, meta_loop


def create_model(**kwargs):

    if "BASESVGP" in kwargs["model_name"]:

        Z = np.random.randn(kwargs["n_inducing"], kwargs["dim_in"])
        mean_func = None
        kernel = kernels.RBF(kwargs["dim_in"], ARD=True)
        likelihood = likelihoods.MultiGaussian(dim=kwargs["dim_out"])

        model = BASESVGP(
            dim_in=kwargs["dim_in"], dim_out=kwargs["dim_out"],
            kern=kernel, likelihood=likelihood, mean_function=mean_func,
            Z=Z, name=kwargs["model_name"])

    elif kwargs["model_name"] == "MLSVGP":

        Z = np.random.randn(kwargs["n_inducing"], kwargs["dim_in"]+kwargs["dim_h"])
        mean_func = None
        kernel = kernels.RBF(kwargs["dim_in"] + kwargs["dim_h"], ARD=True)
        likelihood = likelihoods.MultiGaussian(dim=kwargs["dim_out"])

        model = MLSVGP(
            dim_in=kwargs["dim_in"], dim_out=kwargs["dim_out"],
            dim_h=kwargs["dim_h"], num_h=kwargs["n_envs"],
            kern=kernel, likelihood=likelihood, mean_function=mean_func,
            Z=Z, name=kwargs["model_name"])

    return model


def create_agent(model, cost, **kwargs):

    agent = MultiAgentMPC(
        model=model, cost=cost, n_agents=kwargs["n_envs"],
        dim_states=kwargs["dim_states"], dim_actions=kwargs["dim_actions"],
        dim_angles=kwargs["dim_angles"],
        episode_length=kwargs["episode_length"],
        planning_horizon=kwargs["planning_horizon"],
        name=kwargs["model_name"])

    return agent


def run_PILCO(pilco, training_envs, test_envs, **kwargs):

    if kwargs["model_name"] == "BASESVGP-I":

        all_iters, all_solved = independent_loop(
            pilco=pilco,
            envs=training_envs + test_envs,
            n_iters=kwargs["meta_train_iters"],
            **kwargs)

        n_train = len(training_envs)
        training_iters, training_solved = all_iters[:n_train], all_solved[:n_train]
        test_iters, test_solved = all_iters[n_train:], all_solved[n_train:]

    else:

        training_iters, training_solved = meta_loop(
            pilco=pilco,
            envs=training_envs,
            n_iters=kwargs["meta_train_iters"],
            train=True,
            **kwargs)

        test_iters, test_solved = meta_loop(
            pilco=pilco,
            envs=test_envs,
            n_iters=kwargs["meta_test_iters"],
            train=False,
            **kwargs)

    out_path = kwargs["experiment_path"]
    np.savez(out_path + "/training.npz", training_iters, training_solved)
    np.savez(out_path + "/test.npz", test_iters, test_solved)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="CartpoleSwingup", type=str)
    parser.add_argument("--model_name", default="BASESVGP-I", type=str)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--episode_length", default=30, type=int)
    parser.add_argument("--planning_horizon", default=10, type=int)
    parser.add_argument("--dim_h", default=2, type=int)
    parser.add_argument("--n_inducing", default=180, type=int)
    parser.add_argument("--meta_train_iters", default=10, type=int)
    parser.add_argument("--meta_test_iters", default=10, type=int)
    parser.add_argument("--model_train_steps", default=5000, type=int)
    parser.add_argument("--model_infer_steps", default=100, type=int)
    parser.add_argument("--model_learning_rate", default=1e-2, type=float)
    parser.add_argument("--batch_size", default=5, type=int)

    ARGS = parser.parse_args()
    arg_dict = vars(ARGS)

    if arg_dict["env"] == "CartpoleSwingup":
        arg_dict["dim_in"] = 6
        arg_dict["dim_out"] = 4
        arg_dict["dim_states"] = 4
        arg_dict["dim_actions"] = 1
        arg_dict["dim_angles"] = 1
        arg_dict["target_reward"] = -0.08
    elif arg_dict["env"] == "PendulumEnv":
        arg_dict["dim_in"] = 4
        arg_dict["dim_out"] = 2
        arg_dict["dim_states"] = 2
        arg_dict["dim_actions"] = 1
        arg_dict["dim_angles"] = 1
        arg_dict["target_reward"] = -0.08

    folder_keys = [
        "env",
        "model_name",
        "seed"]

    experiment_path = "experiments/"
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    for key in folder_keys:
        experiment_path += "{}-".format(arg_dict[key])
    experiment_path = experiment_path[:-1]

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        os.makedirs(experiment_path + "/img")
        os.makedirs(experiment_path + "/img/rewards")
        os.makedirs(experiment_path + "/img/H_space")
        os.makedirs(experiment_path + "/img/H_space/live_inference")

    checkpoint_path = experiment_path + "/model.ckpt"
    config_path = experiment_path + "/config.json"

    arg_dict["experiment_path"] = experiment_path
    arg_dict["checkpoint_path"] = checkpoint_path

    training_envs, test_envs, dataset, cost, labels = get_envs(**arg_dict)
    num_envs = len(training_envs) + len(test_envs)
    arg_dict["n_envs"] = num_envs
    arg_dict["n_active_tasks"] = len(training_envs)
    arg_dict["labels"] = labels

    with open(config_path, "w") as f:
        json.dump(arg_dict, f)

    model = create_model(**arg_dict)
    agent = create_agent(model, cost, **arg_dict)

    pilco = PILCO(model, agent, dataset, **arg_dict)

    run_PILCO(pilco, training_envs, test_envs, **arg_dict)
