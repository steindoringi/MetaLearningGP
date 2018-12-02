import numpy as np
from env_utils import initialize_envs, close_envs


def independent_loop(pilco, envs, n_iters, **kwargs):

    n_tasks = len(envs)
    solved_tasks = n_tasks * [False]
    n_trials = n_tasks * [0]

    for env_id, env in enumerate(envs):

        # Initialise env
        env = initialize_envs([env], kwargs["seed"])[0]

        ## Reinitialize model, optimizer, dataset for each env
        pilco.reset()

        for i in range(n_iters):
            if i == 0:
                ### Run random trial
                states, actions = pilco.execute_policy(env, env_id, random=True)
            else:
                ### Run MPC on env
                print("Running MPC on task {}..".format(env_id))
                states, actions = pilco.execute_policy(env, env_id)

            ### Check if env is solved
            solved_tasks[env_id] = env.check_if_solved(states)
            n_trials[env_id] += 1

            pilco.plot_rewards(env_id, target=kwargs["target_reward"])

            if solved_tasks[env_id]:
                break

            ## Re-Train GP
            print("Training model..")
            pilco.train_model()

        env.close()

    n_trials = np.int32(n_trials)
    solved = np.int32(solved_tasks)

    return n_trials, solved


def meta_loop(pilco, envs, n_iters, train, **kwargs):

    # Initialise envs
    envs = initialize_envs(envs, kwargs["seed"])

    if train:
        ## Run random trials
        start_id = 0
        for env_id, env in enumerate(envs):
            states, actions = pilco.execute_policy(env, env_id, random=True)
    else:
        start_id = pilco.n_active_tasks

    # Train GP
    print("Training model..")
    pilco.train_model()

    # Meta Loop over tasks
    n_tasks = len(envs)
    solved_tasks = n_tasks * [False]
    if train:
        n_trials = n_tasks * [1]
    else:
        n_trials = n_tasks * [0]

    for meta_iter in range(n_iters):

        print("Meta iteration {}/{}".format(meta_iter+1, n_iters))

        for env_id, env in enumerate(envs):

            ### Add task to active
            if not train and meta_iter == 0:
                pilco.n_active_tasks += 1

            ### Only run on unsolved tasks
            if solved_tasks[env_id]:
                continue

            ### Run MPC on env
            print("Running MPC on task {}..".format(env_id))
            if kwargs["model_name"] == "MLSVGP" and meta_iter == 0 and not train:
                states, actions = pilco.execute_policy(
                    env, start_id+env_id, live_inference=True)
            else:
                states, actions = pilco.execute_policy(env, start_id+env_id)

            ### Check if env is solved
            solved_tasks[env_id] = env.check_if_solved(states)
            n_trials[env_id] += 1

            pilco.plot_rewards(start_id+env_id, target=kwargs["target_reward"])

        if kwargs["model_name"] == "MLSVGP":
            pilco.plot_H_space()

        num_solved = np.sum(solved_tasks)
        print("{}/{} tasks solved.".format(
            num_solved, n_tasks))

        pilco.n_iters += 1

        if np.all(solved_tasks):
            break

        ## Re-Train GP
        print("Training model..")
        pilco.train_model()

    n_trials = np.int32(n_trials)
    solved = np.int32(solved_tasks)

    # Close envs
    envs = close_envs(envs)

    return n_trials, solved
