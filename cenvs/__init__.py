import numpy as np
from gym.envs.registration import register


def num_to_str(num):
    return "{:.2f}".format(num).replace(".", "")


""" Cartpole """

masses = np.arange(0.4, 1.1, 0.1)
lengths = np.arange(0.4, 1.1, 0.1)

for mass in masses:
    for length in lengths:

        mass_str, length_str = num_to_str(mass), num_to_str(length)
        env_id = "MTDartCartPoleSwingUp_{}-{}-v1".format(mass_str, length_str)
        env_entry_point = "cenvs.cartpole_swingup:MTDartCartPoleSwingUpEnv"
        kwargs = {"m": round(mass, 1), "l": round(length, 1)}
        register(id=env_id, entry_point=env_entry_point, kwargs=kwargs)


""" Pendulum """

masses = np.arange(0.5, 1.6, 0.1)
lengths = np.arange(0.5, 1.6, 0.1)

for mass in masses:
    for length in lengths:

        mass_str, length_str = num_to_str(mass), num_to_str(length)
        env_id = "PendulumEnv_{}-{}-v0".format(mass_str, length_str)
        env_entry_point = "cenvs.pendulum:PendulumEnv"
        kwargs = {"m": round(mass, 1), "l": round(length, 1)}
        register(id=env_id, entry_point=env_entry_point, kwargs=kwargs)
