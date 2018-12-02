import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import os


class MTDartCartPoleSwingUpEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self, m=0.5, l=0.5, visualize=False, disableViewer=True):
        self.control_bounds = np.array([[1.0],[-1.0]])
        self.action_scale = 20
        self.m = m
        self.l = l

        asset_path = "{}/cenvs/assets".format(os.getcwd())
        skel_path = "{}/cartpole_swingup_{}-{}.skel".format(asset_path, m, l)

        dart_env.DartEnv.__init__(
            self, model_paths=skel_path,
            frame_skip=10, observation_size=4,
            action_bounds=self.control_bounds, dt=0.01,
            visualize=visualize, disableViewer=disableViewer)
        utils.EzPickle.__init__(self)

    def step(self, a):
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale

        self.do_simulation(tau, self.frame_skip)

        done = abs(self.robot_skeleton.q[0]) >= 10.

        x = self.robot_skeleton.q[0]
        theta = self.robot_skeleton.q[1] - np.pi
        reward = -np.sqrt(self.squared_distance(x, theta, self.l))

        return bool(done), reward

    def reset_model(self):
        self.dart_world.reset()

        cart_pos = self.np_random.uniform(low=-0.2, high=0.2)
        ang_pos = self.np_random.uniform(low=-0.5, high=0.5)
        qpos = np.float32([cart_pos, ang_pos])
        qvel = self.np_random.uniform(low=-0.2, high=0.2, size=self.robot_skeleton.ndofs)

        #qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)
        #qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        if self.np_random.uniform(low=0, high=1, size=1) > 0.5:
            qpos[1] += np.pi
        else:
            qpos[1] += -np.pi

        self.set_state(qpos, qvel)
        return self.state

    def squared_distance(self, x, theta, l):
        return x**2 + 2*x*l*np.sin(theta) + 2*(l**2) + 2*(l**2)*np.cos(theta)

    def check_if_solved(self, states):
        x = states[:, 1]
        theta = states[:, 0]
        dist = np.sqrt(self.squared_distance(x, theta, self.l))
        solved = np.all(dist[-10:] < 0.08)
        return solved

    @property
    def state(self):
        cart_pos = self.robot_skeleton.q[0]
        theta = self.robot_skeleton.q[1]
        theta -= np.pi #Shift by pi to match PILCO cost
        cart_vel = self.robot_skeleton.dq[0]
        theta_vel = self.robot_skeleton.dq[1]
        return np.hstack([theta, cart_pos, cart_vel, theta_vel])

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
