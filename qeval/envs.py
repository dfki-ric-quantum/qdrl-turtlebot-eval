import numpy as np
import qturtle
from tf_agents.environments import (py_environment, suite_gym,
                                    tf_py_environment, utils, wrappers)
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from qeval.frozenlake import FrozenLake


class QTurtleDefaultEnv(py_environment.PyEnvironment):
    def __init__(self, env, gui=False):
        if env[8] not in ["3", "4", "5"]:
            raise ValueError(f"{env} is not a valid default env")

        self._env = qturtle.make(env, gui=gui)

        min_action = 0
        max_action = min_action + self._env.action_space.n - 1

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=min_action, maximum=max_action
        )

        max_state = float(env[8])

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=np.float32,
            minimum=np.array([0.0, 0.0, -np.pi], dtype=np.float32),
            maximum=np.array([max_state, max_state, np.pi], dtype=np.float32),
        )

        self._state = None
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def close(self):
        self._env.close()

    def _reset(self):
        self._state = self._prep_state(self._env.reset()[0])
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        next_state, reward, done, _ = self._env.step(action)
        self._state = self._prep_state(next_state[0])

        if done:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=1.0)

    def _prep_state(self, state):
        return np.array(state, dtype=np.float32)


def get_default_envs(env, max_steps, validate=False):

    if env == "CartPole-v1":
        train_py_env = suite_gym.load(env)
        eval_py_env = suite_gym.load(env)

    elif env == "FrozenLake-v1":
        train_py_env = wrappers.TimeLimit(FrozenLake(), max_steps)
        eval_py_env = wrappers.TimeLimit(FrozenLake(), max_steps)

    else:
        train_py_env = wrappers.TimeLimit(QTurtleDefaultEnv(env), max_steps)
        eval_py_env = wrappers.TimeLimit(QTurtleDefaultEnv(env), max_steps)

    if validate:
        utils.validate_py_environment(train_py_env, 5)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    return train_env, eval_env
