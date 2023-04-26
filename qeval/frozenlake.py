import math

import gym
import numpy as np
from tf_agents.environments import py_environment  # type: ignore
from tf_agents.specs import array_spec  # type: ignore
from tf_agents.trajectories import time_step as ts  # type: ignore


def to_binary(state, n_states):
    digits = math.ceil(math.log(n_states, 2))
    return np.pi * np.asarray(
        [b for b in bin(state)[2:].rjust(digits, "0")], dtype=np.float32
    )


def process_reward(reward, done):
    if done:
        if reward == 0.0:
            return -0.2
        else:
            return reward
    else:
        return -0.01


class FrozenLake(py_environment.PyEnvironment):
    def __init__(self, encoding="bin"):
        env_name = "FrozenLake-v1"
        self.encoding = encoding
        self._env = gym.make(env_name, is_slippery=False)

        if encoding == "bin":
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int64, minimum=0, maximum=3
            )

            self._observation_spec = array_spec.BoundedArraySpec(
                name="observation",
                shape=(4,),
                dtype=np.float32,
                minimum=to_binary(0, 16),
                maximum=to_binary(15, 16),
            )
        else:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int64, minimum=0, maximum=3
            )

            self._observation_spec = array_spec.BoundedArraySpec(
                name="observation", shape=(4,), dtype=np.int32, minimum=0, maximum=15
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
        self._state = self._prep_state(self._env.reset())
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        next_state, reward, done, info = self._env.step(int(action))

        reward = process_reward(reward, done)
        self._state = self._prep_state(next_state)

        if done:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=1.0)

    def _prep_state(self, state):
        if self.encoding == "bin":
            return to_binary(state, 16)
        else:
            return np.asarray([state, state, state, state], dtype=np.int32)
