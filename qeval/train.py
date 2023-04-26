from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy


class MultiOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def apply_gradients(self, grads_and_vars):
        for opt, grad_var in zip(self.optimizers, grads_and_vars):
            opt.apply_gradients([grad_var])


class EpsilonDecay:
    def __init__(self, initial, minimum, rate):
        self.epsilon = initial
        self.minimum = minimum
        self.rate = rate

    def __call__(self):
        return self.epsilon

    def decay(self):
        self.epsilon = max(self.minimum, self.epsilon*self.rate)


def initial_collect(replay_buffer, env, n_steps):
    random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                    env.action_spec())

    _ = dynamic_step_driver.DynamicStepDriver(
        env,
        random_policy,
        [replay_buffer.add_batch],
        num_steps=n_steps
    ).run()
