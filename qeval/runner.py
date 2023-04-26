import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from qeval.envs import get_default_envs
from qeval.models import create_classical_model, create_quantum_model
from qeval.train import EpsilonDecay, MultiOptimizer, initial_collect
from qeval.utils import save_results, save_run_statistics


def compute_avg_reward(env, policy, n_episodes):
    total_reward = 0.0

    for _ in range(n_episodes):
        time_step = env.reset()
        ep_reward = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            ep_reward += time_step.reward
        total_reward += ep_reward

    avg_reward = total_reward / n_episodes
    return avg_reward.numpy()[0]


def run_training(config, args):
    train_env, eval_env = get_default_envs(config.ENV, config.MAX_STEPS)

    if config.ENV == "FrozenLake-v1":
        N_ACTIONS = 4
        if config.TYPE == "quantum":
            config.QUANTUM_MODEL_CONFIG.update({"env": "FrozenLake-v1"})
            config.QUANTUM_MODEL_CONFIG.update({"trainable_output": True})

    elif config.ENV == "CartPole-v1":
        N_ACTIONS = 2
        if config.TYPE == "quantum":
            config.QUANTUM_MODEL_CONFIG.update({"env": "CartPole-v1"})
            config.QUANTUM_MODEL_CONFIG.update({"trainable_output": True})

    else:
        N_ACTIONS = 3

    if config.TYPE == "classic":
        q_online = create_classical_model(N_ACTIONS, config.LAYERS)
        q_target = create_classical_model(N_ACTIONS, config.LAYERS)

        optimizer = tf.keras.optimizers.Adam(config.LRATE)
    elif config.TYPE == "quantum":
        q_online = create_quantum_model(**config.QUANTUM_MODEL_CONFIG)
        q_target = create_quantum_model(**config.QUANTUM_MODEL_CONFIG)

        optimizer = MultiOptimizer(
            [
                tf.keras.optimizers.Adam(config.INPUT_LRATE),
                tf.keras.optimizers.Adam(config.CIRCUIT_LRATE),
                tf.keras.optimizers.Adam(config.OUTPUT_LRATE),
            ]
        )

    train_step_counter = tf.Variable(0)

    epsilon = EpsilonDecay(
        config.EPSILON_START, config.EPSILON_MIN, config.EPSILON_DECAY
    )

    agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_online,
        target_q_network=q_target,
        target_update_period=config.TARGET_UPDATE,
        optimizer=optimizer,
        gamma=config.GAMMA,
        epsilon_greedy=epsilon,
        td_errors_loss_fn=common.element_wise_huber_loss,
        train_step_counter=train_step_counter,
    )

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=config.RB_CAPACITY,
    )

    initial_collect(replay_buffer, train_env, config.INITIAL_COLLECT)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=config.BATCH_SIZE, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    # Training

    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    avg_reward = compute_avg_reward(eval_env, agent.policy, config.EVAL_RUNS)
    rewards = [avg_reward]

    time_step = train_env.reset()
    policy_state = None

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        [replay_buffer.add_batch],
        num_steps=config.COLLECT_STEPS,
    )

    for _ in range(config.N_ITERATIONS):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        experience, _ = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % config.EVAL_EVERY == 0:
            avg_reward = compute_avg_reward(eval_env, agent.policy, config.EVAL_RUNS)
            print(
                f"Step {step} - eval reward {round(avg_reward, 2)}"
                f" - loss {train_loss}"
                f" - epsilon {epsilon()}"
            )
            rewards.append(avg_reward)

            if avg_reward > config.EVAL_THRESHOLD:
                print("Done")
                break

        if step % config.DECAY_EVERY == 0:
            epsilon.decay()

    train_env.close()
    eval_env.close()

    save_run_statistics(args.number, args, step, rewards[-1])

    filename = save_results(args.number, args, config, rewards)
    print(f"Results saved to {filename}")
