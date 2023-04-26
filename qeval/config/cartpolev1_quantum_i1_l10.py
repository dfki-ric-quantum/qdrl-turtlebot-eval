import tensorflow as tf

CONFIG = dict(
    ENV="CartPole-v1",
    TYPE="quantum",

    N_ITERATIONS=7500,
    COLLECT_STEPS=1,
    MAX_STEPS=500,
    INITIAL_COLLECT=5000,

    EVAL_RUNS=100,
    EVAL_EVERY=100,
    EVAL_THRESHOLD=475.,

    BATCH_SIZE=64,
    RB_CAPACITY=20000,

    EPSILON_START=1.0,
    EPSILON_MIN=0.05,
    EPSILON_DECAY=0.95,
    DECAY_EVERY=100,

    TARGET_UPDATE=100,
    GAMMA=0.99,

    INPUT_LRATE=0.01,
    CIRCUIT_LRATE=0.001,
    OUTPUT_LRATE=0.01,

    QUANTUM_MODEL_CONFIG={
        "n_qubits": 4,
        "layers": 10,
        "input_style": 1,
        "rot_per_unitary": 3,
        "data_reupload": True,
        "trainable_input": True,
        "zero_layer": True,
        "rescale": False,
        "activition": tf.math.atan,
        "n_states": 4,
        "n_actions": 2
    }
)
