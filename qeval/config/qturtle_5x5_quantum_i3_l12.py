import tensorflow as tf

CONFIG = dict(
    ENV="DEFAULT_5x5_V0",
    TYPE="quantum",

    N_ITERATIONS=50000,
    COLLECT_STEPS=1,
    MAX_STEPS=200,
    INITIAL_COLLECT=5000,

    EVAL_RUNS=10,
    EVAL_EVERY=100,
    EVAL_THRESHOLD=10.,

    BATCH_SIZE=64,
    RB_CAPACITY=20000,

    EPSILON_START=1.0,
    EPSILON_MIN=0.1,
    EPSILON_DECAY=0.99,
    DECAY_EVERY=250,

    TARGET_UPDATE=100,
    GAMMA=0.99,

    INPUT_LRATE=0.01,
    CIRCUIT_LRATE=0.001,
    OUTPUT_LRATE=0.01,

    QUANTUM_MODEL_CONFIG={
        "n_qubits": 3,
        "layers": 12,
        "input_style": 3,
        "rot_per_unitary": 3,
        "data_reupload": True,
        "trainable_input": True,
        "zero_layer": True,
        "rescale": False,
        "activition": tf.math.atan,
        "n_states": 3,
        "n_actions": 3
    }
)
