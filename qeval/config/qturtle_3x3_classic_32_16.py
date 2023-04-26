CONFIG = dict(
    ENV="DEFAULT_3x3_V0",
    TYPE="classic",

    N_ITERATIONS=50000,
    COLLECT_STEPS=1,
    MAX_STEPS=200,
    INITIAL_COLLECT=5000,

    EVAL_RUNS=10,
    EVAL_EVERY=100,
    EVAL_THRESHOLD=10.5,

    BATCH_SIZE=64,
    RB_CAPACITY=20000,

    EPSILON_START=1.0,
    EPSILON_MIN=0.1,
    EPSILON_DECAY=0.99,
    DECAY_EVERY=250,

    TARGET_UPDATE=100,
    GAMMA=0.99,
    LRATE=0.001,
    LAYERS=[32, 16],
)
