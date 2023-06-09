CONFIG = dict(
    ENV="CartPole-v1",
    TYPE="classic",

    N_ITERATIONS=7500,
    COLLECT_STEPS=1,
    MAX_STEPS=500,
    INITIAL_COLLECT=5000,

    EVAL_RUNS=100,
    EVAL_EVERY=100,
    EVAL_THRESHOLD=475,

    BATCH_SIZE=64,
    RB_CAPACITY=20000,

    EPSILON_START=1.0,
    EPSILON_MIN=0.05,
    EPSILON_DECAY=0.95,
    DECAY_EVERY=100,

    TARGET_UPDATE=100,
    GAMMA=0.99,
    LRATE=0.001,
    LAYERS=[64, 64],
)
