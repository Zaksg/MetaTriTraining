class Config(object):
    # Tri training
    BATCH_SIZE = 8
    NUM_BATCHES = 100
    RESAMPLE_LABELED_RATIO = 0.9
    RANDOM_STATE = 42
    TOP_CONFIDENCE_RATIO = 0.05
    MAX_ITERATIONS = 10

    # Experiment
    LABEL_RATE = 0.8
    TEST_RATE = 0.25
    DATASET_NAME = 'phoneme' # 'phoneme','german_credit'
    MODEL_TYPE = 'original' # ['original', 'batch', 'meta']
    CLASSIFIER = 'logistic_regression'