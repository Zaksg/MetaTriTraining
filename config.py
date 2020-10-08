class Config(object):
    # Tri training
    BATCH_SIZE = 8
    NUM_BATCHES = 100
    RESAMPLE_LABELED_RATIO = 0.9
    RANDOM_STATE = 42
    TOP_CONFIDENCE_RATIO = 0.05
    MAX_ITERATIONS = 10
    NUM_CLASSIFIERS = 3

    # Experiment
    LABEL_RATE = 0.8
    TEST_RATE = 0.25
    MODEL_TYPE = 'batch' # ['original', 'batch', 'meta']
    CLASSIFIER = 'logistic_regression'
    DATASET_NAME = 'phoneme' # 'phoneme','german_credit'
    DATASET_NAME_LIST = [
        'abalone',
        'adult',
        'ailerons',
        'australian',
        'auto_univ',
        'blood_transfusion',
        'cancer',
        'colic',
        'cpu_act',
        'delta_elevators',
        'diabetes',
        'flare',
        'fri_c0_1000_10',
        'fri_c0_1000_25',
        'fri_c0_1000_50',
        'german_credit',
        'ilpd',
        'ionosphere',
        'japanese_vowels',
        'kc2',
        'kr_vs_kp',
        'mammography',
        'mfeat_karhunen',
        'monk',
        'ozone_level',
        'page_blocks',
        'phoneme',
        'puma32H',
        'puma8NH',
        'qsar_biodeg',
        'sick',
        'space_ga',
        'spambase',
        'threeof9',
        'tic_tac_toe',
        'vote',
        'wdbc',
        'wilt',
        'wind',
        'xd6',
    ]

    DATASET_NAME_LIST_TEST = [
        'german_credit'
    ]