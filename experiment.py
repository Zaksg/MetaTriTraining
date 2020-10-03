from tri_training import TriTraining
from data_handler import DataHandler
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import datetime
from meta_features_extractor import MetaFeaturesExtracion
from classifier import Classifier


class Experiment:
    def __init__(self, label_rate = 0.8):
        self.exp_id = datetime.now()
        self.is_extract_meta_features = True
        self.is_use_meta_model = False
        self.label_rate = label_rate

    def start(self):
        # 'phoneme','german_credit'
        dataset = DataHandler(dataset_name = 'phoneme') # , target_col_name = 'Class'
        L_X, L_y, U_X, X_test, y_test = dataset.data_split(label_rate=self.label_rate, test_rate=0.25)
        
        classifier = Classifier('logistic_regression')
        t_training = TriTraining(classifier.get_classifier(), self.is_extract_meta_features, self.is_use_meta_model)

        if self.is_extract_meta_features:
            self.meta_features = MetaFeaturesExtracion()
            self.meta_features.dataset_based_mf(dataset, classifier)
            t_training.set_meta_features_extractor(self.meta_features)

        t_training.fit(L_X, L_y, U_X)
        self.res = t_training.predict(X_test)
        self.evaluation = t_training.score(X_test, y_test)
    
    def export_results(self):
        print(self.evaluation)

    def export_meta_features(self):
        return self.meta_features
