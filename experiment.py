from tri_training import TriTraining
from data_loader import DataLoader
import pandas as pd
from sklearn import tree
import numpy as np
from datetime import datetime


class Experiment:
    def __init__(self):
        self.exp_id = datetime.now()

    def start(self):
        dataset = DataLoader(dataset_name = 'german_credit', target_col_name = 'class')
        L_X, L_y, U_X, X_test, y_test = dataset.data_split(label_rate=0.8, test_rate=0.25)
        classifier = tree.DecisionTreeClassifier()
        t_training = TriTraining(classifier)
        t_training.fit(L_X, L_y, U_X)
        self.res = t_training.predict(X_test)
        self.evaluation = t_training.score(X_test, y_test)
    
    def export_results(self):
        print(self.evaluation)

    def export_meta_features(self):
        pass
