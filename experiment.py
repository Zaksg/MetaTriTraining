from tri_training import TriTraining
from data_handler import DataHandler
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import datetime
from meta_features_extractor import MetaFeaturesExtracion
from classifier import Classifier
from config import Config

class Experiment:
    def __init__(self, label_rate = Config.LABEL_RATE):
        self.exp_id = datetime.now()
        self.is_extract_meta_features = True
        self.model_type = Config.MODEL_TYPE
        self.label_rate = label_rate
        self.exp_results = {}

    def start(self, ds = Config.DATASET_NAME):
        self.dataset_name = ds
        dataset = DataHandler(dataset_name = ds)
        L_X, L_y, U_X, X_test, y_test = dataset.data_split(label_rate=self.label_rate, test_rate=Config.TEST_RATE)
        
        classifier = Classifier(Config.CLASSIFIER)
        t_training = TriTraining(classifier.get_classifier(), self.is_extract_meta_features, self.model_type)

        if self.is_extract_meta_features:
            self.meta_features = MetaFeaturesExtracion()
            self.meta_features.dataset_based_mf(dataset, classifier)
            t_training.set_meta_features_extractor(self.meta_features)

        t_training.fit(dataset)
        self.res = t_training.predict(X_test)
        self.evaluation = t_training.score(X_test, y_test)
    
    def export_results(self):
        self.exp_results[self.dataset_name] = self.evaluation
        print("Accuracy for dataset {}: {}".format(self.dataset_name, self.evaluation))

    def export_meta_features(self):
        # meta_dataset = pd.DataFrame()
        list_meta_features = []
        for iteration in range(Config.MAX_ITERATIONS):
            for view in range(Config.NUM_CLASSIFIERS):
                for batch in range(Config.NUM_BATCHES):
                    temp_dict = {}
                    # additional data
                    temp_dict['dataset'] = self.dataset_name
                    temp_dict['iteration'] = iteration
                    temp_dict['view'] = view
                    temp_dict['exp_id'] = self.exp_id
                    temp_dict['label_rate'] = self.label_rate
                    # add meta features
                    temp_dict.update(self.meta_features.instance_based_meta_features[iteration][view][batch])
                    temp_dict.update(self.meta_features.view_based_meta_features[iteration][view])
                    temp_dict.update(self.meta_features.dataset_based_meta_features)
                    list_meta_features.append(temp_dict)
        meta_dataset = pd.DataFrame.from_dict(list_meta_features)
        meta_dataset.to_csv('./meta_datasets/{}_meta_features.csv'.format(self.dataset_name))
        return self.meta_features, meta_dataset
