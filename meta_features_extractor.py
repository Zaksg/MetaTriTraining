import numpy as np

class MetaFeaturesExtracion:
    def __init__(self):
        self.view_based_meta_features = dict()
        self.instance_based_meta_features = dict()
        self.dataset_based_meta_features = dict()

    def view_based_mf(self, iteration, view_1_presictions, view_2_presictions, X_pseudo_label, y_pseudo_label):
        view_based_meta_features_current = dict()

        view_based_meta_features_current['avg_confidence_view_1'] = np.mean(view_1_presictions)
        view_based_meta_features_current['avg_confidence_view_2'] = np.mean(view_2_presictions)

        self.view_based_meta_features[iteration] = view_based_meta_features_current

    def instance_based_mf(self, iteration):
        pass

    def dataset_based_mf(self, dataset):
        self.dataset_based_meta_features['num_categorical_cols'] = len(dataset.categorical_cols)
        self.dataset_based_meta_features['num_numeric_cols'] = len(dataset.numeric_cols)
