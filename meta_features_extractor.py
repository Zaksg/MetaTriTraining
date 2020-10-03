import numpy as np

class MetaFeaturesExtracion:
    def __init__(self):
        self.view_based_meta_features = dict()
        self.instance_based_meta_features = dict()
        self.dataset_based_meta_features = dict()

    def view_based_mf(self, iteration, classifier_number, view_j_presictions, view_k_presictions, view_i_presictions, X_pseudo_label, y_pseudo_label 
        , view_j_presictions_proba, view_k_presictions_proba, view_i_presictions_proba):
        
        view_based_meta_features_current = dict()

        view_based_meta_features_current['avg_confidence_view_j_classifier_{}'.format(classifier_number)] = np.mean(view_j_presictions_proba[:,1])
        view_based_meta_features_current['avg_confidence_view_k_classifier_{}'.format(classifier_number)] = np.mean(view_k_presictions_proba[:,1])

        self.view_based_meta_features[iteration] = view_based_meta_features_current

    def instance_based_mf(self, iteration):
        pass

    def dataset_based_mf(self, dataset):
        self.dataset_based_meta_features['num_categorical_cols'] = len(dataset.categorical_cols)
        self.dataset_based_meta_features['num_numeric_cols'] = len(dataset.numeric_cols)
