import numpy as np
from scipy.stats import ttest_ind, normaltest, skewtest
from sklearn.cluster import KMeans
from collections import Counter
from classifier import Classifier
from sklearn.metrics import accuracy_score

class MetaFeaturesExtracion:
    def __init__(self):
        self.view_based_meta_features = dict()
        self.instance_based_meta_features = dict()
        self.dataset_based_meta_features = dict()

    def view_based_mf(self, iteration, classifier_number
        , view_j_presictions, view_k_presictions, view_i_presictions
        , X_pseudo_label, y_pseudo_label 
        , view_j_presictions_proba, view_k_presictions_proba, view_i_presictions_proba):
        
        view_based_meta_features_current = dict()
        
        # Unlabeled sets size
        view_based_meta_features_current['view_j_unlabeled_instances_classifier_{}'.format(classifier_number)] = len(view_j_presictions)
        view_based_meta_features_current['view_k_unlabeled_instances_classifier_{}'.format(classifier_number)] = len(view_k_presictions)
        view_based_meta_features_current['view_i_unlabeled_instances_classifier_{}'.format(classifier_number)] = len(view_i_presictions)
        
        # Class 1 per unlabeled set, assuming 0,1 classes in the dataset
        view_based_meta_features_current['view_j_unlabeled_instances_class_1_classifier_{}'.format(classifier_number)] = np.sum(view_j_presictions)
        view_based_meta_features_current['view_k_unlabeled_instances_class_1_classifier_{}'.format(classifier_number)] = np.sum(view_k_presictions)
        view_based_meta_features_current['view_i_unlabeled_instances_class_1_classifier_{}'.format(classifier_number)] = np.sum(view_i_presictions)
        
        # T-test means comparison for each pairs of views
        ttest_stat, ttest_pval = ttest_ind(view_j_presictions_proba[:,1], view_k_presictions_proba[:,1])
        view_based_meta_features_current['view_j_k_ttest_stat_classifier_{}'.format(classifier_number)] = ttest_stat
        view_based_meta_features_current['view_j_k_ttest_pval_classifier_{}'.format(classifier_number)] = ttest_pval
        ttest_stat, ttest_pval = ttest_ind(view_j_presictions_proba[:,1], view_i_presictions_proba[:,1])
        view_based_meta_features_current['view_j_i_ttest_stat_classifier_{}'.format(classifier_number)] = ttest_stat
        view_based_meta_features_current['view_j_i_ttest_pval_classifier_{}'.format(classifier_number)] = ttest_pval
        ttest_stat, ttest_pval = ttest_ind(view_i_presictions_proba[:,1], view_k_presictions_proba[:,1])
        view_based_meta_features_current['view_i_k_ttest_stat_classifier_{}'.format(classifier_number)] = ttest_stat
        view_based_meta_features_current['view_i_k_ttest_pval_classifier_{}'.format(classifier_number)] = ttest_pval

        # Descriptive statistics for each view
        view_based_meta_features_current.update(
            self.descriptive_statistics('confidence_view_j_classifier_{}'.format(classifier_number), view_j_presictions_proba[:,1]))
        view_based_meta_features_current.update(
            self.descriptive_statistics('confidence_view_k_classifier_{}'.format(classifier_number), view_k_presictions_proba[:,1]))
        view_based_meta_features_current.update(
            self.descriptive_statistics('confidence_view_i_classifier_{}'.format(classifier_number), view_i_presictions_proba[:,1]))

        # Check distributions
        _, view_based_meta_features_current['view_j_norm_dist_pval_classifier_{}'.format(classifier_number)] = normaltest(view_j_presictions_proba[:,1])
        _, view_based_meta_features_current['view_k_norm_dist_pval_classifier_{}'.format(classifier_number)] = normaltest(view_k_presictions_proba[:,1])
        _, view_based_meta_features_current['view_i_norm_dist_pval_classifier_{}'.format(classifier_number)] = normaltest(view_i_presictions_proba[:,1])
        
        view_based_meta_features_current['view_j_skew_classifier_{}'.format(classifier_number)], _ = skewtest(view_j_presictions_proba[:,1])
        view_based_meta_features_current['view_k_skew_classifier_{}'.format(classifier_number)], _ = skewtest(view_k_presictions_proba[:,1])
        view_based_meta_features_current['view_i_skew_classifier_{}'.format(classifier_number)], _ = skewtest(view_i_presictions_proba[:,1])
        
        # Agreement features
        view_based_meta_features_current['agreement_set_size_classifier_{}'.format(classifier_number)] = len(y_pseudo_label)
        view_based_meta_features_current['agreement_set_class_1_size_classifier_{}'.format(classifier_number)] = np.sum(y_pseudo_label)

        # Cluster (k=3) the agreement features
        kmeans = KMeans(n_clusters=3).fit(X_pseudo_label)
        kmeans_labels = Counter(kmeans.labels_)
        view_based_meta_features_current['agreement_set_cluster_0_classifier_{}'.format(classifier_number)] = kmeans_labels[0]
        view_based_meta_features_current['agreement_set_cluster_1_classifier_{}'.format(classifier_number)] = kmeans_labels[1]
        view_based_meta_features_current['agreement_set_cluster_2_classifier_{}'.format(classifier_number)] = kmeans_labels[2]

        if classifier_number == 0:
            self.view_based_meta_features[iteration] = view_based_meta_features_current
        else:
            self.view_based_meta_features[iteration].update(view_based_meta_features_current)


    def instance_based_mf(self, iteration):
        pass

    def dataset_based_mf(self, dataset, classifier):
        self.dataset_based_meta_features['num_categorical_cols'] = len(dataset.categorical_cols)
        self.dataset_based_meta_features['num_numeric_cols'] = len(dataset.numeric_cols)
        self.dataset_based_meta_features['num_instances'] = len(dataset.label)
        self.dataset_based_meta_features['num_labeled_instances'] = len(dataset.L_y)

        # Clusters
        kmeans = KMeans(n_clusters=3).fit(dataset.data)
        kmeans_labels = Counter(kmeans.labels_)
        self.dataset_based_meta_features['instances_cluster_0'] = kmeans_labels[0]
        self.dataset_based_meta_features['instances_cluster_1'] = kmeans_labels[1]
        self.dataset_based_meta_features['instances_cluster_2'] = kmeans_labels[2]

        # Initial AUC
        temp_classifier = Classifier(classifier.get_classifier_name).get_classifier()
        temp_classifier.fit(dataset.L_X, dataset.L_y)
        self.dataset_based_meta_features['initial_auc'] = accuracy_score(dataset.y_test, temp_classifier.predict(dataset.X_test))

        # Mean skewness of numeric attributes

        # Classes ratio (test set)



    def descriptive_statistics(self, feature_name_prefix, numbers_list):
        desc_features = dict()
        desc_features['{}_avg'.format(feature_name_prefix)] = np.mean(numbers_list)
        desc_features['{}_min'.format(feature_name_prefix)] = np.min(numbers_list)
        desc_features['{}_max'.format(feature_name_prefix)] = np.max(numbers_list)
        desc_features['{}_median'.format(feature_name_prefix)] = np.median(numbers_list)
        desc_features['{}_std'.format(feature_name_prefix)] = np.std(numbers_list)

        return desc_features
