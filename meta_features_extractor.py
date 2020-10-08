import numpy as np
from scipy.stats import ttest_ind, normaltest, skewtest
from sklearn.cluster import KMeans
from collections import Counter
from classifier import Classifier
from sklearn.metrics import accuracy_score
from config import Config

class MetaFeaturesExtracion:
    def __init__(self):
        self.view_based_meta_features = dict()
        self.instance_based_meta_features = dict()
        self.dataset_based_meta_features = dict()

        # init dictionaries
        for iteration in range(Config.MAX_ITERATIONS + 1):
            self.view_based_meta_features[iteration] = []
            self.instance_based_meta_features[iteration] = []
            for clf in range(Config.NUM_CLASSIFIERS):
                self.view_based_meta_features[iteration].append(dict())
                self.instance_based_meta_features[iteration].append([])
                for batch in range(Config.NUM_BATCHES):
                    self.instance_based_meta_features[iteration][clf].append(dict())

                

    def view_based_mf(self, iteration, classifier_number
        , view_j_presictions, view_k_presictions, view_i_presictions
        , X_pseudo_label, y_pseudo_label 
        , view_j_presictions_proba, view_k_presictions_proba, view_i_presictions_proba):
        
        view_based_meta_features_current = dict()
        
        # Unlabeled sets size
        view_based_meta_features_current['view_j_unlabeled_instances'] = len(view_j_presictions)
        view_based_meta_features_current['view_k_unlabeled_instances'] = len(view_k_presictions)
        view_based_meta_features_current['view_i_unlabeled_instances'] = len(view_i_presictions)
        
        # Class 1 per unlabeled set, assuming 0,1 classes in the dataset
        view_based_meta_features_current['view_j_unlabeled_instances_class_1'] = np.sum(view_j_presictions)
        view_based_meta_features_current['view_k_unlabeled_instances_class_1'] = np.sum(view_k_presictions)
        view_based_meta_features_current['view_i_unlabeled_instances_class_1'] = np.sum(view_i_presictions)
        
        # T-test means comparison for each pairs of views
        ttest_stat, ttest_pval = ttest_ind(view_j_presictions_proba[:,1], view_k_presictions_proba[:,1])
        view_based_meta_features_current['view_j_k_ttest_stat'] = ttest_stat
        view_based_meta_features_current['view_j_k_ttest_pval'] = ttest_pval
        ttest_stat, ttest_pval = ttest_ind(view_j_presictions_proba[:,1], view_i_presictions_proba[:,1])
        view_based_meta_features_current['view_j_i_ttest_stat'] = ttest_stat
        view_based_meta_features_current['view_j_i_ttest_pval'] = ttest_pval
        ttest_stat, ttest_pval = ttest_ind(view_i_presictions_proba[:,1], view_k_presictions_proba[:,1])
        view_based_meta_features_current['view_i_k_ttest_stat'] = ttest_stat
        view_based_meta_features_current['view_i_k_ttest_pval'] = ttest_pval

        # Descriptive statistics for each view
        view_based_meta_features_current.update(
            self.descriptive_statistics('confidence_view_j', view_j_presictions_proba[:,1]))
        view_based_meta_features_current.update(
            self.descriptive_statistics('confidence_view_k', view_k_presictions_proba[:,1]))
        view_based_meta_features_current.update(
            self.descriptive_statistics('confidence_view_i', view_i_presictions_proba[:,1]))

        # Check distributions
        _, view_based_meta_features_current['view_j_norm_dist_pval'] = normaltest(view_j_presictions_proba[:,1])
        _, view_based_meta_features_current['view_k_norm_dist_pval'] = normaltest(view_k_presictions_proba[:,1])
        _, view_based_meta_features_current['view_i_norm_dist_pval'] = normaltest(view_i_presictions_proba[:,1])
        
        view_based_meta_features_current['view_j_skew'], _ = skewtest(view_j_presictions_proba[:,1])
        view_based_meta_features_current['view_k_skew'], _ = skewtest(view_k_presictions_proba[:,1])
        view_based_meta_features_current['view_i_skew'], _ = skewtest(view_i_presictions_proba[:,1])
        
        # Agreement features
        view_based_meta_features_current['agreement_set_size'] = len(y_pseudo_label)
        view_based_meta_features_current['agreement_set_class_1_size'] = np.sum(y_pseudo_label)

        # Cluster (k=3) the agreement features
        kmeans = KMeans(n_clusters=3).fit(X_pseudo_label)
        kmeans_labels = Counter(kmeans.labels_)
        view_based_meta_features_current['agreement_set_cluster_0'] = kmeans_labels[0]
        view_based_meta_features_current['agreement_set_cluster_1'] = kmeans_labels[1]
        view_based_meta_features_current['agreement_set_cluster_2'] = kmeans_labels[2]

        # if classifier_number == 0:
        #     self.view_based_meta_features[iteration] = view_based_meta_features_current
        # else:
        #     self.view_based_meta_features[iteration].update(view_based_meta_features_current)
        self.view_based_meta_features[iteration][classifier_number].update(view_based_meta_features_current)


    def instance_based_mf(self, iteration, classifier_number, batches, X_unlabel, 
        view_j_presictions_proba, view_k_presictions_proba, view_i_presictions_proba):
        
        instance_based_meta_features_current = dict()
        

        for index, batch in enumerate(batches):
            if isinstance(batch[0], list):
                batch_union = batch[0] + batch[1]
            else:
                batch_union = batch[0].tolist() + batch[1].tolist()
            instances_j = view_j_presictions_proba[batch_union]
            instances_k = view_k_presictions_proba[batch_union]
            instances_i = view_i_presictions_proba[batch_union]
            
            # Descriprive
            instance_based_meta_features_current.update(
                self.descriptive_statistics('batch_confidence_view_j', instances_j[:,1]))
            instance_based_meta_features_current.update(
                self.descriptive_statistics('batch_confidence_view_k', instances_k[:,1]))
            instance_based_meta_features_current.update(
                self.descriptive_statistics('batch_confidence_view_i', instances_i[:,1]))
            
            # T-Test
            ttest_stat, ttest_pval = ttest_ind(instances_j[:,1], instances_k[:,1])
            instance_based_meta_features_current['batch_view_j_k_ttest_stat'] = ttest_stat
            instance_based_meta_features_current['batch_view_j_k_ttest_pval'] = ttest_pval
            ttest_stat, ttest_pval = ttest_ind(instances_j[:,1], instances_i[:,1])
            instance_based_meta_features_current['batch_view_j_i_ttest_stat'] = ttest_stat
            instance_based_meta_features_current['batch_view_j_i_ttest_pval'] = ttest_pval
            ttest_stat, ttest_pval = ttest_ind(instances_i[:,1], instances_k[:,1])
            instance_based_meta_features_current['batch_view_i_k_ttest_stat'] = ttest_stat
            instance_based_meta_features_current['batch_view_i_k_ttest_pval'] = ttest_pval

            self.instance_based_meta_features[iteration][classifier_number][index].update(instance_based_meta_features_current)

        # if classifier_number == 0:
        #     self.instance_based_meta_features['iteration_{}'.format(iteration)] = meta_features_per_batch
        # else:
        #     self.instance_based_meta_features['iteration_{}'.format(iteration)].update(meta_features_per_batch)



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
