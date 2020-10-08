import numpy as np
import sklearn
import math
import pandas as pd
from itertools import combinations, product
from config import Config

# Default parameters
# BATCH_SIZE = 8
# NUM_BATCHES = 100
# RESAMPLE_LABELED_RATIO = 0.9
# RANDOM_STATE = 42
# TOP_CONFIDENCE_RATIO = 0.05
# MAX_ITERATIONS = 10

class TriTraining:
    def __init__(self, classifier, is_extract_meta_features = False, model_type = 'original'):
        '''
        model_type: {'original', 'batch', 'meta'}. The model type usage - 
            i. original is the original tri-training model. 
            ii. batch is the combination of models agreement and top confidence instancas per classifier and their combination to a batch of 8 instances
            iii. meta is the usage of the meta model, after generation the batches to add
        is_extract_meta_features: bool. Use the meta-feature extraction functions. 
        '''
        self.is_extract_meta_features = is_extract_meta_features
        self.model_type = model_type

        if sklearn.base.is_classifier(classifier):
            self.classifiers = [sklearn.base.clone(classifier) for i in range(Config.NUM_CLASSIFIERS)]
        else:
            self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(Config.NUM_CLASSIFIERS)]
            
    def fit(self, dataset):
        X_label, y_label, X_unlabel, X_test, y_test = dataset.L_X, dataset.L_y, dataset.U_X, dataset.X_test, dataset.y_test
        self.dataset_exp = dataset
        self.X_label = X_label
        self.y_label = y_label

        # BootstrapSample
        views_samples = []
        for i in range(Config.NUM_CLASSIFIERS):
            sample = sklearn.utils.resample(self.X_label, self.y_label, n_samples=int(Config.RESAMPLE_LABELED_RATIO*len(self.y_label))
                , random_state=Config.RANDOM_STATE)
            views_samples.append(sample)
            self.classifiers[i].fit(*sample)

        # Initial variables 
        classification_error_current = [0.5]*Config.NUM_CLASSIFIERS
        pseudo_label_size_current = [0]*Config.NUM_CLASSIFIERS
        classification_error = [0]*Config.NUM_CLASSIFIERS
        pseudo_label_size = [0]*Config.NUM_CLASSIFIERS
        X_pseudo_label_index = [[]]*Config.NUM_CLASSIFIERS
        X_pseudo_label_index_current = [[]]*Config.NUM_CLASSIFIERS
        update = [False]*Config.NUM_CLASSIFIERS
        improve = True
        self.iter = -1
        
        while improve:
            self.iter += 1
            # Test set evaluation
            # print ("Test set score for iteration {} is: {}".format(self.iter, self.score(X_test, y_test)))
            for i in range(Config.NUM_CLASSIFIERS):
                X_pseudo_label_index_current[i] = X_pseudo_label_index[i]

            # The new pseudo label set determined by tri-training iteration for classifier i
            # X_pseudo_label_index, contains the data record index (in the full unlabelled set) of the new pseudo label set determined by tri-training iteration for classifier i
            # X_pseudo_label, contains the features for new pseudo label set determined by tri-training iteration for classifier i
            # y_pseudo_label, contains the labels (not ground truth label, but pseudo label calculated by tri-training iteration) for new pseudo label set determined by tri-training iteration for classifier i
            X_pseudo_label_index = [[]]*Config.NUM_CLASSIFIERS
            X_pseudo_label = [[]]*Config.NUM_CLASSIFIERS
            y_pseudo_label = [[]]*Config.NUM_CLASSIFIERS
            
            # Loop unlabeled set
            for i in range(Config.NUM_CLASSIFIERS):    
                j, k = np.delete(np.array([0,1,2]),i)
                update[i] = False
                classification_error[i] = self.measure_error(self.X_label, self.y_label, j, k)
                               
                if self.model_type == 'original':
                    stop_criteria = classification_error[i] < classification_error_current[i]
                else:
                    stop_criteria = self.iter < Config.MAX_ITERATIONS

                if stop_criteria:
                    U_y_j = self.classifiers[j].predict(X_unlabel)
                    U_y_k = self.classifiers[k].predict(X_unlabel)
                    U_y_i = self.classifiers[i].predict(X_unlabel)
                    X_pseudo_label[i] = X_unlabel[U_y_j == U_y_k] # models agreement
                    y_pseudo_label[i] = U_y_j[U_y_j == U_y_k]
                    # X_pseudo_label_index[i] = np.where(U_y_j==U_y_k)
                    pseudo_label_size[i] = len(y_pseudo_label[i])

                    # Confidence score
                    U_y_j_proba = self.classifiers[j].predict_proba(X_unlabel)
                    U_y_k_proba = self.classifiers[k].predict_proba(X_unlabel)
                    U_y_i_proba = self.classifiers[i].predict_proba(X_unlabel)

                    # Get meta features
                    if self.is_extract_meta_features:
                        self.meta_features_extractor.view_based_mf(self.iter, i, U_y_j, U_y_k, U_y_i, X_pseudo_label[i], y_pseudo_label[i], 
                            U_y_j_proba, U_y_k_proba, U_y_i_proba)

                    if self.model_type == 'original':
                        # Continue tri-training flow
                        if pseudo_label_size_current[i] == 0: # first updated
                            pseudo_label_size_current[i]  = int(classification_error[i]/(classification_error_current[i] - classification_error[i]) + 1)
                        if pseudo_label_size_current[i] < pseudo_label_size[i]:
                            if classification_error[i] * pseudo_label_size[i] < classification_error_current[i] * pseudo_label_size_current[i]:
                                update[i] = True
                            elif pseudo_label_size_current[i] > classification_error[i]/(classification_error_current[i] - classification_error[i]):
                                resample_size = math.ceil(classification_error_current[i] * pseudo_label_size_current[i] / classification_error[i] - 1)
                                X_pseudo_label[i], y_pseudo_label[i] = sklearn.utils.resample(
                                    X_pseudo_label[i],y_pseudo_label[i],replace=False,n_samples=resample_size, random_state=Config.RANDOM_STATE)
                                pseudo_label_size[i] = len(y_pseudo_label[i])
                                update[i] = True
                    
                    else:
                        ### Generate pseudo-label candidates
                        generated_batches = self.generate_labeling_candidates(U_y_j_proba, U_y_k_proba, y_pseudo_label[i], batch_size=Config.BATCH_SIZE)
                        self.meta_features_extractor.instance_based_mf(self.iter, i, generated_batches, X_unlabel, U_y_j_proba, U_y_k_proba, U_y_i_proba)
                    
                        # Batch selection
                        if self.model_type == 'batch':
                            # Take the first batch
                            selected_batch = generated_batches[0][0].tolist() + generated_batches[0][1].tolist()
                            X_pseudo_label[i] = X_unlabel.iloc[selected_batch]
                            y_pseudo_label[i] = [0]*len(generated_batches[0][0]) + [1]*len(generated_batches[0][1].tolist())
                        elif self.model_type == 'meta':
                            pass

                        pseudo_label_size[i] = Config.BATCH_SIZE
                        update[i] = True

             
            for i in range(Config.NUM_CLASSIFIERS):
                if update[i]:
                    self.classifiers[i].fit(np.append(self.X_label,X_pseudo_label[i],axis=0), np.append(self.y_label, y_pseudo_label[i], axis=0))
                    classification_error_current[i] = classification_error[i]
                    pseudo_label_size_current[i] = pseudo_label_size[i]
    
            if update == [False]*3:
                improve = False


    def predict(self, X):
        pred = np.asarray([self.classifiers[i].predict(X) for i in range(Config.NUM_CLASSIFIERS)])
        pred[0][pred[1]==pred[2]] = pred[1][pred[1]==pred[2]]
        return pred[0]
        
    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))
        
    def measure_error(self, X, y, j, k):
        j_pred = self.classifiers[j].predict(X)
        k_pred = self.classifiers[k].predict(X)
        wrong_index =np.logical_and(j_pred != y, k_pred==j_pred)
        return sum(wrong_index)/sum(j_pred == k_pred)

    def set_meta_features_extractor(self, mf_extractor):
        self.meta_features_extractor = mf_extractor

    def top_confidence(self, confidence_list):
        class_1_conf = confidence_list[:,1]
        
        condidates_per_class = int(len(class_1_conf) * Config.TOP_CONFIDENCE_RATIO)
        top_class_0 = np.argsort(class_1_conf)[:condidates_per_class]
        top_class_1 = np.argsort(class_1_conf)[-condidates_per_class:]
        return top_class_0, top_class_1

    def generate_labeling_candidates(self, confidence_list_j, confidence_list_k, agree_list, batch_size):
        # ToDo: add the elements to the agreed: if the agreed>= batch size, than sample. else: add all agreed and add top instances.
        candidates = []
        classes_ratio = self.dataset_exp.class_ratio
        class_1_ratio = int(Config.BATCH_SIZE*classes_ratio)
        class_0_ratio = Config.BATCH_SIZE - class_1_ratio
        
        # Get instances with highest labeling confidence
        candidates_j_class_0, candidates_j_class_1 = self.top_confidence(confidence_list_j)
        candidates_k_class_0, candidates_k_class_1 = self.top_confidence(confidence_list_k)
        agree_class_0, agree_class_1 = np.where(agree_list==0), np.where(agree_list==1)

        agree_candidates = []
        if len(agree_class_1[0]) >= class_1_ratio and len(agree_class_0[0]) >= class_0_ratio:
            for batch_i in range(int(0.1*Config.NUM_BATCHES)):
                candidates_classes = []
                agree_class_1_tmp = np.random.choice(agree_class_1[0].tolist(), class_1_ratio)
                agree_class_0_tmp = np.random.choice(agree_class_0[0].tolist(), class_0_ratio)
                candidates_classes.append(agree_class_0_tmp)
                candidates_classes.append(agree_class_1_tmp)
                agree_candidates.append(candidates_classes)

        if len(agree_candidates) >= Config.NUM_BATCHES:
            candidates = agree_candidates[:Config.NUM_BATCHES]
            return candidates

        for batch_i in range(Config.NUM_BATCHES - len(agree_candidates)):
            candidates_classes = []
            class_1_concat = list(dict.fromkeys(candidates_j_class_1.tolist() + candidates_k_class_1.tolist() + agree_class_1[0].tolist()))
            class_0_concat = list(dict.fromkeys(candidates_j_class_0.tolist() + candidates_k_class_0.tolist() + agree_class_0[0].tolist()))
            
            class_1 = sklearn.utils.resample(class_1_concat
                ,replace=False,n_samples=class_1_ratio, random_state=Config.RANDOM_STATE + batch_i)
            class_0 = sklearn.utils.resample(class_0_concat
                ,replace=False,n_samples=class_0_ratio, random_state=Config.RANDOM_STATE + batch_i)

            candidates_classes.append(class_0)
            candidates_classes.append(class_1)
            candidates.append(candidates_classes)

        if len(agree_candidates)> 0:
            candidates = agree_candidates + candidates

        return np.asarray(candidates)
