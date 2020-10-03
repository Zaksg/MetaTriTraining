import numpy as np
import sklearn
import math
import pandas as pd 

class TriTraining:
    def __init__(self, classifier, is_extract_meta_features = False, is_use_meta_model = False):
        self.is_extract_meta_features = is_extract_meta_features
        self.is_use_meta_model = is_use_meta_model

        if sklearn.base.is_classifier(classifier):
            self.classifiers = [sklearn.base.clone(classifier) for i in range(3)]
        else:
            self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(3)]
            
    def fit(self, X_label, y_label, X_unlabel):
        self.X_label = X_label
        self.y_label = y_label

        # BootstrapSample
        for i in range(3):
            sample = sklearn.utils.resample(self.X_label, self.y_label)  
            self.classifiers[i].fit(*sample)

        # Initial variables 
        classification_error_current = [0.5]*3
        pseudo_label_size_current = [0]*3
        classification_error = [0]*3
        pseudo_label_size = [0]*3
        X_pseudo_label_index = [[]]*3
        X_pseudo_label_index_current = [[]]*3
        update = [False]*3
        improve = True
        self.iter = 0
        
        while improve:
            print("Iteration: {}".format(self.iter))
            self.iter += 1
            for i in range(3):
                X_pseudo_label_index_current[i] = X_pseudo_label_index[i]

            # The new pseudo label set determined by tri-training iteration for classifier i
            # X_pseudo_label_index, contains the data record index (in the full unlabelled set) of the new pseudo label set determined by tri-training iteration for classifier i
            # X_pseudo_label, contains the features for new pseudo label set determined by tri-training iteration for classifier i
            # y_pseudo_label, contains the labels (not ground truth label, but pseudo label calculated by tri-training iteration) for new pseudo label set determined by tri-training iteration for classifier i
            X_pseudo_label_index = [[]]*3
            X_pseudo_label = [[]]*3
            y_pseudo_label = [[]]*3
            
            # Loop unlabeled set
            for i in range(3):    
                j, k = np.delete(np.array([0,1,2]),i)
                update[i] = False
                classification_error[i] = self.measure_error(self.X_label, self.y_label, j, k)
                # ToDo: change stop criteria
                if classification_error[i] < classification_error_current[i] or self.iter <= 3:
                    U_y_j = self.classifiers[j].predict(X_unlabel)
                    U_y_k = self.classifiers[k].predict(X_unlabel)
                    U_y_i = self.classifiers[i].predict(X_unlabel)
                    X_pseudo_label[i] = X_unlabel[U_y_j == U_y_k] # models agreement
                    y_pseudo_label[i] = U_y_j[U_y_j == U_y_k]
                    # X_pseudo_label_index[i] = np.where(U_y_j==U_y_k)
                    pseudo_label_size[i] = len(y_pseudo_label[i])
                    
                    # Get meta features
                    if self.is_extract_meta_features:
                        U_y_j_proba = self.classifiers[j].predict_proba(X_unlabel)
                        U_y_k_proba = self.classifiers[k].predict_proba(X_unlabel)
                        U_y_i_proba = self.classifiers[i].predict_proba(X_unlabel)
                        self.meta_features_extractor.view_based_mf(self.iter, i, U_y_j, U_y_k, U_y_i, X_pseudo_label[i], y_pseudo_label[i], 
                            U_y_j_proba, U_y_k_proba, U_y_i_proba)

                    # Continue tri-training flow
                    if pseudo_label_size_current[i] == 0: # first updated
                        pseudo_label_size_current[i]  = int(classification_error[i]/(classification_error_current[i] - classification_error[i]) + 1)
                    if pseudo_label_size_current[i] < pseudo_label_size[i]:
                        if classification_error[i] * pseudo_label_size[i] < classification_error_current[i] * pseudo_label_size_current[i]:
                            update[i] = True
                        elif pseudo_label_size_current[i] > classification_error[i]/(classification_error_current[i] - classification_error[i]):
                            resample_size = math.ceil(classification_error_current[i] * pseudo_label_size_current[i] / classification_error[i] - 1)
                            X_pseudo_label[i], y_pseudo_label[i] = sklearn.utils.resample(
                                X_pseudo_label[i],y_pseudo_label[i],replace=False,n_samples=resample_size)
                            pseudo_label_size[i] = len(y_pseudo_label[i])
                            update[i] = True
             
            for i in range(3):
                if update[i]:
                    self.classifiers[i].fit(np.append(self.X_label,X_pseudo_label[i],axis=0), np.append(self.y_label, y_pseudo_label[i], axis=0))
                    classification_error_current[i] = classification_error[i]
                    pseudo_label_size_current[i] = pseudo_label_size[i]
    
            if update == [False]*3:
                improve = False


    def predict(self, X):
        pred = np.asarray([self.classifiers[i].predict(X) for i in range(3)])
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