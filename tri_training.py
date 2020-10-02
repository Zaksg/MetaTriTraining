import numpy as np
import sklearn
import math
import pandas as pd 

class TriTraining:
    def __init__(self, classifier):
        if sklearn.base.is_classifier(classifier):
            self.classifiers = [sklearn.base.clone(classifier) for i in range(3)]
        else:
            self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(3)]
            
    def fit(self, X_label, y_label, X_unlabel):
        self.X_label = X_label
        self.y_label = y_label

        for i in range(3):
            sample = sklearn.utils.resample(self.X_label, self.y_label)  # BootstrapSample(L)
            self.classifiers[i].fit(*sample)  # Learn(Si)   
        classification_error_current = [0.5]*3
        pseudo_label_size_current = [0]*3
        classification_error = [0]*3
        pseudo_label_size = [0]*3
        X_pseudo_label_index = [[]]*3
        X_pseudo_label_index_current = [[]]*3
        update = [False]*3
        # Li_X, Li_y = [[]]*3, [[]]*3#to save proxy labeled data
        improve = True
        self.iter = 0
        
        while improve:
            self.iter += 1#count iterations 
            for i in range(3):
                X_pseudo_label_index_current[i] = X_pseudo_label_index[i]

            # Step3.1 Set Li = empty set, Li denotes the new pseudo label set determined by tri-training iteration for classifier i
            # X_pseudo_label_index, contains the data record index (in the full unlabelled set) of the new pseudo label set determined by tri-training iteration for classifier i
            # X_pseudo_label, contains the features for new pseudo label set determined by tri-training iteration for classifier i
            # y_pseudo_label, contains the labels (not ground truth label, but pseudo label calculated by tri-training iteration) for new pseudo label set determined by tri-training iteration for classifier i
            X_pseudo_label_index = [[]]*3
            X_pseudo_label = [[]]*3
            y_pseudo_label = [[]]*3
            
            # Step 3.2 Loop through all the data record in unlabelled set
            for i in range(3):    
                j, k = np.delete(np.array([0,1,2]),i)
                update[i] = False
                classification_error[i] = self.measure_error(self.X_label, self.y_label, j, k)
                if classification_error[i] < classification_error_current[i]:
                    U_y_j = self.classifiers[j].predict(X_unlabel)
                    U_y_k = self.classifiers[k].predict(X_unlabel)
                    X_pseudo_label[i] = X_unlabel[U_y_j == U_y_k]#when two models agree on the label, save it
                    y_pseudo_label[i] = U_y_j[U_y_j == U_y_k]
                    X_pseudo_label_index[i] = np.where(U_y_j==U_y_k)
                    pseudo_label_size[i] = len(X_pseudo_label_index[i])
                    if pseudo_label_size_current[i] == 0:#no updated before
                        pseudo_label_size_current[i]  = int(classification_error[i]/(classification_error_current[i] - classification_error[i]) + 1)
                    if pseudo_label_size_current[i] < pseudo_label_size[i]:
                        if classification_error[i] * pseudo_label_size[i] < classification_error_current[i] * pseudo_label_size_current[i]:
                            update[i] = True
                        elif pseudo_label_size_current[i] > classification_error[i]/(classification_error_current[i] - classification_error[i]):
                            # L_index = np.random.choice(len(Li_y[i]), int(classification_error_current[i] * pseudo_label_size_current[i]/classification_error[i] -1))#subsample from proxy labeled data
                            # Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                            resample_size = math.ceil(classification_error_current[i] * pseudo_label_size_current[i] / classification_error[i] - 1)
                            X_pseudo_label_index[i], y_pseudo_label[i] = sklearn.utils.resample(X_pseudo_label_index[i],y_pseudo_label[i],replace=False,n_samples=resample_size)
                            pseudo_label_size[i] = len(X_pseudo_label_index[i])
                            update[i] = True
             
            for i in range(3):
                if update[i]:
                    # self.classifiers[i].fit(np.append(self.X_label,Li_X[i],axis=0), np.append(self.y_label, Li_y[i], axis=0))#train the classifier on integrated dataset
                    X_pseudo_label[i] = np.array(X_unlabel[X_pseudo_label_index[i]])
                    self.classifiers[i].fit(np.concatenate((X_pseudo_label[i], self.X_label), axis=0),np.concatenate((np.array(y_pseudo_label[i]), self.y_label), axis=0))
                    classification_error_current[i] = classification_error[i]
                    pseudo_label_size_current[i] = pseudo_label_size[i]
    
            if update == [False]*3:
                improve = False#if no classifier was updated, no improvement


    def predict(self, X):
        pred = np.asarray([self.classifiers[i].predict(X) for i in range(3)])
        pred[0][pred[1]==pred[2]] = pred[1][pred[1]==pred[2]]
        return pred[0]
        
    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))
        
    def measure_error(self, X, y, j, k):
        j_pred = self.classifiers[j].predict(X)
        k_pred = self.classifiers[k].predict(X)
        wrong_index =np.logical_and(j_pred != y, k_pred==j_pred)#model_j and model_k make the same wrong prediction
        #wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index)/sum(j_pred == k_pred)