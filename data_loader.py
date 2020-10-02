import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, dataset_name, target_col_name):
        self.dataset = pd.read_csv('./datasets/{}.csv'.format(dataset_name))
        self.original_dataset = self.dataset # keep the original dataset
        
        self.handle_categorical_data()
        
        if target_col_name is not None:
            self.data = self.dataset.drop([target_col_name], axis=1)
            self.label = self.dataset[target_col_name]
        else:
            self.data = self.dataset[:, :-1]
            self.label = self.dataset[:,-1]

        
    def data_split(self, label_rate, test_rate=0.25):
        self.test_rate = test_rate
        self.label_rate = label_rate
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size = self.test_rate, random_state=0)
        rng = np.random.RandomState(42)
        self.labeled_index = rng.rand(len(y_train)) < self.label_rate
        self.unlabeled_index = np.logical_not(self.labeled_index)
        self.L_X = X_train[self.labeled_index]#data of L
        self.L_y = y_train[self.labeled_index]#lable of L
        self.U_X = X_train[self.unlabeled_index]#data of U
        return self.L_X, self.L_y, self.U_X, X_test, y_test

    def handle_categorical_data(self):
        self.numeric_cols = self.dataset._get_numeric_data().columns
        self.categorical_cols = list(set(self.dataset.columns) - set(self.numeric_cols))
        for cat_col in self.categorical_cols:
            self.dataset[cat_col] = self.dataset[cat_col].astype('category')
        self.dataset[self.categorical_cols] = self.dataset[self.categorical_cols].apply(lambda x: x.cat.codes)