from sklearn import tree
from sklearn.linear_model import LogisticRegression

class Classifier:
    def __init__(self, classifier_name = None):
        self.classifier_name = classifier_name
        if classifier_name=='autoML':
            pass
        elif classifier_name == 'desicition_tree':
            self.classifier = tree.DecisionTreeClassifier()
        elif classifier_name == 'logistic_regression':
            self.classifier = LogisticRegression()
        else:
            self.classifier = LogisticRegression()

    def get_classifier_name(self):
        return self.classifier_name

    def get_classifier(self):
        return self.classifier