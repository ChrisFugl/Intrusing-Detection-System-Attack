from ids.abstract_model import AbstractModel
from sklearn import tree

class DecisionTree(AbstractModel):

    def __init__(self):
        self.classifier = tree.DecisionTreeClassifier()

    def train(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)
