from ids.abstract_model import AbstractModel
from joblib import dump, load
from sklearn import tree

class DecisionTree(AbstractModel):

    def __init__(self):
        super(DecisionTree, self).__init__()
        self.classifier = tree.DecisionTreeClassifier()

    def train(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def save(self, path):
        dump(self.classifier, path)

    def load(self, path):
        self.classifier = load(path)
