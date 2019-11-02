from ids.abstract_model import AbstractModel
from joblib import dump, load
from sklearn import linear_model

class LogisticRegression(AbstractModel):

    def __init__(self, max_iter=1000):
        super(LogisticRegression, self).__init__()
        self.classifier = linear_model.LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=max_iter
    )

    def train(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def save(self, path):
        dump(self.classifier, path)

    def load(self, path):
        self.classifier = load(path)
