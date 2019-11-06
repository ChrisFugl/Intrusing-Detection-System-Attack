from joblib import dump, load

class AbstractModel:

    def train(self, X, y):
        raise NotImplementedError('train is not implemented yet')

    def predict(self, X):
        raise NotImplementedError('predict is not implemented yet')

    def save(self, path):
        dump(self.classifier, path)

    def load(self, path):
        self.classifier = load(path)
