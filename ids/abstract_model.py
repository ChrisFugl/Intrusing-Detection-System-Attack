class AbstractModel:

    def train(self, X, y):
        raise Exception('Method not implemented!')

    def predict(self, X):
        raise Exception('Method not implemented!')

    def save(self, path):
        raise Exception('Method not implemented!')

    def load(self, path):
        raise Exception('Method not implemented!')
