from ids.abstract_model import AbstractModel
import numpy as np
from scipy import stats

class Baseline(AbstractModel):

    def __init__(self):
        self.classifier = ModeClassifier()

class ModeClassifier:

    def fit(self, X, y):
        self.prediction, _ = stats.mode(y, axis=None)

    def predict(self, X):
        n_observations = len(X)
        predictions = np.array([self.prediction] * n_observations)
        return predictions
