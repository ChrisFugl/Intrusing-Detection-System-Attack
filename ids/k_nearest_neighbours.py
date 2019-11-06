from ids.abstract_model import AbstractModel
from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighbours(AbstractModel):

    def __init__(self, n_neighbors=5):
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
