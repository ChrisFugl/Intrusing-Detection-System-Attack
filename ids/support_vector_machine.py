from ids.abstract_model import AbstractModel
from sklearn.svm import LinearSVC

class SupportVectorMachine(AbstractModel):

    def __init__(self, max_iter=1000):
        self.classifier = LinearSVC(max_iter=max_iter)
