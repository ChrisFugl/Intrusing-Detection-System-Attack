from ids.abstract_model import AbstractModel
from sklearn.svm import SVC

class SupportVectorMachine(AbstractModel):

    def __init__(self, max_iter=1000):
        self.classifier = SVC(max_iter=max_iter, gamma='auto')
