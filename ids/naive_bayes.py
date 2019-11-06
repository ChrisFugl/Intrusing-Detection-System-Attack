from ids.abstract_model import AbstractModel
from sklearn.naive_bayes import GaussianNB

class NaiveBayes(AbstractModel):

    def __init__(self):
        self.classifier = GaussianNB()
