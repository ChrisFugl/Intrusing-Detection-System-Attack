from ids.abstract_model import AbstractModel
from sklearn.naive_bayes import BernoulliNB

class NaiveBayes(AbstractModel):

    def __init__(self):
        self.classifier = BernoulliNB()
