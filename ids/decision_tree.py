from ids.abstract_model import AbstractModel
from sklearn import tree

class DecisionTree(AbstractModel):

    def __init__(self):
        self.classifier = tree.DecisionTreeClassifier()
