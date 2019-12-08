from ids.abstract_model import AbstractModel
from sklearn import tree

class DecisionTree(AbstractModel):

    def __init__(self,
            split_criterion='gini',
            splitter='best',
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2
        ):
        self.classifier = tree.DecisionTreeClassifier(
            criterion=split_criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split
        )
