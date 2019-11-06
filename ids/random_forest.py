from ids.abstract_model import AbstractModel
from sklearn.ensemble import RandomForestClassifier

class RandomForest(AbstractModel):

    def __init__(self,
            n_trees=10,
            split_criterion='gini',
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2
        ):
        self.classifier = RandomForestClassifier(
            n_estimators=n_trees,
            criterion=split_criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split
        )
