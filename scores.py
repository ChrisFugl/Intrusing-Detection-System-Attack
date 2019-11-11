from sklearn import metrics
from tabulate import tabulate

def get_binary_class_scores(labels, predictions):
    accuracy = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    return accuracy, f1, precision, recall

def print_scores(scores):
    scores = list(map(lambda score: f'{score:0.4f}', scores))
    headers = ['accuracy', 'f1', 'precision', 'recall']
    print(tabulate([scores], headers=headers))
