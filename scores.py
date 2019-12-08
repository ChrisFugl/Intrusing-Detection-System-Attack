from sklearn import metrics
from tabulate import tabulate

def get_binary_class_scores(labels, predictions):
    accuracy = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    detection_rate = get_detection_rate(labels, predictions)
    return accuracy, f1, precision, recall, detection_rate

def print_scores(scores):
    scores = list(map(lambda score: f'{score:0.4f}', scores))
    headers = ['accuracy', 'f1', 'precision', 'recall', 'detection_rate']
    print(tabulate([scores], headers=headers))

def get_detection_rate(labels, predictions):
    prediction_mask = predictions == 1
    if not any(prediction_mask):
        return 0
    number_correctly_detected_attacks = labels[prediction_mask].sum()
    number_total_attacks = labels.sum()
    if number_total_attacks == 0:
        return 0
    return number_correctly_detected_attacks / number_total_attacks
