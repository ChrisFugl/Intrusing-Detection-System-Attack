from tabulate import tabulate
from train_ids import parse_arguments, train

IDS_CONFIGS = [
    ('baseline', 'configs/baseline.yaml'),
    ('decision_tree', 'configs/decision_tree.yaml'),
    ('k_nearest_neighbors', 'configs/k_nearest_neighbors.yaml'),
    ('logistic_regression', 'configs/logistic_regression.yaml'),
    ('multi_layer_perceptron', 'configs/multi_layer_perceptron.yaml'),
    ('naive_bayes', 'configs/naive_bayes.yaml'),
    ('random_forest', 'configs/random_forest.yaml'),
    ('support_vector_machine', 'configs/support_vector_machine.yaml')
]

def main():
    results = []
    for name, config_path in IDS_CONFIGS:
        options = parse_arguments(['--config', config_path])
        print(name)
        scores_val = train(options)
        named_scores = [name, *scores_val]
        results.append(named_scores)
    print_results(results)

def print_results(results):
    rows = list(map(format_result, results))
    headers = ['algorithm', 'accuracy', 'f1', 'precision', 'recall', 'detection_rate']
    print(tabulate(rows, headers=headers))

def format_result(result):
    scores = map(lambda score: f'{score:0.4f}', result[1:])
    formatted_result = [result[0], *scores]
    return formatted_result

if __name__ == '__main__':
    main()
