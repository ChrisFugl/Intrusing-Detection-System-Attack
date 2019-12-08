import argparse
import pandas as pd
import sys
from tabulate import tabulate
from test_ids import parse_arguments as parse_test_ids_arguments, test

IDS_CONFIGS = [
    ('decision_tree', 'configs/decision_tree.yaml'),
    ('k_nearest_neighbors', 'configs/k_nearest_neighbors.yaml'),
    ('logistic_regression', 'configs/logistic_regression.yaml'),
    ('multi_layer_perceptron', 'configs/multi_layer_perceptron.yaml'),
    ('naive_bayes', 'configs/naive_bayes.yaml'),
    ('random_forest', 'configs/random_forest.yaml'),
    ('support_vector_machine', 'configs/support_vector_machine.yaml')
]

def main():
    arguments = sys.argv[1:]
    options = parse_own_arguments(arguments)
    tester = get_tester(options)
    results = list(map(tester, IDS_CONFIGS))
    save_results(results, options)
    print_results(results)

def parse_own_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add('--attack', required=False, default=None, choices=['DoS', 'Probe', 'U2R_R2L'], help='select attack class to only evaluate on this attack class (default evaluate on all)')
    parser.add('--save_result', required=False, default=None, type=str, help='path to where to save result (csv)')
    options = parser.parse_args(arguments)
    return options

def get_tester(options):
    attack = options.attack
    def tester(configuration):
        name, config_path = configuration
        arguments = ['--config', config_path]
        if attack is not None:
            arguments.append('--attack')
            arguments.append(attack)
        options = parse_test_ids_arguments(arguments)
        scores = test(options)
        named_scores = [name, *scores]
        return named_scores
    return tester

def save_results(results, options):
    columns = ['algorithm', 'accuracy', 'f1', 'precision', 'recall', 'detection_rate']
    dataframe = pd.DataFrame(results, columns=columns)
    with open(options.save_result, 'w') as result_file:
        dataframe.to_csv(result_file, index=False)

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
