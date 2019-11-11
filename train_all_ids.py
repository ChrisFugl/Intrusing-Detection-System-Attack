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
    for name, config_path in IDS_CONFIGS:
        options = parse_arguments(['--config', config_path])
        print(name)
        train(options)

if __name__ == '__main__':
    main()
