import configargparse
import ids
import torch
from data import load_test, preprocess, split_features
from model import WGAN
from train_wgan import parse_arguments
import numpy as np
import pandas as pd
from tabulate import tabulate
from scores import get_binary_class_scores, print_scores
from test_ids import parse_arguments as parse_test_ids_arguments

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
    options = parse_arguments()
    scores = test_ids(options)

def test(options):
    functional_features, non_functional_features, normal_ff, normal_nff = split_features(load_test(), selected_attack_class=options.attack)
    nff_attributes, labels_mal = preprocess(non_functional_features, normalize=options.normalize)
    normal_attributes, labels_nor = preprocess(normal_nff, normalize=options.normalize)

    n_attributes = nff_attributes.shape[1]

    save_model_directory = os.path.join(options.save_model, options.name)
    model = WGAN(options, n_attributes)
    model.load(save_model_directory)
    predictions = model.predict_normal_and_adversarial(normal_attributes, nff_attributes)
    labels = np.concatenate((labels_nor, labels_mal), axis=0)
    return get_binary_class_scores(labels, predictions)

def test_ids(options):
    functional_features, non_functional_features, normal_ff, normal_nff = split_features(load_test(), selected_attack_class=options.attack)
    adversarial_ff, _ = preprocess(functional_features, normalize=options.normalize)
    adversarial_nff, labels_mal = preprocess(non_functional_features, normalize=options.normalize)
    nor_nff, labels_nor = preprocess(normal_nff, normalize=options.normalize)
    nor_ff, _ = preprocess(normal_ff, normalize=options.normalize)

    n_attributes = adversarial_nff.shape[1]

    model = WGAN(options, n_attributes)
    #model.load(options.save_model)
    model.load_checkpoint('checkpoints/dos1/epoch_1472.pt')
    adversarial = model.generate(adversarial_nff).detach()

    data = reassemble(options.attack, adversarial, adversarial_ff, nor_nff, nor_ff)
    labels = np.concatenate((labels_mal, labels_nor), axis=0)

    tester = get_tester(options.attack, data, labels)
    results = list(map(tester, IDS_CONFIGS))
    save_results(results, f'results/{options.name}.csv')
    print_results(results)

def reassemble(type, adversarial_nff, adversarial_ff, normal_nff, normal_ff):
    if type == "DoS":
        intrinsic = adversarial_ff[:,:6]
        content = adversarial_nff[:,:13]
        time_based = adversarial_ff[:,6:15]
        host_based = adversarial_nff[:,13:]
        categorical = adversarial_ff[:,15:]
        adversarial_traffic = np.concatenate((intrinsic, content, time_based, host_based, categorical), axis=1)

        intrinsic_normal = normal_ff[:,:6]
        content_normal = normal_nff[:,:13]
        time_based_normal = normal_ff[:,6:15]
        host_based_normal = normal_nff[:,13:]
        categorical_normal = normal_ff[:,15:]
        normal_traffic = np.concatenate((intrinsic_normal, content_normal, time_based_normal, host_based_normal, categorical_normal), axis=1)
    elif type == "Probe":
        intrinsic = adversarial_ff[:,:6]
        content = adversarial_nff[:,:13]
        time_based = adversarial_ff[:,6:15]
        host_based = adversarial_ff[:,15:25]
        categorical = adversarial_ff[:,25:]
        frame = np.concatenate((intrinsic, content, time_based, host_based, categorical), axis=1)

        intrinsic_normal = normal_ff[:,:6]
        content_normal = normal_nff[:,:13]
        time_based_normal = normal_ff[:,6:15]
        host_based_normal = normal_ff[:,15:25]
        categorical_normal = normal_ff[:,25:]
        normal_traffic = np.concatenate((intrinsic_normal, content_normal, time_based_normal, host_based_normal, categorical_normal), axis=1)

    output = np.concatenate((adversarial_traffic, normal_traffic), axis=0)
    return output

def get_tester(attack, data, labels):
    def tester(configuration):
        name, config_path = configuration
        arguments = ['--config', config_path]
        if attack is not None:
            arguments.append('--attack')
            arguments.append(attack)
        options = parse_test_ids_arguments(arguments)
        scores = test(options, data, labels)
        named_scores = [name, *scores]
        return named_scores
    return tester

def test(options, data, labels):
    n_attributes = data.shape[1]
    model = get_model(options, n_attributes)
    model.load(options.save_model)
    predictions = model.predict(data)
    return get_binary_class_scores(labels, predictions)

def get_model(options, n_features):
    algorithm = options.algorithm
    if algorithm == 'baseline':
        return ids.Baseline()
    elif algorithm == 'dt':
        return ids.DecisionTree(
            max_depth=options.max_depth,
            split_criterion=options.split_criterion,
            splitter=options.splitter,
            min_samples_leaf=options.min_samples_leaf,
            min_samples_split=options.min_samples_split
        )
    elif algorithm == 'knn':
        return ids.KNearestNeighbours(
            algorithm=options.knn_algorithm,
            n_neighbors=options.n_neighbors,
            weights=options.knn_weights
        )
    elif algorithm == 'lr':
        return ids.LogisticRegression(
            max_iter=options.iterations,
            solver=options.lr_solver
        )
    elif algorithm == 'mlp':
        return ids.MultiLayerPerceptron(
            input_size=n_features,
            log_dir=options.log_dir,
            log_every=options.log_every,
            evaluate_every=options.evaluate_every,
            epochs=options.epochs,
            batch_size=options.batch_size,
            learning_rate=options.learning_rate,
            weight_decay=options.weight_decay,
            dropout_rate=options.dropout_rate,
            hidden_size=options.hidden_size
        )
    elif algorithm == 'nb':
        return ids.NaiveBayes()
    elif algorithm == 'rf':
        return ids.RandomForest(
            n_trees=options.n_trees,
            max_depth=options.max_depth,
            split_criterion=options.split_criterion,
            min_samples_leaf=options.min_samples_leaf,
            min_samples_split=options.min_samples_split
        )
    elif algorithm == 'svm':
        return ids.SupportVectorMachine(max_iter=options.iterations)
    else:
        raise Exception(f'"{algorithm}" is not a valid choice of algorithm.')

def print_results(results):
    rows = list(map(format_result, results))
    headers = ['algorithm', 'accuracy', 'f1', 'precision', 'recall', 'detection_rate']
    print(tabulate(rows, headers=headers))

def save_results(results, options):
    columns = ['algorithm', 'accuracy', 'f1', 'precision', 'recall', 'detection_rate']
    dataframe = pd.DataFrame(results, columns=columns)
    with open(options, 'w') as result_file:
        dataframe.to_csv(result_file, index=False)

def format_result(result):
    scores = map(lambda score: f'{score:0.4f}', result[1:])
    formatted_result = [result[0], *scores]
    return formatted_result

if __name__ == '__main__':
    main()
