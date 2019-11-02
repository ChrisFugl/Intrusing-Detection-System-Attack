import configargparse
from data import load_test, preprocess
import ids
from sklearn import metrics
from tabulate import tabulate

def main():
    options = parse_arguments()
    accuracy, f1, precision, recall = test(options)
    scores = [accuracy, f1, precision, recall]
    scores = list(map(lambda score: f'{score:0.4f}', scores))
    headers = ['accuracy', 'f1', 'precision', 'recall']
    print(tabulate([scores], headers=headers))

def parse_arguments():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--save_model', required=False, default=None, type=str, help='path of file to pretrained model')
    parser.add('--algorithm', required=True, choices=['dt', 'knn', 'lr', 'mlp', 'nb', 'rf', 'svm'], help='algorithm to test')
    parser.add('--normalize', required=False, action='store_true', default=False, help='normalize data (default false)')
    options, _ = parser.parse_known_args()
    return options

def test(options):
    attributes, attack_class = load_data(options)
    model = get_model(options)
    model.load(options.save_model)
    predictions = model.predict(attributes)
    return get_scores(attack_class, predictions)

def load_data(options):
    attributes_dataframe, _, attack_class_dataframe = preprocess(load_test(), normalize=options.normalize)
    attributes = attributes_dataframe.to_numpy()
    attack_class = attack_class_dataframe.to_numpy()
    return attributes, attack_class

def get_model(options):
    algorithm = options.algorithm
    if algorithm == 'dt':
        return ids.DecisionTree()
    elif algorithm == 'knn':
        raise Exception(f'Not implemented yet ({algorithm}).')
    elif algorithm == 'lr':
        return ids.LogisticRegression()
    elif algorithm == 'mlp':
        raise Exception(f'Not implemented yet ({algorithm}).')
    elif algorithm == 'nb':
        raise Exception(f'Not implemented yet ({algorithm}).')
    elif algorithm == 'rf':
        raise Exception(f'Not implemented yet ({algorithm}).')
    elif ids == 'svm':
        raise Exception(f'Not implemented yet ({algorithm}).')
    else:
        raise Exception(f'"{algorithm}" is not a valid choice of algorithm.')

def get_scores(attack_class, predictions):
    accuracy = metrics.accuracy_score(attack_class, predictions)
    f1 = metrics.f1_score(attack_class, predictions, average='micro')
    precision = metrics.precision_score(attack_class, predictions, average='micro')
    recall = metrics.recall_score(attack_class, predictions, average='micro')
    return accuracy, f1, precision, recall

if __name__ == '__main__':
    main()
