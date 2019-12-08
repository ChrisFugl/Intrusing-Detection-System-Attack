import configargparse
from data import load_test, preprocess, split_features
from model import WGAN
from train_wgan import parse_arguments
import numpy as np
from scores import get_binary_class_scores, print_scores

def main():
    options = parse_arguments()
    scores = test(options)
    print_scores(scores)

def test(options):
    functional_features, non_functional_features, normal_ff, normal_nff = split_features(load_test(), selected_attack_class=options.attack)
    nff_attributes, labels_mal = preprocess(non_functional_features, normalize=options.normalize)
    normal_attributes, labels_nor = preprocess(normal_nff, normalize=options.normalize)

    n_attributes = nff_attributes.shape[1]
    
    model = WGAN(options, n_attributes)
    model.load(options.save_model)
    #print(non_functional_features.shape)
    #print(nff_attributes.shape)
    predictions = model.predict_normal_and_adversarial(normal_attributes, nff_attributes)
    #print(predictions.shape)
    labels = np.concatenate((labels_nor, labels_mal), axis=0)
    #print(labels.shape)
    return get_binary_class_scores(labels, predictions)

def test_ids(options):
    functional_features, non_functional_features, _, _ = split_features(load_test(), selected_attack_class=options.attack)
    nff_attributes, labels_mal = preprocess(non_functional_features, normalize=options.normalize)
    
    n_attributes = nff_attributes.shape[1]

    model = WGAN(options, n_attributes)
    model.load(options.save_model)
    adversarial = model.generate(nff_attributes)


if __name__ == '__main__':
    main()
