import configargparse
from data import load_test, preprocess, split_features
from model import WGAN
from train_wgan import parse_arguments
import numpy as np
from scores import get_binary_class_scores, print_scores

def main():
    options = parse_arguments()
    scores = test_ids(options)
    #print_scores(scores)

def test(options):
    functional_features, non_functional_features, normal_ff, normal_nff = split_features(load_test(), selected_attack_class=options.attack)
    nff_attributes, labels_mal = preprocess(non_functional_features, normalize=options.normalize)
    normal_attributes, labels_nor = preprocess(normal_nff, normalize=options.normalize)

    n_attributes = nff_attributes.shape[1]

    model = WGAN(options, n_attributes)
    model.load(options.save_model)
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
    model.load(options.save_model)
    adversarial = model.generate(adversarial_nff)

    test = reassemble(options.attack, adversarial, adversarial_ff, nor_nff, nor_ff)

def reassemble(type, adversarial_nff, adversarial_ff, normal_nff, normal_ff):
    length = adversarial_nff.shape[1] + adversarial_ff.shape[1]
    #reconstructed_frame = np.zeros(length)
    #print(reconstructed_frame.shape)
    print(adversarial_nff.shape)

    if type == "DoS":
        content = adversarial_nff[:,:13]
        time_based = adversarial_nff[:,13:]
    print(time_based.shape)
    #return content


if __name__ == '__main__':
    main()
