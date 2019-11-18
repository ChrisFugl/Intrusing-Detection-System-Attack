import configargparse
from data import load_test, preprocess
from model import WGAN
from train_wgan import parse_arguments
from scores import print_scores

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
    model.predict(normal_attributes, nff_attributes)

if __name__ == '__main__':
    main()
