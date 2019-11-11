from scores import get_binary_class_scores
import unittest

class TestScores(unittest.TestCase):

    binary_labels       = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    binary_predictions  = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    binary_tp           = 4
    binary_fp           = 3
    binary_tn           = 2
    binary_fn           = 1
    binary_accuracy     = (binary_tn + binary_tp) / (binary_tp + binary_tn + binary_fp + binary_fn)     # 6 / 10
    binary_precision    = binary_tp / (binary_tp + binary_fp)                                           # 4 / 7
    binary_recall       = binary_tp / (binary_tp + binary_fn)                                           # 4 / 5
    binary_f1           = 2 * (binary_precision * binary_recall) / (binary_precision + binary_recall)   # 2 / 3

    def test_binary_class_scores(self):
        accuracy, f1, precision, recall = get_binary_class_scores(self.binary_labels, self.binary_predictions)
        self.assertAlmostEqual(accuracy, self.binary_accuracy)
        self.assertAlmostEqual(f1, self.binary_f1)
        self.assertAlmostEqual(precision, self.binary_precision)
        self.assertAlmostEqual(recall, self.binary_recall)
