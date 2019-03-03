"""
GA NN training
"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN3.py
import sys

sys.path.append("./ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU
from base import *

TRAINING_ITERATIONS = 5000
OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/NN_{}_LOG.csv'


def main(ds_name, P, mate, mutate):
    """Run this experiment"""
    nn_config, train_file, val_file, test_file = get_problemset(ds_name)
    training_ints = initialize_instances(train_file)
    testing_ints = initialize_instances(test_file)
    validation_ints = initialize_instances(val_file)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    # 50 and 0.000001 are the defaults from RPROPUpdateRule.java
    rule = RPROPUpdateRule(0.064, 50, 0.000001)
    oa_name = "GA_{}_{}_{}_{}".format(ds_name, P, mate, mutate)
    with open(OUTFILE.format(oa_name), 'w') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg',
                                                            'acc_val', 'acc_tst', 'f1_trg', 'f1_val', 'f1_tst',
                                                            'elapsed'))
    classification_network = factory.createClassificationNetwork(nn_config, relu)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = StandardGeneticAlgorithm(P, mate, mutate, nnop)
    train(oa, classification_network, oa_name, training_ints, validation_ints, testing_ints, measure,
          TRAINING_ITERATIONS, OUTFILE.format(oa_name))


if __name__ == "__main__":
    for name in ['phishingwebsite']:
        for p, mate, mutate in [ [50, 10, 10], [50, 20, 10], [50, 20, 20], [100, 20, 20] ]:
            args = (name, p, mate, mutate)
            main(*args)
