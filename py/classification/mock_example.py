# -*- coding: utf-8 -*-
"""
    This module contains the reproducible mock example described in Section IV of the paper.
"""
import py.classification.neural_net as neural_net
import py.utilities.dataset as dataset_utils


def mock_example() -> None:
    """
        This function reproduces a mock example described in the readme file.
    """
    n_features = 7
    dataset = dataset_utils.Amr_Uti_Dataset(n_features=n_features)
    learning_rate = 1e-2
    weight_decay = 1e-3
    momentum = 0.6

    n_neurons_l1 = 5
    layer1 = {'n_neurons': n_neurons_l1, 'activation': 'relu', 'other_hyper_params': {'learning_rate': learning_rate,
                                                                                      'weight_decay': weight_decay,
                                                                                      'momentum': momentum}}
    n_neurons_l2 = 2
    layer2 = {'n_neurons': n_neurons_l2, 'activation': 'softmax', 'other_hyper_params': {'learning_rate': learning_rate,
                                                                                         'weight_decay': weight_decay,
                                                                                         'momentum': momentum}}
    architecture = [layer1, layer2]

    net = neural_net.NeuralNetwork(architecture, dataset, random_seed=16)

    n_epochs = 100
    batch_size = -1
    values_of_psi = [1]

    net.train(dataset.train_X, dataset.train_Y_one_hot, n_epochs,
              batch_size=batch_size, test_x=dataset.test_X, test_y=dataset.test_Y_one_hot, attack=True,
              values_of_psi=values_of_psi)

    net.print_parameters()


if __name__ == '__main__':
    mock_example()
