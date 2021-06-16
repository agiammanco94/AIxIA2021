# -*- coding: utf-8 -*-
"""
    This module performs the experimental evaluation described in Section V of the paper, by means of the Random search
    of the hyperparameters.

    References:
        [1] GOODFELLOW, Ian, et al. Deep learning. Cambridge: MIT press, 2016.
"""
import math
import multiprocessing

import numpy as np

import py.classification.neural_net as neural_net
import py.utilities.dataset as dataset_utils
import py.utilities.miscellaneous as misc_utils


def random_search_main(run: int, simulations_for_architecture: int) -> None:
    """
        This function performs the random search for the hyperparameters of the problem [1].

        Args:
            run: An integer representing the current run. A single run keeps fixed the hyperparameters which we
                categorized as "Network and dataset" in Table II of the paper, namely: the number of layers of the neural
                network, the number of neurons per layer, the number of features considered in the subset of the dataset.

            simulations_for_architecture: An integer representing how many times, having fixed the hyperparameters of
                category "Network and dataset", the hyperparameters of the category "Training" have to be explored.
    """
    values_of_psi = [1, 2, 3]
    np.random.seed(run)
    n_epochs = 500
    n_features_dataset = round(np.random.uniform(20, 70))
    dataset = dataset_utils.Amr_Uti_Dataset(n_features=n_features_dataset)
    batch_size = -1

    log_n_layers = np.random.uniform(math.log(2, 10), math.log(7, 10))
    n_layers = round(10 ** log_n_layers)
    n_neurons = []
    for layer in range(n_layers - 1):
        if layer == 0:
            log_n_neurons = np.random.uniform(math.log(50, 10), math.log(500, 10))
            neurons = round(10 ** log_n_neurons)
        else:
            neurons = round(n_neurons[-1] / 2)
        n_neurons.append(neurons)

    # fixing one architecture in terms of number of layers, and number of neurons per layer,
    # we perform at least simulations_for_architecture simulations
    for j in range(simulations_for_architecture):

        np.random.seed(run*j)
        log_learning_rate = np.random.uniform(-4, -3)
        learning_rate = 10 ** log_learning_rate

        log_weight_decay = np.random.uniform(-5, -3)
        weight_decay = 10 ** log_weight_decay

        log_momentum = np.random.uniform(-3, math.log(0.8, 10))
        momentum = 10 ** log_momentum

        misc_utils.print_with_timestamp(f'Run={run + 1} Simulation={j + 1}/{simulations_for_architecture} '
                             f'alpha={learning_rate} lambda={weight_decay} nu={momentum}')

        architecture = []
        for i in range(n_layers - 1):
            layer = {'n_neurons': n_neurons[i], 'activation': 'relu', 'other_hyper_params': {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'momentum': momentum}}
            architecture.append(layer)
        final_layer = {'n_neurons': 2, 'activation': 'softmax', 'other_hyper_params': {'learning_rate': learning_rate,
                                                                                       'weight_decay': weight_decay,
                                                                                       'momentum': momentum}}
        architecture.append(final_layer)

        net = neural_net.NeuralNetwork(architecture, dataset, random_seed=run*j)
        net.train(dataset.train_X, dataset.train_Y_one_hot, n_epochs, debug_step=False,
                  batch_size=batch_size, test_x=dataset.test_X, test_y=dataset.test_Y_one_hot,
                  attack=True, values_of_psi=values_of_psi)


def parallel_main() -> None:
    """
        This function performs multiple runs of random search in parallel, exploiting all the available cores in the
        executing machine.
    """
    runs = 4
    simulations_for_architecture = 1
    pool = multiprocessing.Pool()
    for i in range(runs):
        pool.apply_async(random_search_main, args=(i, simulations_for_architecture,))
    pool.close()
    pool.join()


def serial_main() -> None:
    """
        This function performs the runs of random search once at a time.
    """
    runs = 50
    simulations_for_architecture = 50
    for i in range(runs):
        random_search_main(i, simulations_for_architecture)


if __name__ == '__main__':
    parallel_main()
