# -*- coding: utf-8 -*-
"""
    This module provides an implementation of neural network, which is based on the Numpy implementation in [0].

    The Fast Gradient Sign Method, and the attack algorithm proposed in our paper are implemented as methods of the
    NeuralNetwork class.

    References:
        [0] https://github.com/RafayAK/NothingButNumPy
        [1] HOWARD, Jeremy; GUGGER, Sylvain. Deep Learning for Coders with fastai and PyTorch. O'Reilly Media, 2020.
        [2] GOODFELLOW, Ian, et al. Deep learning. Cambridge: MIT press, 2016.
        [3] https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html
        [4] http://cs229.stanford.edu/summer2020/cs229-notes-deep_learning.pdf
        [5] http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        [6] https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html
        [7] https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
        [8] https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        [9] https://towardsdatascience.com/weight-decay-l2-regularization-90a9e17713cd
        [10] https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
        [11] https://gombru.github.io/2018/05/23/cross_entropy_loss/
        [12] https://stats.stackexchange.com/questions/149139/vectorization-of-cross-entropy-loss
        [13] https://stats.stackexchange.com/questions/70101/neural-networks-weight-change-momentum-and-weight-decay
        [14] HE, Kaiming, et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet
        classification. In: Proceedings of the IEEE international conference on computer vision. 2015. p. 1026-1034.
        [15] GLOROT, Xavier; BENGIO, Yoshua. Understanding the difficulty of training deep feedforward neural networks.
        In: Proceedings of the thirteenth international conference on artificial intelligence and statistics.
        JMLR Workshop and Conference Proceedings, 2010. p. 249-256.
        [16] GOODFELLOW, Ian J.; SHLENS, Jonathon; SZEGEDY, Christian. Explaining and harnessing adversarial examples.
        arXiv preprint arXiv:1412.6572, 2014.
        [17] https://cs230.stanford.edu/files/C2M2.pdf
        [18] https://www.deeplearning.ai/ai-notes/optimization/
        [19] https://github.com/limberc/deeplearning.ai/blob/master/
        COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%
        20Optimization/week%2001/week%2001%20Setting%20up%20your%20Machine%20Learning%20Application.pptx
        [20] https://stackoverflow.com/questions/47377222/
        what-is-the-problem-with-my-implementation-of-the-cross-entropy-function
        [21] https://stackoverflow.com/questions/42042561/relu-derivative-in-backpropagation
"""
import math
import pickle
from typing import List, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

import py.classification.class_utils as class_utils
import py.utilities.dataset as dataset_class
import py.utilities.miscellaneous as misc_utils


class NeuralNetwork:
    """
        This class models a neural network, providing the methods for training and testing, as well as the adversarial
        attacks FGSM and the one proposed in the paper.

        Attributes:
            architecture: It is a list of dictionaries, where each dictionary represents a layer of the network.
                          A single layer (which is a dictionary) has the following structure:
                              n_neurons: the number of neurons in the layer
                              activation: the activation function used in the layer
                              learning_rate: the learning rate for the layer (optional)
                              init_type: the initialization type for the weight matrix of the layer (optional)
                              other_hyper_params: it is a nested dictionary containing other hyper parameters for
                              the linear layer in particular:
                                    init_type: the initialization type for the weight matrix of the layer
                                    alpha: the learning rate for the layer
                                    lambda: the regularization factor for the layer
                                    W: a manually inserted weight matrix
                                    b: a manually inserted bias vector

            random_seed: An integer which controls the initial random initialization of the parameters (bias and
                weights). It is useful for reproducibility of results.

            layers: It is a list which contains Layer objects, representing the linear layers and the non linear
                (activation) layers of the neural network.

            features_in_dataset: An integer representing the total number of features present in the dataset.

            dataset_name: A string representing the name of the current dataset, e.g., "AMR-UTI".

            class_labels: A list of strings representing the name of the target labels in the dataset.

            perturbation_mask: A binary ndarray of shape (n_features_in_dataset,) containing values equal to 0 for
                the features whose value is not corruptible.

            train_costs: A ndarray containing the cost that the neural network obtained across the different training
                epochs.

            train_times: A ndarray containing the times (in minutes) the neural network employed to perform the
                different training epochs.

            train_shutdown_neurons: An integer tracking the number of neurons that do not "fire", in the sense that
                their activation (and thus their derivative) is zero.

            predictions: A ndarray which buffers the last predictions returned by the neural network. It has shape
                (n_samples_in_test_set, n_classes).

            model_path: A string representing the parent folder containing the path of a specific neural network
                architecture.

            train_path: A string representing the specific path of a model trained with a certain set of
                hyperparameters.

    """
    def __init__(self, architecture: List[Dict], dataset: dataset_class.Dataset, random_seed: int = 42) -> None:
        """
            Inits an instance of NeuralNetwork.

            Args:
                architecture: It is a list of dictionaries, where each dictionary represents a layer of the network.

                dataset: An object of class Dataset which will be used to initialize some utilities parameters.

                random_seed: An integer which controls the initial random initialization of the parameters (bias and
                    weights). It is useful for reproducibility of results.
        """
        np.random.seed(random_seed)
        self.architecture = architecture
        self.random_seed = random_seed
        self.layers = []
        self.features_in_dataset = dataset.features_in_dataset
        self.dataset_name = dataset.dataset_name
        self.class_labels = dataset.classes_names
        self.perturbation_mask = dataset.perturbation_mask
        self.parse_architecture(architecture)
        self.train_costs, self.train_times = None, None
        self.train_shutdown_neurons = None
        self.predictions = None
        self.model_path = self.init_model_path()
        self.train_path = None

    @property
    def n_neurons_str(self) -> str:
        """
            This method returns a string representing the neurons amount for each single layer. E.g., "215_2".

            Returns:
                n_neurons: The string representing the neurons of the network.
        """
        n_neurons = ''
        for layer in self.layers:
            linear_layer, _activation_layer = layer
            n_neurons = linear_layer.n_neurons
            if len(n_neurons) > 0:
                n_neurons += '\_'
            n_neurons += str(n_neurons)
        return n_neurons

    def init_model_path(self) -> str:
        """
            This method returns a string representing the parent folder containing the path of a specific neural network
            architecture. E.g., "py/classification/results/59_features_considered/2_layers/215_2".

            Returns:
                parameters_path: A string representing the parent folder for all the storage files of a network.
        """
        prefix = misc_utils.get_relative_path()
        parameters_path = prefix + 'py/classification/results'
        misc_utils.create_dir(parameters_path)
        parameters_path = parameters_path + '/' + self.dataset_name
        misc_utils.create_dir(parameters_path)
        parameters_path = parameters_path + '/' + str(self.features_in_dataset) + '_features_considered'
        misc_utils.create_dir(parameters_path)
        n_layers = len(self.layers)
        parameters_path = parameters_path + '/' + str(n_layers) + '_layers'
        misc_utils.create_dir(parameters_path)
        neurons_str = ''
        for layer in self.architecture:
            n_neurons = layer['n_neurons']
            if len(neurons_str) > 0:
                neurons_str += '_'
            neurons_str += str(n_neurons)
        parameters_path = parameters_path + '/' + neurons_str
        misc_utils.create_dir(parameters_path)
        return parameters_path

    def train_model_path(self, train_parameters: Dict) -> str:
        """
            This method returns a string representing the path of the neural network trained with a specific set of
            hyperparameters. E.g.,
            "py/classification/results/59_features_considered/2_layers/215_2/500_epochs_601_batch_size/
            5.18e-04_3.25e-05_7.65e-01".

            Args:
                train_parameters: A dictionary containing the number of epochs of training, and the size of a
                    single batch.

            Returns:
                train_path: A string representing the path for all the storage files of a trained neural network.
        """
        n_epochs = train_parameters['n_epochs']
        batch_size = train_parameters['batch_size']
        train_path = self.model_path + '/' + str(n_epochs) + '_epochs_' + str(batch_size) + '_batch_size'
        misc_utils.create_dir(train_path)
        learning_rate = self.layers[0][0].learning_rate
        weight_decay = self.layers[0][0].weight_decay
        momentum = self.layers[0][0].momentum
        train_path += '/' + '{:.2e}'.format(learning_rate) + '_{:.2e}'.format(weight_decay) + '_{:.2e}'.format(momentum)
        misc_utils.create_dir(train_path)
        self.train_path = train_path
        return train_path

    def print_all_dimensions(self) -> None:
        """
            Utility function to print the number of layers in a network, as well as their number of neurons, and the
            activation function employed.
        """
        misc_utils.print_with_timestamp(f'*** Neural Network with {len(self.layers)} layers. ***')
        for i, layer in enumerate(self.layers):
            misc_utils.print_with_timestamp(f'*** Layer {i+1} ***')
            misc_utils.print_with_timestamp(f'\t{layer[0]}')
            misc_utils.print_with_timestamp(f'\t{layer[1]}')

    def print_parameters(self) -> None:
        """
            Utility function to print the parameters of the neural network, i.e., the weight matrix and the bias
            vectors, for each layer of the network.
        """
        self.print_all_dimensions()
        for i, layer in enumerate(self.layers):
            misc_utils.print_with_timestamp(f'*** Layer {i+1} ***')
            linear_layer, activation_layer = layer
            misc_utils.print_with_timestamp(f'W_{i+1} = ')
            misc_utils.print_with_timestamp(linear_layer.W)
            misc_utils.print_with_timestamp(f'b_{i+1} = ')
            misc_utils.print_with_timestamp(linear_layer.b)

    def print_train_shutdown_neurons(self) -> None:
        """
            Utility function to print the amount of neurons that has value 0.
        """
        misc_utils.print_with_timestamp('*** Shutdown neurons report ***')
        for epoch in range(self.train_shutdown_neurons.shape[0]):
            misc_utils.print_with_timestamp(f'Epoch: {epoch}')
            for batch_ctr in range(self.train_shutdown_neurons.shape[1]):
                misc_utils.print_with_timestamp(f'\tBatch: {batch_ctr}')
                for count_l, layer in enumerate(self.layers):
                    misc_utils.print_with_timestamp(f'\t\tLayer: {count_l}')
                    linear_layer, _ = layer
                    n_neurons = linear_layer.n_neurons
                    misc_utils.print_with_timestamp(f'\t\t\t{self.train_shutdown_neurons[epoch][batch_ctr][count_l]}'
                                                    f'/{n_neurons} were zero.')

    def print_train_costs_and_times(self) -> None:
        """
            Utility function to print all the training costs and minutes needed for each single epoch of training.
        """
        misc_utils.print_with_timestamp('*** Train costs and times report ***')
        for epoch in range(self.train_costs.shape[0]):
            misc_utils.print_with_timestamp(f'Epoch: {epoch}')
            for batch_ctr in range(self.train_costs.shape[1]):
                misc_utils.print_with_timestamp(f'\tBatch: {batch_ctr}')
                misc_utils.print_with_timestamp(f'\t\tCost: {self.train_costs[epoch][batch_ctr]}')
                misc_utils.print_with_timestamp(f'\t\tTime: {self.train_times[epoch][batch_ctr]}')

    def save_parameters(self, parameters: Dict, fold_ctr: int = None) -> None:
        """
            Utility function to save the parameters (weights and biases) of the neural network, along with a .txt file
            containing all the hyperparameters used during training.

            Args:
                parameters: A dictionary containing the number of epochs of training, and the size of a
                    single batch.

                fold_ctr: An optional parameter useful to separate the parameters of network across the different fold
                    during k-fold cross validation.
        """
        model_path = self.train_model_path(parameters)
        fname = model_path + '/parameters.txt'
        with open(fname, 'w') as f:
            f.write('***** Architecture *****\n')
            for i, layer in enumerate(self.architecture):
                f.write(f'Layer {i}\n')
                for key, value in layer.items():
                    f.write(f'{key}: {value}\n')
                f.write(f'Learning rate: {self.layers[i][0].learning_rate}\n')
                f.write(f'Weight decay: {self.layers[i][0].weight_decay}\n')
                f.write(f'Momentum: {self.layers[i][0].momentum}\n')
            f.write('\n\n***** Other parameters *****\n')
            for key, value in parameters.items():
                f.write(f'{key}: {value}\n')
            f.write(f'dataset features: {self.features_in_dataset}\n')
            f.write(f'random seed: {self.random_seed}\n')
        if fold_ctr is not None:
            model_path += ('/fold_' + str(fold_ctr))
            misc_utils.create_dir(model_path)
        fname = model_path + '/final_model.pkl'
        with open(fname, 'wb') as out_f:  # wb stands for write and binary
            pickle.dump(self, out_f, pickle.HIGHEST_PROTOCOL)

    def parse_architecture(self, architecture: List[Dict]) -> None:
        """
            This function initializes the layers of the Neural Network.
            Each single layer is a list of tuples, where the first element is the forward linear layer, and the second
            element is the activation layer.

            Args:
                architecture: A list of dictionaries, where each dictionary represents a layer of the network.
        """
        for i, layer in enumerate(architecture):
            n_neurons = layer['n_neurons']
            n_features = self.features_in_dataset if i == 0 else self.layers[i-1][0].n_neurons
            if 'other_hyper_params' in layer:
                lin_layer = LinearLayer(n_features, n_neurons, n_layer=i+1, hyperparams=layer['other_hyper_params'])
            else:
                lin_layer = LinearLayer(n_features, n_neurons, n_layer=i+1)
            try:
                activation = layer['activation']
                if activation == 'relu':
                    activation_layer = ReLULayer((n_features, n_neurons), n_layer=i+1)
                elif activation == 'softmax':
                    activation_layer = SoftMaxLayer((n_features, n_neurons), n_layer=i+1)
                else:
                    raise KeyError
            except KeyError:
                misc_utils.print_with_timestamp(f'No activation function is being used for layer {i+1}')
                activation_layer = ActivationLayer((n_features, n_neurons), n_layer=i+1)
            self.layers.append((lin_layer, activation_layer))

    def fast_gradient_sign_method(self, input_sample: np.ndarray, input_label: np.ndarray,
                                  modified: bool = False, return_perturbation: bool = True,
                                  epsilon: float = 0.1, loss_function: str = 'cross_entropy',
                                  verbose: bool = False) -> np.ndarray:
        """
            This method implements the Fast Gradient Sign Method [16].

            Args:
                input_sample: A ndarray of features with shape (n_features_in_dataset,).

                input_label: A ndarray of labels with shape (n_classes_in_dataset,).

                modified: A boolean which indicates whether to perform the original method proposed in [16], or the
                    "early stopped" version we adopted in our paper.

                return_perturbation: A boolean which indicates whether to return the perturbation vector alone,
                    or its sum with the input sample.

                epsilon: A float representing the amount of perturbation to add to samples.

                loss_function: A string which represents the loss function to use for computing the final loss.

                verbose: A boolean which indicates whether to print useful information for debugging.

            Returns:
                If return_perturbation is True, it returns:
                    perturbation: A ndarray containing the perturbation values obtained with the algorithm.
                else:
                    perturbed_sample: A ndarray containing the input_sample already perturbed with the result of the
                        algorithm.
        """
        if len(input_sample.shape) == 1:
            input_sample = input_sample.reshape(1, input_sample.shape[0])  # here I want a row vector
            input_label = input_label.reshape(1, input_label.shape[0])
        if verbose:
            misc_utils.print_with_timestamp(f'Fast gradient sign method for {input_sample.shape[0]} samples...')
        # now I do the forward pass with the trained parameters
        A_prev = input_sample  # initially, the output of the previous layer is the input layer
        for count_layer, layer in enumerate(self.layers):
            linear_layer, activation_layer = layer
            Z = linear_layer.forward(A_prev, attack=True)
            A = activation_layer.forward(Z, attack=True)
            A_prev = A

        # I compute the loss and the derivative of the loss
        _, dLoss_over_dOut = self.compute_cost(ground_truth=input_label, predictions=A_prev,
                                               loss_function=loss_function, attack=True)

        # now, I propagate the loss backwards
        dA_prev = dLoss_over_dOut
        for layer in reversed(self.layers):
            linear_layer, activation_layer = layer
            dZ_prev = activation_layer.backward(dA_prev, attack=True)
            dA_prev = linear_layer.backward(dZ_prev, attack=True)

        # the sign of the gradient times the epsilon coefficient
        # this is the original Fast Gradient Sign Method
        if not modified:
            perturbation = np.sign(dA_prev) * epsilon
        else:
            # this is the version we adopted in our paper, performing a kind of "early stopping" of the algorithm,
            # in order to obtain a vector composed of real values.
            perturbation = dA_prev
        if return_perturbation:
            return perturbation
        # I add the perturbation, in order to climb over the gradient
        perturbed_sample = input_sample + perturbation
        return perturbed_sample

    def aiXia2021_attack(self, test_X: np.ndarray, test_Y: np.ndarray,
                         predicted_Y_probabilities: np.ndarray, epoch: int,
                         values_of_psi: List[int]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
            The implementation of the attack algorithm proposed in Section IV of the paper.
            Args:
                test_X: A ndarray containing the input samples in the test set. It has shape
                (n_samples_in_test_set, n_features).

                test_Y: A ndarray containing the labels associated with the samples in the test set. It has shape
                (n_samples_in_test_set, n_classes).

                predicted_Y_probabilities: A ndarray containing the class probabilities predicted from the neural
                network. It has shape (n_samples_in_test_set, n_classes).

                epoch: It is an integer which represents the current epoch of training.

                values_of_psi: A list of integers which represent the amount of maximum features that the adversary
                can alter during the attack.

            Returns:
                results: A list of ndarrays containing the error% for the classes, for each single psi value.
                It has shape (values_of_psi, n_classes).

                history_of_altered_features: A numpy array containing the history of altered features.

                list_of_corrupted_X: A list of numpy arrays containing the altered input vectors for each value of psi.
        """
        predicted_Y = class_utils.from_probabilities_to_labels(predicted_Y_probabilities)

        # we are in the binary classification case, thus, the desired_y we suppose to be always
        # the contrary of the predicted
        perturbations = self.fast_gradient_sign_method(test_X, test_Y, modified=True)

        perturbable_features_mask = np.array(self.perturbation_mask, dtype=bool)

        # First, we want to evaluate the attack only on those samples which are correctly classified
        correctly_classified_mask = (predicted_Y == test_Y)[:, 0]
        results = []
        list_of_corrupted_X = []
        test_X = test_X[correctly_classified_mask]
        perturbations = perturbations[correctly_classified_mask]
        predicted_Y = predicted_Y[correctly_classified_mask]
        history_of_altered_features = np.zeros((len(values_of_psi), len(perturbations), self.features_in_dataset))
        for counter_psi, psi in enumerate(values_of_psi):
            psi_perturbations = np.copy(perturbations)
            perturbations_signs = np.sign(psi_perturbations)
            for i, single_perturbation in enumerate(psi_perturbations):
                input_vector = test_X[i]
                # here we compute what are the psi highest perturbations in terms of absolute value
                # we also want to consider only those perturbations which are legal, that is,
                # if the input vector has the value 1 in a particular binary feature, than, we can only wonder about
                # its possibility of being disrupted if and only if the correspondent value in the perturbation has
                # negative sign
                # first, we set to 0 all those features which we labeled as "not perturbable"
                # then, we perform the "signs" check as discussed above
                perturbation_signs_vector = perturbations_signs[i]
                perturbation_signs_vector[perturbation_signs_vector == -1] = 0
                flippable = np.logical_and(perturbable_features_mask,
                                           np.logical_xor(input_vector, perturbation_signs_vector))
                # here, inside flippable, we have value 1 for all those features which respect either one of these
                # two criteria:
                # 1) the input feature has value 1 and the perturbation has sign "-"
                # 2) the input feature has value 0 and the perturbation has sign "+"
                # in the end, we consider the psi highest perturbations
                ranked_perturbation_indices = np.flip(np.argsort(np.abs(single_perturbation)))
                counter = 0
                modified_indices = []
                for index in ranked_perturbation_indices:
                    if counter == psi:
                        break
                    if flippable[index] == 1:
                        if input_vector[index] == 0:
                            psi_perturbations[i][index] = 1
                            counter += 1
                            modified_indices.append(index)
                            history_of_altered_features[counter_psi][i][index] += 1
                        elif input_vector[index] == 1:
                            psi_perturbations[i][index] = -1
                            counter += 1
                            modified_indices.append(index)
                            history_of_altered_features[counter_psi][i][index] += 1
                not_modified_indices = [x for x in range(len(single_perturbation)) if x not in modified_indices]
                psi_perturbations[i][not_modified_indices] = 0
            corrupted_X = test_X + psi_perturbations
            list_of_corrupted_X.append(corrupted_X)
            corrupted_Y_probabilities = self.predict(corrupted_X)
            corrupted_Y = class_utils.from_probabilities_to_labels(corrupted_Y_probabilities)
            # We now want to count how many times the neural network was right, and with the corrupted samples is
            # induced into error instead.
            result = self.evaluate_predictions(predicted_Y, corrupted_Y, n_epoch=epoch + 1, attack=True, psi=psi)
            results.append(result)
        return results, history_of_altered_features, list_of_corrupted_X

    def train(self, train_x: np.ndarray, train_y: np.ndarray,
              n_epochs: int, batch_size: int = -1,
              loss_function: str = 'cross_entropy',
              debug_step: int = -1, save: bool = True, shuffle: bool = True,
              fold_ctr: int = None, test_x: np.ndarray = None, test_y: np.ndarray = None,
              attack: bool = False, values_of_psi: List[int] = None) -> None:
        """
            This function performs the gradient descent algorithm for tuning the parameters of the network [3].
            Args:
                train_x: A ndarray containing the samples belonging to the training set. It has shape
                    (n_samples_in_train_set, n_features).

                train_y: A ndarray containing the class labels for the samples belonging to the training set. It has
                    shape (n_samples_in_train_set, n_classes).

                n_epochs: An integer representing the number of epochs for the neural network training.

                batch_size: An integer representing the number of samples to consider in each batch. If it has value
                    -1, the entire training set is considered as a single batch during training, performing standard
                    gradient descent.

                loss_function: A string representing the loss function used for the computation of the cost,
                    which is the average loss between the prediction and the ground truth between all the examples in
                    the batch.

                debug_step: An integer which measures the number of epochs to scan before printing some output
                    information.

                save: A boolean which indicates whether to save the trained parameters of the neural network.

                shuffle: A boolean which indicates whether to shuffle the order of batches feeded during the different
                    epochs of training.

                fold_ctr: An optional parameter useful to separate the parameters of network across the different fold
                    during k-fold cross validation.

                test_x: A ndarray containing the samples belonging to the test set. It has shape
                    (n_samples_in_test_set, n_features).

                test_y: A ndarray containing the class labels for the samples belonging to the training set. It has
                    shape (n_samples_in_test_set, n_classes).

                attack: A boolean which indicates whether to perform the attack proposed in the paper at each training
                    epoch.

                values_of_psi: A list of integers which represent the maximum values of features that the adversary
                    may perturb with the attack.
        """
        legal_loss_functions = ['cross_entropy', 'mse']
        try:
            assert loss_function in legal_loss_functions
        except AssertionError:
            exit(f'Unrecognized loss function. Please provide one between {legal_loss_functions}')
        if debug_step == -1:  # default
            debug_step = n_epochs / 10
        if batch_size == -1:  # only use one batch (full batch gradient descent)
            batch_size = len(train_x)
        if debug_step is not False:
            misc_utils.print_with_timestamp(f'Start training for {n_epochs} epochs...')
        batches = class_utils.batches_generator(train_x, batch_size)
        batches_y = class_utils.batches_generator(train_y, batch_size)
        parameters = dict()
        parameters['batch_size'] = batch_size
        parameters['n_epochs'] = n_epochs
        parameters['loss_function'] = loss_function
        n_batches = len(batches)
        self.train_costs = np.zeros((n_epochs, n_batches))
        self.train_times = np.zeros(n_epochs)
        self.train_shutdown_neurons = np.zeros((n_epochs, n_batches, len(self.layers)))
        results_to_plot = []

        for epoch in range(n_epochs):
            if shuffle:
                batches, batches_y = class_utils.shuffle_batches(batches, batches_y)
            timer = misc_utils.Timer()

            for batch_ctr, batch in enumerate(batches):
                train_x_batch = batch
                train_y_batch = batches_y[batch_ctr]

                # forward-propagation
                A_prev = train_x_batch  # initially, the output of the previous layer is the input layer
                for count_layer, layer in enumerate(self.layers):
                    linear_layer, activation_layer = layer
                    linear_layer.forward(A_prev)
                    activation_layer.forward(linear_layer.Z)
                    A_prev = activation_layer.A
                    self.train_shutdown_neurons[epoch][batch_ctr][count_layer] = np.count_nonzero(A_prev == 0)
                # here, A_prev contains the predictions of the model

                # Compute loss for every sample
                cost, dLoss_over_dOut = self.compute_cost(ground_truth=train_y_batch, predictions=A_prev,
                                                          loss_function=loss_function)
                self.train_costs[epoch][batch_ctr] = cost

                # back-propagation
                # in the first step of backpropagation, the uppermost gradient is the gradient of the cost function
                # with respect to the predictions of the model.
                dA_prev = dLoss_over_dOut
                for layer in reversed(self.layers):
                    linear_layer, activation_layer = layer
                    activation_layer.backward(dA_prev)
                    linear_layer.backward(activation_layer.dZ_prev)
                    dA_prev = linear_layer.dA_prev

                # Parameters update
                for layer in reversed(self.layers):
                    linear_layer, _ = layer
                    linear_layer.update_params()

            # epoch end
            time_for_epoch = timer.stop()
            self.train_times[epoch] = time_for_epoch
            if debug_step is not False and epoch % debug_step == 0:
                misc_utils.print_with_timestamp(f'{epoch + 1}/{n_epochs}')
                misc_utils.print_with_timestamp(f'\t\tCost: {self.train_costs[epoch].mean()}')
                partial_elapsed_time = self.train_times.sum()
                misc_utils.print_with_timestamp(f'\t\tElapsed time until now {partial_elapsed_time} minutes')

            # save predictions if needed
            if save:
                self.save_parameters(parameters, fold_ctr)
            if test_x is not None and test_y is not None:
                y_hat = self.predict(test_x)
                results = dict()
                results['epoch'] = epoch + 1
                fscore_values = self.evaluate_predictions(y_hat, test_y, n_epoch=epoch+1)
                for i, label in enumerate(self.class_labels):
                    results['f-score ' + label] = fscore_values[i]
                if attack:
                    error_values, _history_of_altered_features, _corrupted_X = \
                        self.aiXia2021_attack(test_x, test_y, y_hat, epoch, values_of_psi=values_of_psi)
                    for psi_idx, psi in enumerate(values_of_psi):
                        psi_str = r'$\psi=$' + str(psi)
                        for i, label in enumerate(self.class_labels):
                            results['error ' + label + ' (' + psi_str + ')'] = error_values[psi_idx][i]
                results_to_plot.append(results)
        total_elapsed_time = self.train_times.sum()
        if debug_step is not False:
            misc_utils.print_with_timestamp('\n*********************************************************\n')
            misc_utils.print_with_timestamp(f'Training ended in {total_elapsed_time} minutes')
            misc_utils.print_with_timestamp(f'Mean train time for epoch {total_elapsed_time.mean()} minutes')
            final_cost = self.train_costs[-1].mean()
            misc_utils.print_with_timestamp(f'Final cost: {final_cost}')
            misc_utils.print_with_timestamp('\n*********************************************************\n')
        if save:
            self.save_parameters(parameters, fold_ctr)
        if test_x is not None and test_y is not None:
            y_hat = self.predict(test_x)
            self.evaluate_predictions(y_hat, test_y)
            self.plot(pd.DataFrame(results_to_plot), values_of_psi)

    def compute_cost(self, ground_truth: np.ndarray, predictions: np.ndarray,
                     loss_function: str = 'cross_entropy', epsilon: float = 1e-12,
                     attack: bool = False) -> Tuple[float, np.ndarray]:
        """
            This function computes the cost of the neural network during the training, which is defined as the sum
            of the loss averaged across all the training samples [11].
            Args:
                ground_truth: It is a ndarray which represents the ground truth class labels. The labels are in their
                    one-hot encoded form, and the array has shape (n_samples_in_train_set, n_classes).

                predictions: It is a ndarray which contains the classes probabilities predicted by the neural network.
                    It has shape (n_samples_in_train_set, n_classes).

                loss_function: It is a string which represents the type of loss function to compute.

                epsilon: It is a float useful to avoid taking the logarithm of 0.

                attack: It is a boolean which indicates whether the calling function is the plain gradient descent
                    algorithm for training, or the adversarial attack algorithm. Indeed, in the second case, the
                    Jacobian does not have to be divided by the number of samples in the batch, which is then supposed
                    to be of size 1 during the execution of an adversarial attack algorithm.

            Returns:
                cost_value: It is a float which represents the average of the loss between all the samples in the batch.

                dLoss_over_dOut: It is a ndarray which represents the Jacobian of the cost, that is, it is the vector
                    of all the derivatives of the cost function with respect to every single predictions. It has the
                    same shape as the ground_truth and the predictions.
        """
        try:
            assert ground_truth.shape[0] == predictions.shape[0]
        except AssertionError:
            misc_utils.print_with_timestamp(f'ground_truth: {ground_truth.shape}')
            misc_utils.print_with_timestamp(f'predictions: {predictions.shape}')
            exit()
        N = predictions.shape[0]  # this is the number of samples in the batch
        if loss_function == 'cross_entropy':
            # useful for classification
            # To avoid numerical issues with logarithm, clip the predictions to [10^{−12}, 1 − 10^{−12}] range [20].
            predictions = np.clip(predictions, epsilon, 1. - epsilon)
            ce = -np.sum(ground_truth * np.log(predictions)) / N
            # if the loss function is the categorical cross entropy
            # L(y_hat, y) = - \sum_{i=1}^{|Classes|} y_i log(y_hat_i)
            # the derivative is:
            # dL / dy_hat = dL = - (y/y_hat)
            # then we take the mean for all the samples in the batch
            # dLoss_over_dOut = - (ground_truth / predictions) / N
            # [12] when combining cross entropy and softmax in the output layer
            # the combined derivative has the simple expression:
            dLoss_over_dOut = (predictions - ground_truth)
            if not attack:
                dLoss_over_dOut /= N
            # this is also confirmed in [2] (pdf page 226, book page 215)
            return ce, dLoss_over_dOut
        elif loss_function == 'mse':
            mse = np.power((predictions - ground_truth), 2) / N
            dLoss_over_dOut = 2 * (predictions - ground_truth) / N
            return mse, dLoss_over_dOut

    def predict(self, x: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
            This function performs the forward propagation of the neural network in order to obtain the predictions.

            Args:
                x: A ndarray which represents the input samples to associate with a label. It has shape
                    (n_samples_in_test_set, n_features).

                verbose: A boolean which indicates whether to print useful output information.

            Returns:
                predictions: A ndarray which contains the predictions computed by the neural network for the samples
                    in the test set. It has shape (n_samples_in_test_set, n_features).
        """
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        timer = misc_utils.Timer()
        n_predictions = x.shape[0]
        if verbose:
            misc_utils.print_with_timestamp(f'Predicting {n_predictions} samples...')
        A_prev = x
        for layer in self.layers:
            linear_layer, activation_layer = layer
            linear_layer.forward(A_prev)
            activation_layer.forward(linear_layer.Z)
            A_prev = activation_layer.A
        predictions = A_prev
        if verbose:
            misc_utils.print_with_timestamp(f'Ended in {timer.stop()} minutes.')
        self.predictions = predictions
        return self.predictions

    def evaluate_predictions(self, predictions: np.ndarray, ground_truth: np.ndarray,
                             save: bool = True, attack: bool = None,
                             fold_ctr: int = None, n_epoch: int = None, psi: int = None) -> np.ndarray:
        """
            Given the ground truth labels and those predicted by the neural network, this method computes the f-score
            values (or the error% values if attack is True) for the classes in the dataset.

            Args:
                predictions: A ndarray containing the predicted labels from the model. It has shape
                    (n_samples, n_classes).

                ground_truth: A ndarray containing the ground truth labels for the samples. It has shape
                    (n_samples, n_classes).

                save: A boolean which indicates whether to save the results computed.

                attack: A boolean which indicates whether the calling function is an attack algorithm, thus computing
                    the error% of the classes.

                fold_ctr: An optional parameter useful to separate the parameters of network across the different fold
                    during k-fold cross validation.

                n_epoch: An integer which represent the current number of training epoch.

                psi: An integer which represent the maximum number of bits that the adversary may perturb.

            Returns:
                If attack is False it returns:
                    fscore_values: A ndarray which contains the f-score values of the classes for the current epoch.
                else:
                    error_values: A ndarray which contains the error% values of the classes for the current epoch, with
                        the specific psi value in input.
        """
        predicted_classes = class_utils.from_probabilities_to_labels(predictions)
        # now I have to pass from the one-hot encoded version to a 1D version
        predicted_classes = class_utils.one_hot(predicted_classes, encode_decode=1)
        ground_truth = class_utils.one_hot(ground_truth, encode_decode=1)
        y_ground = pd.Series(ground_truth, name='Ground_truth')
        y_pred = pd.Series(predicted_classes, name='Predicted')
        try:
            df_confusion = pd.crosstab(y_ground, y_pred, margins=True)
        except Exception:
            df_confusion = pd.DataFrame(0, index=['0', '1', 'All'], columns=['0', '1', 'All'])
        if self.class_labels is not None:
            for i, column in enumerate(df_confusion):
                if i == len(self.class_labels):
                    break
                x1 = 'predicted'
                x2 = 'ground'
                if attack is not None:
                    x1 = 'perturbed'
                    x2 = 'predicted'
                df_confusion.rename(columns={i: x1+'_' + self.class_labels[i]}, inplace=True)
                df_confusion.rename(index={i: x2+'_' + self.class_labels[i]}, inplace=True)
        if len(df_confusion) != len(df_confusion.columns):
            rows_index = df_confusion.index.to_list()
            columns_index = df_confusion.columns.to_list()
            columns_to_add = []
            for x in rows_index:
                class_label = x.split('_')[-1]
                found = False
                for y in columns_index:
                    tmp = y.split('_')[-1]
                    if tmp == class_label:
                        found = True
                        break
                if not found:
                    columns_to_add.append(x)
            for c in columns_to_add:
                idx = rows_index.index(c)
                t = c.split('_')[1]
                x1 = 'predicted'
                if attack is not None:
                    x1 = 'perturbed'
                df_confusion.insert(loc=idx, column=x1+'_'+t, value=[0]*len(df_confusion))
        if not attack:
            try:
                metrics = classification_report(y_ground, y_pred, target_names=self.class_labels,
                                                output_dict=True, zero_division=0)
            except Exception as e:
                print(e)
            df_metrics = pd.DataFrame(metrics).transpose()
        if save:  # save predictions
            model_path = self.train_path
            if attack:
                model_path += '/attacks'
                misc_utils.create_dir(model_path)
            if fold_ctr is not None:
                model_path += '/fold_' + str(fold_ctr)
            if n_epoch is None:
                fname = model_path + '/final_predictions.csv'
                fname_metrics = model_path + '/final_metrics.csv'
                if attack:
                    attack_str = '_attack.csv'
                    fname = fname[:-4] + attack_str
            else:
                fname = model_path + '/single_epoch_results'
                misc_utils.create_dir(fname)
                fname = fname + '/' + str(n_epoch) + '_epoch'
                misc_utils.create_dir(fname)
                fname = fname + '/' + str(n_epoch) + '_epoch_predictions.csv'
                fname_metrics = f'{model_path}/single_epoch_results/{n_epoch}_epoch/{n_epoch}_epoch_metrics.csv'
                if attack:
                    attack_str = '_attack.csv'
                    if psi is not None:
                        attack_str = str(psi) + '_psi_' + attack_str
                    fname = fname[:-4] + attack_str
            df_confusion.to_csv(fname, index=True)
            if not attack:
                df_metrics.to_csv(fname_metrics, index=True)
                fscore_values = np.zeros(len(self.class_labels))
                for i, _label in enumerate(self.class_labels):
                    fscore_values[i] = df_metrics.iloc[i]['f1-score']
                return fscore_values
            else:
                n_wrong_samples = np.zeros(len(self.class_labels))
                all_samples = np.zeros(len(self.class_labels))
                for i, label in enumerate(self.class_labels):
                    correct_samples_of_label = df_confusion.iloc[i, i]
                    all_samples_of_label = df_confusion.loc['All'].iloc[i]
                    all_samples[i] = all_samples_of_label
                    n_wrong_samples[i] = all_samples_of_label - correct_samples_of_label
                error_values = np.divide(n_wrong_samples, all_samples, out=np.zeros_like(n_wrong_samples),
                                         where=all_samples != 0)
                return error_values

    def plot(self, results_to_plot: pd.DataFrame, values_of_psi: List[int]) -> None:
        """
            This function plots the f-score values and the error% values for the classes along all the different
            training epochs.

            Args:
                results_to_plot: A pandas DataFrame which contains all the fscore and error% values for the classes.

                values_of_psi: A list of integers which represent the number of bits that the adversary may perturb with
                    an attack.
        """
        fig, ax = plt.subplots(figsize=(13, 13))
        textstr = '\t\t'.join((
            r'$\alpha=%.2e$' % (self.layers[0][0].learning_rate,),
            r'$\lambda=%.2e$' % (self.layers[0][0].weight_decay,),
            r'$\nu=%.2e$' % (self.layers[0][0].momentum,),
            r'$\kappa=%s$' % (str(self.features_in_dataset),),
            r'$\eta=%s$' % (self.n_neurons_str,)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.1)
        columns = ['epoch', 'f-score NIT', 'f-score SXT']
        plain_results = results_to_plot[columns]
        plain_results = plain_results.melt('epoch', var_name='cols', value_name='%')
        sns.lineplot(data=plain_results, x='epoch', y='%', hue='cols', style='cols')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, fontsize=14, ncol=2)
        plt.text(0.01, 1.04, textstr, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('%', fontsize=14)
        plt.savefig(self.train_path+'/fscore_results.pdf')
        plt.close(fig)
        plt.close()
        fig, ax = plt.subplots(figsize=(13, 13))
        global_results = results_to_plot.melt('epoch', var_name='cols', value_name='%')
        sns.lineplot(data=global_results, x='epoch', y='%', hue='cols', style='cols')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, fontsize=14, ncol=4)
        plt.text(0.01, 1.04, textstr, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('%', fontsize=14)
        plt.savefig(self.train_path+'/global_results.pdf')
        plt.close(fig)
        plt.close()
        fig, ax = plt.subplots(figsize=(13, 13))
        error_results = results_to_plot[results_to_plot.columns[~results_to_plot.columns.
                                        isin(['f-score NIT', 'f-score SXT'])
                                        ]].melt('epoch', var_name='cols', value_name='%')
        sns.lineplot(data=error_results, x='epoch', y='%', hue='cols', style='cols')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, fontsize=14, ncol=3)
        plt.text(0.01, 1.04, textstr, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('%', fontsize=14)
        plt.savefig(self.train_path+'/error_results.pdf')
        plt.close(fig)
        plt.close()
        columns = ['epoch', 'f-score NIT', 'f-score SXT']
        for psi in values_of_psi:
            psi_str = r'$\psi=$' + str(psi)
            columns_with_psi = columns[:]
            for i, label in enumerate(self.class_labels):
                columns_with_psi += ['error ' + label + ' (' + psi_str + ')']
            fig, ax = plt.subplots(figsize=(13, 13))
            psi_error_results = results_to_plot[columns_with_psi].melt('epoch', var_name='cols', value_name='%')
            sns.lineplot(data=psi_error_results, x='epoch', y='%', hue='cols', style='cols')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, fontsize=14, ncol=2)
            plt.text(0.01, 1.04, textstr, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('%', fontsize=14)
            plt.savefig(self.train_path + '/' + str(psi) + '_psi_error_results_with_fscore.pdf')
            plt.close(fig)
            plt.close()
        columns = ['epoch']
        for psi in values_of_psi:
            psi_str = r'$\psi=$' + str(psi)
            columns_with_psi = columns[:]
            for i, label in enumerate(self.class_labels):
                columns_with_psi += ['error ' + label + ' (' + psi_str + ')']
            fig, ax = plt.subplots(figsize=(13, 13))
            psi_error_results = results_to_plot[columns_with_psi].melt('epoch', var_name='cols', value_name='%')
            sns.lineplot(data=psi_error_results, x='epoch', y='%', hue='cols', style='cols')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, fontsize=14, ncol=2)
            plt.text(0.01, 1.04, textstr, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('%', fontsize=14)
            plt.savefig(self.train_path + '/' + str(psi) + '_psi_error_results.pdf')
            plt.close(fig)
            plt.close()


class LinearLayer:
    """
        This class models a Linear Layer of the neural network, which performs the forward computation of the input
        for the weight matrix, adding the bias vector. It is based on the implementation given in [3].
        Following notation in [4], a LinearLayer computes the dot product between weight and inputs, and add bias.
        This is the value of Z.

        Attributes:
            n_features: An integer which represents the number of input features. For the first LinearLayer, this number
                coincides with the features contained in the dataset. For intermediate layers, this number coincides
                with the number of neurons of the previous layers.

            n_neurons: An integer which represents the number of neurons in the layer.

            n_layer: An integer which represents the index of the layer in the list of all the layers of the neural
                network.

            W: A ndarray which represents the weight matrix of the layer. It has shape (n_neurons, n_inputs), where
                n_inputs coincides with the number of features in the dataset for the first layer, whereas it coincides
                with the number of neurons in the previous layer for intermediate layers.

            b: A ndarray which represents the bias vector of the layer. It has shape (n_neurons,)

            Z: A ndarray which contains the dot product between the weight matrix with the input, by also adding the
                bias vector in the end.

            dW: A ndarray which represents the Jacobian of the weight matrix. It has the same shape as W.

            db: A ndarray which represents the Jacobian of the bias vector. It has the same shape as b.

            A_prev: A ndarray which contains the input coming from the previous layer. It has shape
                (n_inputs_previous_layer, n_neurons_previous_layer).

            dA_prev: A ndarray which contains the Jacobian of A_prev.

            weight_velocity: A ndarray which contains the weight step of the previous gradient descent iteration. It is
                useful for gradient descent with momentum.

            bias_velocity: A ndarray which contains the bias step of the previous gradient descent iteration. It is
                useful for gradient descent with momentum.
    """
    def __init__(self, n_incoming_features: int, n_neurons: int, n_layer: int, hyperparams: dict = None) -> None:
        """
            Inits the LinearLayer performing random initialization of weights and biases according to a specific rule.

            Args:
                n_incoming_features: An integer which represents the number of input features.

                n_neurons: An integer which represents the number of neurons in the layer.

                n_layer: An integer which represents the index of the layer in the list of all the layers of the neural
                network.

                hyperparams: A dictionary which contains the hyperparameters for the training phase. In particular,
                    the following values can be indicated (they are all optional):
                    init_type: A string which represents the initialization type for the weight matrix of the layer.
                    alpha: A float which represents the learning rate for the layer.
                    lambda: A float which represents the regularization factor for the layer.
                    W: A ndarray which represents a manually inserted weight matrix.
                    b: A ndarray which represents a manually inserted bias vector.
        """
        self.n_features = n_incoming_features
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.W, self.b, self.scaling_factor = None, None, None
        self.learning_rate, self.weight_decay, self.momentum = None, None, None
        self.initialize_parameters(self.n_features, n_neurons, hyperparams)
        self.Z = np.zeros((self.n_neurons, self.n_features))
        self.A_prev = None
        self.dA_prev = None
        self.dW, self.db = np.zeros((self.n_neurons, self.n_features)), np.zeros((self.n_neurons,))
        self.weight_velocity = np.zeros((self.n_neurons, self.n_features))
        self.bias_velocity = np.zeros(self.n_neurons)

    def __repr__(self) -> str:
        return f'Linear Layer {self.n_layer} learning_rate: {self.learning_rate} weight_decay: {self.weight_decay} ' \
               f'momentum {self.momentum}'

    def print_parameters(self) -> None:
        """
            Utility function to print all the parameters of the neural network.
        """
        misc_utils.print_with_timestamp(self)
        misc_utils.print_with_timestamp(f'W{self.n_layer} {self.W.shape}')  # (n_neurons_layer, n_samples)
        misc_utils.print_with_timestamp(self.W)
        misc_utils.print_with_timestamp(f'b{self.n_layer} {self.b.shape}')  # (n_neurons_layer)
        misc_utils.print_with_timestamp(self.b)
        misc_utils.print_with_timestamp(f'z{self.n_layer} {self.Z.shape}')  # (n_samples, n_neurons_layer)
        misc_utils.print_with_timestamp(str(self.Z))

    def initialize_parameters(self, n_in: int, n_out: int, hyper_params: dict = None) -> None:
        """
            This method initializes the LinearLayer by performing a random init of the weight matrix and the bias
            vector, and by setting other hyper parameters useful for the training phase.
            Args:
                n_in: An integer which represents the size of the input layer.
                n_out: An integer which represents the size of output/number of neurons.
                hyper_params: A dictionary which contains the hyperparameters for the training phase. In particular,
                    the following values can be indicated (they are all optional):
                    init_type: A string which represents the initialization type for the weight matrix of the layer.
                    alpha: A float which represents the learning rate for the layer.
                    lambda: A float which represents the regularization factor for the layer.
                    W: A ndarray which represents a manually inserted weight matrix.
                    b: A ndarray which represents a manually inserted bias vector.
        """
        if hyper_params is None:
            hyper_params = dict()
        init_types = {'plain': 0.01,
                      'xavier': 1 / (math.sqrt(n_in)),  # [15]
                      'he': math.sqrt(2 / n_in)}  # [14]
        if 'init_type' in hyper_params:
            init_type = hyper_params['init_type']
        else:  # default - this is preferred for ReLU [14]
            init_type = 'he'
        try:
            assert init_type in init_types
        except AssertionError:
            misc_utils.print_with_timestamp(f'The inserted init_type {init_type} is not a valid one.')
            keys = ','.join(k for k in init_types.keys())
            misc_utils.print_with_timestamp(f'Please insert a type in: {keys}')
        self.scaling_factor = init_types[init_type]

        if 'learning_rate' in hyper_params:  # learning rate
            self.learning_rate = hyper_params['learning_rate']
        else:  # default
            self.learning_rate = 0.01

        if 'weight_decay' in hyper_params:  # weight decay
            self.weight_decay = hyper_params['weight_decay']
        else:  # default
            self.weight_decay = 1e-5

        if 'momentum' in hyper_params:
            self.momentum = hyper_params['momentum']
        else:
            # with a momentum equal to 0.1, we keep track of a dW up to 10 updates
            self.momentum = 0.1

        if 'W' in hyper_params:  # manual init of weights
            self.W = hyper_params['W']
        else:
            self.W = np.random.randn(n_out, n_in) * self.scaling_factor
        # W^[l]: (n^[l], n^[l-1])
        assert (self.W.shape[0] == self.n_neurons)

        # b^[l]: (n^[l], 1)
        if 'b' in hyper_params:  # manual init of bias
            self.b = hyper_params['b']
        else:
            self.b = np.zeros(n_out)  # a single bias for every output

    def forward(self, A_prev: np.ndarray, attack: bool = False) -> Union[None, np.ndarray]:
        """
            This function performs the forward propagation using activations from previous layer.
            Args:
                A_prev: A ndarray which contains the data coming from the previous layer.

                attack: A boolean which indicates whether the calling function is the gradient descent for training
                    or an attack algorithm. In the latter case, the value of Z is returned instead of being set.
        """
        if not attack:
            self.A_prev = np.copy(A_prev)
            self.Z = np.matmul(self.A_prev, np.transpose(self.W)) + self.b
        else:
            return np.matmul(self.A_prev, np.transpose(self.W)) + self.b

    def backward(self, upstream_grad: np.ndarray, attack: bool = False) -> Union[None, np.ndarray]:
        """
            This function performs the back propagation using upstream gradients.
            Citing from [3]:
                "To perform backpropagation we’ll employ the following technique: at each node, we only have our local
                gradient computed(partial derivatives of that node), then during backpropagation, as we are receiving
                numerical values of gradients from upstream, we take these and multiply with local gradients to pass
                them on to their respective connected nodes."
            Note that the first upstream gradient is dLoss/dLoss = 1
            Recall the chain rule from calculus:
                let h(x) = g(f(x))
                h'(x) = g'(f(x)) * f'(x)
            For this reason, at high level, a backward pass is implemented as a product
            between the gradient coming from the deeper level and the gradient local to the current layer.
            With local gradient we refer to the partial derivatives of that node.

            Args:
                upstream_grad: It is a ndarray which represents the gradient coming in from the upper layer to couple
                    with local gradient. It has shape (N, k) where N is the size of the batch,
                    and k is the number of neurons in the upper layer.

                attack: A boolean which indicates whether the calling function is the gradient descent for training
                    or an attack algorithm. In the latter case, the value of dA_prev is returned instead of being set.
        """
        # gradient of Cost w.r.t W
        # shape (1, k) where k is the number of neurons in the previous layer
        # if the computation of the single neuron is: Z = x * W^T + b
        # the local gradient w.r.t. W is simply x
        # which in our case is the output of the previous layer: A_prev
        dW = np.dot(np.transpose(upstream_grad), self.A_prev)
        if not attack:
            self.dW = dW
        # gradient of Cost w.r.t b
        # if the computation of the single neuron is: Z = x * W^T + b
        # the local gradient w.r.t. b is simply 1
        # [3]
        # "Since the ∂Z/∂b (local gradient at the Z node) is equal to 1,
        # the total gradient at b is the sum of gradients from each example with respect to the Cost."
        db = np.sum(upstream_grad, axis=0)
        if not attack:
            self.db = db
        # gradient of Cost w.r.t A_prev
        # this will be useful for adversarial attacks (dA_prev of the first layer, means the gradient w.r.t. the input)
        # if the computation of the single neuron is: Z = x * W^T + b
        # the local gradient w.r.t. x is simply W^T
        dA_prev = np.dot(upstream_grad, self.W)
        if not attack:
            self.dA_prev = dA_prev
        if attack:
            return dA_prev

    def update_params(self) -> None:
        """
            This function performs the gradient descent update with weight decay and momentum.
            For the regularization term, consult [2] page 224, [19].
            For the momentum term, consult [2] page 288, [17].
        """
        weight_decay_term = -1 * (self.learning_rate * self.weight_decay) * self.W
        momentum_term = self.momentum * self.weight_velocity
        gradient_descent_term = -1 * self.learning_rate * self.dW
        new_increment = gradient_descent_term + weight_decay_term + momentum_term
        self.W = self.W + new_increment
        self.weight_velocity = np.copy(new_increment)
        bias_velocity = self.momentum * self.bias_velocity + self.db
        self.b = self.b - self.learning_rate * bias_velocity
        self.bias_velocity = np.copy(bias_velocity)


class ActivationLayer:
    """
        This class models an Identity Activation function.
        It will be extended with multiple different activation functions.

        Attributes:
            n_layer: An integer which represents the index of the layer in the list of all the layers of the neural
                network.

            A: A ndarray which represents the output of the activation function layer applied to the output of the
                corresponding linear layer. It has shape (n_neurons,).

            dA: A ndarray containing the Jacobian of A. It has the same shape of A.
    """
    def __init__(self, shape: int, n_layer: int) -> None:
        """
            Inits an ActivationLayer with zero values.

            Args:
                shape: An integer which represents the number of neurons in the corresponding LinearLayer.

                n_layer: An integer which represents the index of the layer in the list of all the layers of the neural
                    network.
        """
        self.n_layer = n_layer
        self.A = np.zeros(shape)
        self.dZ_prev = np.zeros(shape)

    def __repr__(self) -> str:
        return f'No activation function Layer {self.n_layer} ' + self.internal_shape__str__()

    def print_parameters(self) -> None:
        """
            Utility function to print all the parameters of the neural network.
        """
        misc_utils.print_with_timestamp(self)
        misc_utils.print_with_timestamp(f'a{self.n_layer} {self.A.shape}')  # (n_samples, n_neurons_layer1)
        misc_utils.print_with_timestamp(str(self.A))

    def internal_shape__str__(self) -> str:
        return f'{self.A.shape}'

    def forward(self, Z: np.ndarray) -> None:
        """
            This function models the identity activation function forward pass.

            Args:
                Z: A ndarray containing the output of the corresponding LinearLayer.
        """
        self.A = Z

    def backward(self, upstream_grad: np.ndarray, attack: bool = False) -> Union[None, np.ndarray]:
        """
            This function models the backward pass of the identity activation function.

            Args:
                upstream_grad: A ndarray containing the Jacobian coming from the upper layer. It has to be multiplied
                    for the local gradient in order to obtain the Jacobian of the current layer.

                attack: A boolean which indicates whether the calling function is the gradient descent for training
                    or an attack algorithm. In the latter case, the value of dZ_prev is returned instead of being set.
        """
        if not attack:
            self.dZ_prev = upstream_grad
        else:
            return upstream_grad


class ReLULayer(ActivationLayer):
    """
        This class implements the ReLU Layer.
        Following notation in [4], a ReLU computes the activation of Z (the output of the current layer) by
        computing the rectified function of Z.
        This is the value of A.
    """
    def __init__(self, shape: int, n_layer: int) -> None:
        """
            Inits the ReLULayer with zero values.

            Args:
                shape: An integer which represents the number of neurons in the corresponding LinearLayer.

                n_layer: An integer which represents the index of the layer in the list of all the layers of the neural
                    network.
        """
        super().__init__(shape, n_layer)

    def __repr__(self) -> str:
        return f'ReLU Layer {self.n_layer} ' + self.internal_shape__str__()

    def forward(self, Z: np.ndarray, attack: bool = False) -> Union[None, np.ndarray]:
        """
            This function models the ReLU activation function forward pass.

            Args:
                Z: A ndarray containing the output of the corresponding LinearLayer.

                attack: A boolean which indicates whether the calling function is the gradient descent for training
                    or an attack algorithm. In the latter case, the value of A is returned instead of being set.
        """
        Z_copy = np.copy(Z)
        Z_copy[Z_copy < 0] = 0
        if not attack:
            self.A = Z_copy
        else:
            return Z_copy

    def backward(self, upstream_grad: np.ndarray, attack: bool = False) -> Union[None, np.ndarray]:
        """
            This function performs the back propagation step of the ReLU activation function [21].

            Args:
                upstream_grad: A ndarray containing the Jacobian coming from the upper layer. It has to be multiplied
                    for the local gradient in order to obtain the Jacobian of the current layer.

                attack: A boolean which indicates whether the calling function is the gradient descent for training
                    or an attack algorithm. In the latter case, the value of dZ_prev is returned instead of being set.
        """
        local_gradient = np.copy(self.A)
        local_gradient[local_gradient > 0] = 1
        local_gradient[local_gradient <= 0] = 0
        if not attack:
            self.dZ_prev = upstream_grad * local_gradient
        else:
            return upstream_grad * local_gradient


class SigmoidLayer(ActivationLayer):
    """
        This class implements the Sigmoid Layer. The base implementation is given in [3].
    """
    def __init__(self, shape: int, n_layer: int) -> None:
        """
            Inits the SigmoidLayer with zero values.

            Args:
                shape: An integer which represents the number of neurons in the corresponding LinearLayer.

                n_layer: An integer which represents the index of the layer in the list of all the layers of the neural
                    network.
        """
        super().__init__(shape, n_layer)

    def __repr__(self) -> str:
        return f'Sigmoid Layer {self.n_layer} ' + self.internal_shape__str__()

    def forward(self, Z: np.ndarray, attack: bool = False) -> Union[None, np.ndarray]:
        """
            This function models the Sigmoid activation function forward pass.

            Args:
                Z: A ndarray containing the output of the corresponding LinearLayer.

                attack: A boolean which indicates whether the calling function is the gradient descent for training
                    or an attack algorithm. In the latter case, the value of A is returned instead of being set.
        """
        if not attack:
            self.A = 1 / (1 + np.exp(-Z))
        else:
            return 1 / (1 + np.exp(-Z))

    def backward(self, upstream_grad: np.ndarray, attack: bool = False) -> Union[None, np.ndarray]:
        """
            This function performs the back propagation step through the Sigmoid activation function.
            Local gradient => derivative of Sigmoid [3]:
                sigmoid(z) = 1 / (1 + e^(-z))
                let y_hat = 1 / (1 + e^(-z))
                sigmoid'(z) = y_hat * (1 - y_hat)

            Args:
                upstream_grad: A ndarray containing the Jacobian coming from the upper layer. It has to be multiplied
                    for the local gradient in order to obtain the Jacobian of the current layer.

                attack: A boolean which indicates whether the calling function is the gradient descent for training
                    or an attack algorithm. In the latter case, the value of dZ_prev is returned instead of being set.
        """
        local_gradient = self.A*(1-self.A)
        if not attack:
            self.dZ_prev = upstream_grad * local_gradient
        else:
            return upstream_grad * local_gradient


class SoftMaxLayer(ActivationLayer):
    """
        This class implements the SoftMax activation function Layer.
    """
    def __init__(self, shape: int, n_layer: int) -> None:
        """
            Inits the SoftmaxLayer with zero values.

            Args:
                shape: An integer which represents the number of neurons in the corresponding LinearLayer.

                n_layer: An integer which represents the index of the layer in the list of all the layers of the neural
                    network.
        """
        super().__init__(shape, n_layer)

    def __repr__(self) -> str:
        return f'Softmax {self.n_layer} ' + self.internal_shape__str__()

    def forward(self, Z: np.ndarray, attack: bool = False) -> Union[None, np.ndarray]:
        """
            This function models the Softmax activation function forward pass.
            In [8] this is the suggested implementation for having a mathematically stable softmax

            Args:
                Z: A ndarray containing the output of the corresponding LinearLayer.

                attack: A boolean which indicates whether the calling function is the gradient descent for training
                    or an attack algorithm. In the latter case, the value of A is returned instead of being set.
        """
        shiftx = Z - np.max(Z, axis=1).reshape(Z.shape[0], 1)
        exps = np.exp(shiftx.astype(float))
        sums = np.sum(exps, axis=1).reshape(exps.shape[0], 1)
        if not attack:
            self.A = np.divide(exps, sums)
        else:
            return np.divide(exps, sums)

    def backward(self, upstream_grad: np.ndarray, attack: bool = False) -> Union[None, np.ndarray]:
        """
            This function performs the back propagation step of the Softmax activation function [10].

            Args:
                upstream_grad: A ndarray containing the Jacobian coming from the upper layer. It has to be multiplied
                    for the local gradient in order to obtain the Jacobian of the current layer.

                attack: A boolean which indicates whether the calling function is the gradient descent for training
                    or an attack algorithm. In the latter case, the value of dZ_prev is returned instead of being set.
        """
        # [12] the derivative of the cross entropy loss, and the softmax, can be combined in the elegant form:
        # y_hat - y
        # which is exactly the upstream_grad
        if not attack:
            self.dZ_prev = upstream_grad
        else:
            return upstream_grad
