# -*- coding: utf-8 -*-
"""
    This module contains utilities functions useful for the classification task, e.g.,
    for computing the one-hot encoding of class labels, for splitting the train dataset in batches, etc.

    References:
        [1] HOWARD, Jeremy; GUGGER, Sylvain. Deep Learning for Coders with fastai and PyTorch. O'Reilly Media, 2020.
        [2] https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
        [3] https://stackoverflow.com/questions/58676588/how-do-i-one-hot-encode-an-array-of-strings-with-numpy
        [4] https://stackoverflow.com/questions/42497340/how-to-convert-one-hot-encodings-into-integers
        [5] https://stackoverflow.com/questions/39622639/
        how-to-break-numpy-array-into-smaller-chunks-batches-then-iterate-through-them
"""
from typing import Tuple

import numpy as np


def from_probabilities_to_labels(predictions: np.ndarray, one_hot_encoded: bool = True) -> np.ndarray:
    """
        This function transforms a vector of predicted probabilities (e.g., coming from a Softmax layer) into
        the class with maximum likelihood probability.

        Args:
            predictions: A ndarray of shape (n_predicted_samples, n_classes) containing the probabilities predicted
                by the model for each single class.

            one_hot_encoded: A boolean which indicates whether to transform the class label in the one-hot encoded
                version.

        Returns:
            predicted_classes: A ndarray which can be either of shape (n_predicted_samples, 1) if one_hot_encoded is
                False, meaning that the classes are encoded for example with an integer encoding; otherwise, it is a ndarray
                of shape (n_predicted_samples, n_classes) with the one-hot encoded version of the predicted class.
    """
    predicted_classes = np.zeros_like(predictions)
    predicted_classes[np.arange(len(predictions)), predictions.argmax(1)] = 1
    if not one_hot_encoded:
        predicted_classes = one_hot(predicted_classes, encode_decode=1)
    return predicted_classes


def one_hot(arr: np.ndarray, encode_decode: int = 0) -> np.ndarray:
    """
        This function performs the one-hot encoding/decoding of an input vector.

        Args:
            arr: A ndarray to encode/decode.

            encode_decode: An integer which may assume values in {0, 1} and represents whether to perform the encoding
                or the decoding.

        Returns:
            res: A ndarray containing the encoded/decoded version of the input vector.
    """
    try:
        assert encode_decode in range(2)
    except AssertionError:
        exit('Please specify if you want to encode (type=0) or decode (type=1)')
    res = None
    if encode_decode == 0:  # encoding
        # [3]
        unique, inverse = np.unique(arr, return_inverse=True)
        one_hot_encode = np.eye(unique.shape[0])[inverse]
        res = one_hot_encode
    elif encode_decode == 1:  # decoding
        # [4]
        one_hot_decode = np.array([np.where(r == 1)[0][0] for r in arr])
        res = one_hot_decode
    return res


def batches_generator(data: np.ndarray, batch_size: int) -> np.ndarray:
    """
        This function splits the data into multiple batches [5].
        The policy implemented is the "rollout", which is the policy for the missing samples in order to fill the
        last batch of data. For example, the rollout policy fills the missing samples by starting from the beginning
        of the input data.

        Args:
            data: A ndarray containing the data to split. It is a ndarray (NumPy array) of shape (N, K),
                  where N is the number of samples in the dataset, and k is the number of features in the dataset.

            batch_size: An integer which represents the number of samples that we want in each batch of data.

        Returns:
            batches: A ndarray containing the input data split in batches.
    """
    rem = len(data) % batch_size
    if rem != 0:
        # I have to augment the dataset, in order to have the last batch of the same size of the others
        # in particular, the dataset is augmented with the roll-out policy, repeating the first elements
        data = np.vstack([data, data[:(batch_size - rem)]])
    n_batches = data.shape[0] // batch_size
    batches = np.split(data, n_batches)
    return batches


def shuffle_batches(batches_x: np.ndarray, batches_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
        This function performs an equal shuffling to both x and y of the batches.
        Citing from [1]:
            "One simple and effective thing we can vary is what data items we put in each mini-batch.
             Rather than simply enumerating our dataset in order for every epoch, instead what we normally do is
             randomly shuffle it in every epoch[...]"

        Args:
            batches_x: A ndarray containing the input features split in batches. It has shape
                (n_batches, n_samples_in_batch, n_features).

            batches_y: A ndarray containing the labels associated with the input samples in the batches. It has shape
                (n_batches, n_samples_in_batch, n_classes).

        Returns:
            shuffled_x, shuffled_y: The batches shuffled in a new random order.

    '''
    n = len(batches_x)
    p = np.random.permutation(n)
    shuffled_x, shuffled_y = [], []
    for i in p:
        shuffled_x.append(batches_x[i])
        shuffled_y.append(batches_y[i])
    return np.array(shuffled_x), np.array(shuffled_y)
