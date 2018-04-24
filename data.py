# -*- coding: utf-8 -*-
# @Author: alexis
# @Date:   2018-04-24 08:12:52
# @Last Modified by:   Alexis Tremblay
# @Last Modified time: 2018-04-24 08:42:05


import random
import numpy as np
from itertools import combinations


def warp(sequence, max_repeat=4, uniform_warp=False, pad=0):
    """
    Warp a sequence of elements either of random warp or a uniform one. It also
    creates a warped output based on the sequence but shifted by one place.

    e.g.

    $ warp([1,2,3], max_repeat=2, uniform_warp=True, pad=0)
    > ([1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2])

    $ warp([1,2,3], max_repeat=2, uniform_warp=False, pad=0)
    > ([1, 2, 2, 3], [0, 1, 1, 2])

    Most likely use case is to provide a list of indexes to be used by an
    Embedding layer (e.g. torch.nn.Embedding or keras.layers.Embedding)

    Params
    ------
    sequence: list of int
        List of elements

    max_repeat: int (default 4)
        Every elements of the sequence to be warped will be done a max_repeat
        times.

    uniform_warp: boolean (default True)
        If this is set to True then all element of the sequence will be repeated
        the same number of times.

    pad: int (default 0)
        Padding symbol. Typically zero.

    Returns
    -------
    warped_input: list of int
        The original sequence warped according to the parameters provided

    target: list of int

    """
    warped_input = []
    target = []
    prev_c = []

    for i, c in enumerate(sequence):
        if uniform_warp:
            repeat = max_repeat
        else:
            repeat = random.randint(1, max_repeat)
        warped_input.extend([c] * repeat)
        if i == 0:
            # insert spaces if this is the first caracter of the sequence
            target.extend([pad] * repeat)
        else:
            target.extend([prev_c] * repeat)
        prev_c = c

    return warped_input, target


def warp_data(
    T,
    alphabet=range(1, 11),
    max_repeat=4,
    uniform_warp=False,
    pad=0,
    batch_size=32
):
    """
    This will return two matrix of indexes to use as inputs and targets.

    Most likely use case is to provide the inputs to an
    Embedding layer (e.g. torch.nn.Embedding or keras.layers.Embedding) and
    convert the outputs to one-hot vectors to use as softmax targets

    Params
    ------

    T: int
        Sequences length

    alphabet: list of elements (default list from 1 to 10)
        These elements will be randomly picked to form the sequences

    max_repeat: int (default 4)
        Every elements of the sequence to be warped will be done a max_repeat
        times.

    uniform_warp: boolean (default True)
        If this is set to True then all element of the sequence will be repeated
        the same number of times.

    pad: int (default 0)
        Padding symbol. Typically zero.

    Returns
    -------
    warped_inputs: numpy array of int of size (batch_size x T)
        The original sequence warped according to the parameters provided

    targets: numpy array of int of size (batch_size x T)

    """

    # This gives n-1 list of elements with the i-th element removed
    # This is to avoid consecutive repeating elements
    indexes = range(len(alphabet))

    idx_to_choose = list(
        reversed(list(combinations(indexes, len(indexes) - 1)))
    )

    sequences = []
    inputs, targets = [], []
    for line in range(batch_size):
        idx = random.choice(indexes)
        sequence = [alphabet[idx]]
        for _ in range(T - 1):
            idx = random.choice(idx_to_choose[idx])
            sequence.append(alphabet[idx])

        warped_input, target = warp(sequence, max_repeat, uniform_warp, pad)
        inputs.append(warped_input[:T])
        targets.append(target[:T])
        sequences.append(sequence)
    return np.array(inputs), np.array(targets)


def copy_data(
    T,
    alphabet=range(1, 9),
    dummy=9,
    eos=10,
    batch_size=32,
    variable=False
):
    """
    For a given 𝑇 , input sequences consist of 𝑇 + 20 characters. The first 10
    characters are drawn uniformly randomly from the first 8 letters of the
    alphabet. These first characters are followed by 𝑇 − 1 dummy characters,
    a signal character, whose aim is to signal the network that it has to
    provide its outputs, and the last 10 characters are dummy characters.

    The target sequence consists of 𝑇 + 10 dummy characters, followed by the
    first 10 characters of the input. This dataset is thus about remembering an
    input sequence for exactly 𝑇 timesteps.

    We also provide results for the variable copy task setup presented in
    (Henaff et al., 2016), where the number of characters between the end of
    the sequence to copy and the signal character is drawn at random between
    1 and 𝑇 .

    Params
    ------

    T: int
        Length

    alphabet: list of elements (default list from 1 to 8)
        These elements will be randomly picked from to start the sequence

    dummy: any (default 9)
        Dummy element to use

    eos: any (default 10)
        End of sequence to use as a signal for the network that it has to
        provide its outputs

    batch_size: int (default 32)
        Names says it all

    variable: boolean (default False)
        The number of characters between the end of the sequence to copy and
        the signal character is drawn at random between 1 and 𝑇.

    Returns
    -------

    input: numpy array of size (batch_size x R)
        If 'variable' is set to False, this will be a matrix of size
        batch_size X (T + 20)

        If 'variable' is set to True, this will be a matrix of size
        batch_size X (T + 10 + 𝒰([1, T]) )

    output: numpy
        𝑇 + 10 dummy characters, followed by the first 10 characters of the
        input
    """
    sequences = np.random.choice(alphabet, size=(batch_size, 10))
    dummies = np.ones(shape=(batch_size, T - 1)) * dummy
    signal = np.ones(shape=(batch_size, 1)) * eos
    values = np.concatenate([sequences, dummies, signal], axis=1)

    output_values = np.concatenate(
        [
            np.ones(shape=(batch_size, T + 10)) * dummy,
            sequences
        ],
        axis=1
    )

    if variable:
        filling = np.ones(shape=(batch_size, random.randint(1, T))) * dummy
    else:
        filling = np.ones(shape=(batch_size, 10)) * dummy

    return np.concatenate([values, filling], axis=1), output_values


def add_data(T, batch_size=32):
    """
    Each training example consists of two input sequences of length 𝑇.
    The first one is a sequence of numbers drawn from 𝒰 ([0, 1]), the second
    is a sequence containing zeros everywhere, except for two locations, one in
    the first half and another in the second half of the sequence.

    The target is a single number, which is the sum of the numbers contained in
    the first sequence at the positions marked in the second sequence.

    Params
    ------

    T: int

    batch_size: int (default 32)

    Returns
    -------

    inputs: numpy array of float32 of size (batch_size x 2T)

    outputs: float
        The sum of the first half where the mask is on in the second half
    """
    first_half = np.random.rand(batch_size, T)
    second_half = np.zeros((batch_size, T), dtype='int32')
    idx = np.array(
        [
            np.random.choice(
                range(T),
                size=2,
                replace=False
            ) for _ in range(batch_size)
        ]
    )
    second_half[range(batch_size), idx[:, 0]] = 1
    second_half[range(batch_size), idx[:, 1]] = 1

    output = (first_half * second_half).sum(axis=1)
    return np.concatenate([first_half, second_half], axis=1), output


def to_categorical(y, num_classes=None):
    """
    *** Copied from keras.utils.to_categorical ***

    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Params
    ------
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    Returns
    -------
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


