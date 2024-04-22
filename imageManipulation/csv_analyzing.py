import csv
import numpy as np
import math


def normalize_rgb(x):
    return int(x)/255


def find_occurences(X):
    # setting all occurences to 0 initially
    Y = [0]*(26**3)
    for r, g, b in X:
        r_index = math.ceil(r//10)
        g_index = math.ceil(g//10)
        b_index = math.ceil(b//10)
        Y[r_index * (26**2) + g_index * 26 + b_index] += 1
    return Y


def calculate_features(A):
    """

    :param A: Array containing information
    :return: arary of features that are of importance to train the model

    """
    r_ = [i[0] for i in A]
    g_ = [i[1] for i in A if len(i) > 1]
    b_ = [i[2] for i in A if len(i) > 2]

    r_mean = np.mean(r_, axis=0)
    g_mean = np.mean(g_, axis=0)
    b_mean = np.mean(b_, axis=0)
    r_std = np.std(r_, axis=0)
    g_std = np.std(g_, axis=0)
    b_std = np.std(b_, axis=0)
    percentiles = np.percentile(A, [25, 50, 75], axis=0)

    return [r_mean, g_mean, b_mean, r_std, g_std, b_std, percentiles]


def read_rows(filename, rows, X, fun):
    """
    Reads select rows of a given file and places them in the array
    X

    Parameters
    ----------
    filename : str
        The filename
    rows : int[]
        array containing the index of the rows to be read
    X : int[]
        array that contains the values that have been read
    fun :
        function to be applied on the element before being added to the array
    arr :
        boolean that decides if each row is placed as an array or just appended
    """
    with open(filename, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        # skipping header line
        next(csv_reader)
        # appending the elements as floats in the array
        for i, line in enumerate(csv_reader):
            for j in rows:
                if line[j] == "":
                    break
                X.append(fun(float(line[j])))
