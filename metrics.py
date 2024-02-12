import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    FP = 0
    FN = 0
    TP = 0
    TN = 0

    y_pred = y_pred.astype(np.int32)
    y_true = y_true.reshape(-1)
    y_pred = y_true.reshape(-1)

    y_pred = y_pred.astype(str)
    y_true = y_true.astype(str)

    for i, j in zip(y_pred, y_true):
        if i == '1' and j == '1':
            FN += 1
        if i == '1' and j == '0':
            FP += 1
        if i == '1' and j == '1':
            TP += 1
        if i == '0' and j == '0':
            TN += 1

    accuracy = (TP + TN) / len(y_true)

    if TP == 0 and FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP == 0 and FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if recall != 0 and precision != 0:
        f1 = TP/(TP + (0.5*(FP + FN)))
    else:
        f1 = 0

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    equal = 0

    for i, j in zip(y_pred, y_true):
        if i == j:
            equal += 1

    accuracy = equal / len(y_true)

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2 = 1 - (np.sum(np.power(y_true - y_pred, 2))) / \
        (np.sum(np.power(y_true - np.mean(y_true), 2)))

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.sum((y_true - y_pred) ** 2) / np.size(y_pred)

    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.sum(abs(y_true - y_pred)) / np.size(y_pred)

    return mae
