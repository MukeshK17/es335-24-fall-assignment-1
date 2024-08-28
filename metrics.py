from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    no_of_right_predc = (y_hat == y).sum()
    accuracy = no_of_right_predc/y.size

    return accuracy

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size

    true_positives = ((y == cls) & (y_hat == cls)).sum()
    false_positives = ((y != cls) & (y_hat == cls)).sum()

    try:
        precision = true_positives/(true_positives + false_positives)
    except:
        return 0
    return precision


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size

    true_positives = ((y == cls) & (y_hat == cls)).sum()
    false_negatives = ((y == cls) & (y_hat != cls)).sum()

    try:
        recall = true_positives/(true_positives + false_negatives)
    except:
        return 0
    return recall


def RMSE(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    sqrd_err = (y - y_hat)**2
    sum_sqrd_err = sqrd_err.sum()
    mean_sqrd_err = sum_sqrd_err/len(y)
    rmse = mean_sqrd_err**0.5
    return rmse


def MAE(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """

    abs_err = abs(y - y_hat)
    sum_abs_err = abs_err.sum()
    mae = sum_abs_err/len(y)
    return mae
