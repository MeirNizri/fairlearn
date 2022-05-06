import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from fairlearn.reductions._moments import ClassificationMoment

logger = logging.getLogger(__name__)


class FairRobustSelection(BaseEstimator, MetaEstimatorMixin):
    """
    "Sample Selection for Fair and Robust Training", by Roh, Y., Lee, K., Whang, S., & Suh, C. (2021).
    https://proceedings.neurips.cc/paper/2021/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf

    An Estimator to train a model using fair and robust sample selection. In each iteration in which the model learns,
    data is sampled with the aim of optimizing the fairness and robustness of the model. This method combines two
    sample selection methods:
    1. Batch selection - In each iteration we adjust the number of samples from each class and a sensitive feature
       improve the fairness. For example, if the model, on objects with a sensitive attribute z is less accurate for
       class y, then in the next iteration we would like to sample more objects from attribute z that belong to
       class y so that the model can be more accurate for this group.
    2. Clean selection - selection of objects for which the loss function is low. This choice helps to achieve
       robustness since usually objects with high loss are considered noisy.
    This method does not require modification on the data and can work in any neural network architecture.
    """

    def __init__(self, estimator, constraints, tau, *,
                 alpha=0.001, sample_weight_name='sample_weight'):
        """
        Parameters
        ----------
        estimator : estimator
            An estimator implementing methods :code:`fit(X, y, sample_weight)` and
            :code:`predict(X)`, where `X` is the matrix of features, `y` is the
            vector of labels (binary classification) or continuous values
            (regression), and `sample_weight` is a vector of weights.
            In binary classification labels `y` and predictions returned by
            :code:`predict(X)` are either 0 or 1.
            In regression values `y` and predictions are continuous.
        constraints : fairlearn.reductions.Moment
            The fairness constraints expressed as a :class:`~Moment`.
        tau: float
            number in range [0,1] indicating the clean ratio of the data.
        alpha : float
            A positive number for step size that used in the lambda adjustment.
        sample_weight_name : str
            Name of the argument to `estimator.fit()` which supplies the sample weights
            (defaults to `sample_weight`)
        """
        pass

    def fit(self, X, y, **kwargs):
        """
        Return a fair classifier under specified fairness constraints.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data
        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            Label vector
        """
        pass

    def predict(self, X, random_state=None):
        """
        Provide predictions for the given input data.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data
        random_state : int or RandomState instance, default=None
            Controls random numbers used for randomized predictions. Pass an
            int for reproducible output across multiple function calls.

        Returns
        -------
        Scalar or vector
            The prediction. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        """
        pass

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())