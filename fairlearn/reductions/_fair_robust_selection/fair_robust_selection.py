import logging
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin

class FairRobustSelection(BaseEstimator, MetaEstimatorMixin):
    """
    "Sample Selection for Fair and Robust Training", by Roh, Y., Lee, K., Whang, S., & Suh, C. (2021).
    https://proceedings.neurips.cc/paper/2021/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf

    An Estimator to train a model using fair and robust sample selection. In each iteration in which the model learns,
    data is sampled with the aim of optimizing the fairness and robustness of the model.
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
            number in range (0,1] indicating the clean ratio of the data.
        alpha : float
            A positive number for step size that used in the lambda adjustment.
        sample_weight_name : str
            Name of the argument to `estimator.fit()` which supplies the sample weights
            (defaults to `sample_weight`)


        >>> FairRobustSelection(LogisticRegression(), DemographicParity(), tau=0.8) is None
        True

        >>> FairRobustSelection(LogisticRegression(),DemographicParity(), tau=0)
        Traceback (most recent call last):
            ...
        ValueError: tau must be between (0,1]
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

        >>> frs_model.fit(x_adult, y_adult, sensitive_features=adult.data['sex']) is None
        True
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

        >>> frs_model.predict(x_adult).shape[0] is x_adult.shape[0]
        True
        """
        pass

if __name__ == "__main__":
    import doctest
    from sklearn.linear_model import LogisticRegression
    from fairlearn.datasets import fetch_adult
    from fairlearn.reductions._moments import DemographicParity

    adult = fetch_adult(cache=True, as_frame=True)
    x_adult = pd.get_dummies(adult.data)
    y_adult = (adult.target == '>50K') * 1
    frs_model = FairRobustSelection(LogisticRegression(), DemographicParity(), tau=0.8)
    frs_model.fit(x_adult, y_adult, sensitive_features=adult.data['sex'])
    print(doctest.testmod())
