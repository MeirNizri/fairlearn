import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
import torch
import torch.nn as nn

from fairlearn.reductions._fair_robust_selection._sample_selection import FairBatchSampler


class FairRobustSelection(BaseEstimator, MetaEstimatorMixin):
    """
    "Sample Selection for Fair and Robust Training", by Roh, Y., Lee, K., Whang, S., & Suh, C. (2021).
    https://proceedings.neurips.cc/paper/2021/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf

    An Estimator to train a model using fair and robust sample selection. In each iteration in which the model learns,
    data is sampled with the aim of optimizing the fairness and robustness of the model.
    """

    def __init__(self, estimator, constraints, *,
                 optimizer=None, loss_func=nn.BCELoss(),
                 num_epoch=200, batch_size=128, warm_start=100,
                 tau=0.9, alpha=0.001, sample_weight_name='sample_weight'):
        """
        Parameters
        ----------
        estimator : torch.nn.module
            An estimator implementing methods :code:`forward(x)`,
            where `x` is the pytorch tensor of features.
            predictions returned by  :code:`forward(x)`` are either 0 or 1.
        constraints : fairlearn.reductions.Moment
            The fairness constraints expressed as a :class:`~Moment`.
        optimizer : torch.optimizer
             torch optimizer that hold the current state and update
             the parameters based on the computed gradients.
        loss_func : torch.nn.module.loss
            torch loss function
        num_epoch : positive integetr
            number of times to train on all the data
        batch_size : positive integetr
            the size of every batch
        tau: float
            number in range (0,1] indicating the clean ratio of the data.
        alpha : float
            A positive number for step size that used in the lambda adjustment.
        sample_weight_name : str
            Name of the argument to `estimator.fit()` which supplies the sample weights
            (defaults to `sample_weight`)


        >>> FairRobustSelection(LogisticRegression(), DemographicParity()) is None
        True

        >>> FairRobustSelection(LogisticRegression(),DemographicParity(), tau=0)
        Traceback (most recent call last):
            ...
        ValueError: tau must be between (0,1]
        """

        self.estimator = estimator
        self.constraints = constraints
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(estimator.parameters())
        self.loss_func = loss_func
        self.num_epoch = num_epoch
        self.warm_start = warm_start
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha
        self.sample_weight_name = sample_weight_name

    def fit(self, x, y, z):
        """
        Return a fair classifier under specified fairness constraints.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Feature data
        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            Label vector
        z : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            sensitive feature data

        >>> frs_model.fit(x_adult, y_adult, sensitive_features=adult.data['sex']) is None
        True
        """
        # turn the data into tensors
        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)
        z_tensor = torch.tensor(z)

        # create fair sampler
        sampler = FairBatchSampler(self.estimator, x_tensor, y_tensor, z_tensor, fairness_constraint=self.constraints,
                                   loss_func=self.loss_func, batch_size=self.batch_size,
                                   warm_start=self.warm_start, tau=self.tau, alpha=self.alpha)

        # train model
        for _ in range(self.num_epoch):
            for batch_train_x, batch_train_y in sampler:
                self.optimizer.zero_grad()
                pred_y = self.estimator.forward(batch_train_x)
                loss = self.loss_func(pred_y, batch_train_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, x):
        """
        Provide predictions for the given input data.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Feature data

        Returns
        -------
        Scalar or vector
            The prediction. If `x` represents the data for a single example
            the result will be a scalar. Otherwise, the result will be a vector

        >>> frs_model.predict(x_adult).shape[0] is x_adult.shape[0]
        True
        """

        return self.estimator(x).numpy()


if __name__ == "__main__":
    # import numpy as np
    # list = np.array([0, 1, 1, 1, 1, 0])
    # dict = {}
    # index = {}
    # values = [0, 1]
    # for value in values:
    #     dict[value] = (list == value)
    #
    # for value in values:
    #     index[value] = (list == value).nonzero()
    # print(index[0])

    import doctest
    from sklearn.linear_model import LogisticRegression
    from fairlearn.datasets import fetch_adult
    from fairlearn.reductions import DemographicParity

    adult = fetch_adult(cache=True, as_frame=True)
    x_adult = pd.get_dummies(adult.data)
    y_adult = (adult.target == '>50K') * 1
    frs_model = FairRobustSelection(LogisticRegression(), DemographicParity(), tau=0.8)
    frs_model.fit(x_adult, y_adult, adult.data['sex'])
    print(doctest.testmod())
