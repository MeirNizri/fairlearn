# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import FairRobustSelection
from fairlearn.reductions import DemographicParity, ErrorRateParity,\
    TruePositiveRateParity, FalsePositiveRateParity

from fairlearn.datasets import fetch_adult, fetch_bank_marketing

# fetch stored datasets
adult = fetch_adult(cache=True, as_frame=True)
x_adult = adult.data
y_adult = (adult.target == '>50K') * 1

bank_marketing = fetch_bank_marketing(cache=True, as_frame=True)
x_marketing = bank_marketing.data
y_marketing = (bank_marketing.target == 'yes') * 1

# combine the datasets to list
datasets = [(x_adult, y_adult), (x_marketing, y_marketing)]

def run_comparisons(moment, metric_fn):
    # check if metric_fn constraints is satisfied on two different datasets
    for x, y in datasets:
        X_dummy = pd.get_dummies(x)
        # sensitive feature
        sex = x['sex']

        unmitigated = LogisticRegression()
        unmitigated.fit(X_dummy, y)
        y_pred = unmitigated.predict(X_dummy)
        mf_unmitigated = MetricFrame(metrics=metric_fn, y_true=y,
                                     y_pred=y_pred, sensitive_features=sex)

        frs_model = FairRobustSelection(
            LogisticRegression(),
            constraints=moment(),
            tau=1)
        frs_model.fit(X_dummy, y, sensitive_features=sex)
        y_pred = frs_model.predict(X_dummy)
        mf_mitigated = MetricFrame(metrics=metric_fn, y_true=y,
                                   y_pred=y_pred, sensitive_features=sex)

        assert (mf_mitigated.difference(method='to_overall') <=
                mf_unmitigated.difference(method='to_overall')).all()

# run four different fairness constraints on the datasets
def test_demographic_parity():
    run_comparisons(DemographicParity, selection_rate)

def test_error_rate_parity():
    run_comparisons(ErrorRateParity, accuracy_score)

def test_true_positive_rate_parity():
    run_comparisons(TruePositiveRateParity, true_positive_rate)

def test_false_positive_rate_parity():
    run_comparisons(FalsePositiveRateParity, false_positive_rate)

def test_arguments_validation():
    for wrong_values in [0, -2, 1.5]:
        with pytest.raises(ValueError):
            FairRobustSelection(
                LogisticRegression(),
                DemographicParity(),
                tau=wrong_values)

