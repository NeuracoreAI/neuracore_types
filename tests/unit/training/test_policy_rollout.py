"""Unit tests for policy rollout models."""

import pytest
from pydantic import ValidationError

from neuracore_types.training.training import PolicyRolloutSeries


def test_series_accepts_equal_lengths():
    series = PolicyRolloutSeries(predicted=[0.1, 0.2], ground_truth=[0.3, 0.4])
    assert series.predicted == [0.1, 0.2]


def test_series_rejects_unequal_lengths():
    with pytest.raises(ValidationError):
        PolicyRolloutSeries(predicted=[0.1, 0.2], ground_truth=[0.3])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_series_rejects_non_finite_values(value):
    with pytest.raises(ValidationError):
        PolicyRolloutSeries(predicted=[value], ground_truth=[0.0])
    with pytest.raises(ValidationError):
        PolicyRolloutSeries(predicted=[0.0], ground_truth=[value])
