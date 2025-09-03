"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt


from inflammation.models import daily_mean, daily_max, daily_min

@pytest.mark.parametrize(
        "argname1, argname2",
        [
            ([[0, 0], [0, 0], [0, 0]], [0, 0]),
            ([[1, 2], [3, 4], [5, 6]], [3, 4]),
        ])
def test_daily_mean(argname1, argname2):
    """Test that mean function works for an array of zeros."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(np.array(argname1)), np.array(argname2))

@pytest.mark.parametrize(
        "input, expected",
        [
            ([[1, 2, 3], [2, 3, 4], [4, 5, 6]], [4, 5, 6]),
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
            ([[-5, 2, 8], [4, 3, -1], [21, 3, 0]], [21, 3, 8])
        ]
)
def test_daily_max(input, expected):
    """Test that the max function works for an array of zeroes, positive integers and negative integers"""

    npt.assert_array_equal(np.array(expected), daily_max(np.array(input)))

@pytest.mark.parametrize(
        "input, expected",
        [
            ([[1, 2, 3], [2, 3, 4], [4, 5, 6]], [1, 2, 3]),
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
            ([[-5, 2, 8], [4, 3, -1], [21, 3, 0]], [-5, 2, -1])
        ]
)
def test_daily_min(input, expected):
    """Test that the min function works for an array of zeroes, positive integers and negative integers"""

    npt.assert_array_equal(np.array(expected), daily_min(np.array(input)))

def test_daily_min_string():
    """Tests for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([["hallo", "wereld"], ["how", "Istiee?"]])

from inflammation.models import patient_normalise

@pytest.mark.parametrize(
    "test, expected, expected_raises",
    [
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None,
        ),
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]], 
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]], 
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
    ])
def test_patient_normalise(test, expected, expected_raises):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""
    
    # Check whether patient_normalise raises the correct error 
    if expected_raises is not None:
        with pytest.raises(expected_raises):
            patient_normalise(np.array(test))

    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)