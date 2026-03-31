import pytest
import numpy as np
import sys
import os

# Add src to path for absolute imports in tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from feature_extraction import calculate_distance, calculate_angle, extract_features


def test_calculate_distance():
    p1 = (0, 0, 0)
    p2 = (3, 4, 0)
    assert np.isclose(calculate_distance(p1, p2), 5.0)


def test_calculate_angle():
    # 90 degree angle at origin
    p1 = (1, 0, 0)
    p2 = (0, 0, 0)
    p3 = (0, 1, 0)
    assert np.isclose(calculate_angle(p1, p2, p3), 90.0)

    # 180 degree angle
    p1 = (-1, 0, 0)
    p2 = (0, 0, 0)
    p3 = (1, 0, 0)
    assert np.isclose(calculate_angle(p1, p2, p3), 180.0)


def test_extract_features_length():
    # Dummy landmarks: 21 points
    landmarks = [(float(i), float(i), float(i)) for i in range(21)]
    # To prevent ref_dist from being 0, adjust wrist and MCP
    landmarks[0] = (0.0, 0.0, 0.0)
    landmarks[9] = (1.0, 1.0, 1.0)

    features = extract_features(landmarks)
    # 4 thumb-to tips + 3 adjacent tips + 5 tips_to_wrist + 5 angles = 17 features
    assert len(features) == 17


def test_extract_features_invalid_input():
    # Too few landmarks
    with pytest.raises(ValueError):
        extract_features([(0, 0, 0)] * 20)
