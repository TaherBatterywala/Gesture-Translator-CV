import numpy as np


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three 3D points p1, p2, p3.
    p2 is the vertex of the angle.
    Returns angle in degrees.
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def extract_features(landmarks):
    """
    Extract geometric features (distances and angles) from 21 MediaPipe landmarks.

    Args:
        landmarks (list or array): List of 21 (x, y, z) tuples representing hand landmarks.

    Returns:
        list: A flat list of engineered numeric features.
    """
    if len(landmarks) != 21:
        raise ValueError("Expected exactly 21 landmarks.")

    features = []

    # Reference distance (Wrist to Middle Finger MCP) for scale normalization
    ref_dist = calculate_distance(landmarks[0], landmarks[9])
    if ref_dist == 0:
        ref_dist = 1e-6

    # 1. Distances from Thumb Tip (4) to other finger tips
    features.append(
        calculate_distance(landmarks[4], landmarks[8]) / ref_dist
    )  # Thumb-Index
    features.append(
        calculate_distance(landmarks[4], landmarks[12]) / ref_dist
    )  # Thumb-Middle
    features.append(
        calculate_distance(landmarks[4], landmarks[16]) / ref_dist
    )  # Thumb-Ring
    features.append(
        calculate_distance(landmarks[4], landmarks[20]) / ref_dist
    )  # Thumb-Pinky

    # 2. Adjacent finger tip distances
    features.append(
        calculate_distance(landmarks[8], landmarks[12]) / ref_dist
    )  # Index-Middle
    features.append(
        calculate_distance(landmarks[12], landmarks[16]) / ref_dist
    )  # Middle-Ring
    features.append(
        calculate_distance(landmarks[16], landmarks[20]) / ref_dist
    )  # Ring-Pinky

    # 3. Distances from Tips to Wrist (0)
    tip_indices = [4, 8, 12, 16, 20]
    for tip in tip_indices:
        features.append(calculate_distance(landmarks[tip], landmarks[0]) / ref_dist)

    # 4. Finger bending angles
    # Thumb: angle between (1-2) and (2-3) -> vertex is 2
    features.append(calculate_angle(landmarks[1], landmarks[2], landmarks[3]))
    # Index: angle at PIP (6) between (5-6) and (6-7)
    features.append(calculate_angle(landmarks[5], landmarks[6], landmarks[7]))
    # Middle: angle at PIP (10) between (9-10) and (10-11)
    features.append(calculate_angle(landmarks[9], landmarks[10], landmarks[11]))
    # Ring: angle at PIP (14) between (13-14) and (14-15)
    features.append(calculate_angle(landmarks[13], landmarks[14], landmarks[15]))
    # Pinky: angle at PIP (18) between (17-18) and (18-19)
    features.append(calculate_angle(landmarks[17], landmarks[18], landmarks[19]))

    return features
