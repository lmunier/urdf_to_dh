#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kinematics_helpers.py

"""A module containing helper functions for kinematics calculations."""

import math
import numpy as np


def x_rotation(theta: float) -> np.ndarray:
    """The 3x3 rotation matrix for a rotation of `theta` radians about the x-axis.

    Args:
        theta: The angle in radians to rotate about the x-axis.

    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    return np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ])


def y_rotation(theta: float) -> np.ndarray:
    """The 3x3 rotation matrix for a rotation of `theta` radians about the y-axis.

    Args:
        theta: The angle in radians to rotate about the y-axis.

    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    return np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])


def z_rotation(theta: float) -> np.ndarray:
    """The 3x3 rotation matrix for a rotation of `theta` radians about the z-axis.

    Args:
        theta: The angle in radians to rotate about the z-axis.

    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    return np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])


def normalize(axis: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Normalize a vector.

    Args:
        axis: A numpy array representing the vector to normalize.

    Returns:
        A numpy array representing the normalized vector.
    """
    if abs(np.linalg.norm(axis) - 1) < epsilon or np.linalg.norm(axis) == 0:
        return axis
    else:
        return axis / np.linalg.norm(np.array(axis))


def get_extrinsic_rotation(rpy: np.ndarray) -> np.ndarray:
    """Gets the extrinsic rotation matrix defined by roll about x, then pitch about y, then yaw
    about z. This is the rotation matrix used in URDF.

    Args:
        rpy: A numpy array containing the roll, pitch, and yaw angles in radians.

    Returns:
        A 3x3 numpy array for the resulting extrinsic rotation.
    """
    x_rot = x_rotation(rpy[0])
    y_rot = y_rotation(rpy[1])
    z_rot = z_rotation(rpy[2])

    return np.matmul(z_rot, np.matmul(y_rot, x_rot))


def inv_tf(tf: np.ndarray) -> np.ndarray:
    """Get the inverse of a homogeneous transform.

    Args:
        tf: A 4x4 numpy array representing the homogeneous transform.

    Returns:
        A 4x4 numpy array representing the inverse of the homogeneous transform.
    """
    inv_tf = np.eye(4)
    inv_tf[0:3, 0:3] = np.transpose(tf[0:3, 0:3])
    inv_tf[0:3, 3] = -1.0 * np.matmul(np.transpose(tf[0:3, 0:3]), tf[0:3, 3])

    return inv_tf


def get_dh_matrix(dh_params: list) -> np.ndarray:
    """Get the tf for the given dh parameters.

    Args:
        dh_params: A list of the 4 denavit hartenberg parameters.

    Returns:
        A 4x4 numpy array representing the dh matrix.
    """
    d = dh_params[0]
    theta = dh_params[1]
    r = dh_params[2]
    alpha = dh_params[3]

    dh_matrix = np.eye(4)

    dh_matrix[0, 0] = math.cos(theta)
    dh_matrix[0, 1] = -math.sin(theta) * math.cos(alpha)
    dh_matrix[0, 2] = math.sin(theta) * math.sin(alpha)
    dh_matrix[0, 3] = r * math.cos(theta)

    dh_matrix[1, 0] = math.sin(theta)
    dh_matrix[1, 1] = math.cos(theta) * math.cos(alpha)
    dh_matrix[1, 2] = -math.cos(theta) * math.sin(alpha)
    dh_matrix[1, 3] = r * math.sin(theta)

    dh_matrix[2, 0] = 0
    dh_matrix[2, 1] = math.sin(alpha)
    dh_matrix[2, 2] = math.cos(alpha)
    dh_matrix[2, 3] = d

    return dh_matrix
