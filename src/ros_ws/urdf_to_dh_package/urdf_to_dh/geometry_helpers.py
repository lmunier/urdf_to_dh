#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# geometry_helpers.py

"""A module containing helper functions for geometry calculations."""

import numpy as np


def are_parallel(vec1: np.ndarray, vec2: np.ndarray, epsilon: float = 1e-6) -> bool:
    """Determine if two vectors are parallel."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        raise ValueError("One or both of the vectors are zero vectors.")

    vec1_unit = vec1 / np.linalg.norm(vec1)
    vec2_unit = vec2 / np.linalg.norm(vec2)

    return np.all(abs(np.cross(vec1_unit, vec2_unit)) < epsilon)


def are_collinear(point1: np.ndarray, vec1: np.ndarray, point2: np.ndarray, vec2: np.ndarray) -> bool:
    """Determine if vectors are collinear."""

    # To be collinear, vectors must be parallel
    if not are_parallel(vec1, vec2):
        return False

    # If parallel and point1 is coincident with point2, vectors are collinear
    if all(np.isclose(point1, point2)):
        return True

    # If vectors are parallel, point2 can be defined as p2 = p1 + t * v1
    t = np.zeros(3)
    for idx in range(0, 3):
        if vec1[idx] != 0:
            t[idx] = (point2[idx] - point1[idx]) / vec1[idx]
    p2 = point1 + t * vec1

    return np.allclose(p2, point2)


def are_intersecting(point1: np.ndarray, vec1: np.ndarray, point2: np.ndarray, vec2: np.ndarray, epsilon: float = 1e-6) -> bool:
    """Determine if two lines intersect."""
    if are_collinear(point1, vec1, point2, vec2):
        return False

    return np.abs(np.cross(vec1, vec2).dot(point1 - point2)) < epsilon


def lines_intersect(point1: np.ndarray, vec1: np.ndarray, point2: np.ndarray, vec2: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Determine if two lines intersect."""
    x = np.zeros(2)

    # Test if lines intersect. Need to find non-singular pair to solve for coefficients
    for idx in range(0, 3):
        i = idx
        j = (idx + 1) % 3
        mat_a = np.array([[vec1[i], -vec2[i]], [vec1[j], -vec2[j]]])
        vec_b = np.array([[point2[i] - point1[i]], [point2[j] - point1[j]]])

        # If not singular matrix, solve for coefficients
        if not np.isclose(np.linalg.det(mat_a), 0):
            x = np.linalg.solve(mat_a, vec_b)

            # Test if solution generates a point of intersection
            p1 = point1 + x[0] * vec1
            p2 = point2 + x[1] * vec2

            if all(np.less(np.abs(p1 - p2), epsilon * np.ones(3))):
                return x

    return x
