#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# urdf_helpers.py

"""A module containing helper functions URDF parsing."""

import xml.etree.ElementTree as ET
import numpy as np

from anytree import AnyNode


def get_urdf_root(urdf_file: str) -> ET.Element:
    """Parse a URDF for joints.

    Args:
        urdf_path: The absolute path to the URDF to be analyzed.

    Returns:
        root: root node of the URDF.
    """
    try:
        tree = ET.parse(urdf_file)
    except ET.ParseError:
        print('ERROR: Could not parse urdf file.')

    return tree.getroot()


def convert_vec_to_skew(vec):
    """
    Description:
        function to get the skew symmetric form of the vector
    :param vec: 3 x 1 vector
    :return: 3 x 3 skew symmetric matrix
    """
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def convert_ax_ang_to_rot(ax, ang):
    """
    Description:
        function to convert the axis-angle representation to a 3 x 3 rotation matrix
        The function uses the Rodrigues formula
    :param ax: axis of the rotation
    :param ang: angle of rotation in radians
    :return: 3 x 3 rotation matrix
    """
    if np.linalg.norm(ax) > 1e-6:
        ax = ax / np.linalg.norm(ax)
    else:
        return np.eye(3)

    ax_so3 = convert_vec_to_skew(vec=ax)
    return np.identity(3) + np.sin(ang) * ax_so3 + (1 - np.cos(ang)) * ax_so3 @ ax_so3

def get_reference_axis(joint: dict, epsilon: float = 1e-10) -> np.ndarray:
    """Extracts the reference axis from the joint.

    Args:
        joint: The joint element to extract the reference axis from.
        epsilon: The tolerance for floating point comparisons.

    Returns:
        reference_axis: The reference axis of the URDF.
    """
    x_axis = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])

    if np.array_equal(joint['axis'], z_axis):
        return x_axis
    else:
        # Check if the input vector is close to zero to avoid division by zero
        if np.linalg.norm(joint) < epsilon:
            raise ValueError("Input vector is too close to zero.")

        rot_vec = np.cross(joint['axis'], z_axis)
        rot_angle = np.arccos(np.dot(joint['axis'], z_axis))

        rot_mat = convert_ax_ang_to_rot(rot_vec, rot_angle)

        return rot_mat @ x_axis

def get_axis(node: AnyNode, joints: dict, direction: str = 'child') -> np.ndarray:
    """Extracts the axis of rotation from next or previous joint element.

    Args:
        node: The joint node to set axis.
        joints: The dictionary containing the joint info.
        direction: The direction to extract the axis from.

    Returns:
        axis: The axis of rotation of the joint.
    """
    next_node = None
    next_direction = direction

    # Return the axis if it is already set
    if joints[node.id]['axis'] is not None:
        return joints[node.id]['axis']

    # Check if it is the start or the end of the tree
    if node.children[0].is_leaf:
        next_direction = 'parent'
    elif node.parent.is_root:
        next_direction = 'child'

    # Store the next node to extract the axis from
    if next_direction == 'child':
        next_node = node.children[0].children[0]
    elif next_direction == 'parent':
        next_node = node.parent.parent
    else:
        print(
            f"ERROR: {next} is not a known next joint direction. Should be ['child' or 'parent']."
        )

    # Recursive call
    return get_axis(next_node, joints, direction=next_direction)


def process_joint(joint: ET.Element) -> tuple:
    """Extracts the relevant joint info into a dictionary.

    Args:
        joint: The joint element to be processed.

    Returns:
        joint_name: The name of the joint.
        joint_info: A dictionary containing the joint info.
    """
    axis = None
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    parent_link = ''
    child_link = ''

    joint_name = joint.get('name')
    joint_type = joint.get('type')

    for child in joint:
        if child.tag == 'axis':
            axis = np.array(child.get('xyz').split(), dtype=float)
        elif child.tag == 'origin':
            xyz = np.array(child.get('xyz').split(), dtype=float)
            rpy = np.array(child.get('rpy').split(), dtype=float)
        elif child.tag == 'parent':
            parent_link = child.get('link')
        elif child.tag == 'child':
            child_link = child.get('link')

    return joint_name, {
        'axis': axis,
        'xyz': xyz,
        'rpy': rpy,
        'parent': parent_link,
        'child': child_link,
        'dh': np.zeros(4),
        'type': joint_type
    }
