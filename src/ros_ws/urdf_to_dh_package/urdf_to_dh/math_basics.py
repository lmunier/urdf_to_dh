import numpy as np

PI = np.pi


def rot_to_axis_angle(in_mat, tolerance=1e-8):
    """
    Description:
        function to convert the rotation matrix to angle axis representation
    :param in_mat: The rotation matrix or a homogeneous transformation matrix
    :param tolerance: the tolerance value for the eigen value to be near 1 (to account for numerical inaccuracies)
    :return: the angle and the corresponding axis of rotation
    """

    in_mat = np.array([in_mat[0][:3], in_mat[1][:3], in_mat[2][:3]])
    rot_angle = np.arccos((np.trace(in_mat) - 1) * 0.5)
    vals, vecs = np.linalg.eig(in_mat)
    vals, vecs = np.real(vals), np.real(vecs)

    for jj in range(len(vals)):
        if abs(vals[jj] - 1) < tolerance:
            return np.array(vecs[:, jj]), rot_angle

    raise "Could not find an axis for the given rotation matrix"


def rad2deg(val):
    return val * 180 / PI


def deg2rad(val):
    return val * PI / 180


def z_translation_matrix(val):
    k = np.eye(4, 4)
    k[2, 3] = val
    return k


def x_translation_matrix(val):
    k = np.eye(4, 4)
    k[0, 3] = val
    return k


def z_rotation_matrix(in_theta):
    mat_id = np.eye(4, 4)
    mat_id[0, 0] = np.cos(in_theta)
    mat_id[1, 0] = np.sin(in_theta)
    mat_id[0, 1] = - np.sin(in_theta)
    mat_id[1, 1] = np.cos(in_theta)

    return mat_id


def x_rotation_matrix(in_theta):
    mat_id = np.eye(4, 4)
    mat_id[1, 1] = np.cos(in_theta)
    mat_id[2, 1] = np.sin(in_theta)
    mat_id[1, 2] = - np.sin(in_theta)
    mat_id[2, 2] = np.cos(in_theta)

    return mat_id


def get_transformation(theta_list, d_list, a_list, alpha_list, coord_num=6):
    m = np.identity(4)

    for ii in range(coord_num):
        m = (m @ z_rotation_matrix(theta_list[ii]) @ z_translation_matrix(d_list[ii]) @
             x_rotation_matrix(alpha_list[ii]) @ x_translation_matrix(a_list[ii]))

    return m


def get_single_transformation(theta_val, d_val, a_val, alpha_val):
    return z_rotation_matrix(theta_val) @ z_translation_matrix(d_val) @ x_rotation_matrix(
        alpha_val) @ x_translation_matrix(a_val)


def normalize(axis):
    if abs(np.linalg.norm(axis) - 1) < 1e-5 or np.linalg.norm(axis) == 0:
        return axis
    else:
        return axis / np.linalg.norm(np.array(axis))
