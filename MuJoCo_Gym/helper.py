import numpy as np
import collections.abc
from scipy.spatial.transform import Rotation


def mat2euler_scipy(mat: np.array) -> np.array:
    """Converts mat angles to euler angles using scipy

    Parameters:
        mat (np.array): mat angles
    Returns:
        euler (np.array): euler angles
    """
    mat = mat.reshape((3, 3))
    r = Rotation.from_matrix(mat)
    euler = r.as_euler("zyx", degrees=True)
    euler = np.array([euler[0], euler[1], euler[2]])
    return euler


def update_deep(old_dict, new_dict):
    """
    Recursively updates old_dict with values from new_dict. 
    If a key in new_dict corresponds to a dictionary, it updates the corresponding dictionary in old_dict recursively.
    """
    for key, value in new_dict.items():
        if key in old_dict and isinstance(old_dict[key], dict) and isinstance(value, dict):
            update_deep(old_dict[key], value)
        else:
            old_dict[key] = value
    return old_dict
