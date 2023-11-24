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


def update_deep(d, u):
    """ ToDo: no clue what this is for """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_deep(d.get(k, {}), v)
        else:
            d[k] = v
    return d
