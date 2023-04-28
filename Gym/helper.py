from scipy.spatial.transform import Rotation  
import numpy as np
import collections.abc

def mat2eulerScipy(mat):
        """
        Converts mat angles to euler angles using scipy
        Parameters:
            mat (np.array): mat angles
        Returns:
            euler (np.array): euler angles
        """
        mat = mat.reshape((3,3))
        r =  Rotation.from_matrix(mat)
        euler = r.as_euler("zyx",degrees=True)
        euler = np.array([euler[0], euler[1], euler[2]])
        return euler

def updateDeep(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = updateDeep(d.get(k, {}), v)
        else:
            d[k] = v
    return d