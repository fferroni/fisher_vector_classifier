from typing import Tuple
import numpy as np
from sklearn.mixture.gaussian_mixture import GaussianMixture, _compute_precision_cholesky


def get_3d_grid_gmm(subdivisions: Tuple[int, int, int]=(5, 5, 5), variance: float=0.04) -> GaussianMixture:
    """
    Compute the weight, mean and covariance of a gmm placed on a 3D grid
    :param subdivisions: 2 element list of number of subdivisions of the 3D space in each axes to form the grid
    :param variance: scalar for spherical gmm.p
    :return gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    n_gaussians = np.prod(np.array(subdivisions))
    step = [1.0/(subdivisions[0]),  1.0/(subdivisions[1]),  1.0/(subdivisions[2])]

    means = np.mgrid[step[0]-1: 1.0-step[0]: complex(0, subdivisions[0]),
                     step[1]-1: 1.0-step[1]: complex(0, subdivisions[1]),
                     step[2]-1: 1.0-step[2]: complex(0, subdivisions[2])]

    means = np.reshape(means, [3, -1]).T
    covariances = variance*np.ones_like(means)
    weights = (1.0/n_gaussians)*np.ones(n_gaussians)
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
    gmm.weights_ = weights
    gmm.covariances_ = covariances
    gmm.means_ = means
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'diag')
    return gmm
