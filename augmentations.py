import numpy as np
import random
from scipy.spatial import KDTree


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def translate_point_cloud(batch_data, tval = 0.2):
    """ Randomly translate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, translated batch of point clouds
    """
    n_batches = batch_data.shape[0]
    n_points = batch_data.shape[1]
    translation = np.random.uniform(-tval, tval, size=[n_batches,3])
    translation = np.tile(np.expand_dims(translation,1),[1,n_points,1])
    batch_data = batch_data + translation
    return batch_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_x_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along x direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def scale_point_cloud(batch_data, smin = 0.66, smax = 1.5):
    """ Randomly scale the point clouds to augument the dataset
        scale is per shape
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, scaled batch of point clouds
    """
    scaled = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        sx = np.random.uniform(smin, smax)
        sy = np.random.uniform(smin, smax)
        sz = np.random.uniform(smin, smax)
        scale_matrix = np.array([[sx, 0, 0],
                                    [0, sy, 0],
                                    [0, 0, sz]])
        shape_pc = batch_data[k, ...]
        scaled[k, ...] = np.dot(shape_pc.reshape((-1, 3)), scale_matrix)
    return scaled


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def insert_outliers_to_point_cloud(batch_data, outlier_ratio=0.05):
    """ inserts log_noise Randomly distributed in the unit sphere
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array,  batch of point clouds with log_noise
    """
    B, N, C = batch_data.shape
    outliers = np.random.uniform(-1, 1, [B, int(np.floor(outlier_ratio * N)), C])
    points_idx = np.random.choice(list(range(0, N)), int(np.ceil(N * (1 - outlier_ratio))))
    outlier_data = np.concatenate([batch_data[:, points_idx, :], outliers], axis=1)
    return outlier_data


def occlude_point_cloud(batch_data, occlusion_ratio):
    """ Randomly k remove points (number of points defined by the ratio.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          Bx(N-k)x3 array, occluded batch of point clouds
    """
    B, N, C = batch_data.shape
    k = int(np.round(N*occlusion_ratio))
    occluded_batch_point_cloud = []
    for i in range(B):
        point_cloud = batch_data[i, :, :]
        kdt = KDTree(point_cloud)
        center_of_occlusion = random.choice(point_cloud)
        _, occluded_points_idx = kdt.query(center_of_occlusion.reshape(1, -1), k=k)
        point_cloud = np.delete(point_cloud, occluded_points_idx, axis=0)
        occluded_batch_point_cloud.append(point_cloud)
    return np.array(occluded_batch_point_cloud)
