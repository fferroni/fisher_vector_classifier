import tensorflow as tf
import numpy as np
from gmm import get_3d_grid_gmm


class Modified3DFisherVectors(tf.keras.layers.Layer):

    def __init__(self, subdivisions=(5, 5, 5), variance=0.04, flatten=False):
        self.subdivisions = subdivisions
        self.variance = variance
        self.flatten = flatten
        self.gaussian_mixture = get_3d_grid_gmm(subdivisions, variance)

        self.nb_gaussians = self.gaussian_mixture.means_.shape[0]
        self.D = self.gaussian_mixture.means_.shape[1]
        self.w = tf.constant(self.gaussian_mixture.weights_, tf.float32)
        self.mu = tf.constant(self.gaussian_mixture.means_, tf.float32)
        self.sigma = tf.constant(np.sqrt(self.gaussian_mixture.covariances_), tf.float32)
        super().__init__()

    def build(self, input_shape):

        batch_size, nb_points, features = input_shape
        self.nb_points = int(nb_points)
        batch_sig = tf.tile(tf.expand_dims(self.sigma, 0), [nb_points, 1, 1])
        self.batch_sig = tf.tile(tf.expand_dims(batch_sig, 0), [batch_size, 1, 1, 1])
        batch_mu = tf.tile(tf.expand_dims(self.mu, 0), [nb_points, 1, 1])
        self.batch_mu = tf.tile(tf.expand_dims(batch_mu, 0), [batch_size, 1, 1, 1])
        self.batch_w = tf.tile(tf.expand_dims(tf.expand_dims(self.w, 0), 0), [batch_size, nb_points, 1])

        # Compute derivatives
        self.w_per_batch_per_d = tf.tile(tf.expand_dims(tf.expand_dims(self.w, 0), -1), [batch_size, 1, 3 * self.D])

        # Define multivariate normal distributions
        self.mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=self.batch_mu, scale_diag=self.batch_sig)

    def get_config(self):
        return {
            "subdivisions": self.subdivisions,
            "variance": self.variance,
            "flatten": self.flatten
        }

    def call(self, points, mask=None):
        """
        Compute the fisher vectors given the gmm model parameters (w,mu,sigma) and a set of points
        """

        # Expand dimension for batch compatibility
        batch_points = tf.tile(tf.expand_dims(points, -2), [1, 1, self.nb_gaussians, 1])

        # Compute probability per point
        p_per_point = self.mvn.prob(batch_points)

        w_p = tf.multiply(p_per_point, self.batch_w)
        Q = w_p/tf.tile(tf.expand_dims(tf.reduce_sum(w_p, axis=-1), -1), [1, 1, self.nb_gaussians])
        Q_per_d = tf.tile(tf.expand_dims(Q, -1), [1, 1, 1, self.D])

        # Compute derivatives and take max and min
        # Method 2: direct derivative formula (convertible to min-max)
        # s0 = tf.reduce_sum(Q, fv_noise)  # n_batches X n_gaussians
        # d_pi = (s0 - n_points * w_per_batch) / (tf.sqrt(w_per_batch) * n_points)
        d_pi_all = tf.expand_dims((Q - self.batch_w) / (tf.sqrt(self.batch_w) * self.nb_points), -1)
        d_pi = tf.concat(
            [tf.reduce_max(d_pi_all, axis=1), tf.reduce_sum(d_pi_all, axis=1)], axis=2)

        d_mu_all = Q_per_d * (batch_points - self.batch_mu) / self.batch_sig
        d_mu = (1 / (self.nb_points * tf.sqrt(self.w_per_batch_per_d))) * tf.concat(
            [tf.reduce_max(d_mu_all, axis=1), tf.reduce_min(d_mu_all, axis=1), tf.reduce_sum(d_mu_all, axis=1)], axis=2)

        d_sig_all = Q_per_d * (tf.pow((batch_points - self.batch_mu) / self.batch_sig, 2) - 1)
        d_sigma = (1 / (self.nb_points * tf.sqrt(2*self.w_per_batch_per_d))) * tf.concat(
            [tf.reduce_max(d_sig_all, axis=1), tf.reduce_min(d_sig_all, axis=1), tf.reduce_sum(d_sig_all, axis=1)], axis=2)

        # Power normalization
        alpha = 0.5
        d_pi = tf.sign(d_pi) * tf.pow(tf.abs(d_pi), alpha)
        d_mu = tf.sign(d_mu) * tf.pow(tf.abs(d_mu), alpha)
        d_sigma = tf.sign(d_sigma) * tf.pow(tf.abs(d_sigma), alpha)

        # L2 normalization
        d_pi = tf.nn.l2_normalize(d_pi, dim=1)
        d_mu = tf.nn.l2_normalize(d_mu, dim=1)
        d_sigma = tf.nn.l2_normalize(d_sigma, dim=1)

        if self.flatten:
            # flatten d_mu and d_sigma
            d_pi = tf.contrib.layers.flatten(tf.transpose(d_pi, perm=[0, 2, 1]))
            d_mu = tf.contrib.layers.flatten(tf.transpose(d_mu,perm=[0, 2, 1]))
            d_sigma = tf.contrib.layers.flatten(tf.transpose(d_sigma, perm=[0, 2, 1]))
            fv = tf.concat([d_pi, d_mu, d_sigma], axis=1)
        else:
            fv = tf.concat([d_pi, d_mu, d_sigma], axis=2)
            fv = tf.transpose(fv, perm=[0, 2, 1])

        return fv
