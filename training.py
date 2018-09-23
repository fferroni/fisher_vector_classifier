from functools import partial
from glob import glob
import os
import getpass
import h5py
import time
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from tensorflow.python.keras.utils import Sequence
from augmentations import rotate_point_cloud, \
    translate_point_cloud, insert_outliers_to_point_cloud, jitter_point_cloud, scale_point_cloud
from network import build_classification_network


def get_data(files: List[str]):

    X = []
    y = []

    for file in files:

        f = h5py.File(file)

        data, label = f["data"][:], f["label"][:]

        X.append(data)
        y.append(label)

    X = np.concatenate(X, axis=0)
    y_ = np.concatenate(y, axis=0)[:, 0]

    y = np.zeros((len(y_), 40))
    y[np.arange(len(y)), y_] = 1

    return X, y


def augment(X: np.ndarray,
            rotate: bool=False,
            translate: bool=False,
            insert_outliers: bool=False,
            jitter: bool=False,
            scale: bool=False):

    if rotate:
        X = rotate_point_cloud(X)

    if translate:
        X = translate_point_cloud(X)

    if insert_outliers:
        X = insert_outliers_to_point_cloud(X)

    if jitter:
        X = jitter_point_cloud(X)

    if scale:
        X = scale_point_cloud(X)

    return X


class DataGenerator(Sequence):

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int=32, nb_points: int=1024, fn_augment=None):
        super(DataGenerator, self).__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.nb_points = nb_points
        self.nb_batches = X.shape[0] // batch_size
        self.max_points = X.shape[1]
        assert self.max_points >= nb_points
        self.augment = fn_augment

    def __len__(self):
        return self.nb_batches

    def __getitem__(self, item):
        start = item * self.batch_size
        end = (item + 1) * self.batch_size
        return self.augment(self.X[start:end, sorted(np.random.choice(self.max_points, size=self.nb_points,
                                                                      replace=False))]), self.y[start:end]

    def on_epoch_end(self):
        shuffled_ids = np.arange(0, self.X.shape[0])
        np.random.shuffle(shuffled_ids)

        self.X = self.X[shuffled_ids]
        self.y = self.y[shuffled_ids]


def create_log_folder(log_root):
    if not os.path.exists(log_root):
        os.mkdir(log_root)
    run_folder = os.path.join(log_root, "%s_%s" % (getpass.getuser(), time.strftime("%Y-%m-%d-%H:%M:%S")))
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
    return run_folder


def train_3d_fisher_vector_classifier(training_data: List[str],
                                      testing_data: List[str],
                                      nb_points: int=1024,
                                      batch_size: int=32,
                                      subdivisions: Tuple[int, int, int]=(8, 8, 8),
                                      variance: float=0.0156,
                                      GPU: str="0"):

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

    model = build_classification_network(batch_size, nb_points, subdivisions, variance)

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])

    print(model.summary())

    X_train, y_train = get_data(training_data)
    X_val, y_val = get_data(testing_data)

    augmentation = partial(augment, translate=True, insert_outliers=True, jitter=True, rotate=False, scale=True)
    no_augmentation = partial(augment, translate=False, insert_outliers=False, jitter=False, rotate=False, scale=False)

    train_gen = DataGenerator(X_train, y_train, batch_size=batch_size, nb_points=nb_points, fn_augment=augmentation)
    valid_gen = DataGenerator(X_val, y_val, batch_size=batch_size, nb_points=nb_points, fn_augment=no_augmentation)

    log_dir = create_log_folder("logs/")
    model.fit_generator(train_gen,
                        callbacks=[
                              tf.keras.callbacks.TensorBoard(log_dir=log_dir),
                              tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, 'weights.h5'),
                                                                 save_best_only=True),
                              tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
                        ],
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=valid_gen,
                        validation_steps=X_val.shape[0] // batch_size,
                        use_multiprocessing=True,
                        workers=8,
                        epochs=200)


def get_config(root: str="/home/francesco/data/modelnet"):

    return {
        "training_data": glob(os.path.join(root, "*train*.h5")),
        "testing_data": glob(os.path.join(root, "*test*.h5")),
        "GPU": "0"
    }


if __name__ = "__main__":
    train_3d_fisher_vector_classifier(**get_config())

