import h5py
import numpy as np
import tensorflow as tf

from augmentations import rotate_point_cloud, \
    translate_point_cloud, insert_outliers_to_point_cloud, jitter_point_cloud
from network import build_classification_network


def get_data(files):

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


def augment(X, rotate=False, translate=False, insert_outliers=False, jitter=False):

    if rotate:
        X = rotate_point_cloud(X)

    if translate:
        X = translate_point_cloud(X)

    if insert_outliers:
        X = insert_outliers_to_point_cloud(X)

    if jitter:
        X = jitter_point_cloud(X)

    return X


def data_generator(X, y, batch_size=32, nb_points=1024, **kwargs):

    nb_batches = X.shape[0] // batch_size
    max_points = X.shape[1]
    assert max_points >= nb_points

    while True:

        np.random.shuffle(X)
        np.random.shuffle(y)

        for i in range(nb_batches):

            start = i * batch_size
            end = (i + 1) * batch_size

            yield augment(X[start:end, np.random.choice(max_points, size=nb_points)], **kwargs), y[start:end]


def r_train():
    training_data = [
        "/home/francesco/data/modelnet40_ply_hdf5_2048/ply_data_train0.h5",
        "/home/francesco/data/modelnet40_ply_hdf5_2048/ply_data_train1.h5",
        "/home/francesco/data/modelnet40_ply_hdf5_2048/ply_data_train2.h5",
        "/home/francesco/data/modelnet40_ply_hdf5_2048/ply_data_train3.h5",
        "/home/francesco/data/modelnet40_ply_hdf5_2048/ply_data_train4.h5",
    ]

    testing_data = [
        "/home/francesco/data/modelnet40_ply_hdf5_2048/ply_data_test0.h5",
        "/home/francesco/data/modelnet40_ply_hdf5_2048/ply_data_test1.h5"
    ]

    NB_POINTS = 2048
    BATCH_SIZE = 16

    model = build_classification_network(BATCH_SIZE, NB_POINTS, (8, 8, 8), 0.0156)

    optimizer = tf.keras.optimizers.Adam(lr=0.001, clipvalue=1.0)

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    print(model.summary())

    X_train, y_train = get_data(training_data)
    X_val, y_val = get_data(testing_data)

    model.fit_generator(data_generator(X_train, y_train, batch_size=BATCH_SIZE, nb_points=NB_POINTS),
                        callbacks=[
                              tf.keras.callbacks.TensorBoard(log_dir='/home/francesco/test'),
                              tf.keras.callbacks.ModelCheckpoint(filepath='/home/francesco/test/weights_{epoch:02d}_{val_loss:.2f}.h5'),
                              tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
                        ],
                        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                        validation_data=data_generator(X_val, y_val, BATCH_SIZE, nb_points=NB_POINTS),
                        validation_steps=X_val.shape[0] // BATCH_SIZE,
                        epochs=200)

if __name__ == "__main__":
    r_train()
