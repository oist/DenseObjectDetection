from functools import partial

import tensorflow as tf
from tensorflow.python.data import Dataset


def make_dataset(data_generator, data_format, batch_size, mode):
    first = next(data_generator())

    if mode == tf.estimator.ModeKeys.PREDICT:
        types = tf.uint8
        shapes = first.shape
    else:
        types = (tf.uint8, tf.uint8)
        shapes = (first[0].shape, first[1].shape)

    dataset = Dataset.from_generator(data_generator, types, shapes)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.repeat()

    process_data_fn = process_image if mode == tf.estimator.ModeKeys.PREDICT else process_image_label
    dataset = dataset.map(partial(process_data_fn, data_format=data_format),
                          num_parallel_calls=64)

    dataset = dataset.batch(batch_size)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.map(partial(augment_data))

    dataset = dataset.prefetch(batch_size)

    return dataset


def process_image(image_data, data_format):

    image_data = tf.cast(image_data, dtype=tf.float32) / 255.
    image_data = image_data[:, :, tf.newaxis] if data_format == 'channels_last' else image_data[tf.newaxis, :, :]
    return image_data


def process_image_label(image_data, label_data, data_format):

    image_data = process_image(image_data, data_format)

    label_data = tf.cast(label_data, dtype=tf.float32)
    label_data = label_data[:, :, tf.newaxis] if data_format == 'channels_last' else label_data[tf.newaxis, :, :]
    return image_data, label_data


def augment_data(image_data, label_data):
    # random flip horizontally
    flip_cond = tf.less(tf.random_uniform([], 0, 1.0), .5)
    image_data, label_data = tf.cond(flip_cond,
                                     lambda: (
                                         tf.image.flip_left_right(image_data), tf.image.flip_left_right(label_data)),
                                     lambda: (image_data, label_data))

    # random rotation 90
    k = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32)
    image_data = tf.image.rot90(image_data, k=k)
    label_data = tf.image.rot90(label_data, k=k)

    return image_data, label_data
