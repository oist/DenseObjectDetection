import tensorflow as tf


def _create_conv_relu(inputs, data_format, name, filters, strides=1, kernel_size=3, padding='same'):
    return tf.layers.conv2d(inputs=inputs, filters=filters, strides=strides,
                            kernel_size=kernel_size, padding=padding, data_format=data_format,
                            name='{}_conv'.format(name), activation=tf.nn.relu)


def _create_pool(data, data_format, name, pool_size=2, strides=2):
    return tf.layers.max_pooling2d(inputs=data, pool_size=pool_size, strides=strides,
                                   padding='same', name=name, data_format=data_format)


def _contracting_path(data, data_format, num_layers, num_filters):
    interim = []

    dim_out = num_filters
    for i in range(num_layers):
        name = 'c_{}'.format(i)
        conv1 = _create_conv_relu(data, data_format, '{}_1'.format(name), dim_out)
        conv2 = _create_conv_relu(conv1, data_format, '{}_2'.format(name), dim_out)
        pool = _create_pool(conv2, data_format, name)
        data = pool

        dim_out *= 2
        interim.append(conv2)

    return interim, data


def _expansive_path(data, data_format, interim, num_layers, dim_in):
    dim_out = int(dim_in / 2)
    for i in range(num_layers):
        name = "e_{}".format(i)
        upconv = tf.layers.conv2d_transpose(data, filters=dim_out, kernel_size=2, strides=2,
                                            name='{}_upconv'.format(name), data_format=data_format, )

        channels_axis = 1 if data_format == 'channels_first' else -1
        concat = tf.concat([interim[len(interim) - i - 1], upconv], axis=channels_axis)
        conv1 = _create_conv_relu(concat, data_format, '{}_1'.format(name), dim_out)
        conv2 = _create_conv_relu(conv1, data_format, '{}_2'.format(name), dim_out)
        data = conv2
        dim_out = int(dim_out / 2)
    return data


def create_unet(data, num_classes, data_format, num_layers=3, num_filters=32):

    (interim, contracting_data) = _contracting_path(data, data_format, num_layers, num_filters)

    middle_dim = num_filters * (2 ** num_layers)
    middle_conv_1 = _create_conv_relu(contracting_data, data_format, 'm_1', middle_dim)
    middle_conv_2 = _create_conv_relu(middle_conv_1, data_format, 'm_2', middle_dim)
    middle_end = middle_conv_2

    expansive_path = _expansive_path(middle_end, data_format, interim, num_layers, middle_dim)

    conv_last = _create_conv_relu(expansive_path, data_format, 'final', num_classes)
    return conv_last
