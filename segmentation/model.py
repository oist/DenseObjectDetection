import tensorflow as tf

from segmentation.unet import create_unet


def build_model(features, labels, num_classes, data_format, bg_fg_weight, mode):
    # The U-net network generates an output from the images data, network_output is the last layer of the network
    network_output = create_unet(features, num_classes, data_format)

    # Prediction step : return the network output, plus the input data to compare
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'prediction': network_output, 'input_data': features}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate the loss function by comparing network output and labels
    loss = add_loss(network_output, labels, data_format, num_classes=num_classes, weight=bg_fg_weight)

    # Evaluation step : output only the loss
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    # Training step : minimize the loss using an optimizer, outputs the loss too
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def add_loss(logits, labels, data_format, num_classes, weight):
    with tf.name_scope('loss'):
        # Convert the labels to separate binary channels : one for each class
        channels_axis = 1 if data_format == 'channels_first' else -1
        labels = tf.squeeze(labels, axis=channels_axis)
        oh_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes, name="one_hot", axis=channels_axis)

        # Compare the network output with the labels, which will produce the most probable class for each pixel
        loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                              labels=tf.stop_gradient(oh_labels),
                                                              dim=channels_axis)

        # Weigh the loss map
        weight_map = tf.where(tf.equal(labels, 0),
                              tf.fill(tf.shape(labels), 1 - weight),
                              tf.fill(tf.shape(labels), weight))
        weighted_loss = tf.multiply(loss_map, weight_map)

        # Average of the loss for the full batch
        loss = tf.reduce_mean(weighted_loss, name="weighted_loss")
        tf.losses.add_loss(loss)
        return loss
