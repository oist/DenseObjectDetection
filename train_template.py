import argparse
import logging
from functools import partial

import tensorflow as tf

from segmentation import dataset
from segmentation.model import build_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


def train_data_generator():
    # Here generate your own training data : pair of 2D images in uint8 format
    # example :
    # while True:
    #    data = cv2.imread(my_train_data_image_path, cv2.IMREAD_GRAYSCALE)
    #    label = cv2.imread(my_train_label_image_path, cv2.IMREAD_GRAYSCALE)
    #    yield data, label
    raise NotImplementedError()


def eval_data_generator():
    # Same as train_data_generator but with evaluation data, should not loop if using steps=None in EvalSpec
    raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--num_classes', type=int, default=3, help="How many outputs of the model")
    parser.add_argument('--data_format', type=str, default='channels_last', choices={'channels_last', 'channels_first'})

    # training parameters
    parser.add_argument('--bg_fg_weight', type=float, default=0.9,
                        help="How much to weight the foreground objects against the background during training.")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training")
    parser.add_argument('--num_steps', type=int, default=5000, help="Number of training steps")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Save model to this path.")
    args = parser.parse_args()

    logger.info('Training network with settings: {}'.format(vars(args)))

    estimator = tf.estimator.Estimator(model_fn=partial(build_model,
                                                        num_classes=args.num_classes,
                                                        data_format=args.data_format,
                                                        bg_fg_weight=args.bg_fg_weight),
                                       model_dir=args.checkpoint_dir,
                                       config=tf.estimator.RunConfig(save_checkpoints_steps=100,
                                                                     save_summary_steps=100))

    train_spec = tf.estimator.TrainSpec(input_fn=partial(dataset.make_dataset,
                                                         data_generator=train_data_generator,
                                                         data_format=args.data_format,
                                                         batch_size=args.batch_size,
                                                         mode=tf.estimator.ModeKeys.TRAIN), max_steps=args.num_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=partial(dataset.make_dataset,
                                                       data_generator=eval_data_generator,
                                                       data_format=args.data_format,
                                                       batch_size=args.batch_size,
                                                       mode=tf.estimator.ModeKeys.EVAL), steps=None)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
